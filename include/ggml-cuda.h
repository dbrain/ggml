#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

// Stream priority hints — relative, clamped to the device's
// cudaDeviceGetStreamPriorityRange. Passed to cudaStreamCreateWithPriority
// when streams are first created on the returned backend.
//   DEFAULT: same behavior as ggml_backend_cuda_init (cudaStreamCreateWithFlags).
//   LOW:     yield SM time to higher-priority streams on the same device
//            (e.g. async-vocoder backend yielding to the talker backend).
//   HIGH:    preempt LOW/DEFAULT streams on the same device.
// On devices with a single priority level the hints are no-ops.
enum ggml_cuda_stream_priority {
    GGML_CUDA_STREAM_PRIORITY_DEFAULT =  0,
    GGML_CUDA_STREAM_PRIORITY_LOW     =  1,
    GGML_CUDA_STREAM_PRIORITY_HIGH    = -1,
};

GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init_with_priority(int device, int priority);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

// True when `device` has Blackwell-class MMA (the same predicate the mul_mat /
// flash-attn dispatch uses to enable the FP4 cuBLASLt GEMM + cuDNN SDPA paths).
// Model graph builders call this to gate Blackwell-only optimizations at RUNTIME
// (e.g. the GGML_CUDNN_ATTN_F16_OUT F16 residual stream) so ONE binary renders
// correctly on both Blackwell and older GPUs (sm86 falls back instead of emitting
// an F16 island the native kernels can't consume -> solid-white output).
GGML_BACKEND_API bool ggml_backend_cuda_device_has_blackwell_mma(int device);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// VRAM-probe helper. Reports how many cudaGraph_t/cudaGraphExec_t entries
// are currently held in this backend's capture cache (one entry per unique
// first-node ptr / topology key). Returns 0/0 if cuda-graph support is
// disabled at build time. cudaGraph driver-side state is opaque, so the
// total cudaGraphExec memory consumption is not directly reportable — use
// `total_node_count` (sum of num_nodes across cached graphs) as a proxy.
// Pass NULL to skip an output field. Safe to call on non-CUDA backends
// (sets both to 0 and returns).
GGML_BACKEND_API void ggml_backend_cuda_get_graph_cache_stats(
    ggml_backend_t backend,
    int    * out_graph_count,
    size_t * out_total_node_count);

// Return the backend's committed CUDA memory-pool high-water back to the OS.
// The VMM pool (ggml_cuda_pool_vmm) only ever grows its physical commitment
// (set by the largest single transient ever seen) and unmaps solely in its
// destructor — so across a multi-segment video chain the pool's high-water
// stays reserved and shows up as cross-segment VRAM growth even though
// pool_used is 0 between segments. This destroys+lazily-rebuilds every
// device/stream pool on the backend, unmapping the committed blocks
// (cuMemUnmap + cuMemAddressFree). MUST be called only when no pool block is
// live (pool_used==0) and no kernel is in flight — it issues a device sync.
// Safe to call on non-CUDA backends (no-op). Cost: a one-shot re-commit on the
// next segment's first large alloc; never touches params/weights buffers.
GGML_BACKEND_API void ggml_backend_cuda_trim_pools(ggml_backend_t backend);

// Destroy cached cuDNN execution plans (fused SDPA + 3D conv), the current
// thread's cuDNN handles, and trim CUDA async mempools. Current cuDNN can return
// freed internal allocations to CUDA's default/current mempool instead of the
// driver; ggml_backend_cuda_trim_pools cannot reach that memory because it lives
// outside the ggml VMM pool. Call only from the CUDA worker thread, at a boundary
// with nothing in flight; handles/plans/pool pages rebuild lazily on next use.
// No-op on non-CUDA / non-cuDNN builds.
GGML_BACKEND_API void ggml_backend_cuda_release_cudnn_plans(void);

// Free the cuDNN conv2d reordered-weight buffers (raw cudaMalloc, keyed by weight ptr).
// Twin of the conv3d call below; live whenever the cuDNN 2D-conv path is enabled
// (GGML_CUDNN_CONV, or GGML_CUDNN_CONV3D with a conv2d-direct model). Call at a boundary
// where the params were re-offloaded / freed, else the buffers leak and a recycled weight
// address can stale-hit and return the WRONG reordered weights. Synchronizes first.
// No-op on non-CUDA / non-cuDNN builds.
GGML_BACKEND_API void ggml_backend_cuda_release_cudnn_conv2d_weights(void);

// Free the cuDNN conv3d reordered-weight buffers (raw cudaMalloc, keyed by weight ptr).
// Call at a segment boundary after VAE params are re-offloaded (their pointers are
// invalidated) to reclaim the ~1.4 GB/segment continuation leak. Synchronizes first.
// No-op on non-CUDA / non-cuDNN builds.
GGML_BACKEND_API void ggml_backend_cuda_release_cudnn_conv3d_weights(void);

// Register a per-tensor NVFP4 weight global scale (ModelOpt weight_scale_2), keyed by
// tensor name. The FP4 cuBLASLt GEMM folds it into the matmul alpha so the stored
// per-block ue4m3 scales can keep their well-conditioned range (UNFOLDED import)
// instead of underflowing into e4m3 subnormals. Names not registered default to a
// multiplier of 1.0 (legacy FOLDED gguf path stays byte-identical). No-op on
// non-CUDA / non-FP4 builds.
GGML_BACKEND_API void ggml_cuda_nvfp4_register_weight_global(const char * name, float g);
GGML_BACKEND_API void ggml_cuda_nvfp4_clear_weight_globals(void);

// LongCat lap-31.2: CPU-precomputed per-(Q-tile, K-tile) all-deny bitmap for the
// avatar's BSA self-attn. Set the device pointer + dimensions before issuing FA
// calls that should consult it; pass nullptr to disable. The FA dispatcher only
// uses the bitmap when DKQ=DV=128, ncols=64, ncols2=1 (the avatar's hot shape) AND
// the FA op carries a non-null mask AND the bitmap is set — every other FA caller
// is unaffected. Stored as one packed uint32 word per 32 K-tiles, n_qtiles rows,
// row stride = n_kwords words (so bitmap[jt * n_kwords + word_idx]).
GGML_BACKEND_API void ggml_cuda_set_longcat_fa_bsa_bitmap(const void * device_bitmap_u32,
                                                          int n_qtiles, int n_kwords);

// WAN SLA (lightx2v sparse-attention port): enable the FA K-tile bitmap skip on the
// MASK-FREE self-attn path (mask_h == nullptr). Wan2.2 self-attention is dense/maskless;
// carrying an N×N -INF mask only to gate the skip would be ~10 GB at 81f. With this on,
// the (tiny) bitmap set above is the sole sparsity artifact. Pass enabled=0 to restore the
// legacy avatar behaviour (skip requires a real mask). `n_ktiles` scopes the bitmap to FA
// calls whose K-tile count matches (so the self-attn bitmap is NOT applied to the shorter
// cross-attention in the same graph); pass the self-attn ceil(L_k/64).
GGML_BACKEND_API void ggml_cuda_set_longcat_fa_bsa_mask_free(int enabled, int n_ktiles);

// WAN SLA Stage 1 (per-head): when n_heads > 0 the bitmap set above is read as
// [n_heads, n_qtiles, n_kwords] and the kernel offsets its per-Q-tile slice by the
// CTA's Q head (ncols2==1 self-attn: one head per CTA), so each head selects its own
// K-blocks. Pass 0 to restore the single shared bitmap (Stage 0 / avatar) layout.
GGML_BACKEND_API void ggml_cuda_set_longcat_fa_bsa_n_heads(int n_heads);

// FP8 FFN activation-quant reuse cache: bump the generation once per graph
// compute so the cache (which reuses the e4m3-quantized activation across the
// q/k/v Linears that share it) can never serve a stale buffer from a prior
// compute. Cheap relaxed atomic; no-op effect on non-FP8 paths. Call right
// before issuing the graph's ops (execute_graph). Off-switch for the cache
// itself is GGML_FP8_ACT_QUANT_CACHE=0 (default on).
GGML_BACKEND_API void ggml_cuda_fp8_act_cache_new_generation(void);

// Determinism probe: cuBLASLt fast-path vs silent-fallback counters. A bail means the caller
// silently computed DIFFERENT MATH via MMQ/dequant, so a count that VARIES across two
// bit-identical forwards is an intermittent path flip rather than a wobbly kernel.
GGML_BACKEND_API void ggml_cuda_fp8_pathstats(unsigned long long * ok,
                                              unsigned long long * bail_peek,
                                              unsigned long long * bail_other);

// Determinism probe (GGML_FP8_INHASH=1): order-independent hash of the exact weight/activation
// bytes fed to every cuBLASLt FP8 GEMM in a forward. Read+reset once per forward. If whash
// moves across bit-identical forwards, the offload is delivering unstable weight bytes.
GGML_BACKEND_API void ggml_cuda_fp8_inhash_read_reset(unsigned long long * whash,
                                                      unsigned long long * ahash);

// Per-call-index bisection: input/output hash of the first N FP8 GEMMs of a forward, in call
// order. The first index whose input is stable but whose output moves is the culprit GEMM.
GGML_BACKEND_API void ggml_cuda_fp8_idxhash_read_reset(unsigned long long * in_out,
                                                       unsigned long long * out_out,
                                                       int n);

// First GEMM (by call order) whose (raw|qin|scale|out) differs from the PREVIOUS forward, and
// which field moved first. Covers every call, so no window guessing.
GGML_BACKEND_API void ggml_cuda_fp8_first_divergent_gemm(int * out_slot, const char ** out_field,
                                                         int * out_ncalls);

// Which Linear (name + M/K/N) a given per-call-index slot corresponds to.
GGML_BACKEND_API const char * ggml_cuda_fp8_idxname(int slot, int * M, int * K, int * N);
// The per-tensor activation scale that slot's quant actually used.
GGML_BACKEND_API float ggml_cuda_fp8_idxscale(int slot);
// Hash of the RAW (pre-quant) src1 bytes per slot: separates "upstream divergence" from
// "the quantization itself is nondeterministic".
GGML_BACKEND_API void ggml_cuda_fp8_idxraw_read_reset(unsigned long long * raw_out, int n);

// Emit the owned FP8 activation-quant cache allocation for VRAM accounting.
// This is diagnostic-only; callers should gate it with their own trace env.
GGML_BACKEND_API void ggml_cuda_fp8_act_cache_log_stats(const char * phase);

// FP8 FFN e4m3 WEIGHT cache (GGML_FP8_WEIGHT_QUANT_CACHE=1, budget
// GGML_FP8_WEIGHT_CACHE_MB): frees all cached e4m3 weight buffers. Optional —
// the cache otherwise persists until process exit. Call at render/model teardown
// if the ~budget of VRAM should be reclaimed between renders.
GGML_BACKEND_API void ggml_cuda_fp8_weight_cache_clear(void);

#ifdef  __cplusplus
}
#endif
