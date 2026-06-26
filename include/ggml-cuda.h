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

// Register a per-tensor NVFP4 weight global scale (ModelOpt weight_scale_2), keyed by
// tensor name. The FP4 cuBLASLt GEMM folds it into the matmul alpha so the stored
// per-block ue4m3 scales can keep their well-conditioned range (UNFOLDED import)
// instead of underflowing into e4m3 subnormals. Names not registered default to a
// multiplier of 1.0 (legacy FOLDED gguf path stays byte-identical). No-op on
// non-CUDA / non-FP4 builds.
GGML_BACKEND_API void ggml_cuda_nvfp4_register_weight_global(const char * name, float g);

// LongCat lap-31.2: CPU-precomputed per-(Q-tile, K-tile) all-deny bitmap for the
// avatar's BSA self-attn. Set the device pointer + dimensions before issuing FA
// calls that should consult it; pass nullptr to disable. The FA dispatcher only
// uses the bitmap when DKQ=DV=128, ncols=64, ncols2=1 (the avatar's hot shape) AND
// the FA op carries a non-null mask AND the bitmap is set — every other FA caller
// is unaffected. Stored as one packed uint32 word per 32 K-tiles, n_qtiles rows,
// row stride = n_kwords words (so bitmap[jt * n_kwords + word_idx]).
GGML_BACKEND_API void ggml_cuda_set_longcat_fa_bsa_bitmap(const void * device_bitmap_u32,
                                                          int n_qtiles, int n_kwords);

#ifdef  __cplusplus
}
#endif
