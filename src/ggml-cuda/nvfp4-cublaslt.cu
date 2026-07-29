// Phase-1 fast FP4 GEMM via cuBLASLt — see nvfp4-cublaslt.cuh.
//
// Convention (proven in flux2.cpp/spike_cutlass_fp4/nvfp4_repack_golden.cu, cosine=1.0):
//   ggml block_nvfp4 decodes E2M1 nibbles via kvalues_mxfp4={0,1,2,3,4,6,8,12} (2x std
//   E2M1) and UE4M3 scales via /2 (ggml_ue4m3_to_fp32). The two 2x factors cancel, so
//   the stored bytes ARE the standard-e4m3 * standard-e2m1 values cuBLASLt expects.
//   => alpha = 1.0, weight bytes reused verbatim (only re-laid-out).
//
// Repacks (one-time per weight tensor, cached by device pointer):
//   - scales: contiguous block_nvfp4.d[4] -> cuBLASLt SWIZZLE_32_4_4 layout.
//   - nibbles: ggml (elem j low / j+8 high within each 16-sub) -> consecutive (2j,2j+1).
// Activation (src1 f32) is quantized per matmul into the same cuBLASLt layout.

#include "nvfp4-cublaslt.cuh"

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <atomic>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <map>
#include <string>
#include <cstring>
#include <tuple>

// SWIZZLE_32_4_4 offset over a (rows x col_length) UE4M3 scale grid (comfy float_utils)
__device__ __forceinline__ size_t swz_off(size_t row, size_t col, uint32_t col_len) {
    const uint32_t R=128, RC=32, CC=4;
    size_t rb = row/R, rem = row%R, d4 = rem/RC, d3 = rem%RC;
    size_t cbg = col/CC, d5 = col%CC;
    size_t cbg_cnt = (col_len + CC - 1)/CC;
    return ((rb*cbg_cnt + cbg)*RC + d3)*16 + d4*CC + d5;
}

// --- weight repack: one thread per (row r in [0,N), sub ss in [0,nsub)) ---
// reorders nibbles to consecutive (8 bytes) and swizzles the UE4M3 scale byte.
static __global__ void repack_weight_kernel(const block_nvfp4 * __restrict__ W,
                                            uint8_t * __restrict__ out_data,    // N*(K/2) bytes
                                            uint8_t * __restrict__ out_scales,  // swizzled
                                            int N, int K) {
    const int nsub = K/16, nblk = K/64;
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= N*nsub) return;
    const int r  = idx / nsub;
    const int ss = idx % nsub;
    const int block = ss/4, s = ss%4;
    const block_nvfp4 & b = W[(size_t)r*nblk + block];
    out_scales[swz_off(r, ss, nsub)] = b.d[s];                 // std-e4m3 byte, verbatim
    // ggml qs for this 16-sub: qs[s*8 + (local%8)], low=local<8, high=local>=8
    const uint8_t * qs = &b.qs[s*8];
    uint8_t * od = &out_data[(size_t)r*(K/2) + ss*8];
    #pragma unroll
    for (int t=0;t<8;t++) {
        int e0 = 2*t, e1 = 2*t+1;
        uint8_t n0 = (e0<8) ? (qs[e0]&0xF) : (qs[e0-8]>>4);
        uint8_t n1 = (e1<8) ? (qs[e1]&0xF) : (qs[e1-8]>>4);
        od[t] = n0 | (n1<<4);
    }
}

// --- activation quant: one thread per (row r in [0,M), sub ss in [0,nsub)) ---
// Mirrors quantize_mmq_nvfp4 EXACTLY (same ggml helpers + ±1/±2 scale-refinement
// search) so the cuBLASLt activation is bit-identical to the MMQ path; only the
// nibble *packing* differs (consecutive for cuBLASLt vs MMQ tile layout). This
// keeps cuBLASLt output as close to MMQ as the GEMM kernels themselves allow.
//
// Templated on the activation element type so the DiT residual stream can flow
// in F16 (LTX_DIT_F16) straight into the FP4 GEMM with NO per-Linear F16->F32
// cast (stage 1 of the beat-comfy plan). The activation is quantized to E2M1
// regardless, so the input element precision barely affects quality; the amax /
// ±scale-refine math runs in float exactly as the F32 path does.
__device__ __forceinline__ float nvfp4_load_act(const float & v) { return v; }
__device__ __forceinline__ float nvfp4_load_act(const half  & v) { return __half2float(v); }

// REFINE=true runs the ggml-MMQ ±2 scale-refinement search (5 candidate scale codes,
// each re-quantizing the 16-lane sub-block and keeping the min-reconstruction-error one).
// REFINE=false is comfy/ModelOpt's native one-shot scaled_mm quantization: scale =
// ue4m3(amax/6), quantize once, no search. The DiT runs 100% on cuBLASLt (no MMQ to match)
// and comfy — our quality target — ships the one-shot path, so REFINE=false is both faster
// (~5x less quant work) and quantization-equivalent to the reference. Gated by
// GGML_NVFP4_QUANT_NOREFINE; default keeps the refinement so flux2/prod are byte-untouched.
// Per-tensor activation amax (max|x| over the whole M*K matrix) for comfy/ModelOpt's
// two-level NVFP4 scale. atomicMax on the IEEE bit pattern is valid because the values
// are non-negative (positive-float bit patterns are monotonic). Result reported as the
// raw uint bits of the max float.
template <typename act_t>
static __global__ void nvfp4_amax_kernel(const act_t * __restrict__ X,
                                         unsigned int * __restrict__ out_bits, size_t n) {
    float local = 0.f;
    for (size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x*blockDim.x) {
        local = fmaxf(local, fabsf(nvfp4_load_act(X[i])));
    }
    __shared__ float s[256];
    s[threadIdx.x] = local; __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (threadIdx.x < st) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x+st]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(out_bits, __float_as_uint(s[0]));
}

// per_tensor > 0 selects comfy/ModelOpt's TWO-LEVEL one-shot quant (see below); per_tensor
// <= 0 keeps the original single-level path (REFINE search or one-shot).
template <typename act_t, bool REFINE>
static __global__ void quant_act_kernel(const act_t * __restrict__ X,
                                        uint8_t * __restrict__ out_data,    // M*(K/2)
                                        uint8_t * __restrict__ out_scales,  // swizzled
                                        int M, int K, float per_tensor) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int nsub = K/16;
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= M*nsub) return;
    const int r  = idx / nsub;
    const int ss = idx % nsub;
    const act_t * x = &X[(size_t)r*K + ss*16];
    float vals[16], amax = 0.f;
    #pragma unroll
    for (int j=0;j<16;j++) { vals[j]=nvfp4_load_act(x[j]); amax = fmaxf(amax, fabsf(vals[j])); }

    // comfy/ModelOpt TWO-LEVEL one-shot: the per-block scale (amax/6) is normalized by
    // the per-tensor global before being quantized to e4m3, so the e4m3 block-scale code
    // stays in its well-conditioned range (un-normalized amax/6 for low-magnitude blocks
    // lands in e4m3 subnormals -> coarse rounding -> the artifact the ±2 REFINE search was
    // papering over). The stored code is the NORMALIZED e4m3; the per_tensor factor is
    // carried by the cuBLASLt GEMM alpha (our weights already fold their own global into
    // the block scale, so alpha = A_per_tensor only). At per_tensor==1 this is byte-
    // identical to the single-level one-shot below.
    if (per_tensor > 0.f) {
        int tc = (int) ggml_cuda_fp32_to_ue4m3((amax/6.0f)/per_tensor);
        tc = tc < 0 ? 0 : (tc > 0x7e ? 0x7e : tc);
        const uint8_t code = (uint8_t)tc;
        out_scales[swz_off(r, ss, nsub)] = code;
        const float total = per_tensor * ggml_cuda_ue4m3_to_fp32(code);
        const float inv_scale = total > 0.f ? 0.5f/total : 0.f;
        uint8_t * od = &out_data[(size_t)r*(K/2) + ss*8];
        #pragma unroll
        for (int t=0;t<8;t++) {
            uint8_t n0 = ggml_cuda_float_to_fp4_e2m1(vals[2*t],   inv_scale);
            uint8_t n1 = ggml_cuda_float_to_fp4_e2m1(vals[2*t+1], inv_scale);
            od[t] = (n0 & 0xF) | ((n1 & 0xF)<<4);
        }
        return;
    }

    const int first_code = (int) ggml_cuda_fp32_to_ue4m3(amax/6.0f);
    uint8_t fp8_code; float subblock_scale;
    if (REFINE) {
        static constexpr int test_offsets[5] = {0,-1,1,-2,2};
        float best_err = FLT_MAX; fp8_code = 0; subblock_scale = 0.f;
        #pragma unroll
        for (int i=0;i<5;i++) {
            const int tc = first_code + test_offsets[i];
            if (tc < 0 || tc > 0x7e) continue;
            const float ts = ggml_cuda_ue4m3_to_fp32((uint8_t)tc);
            const float tinv = ts > 0.f ? 0.5f/ts : 0.f;
            float err = 0.f;
            #pragma unroll
            for (int k=0;k<16;k++) {
                const uint8_t q = ggml_cuda_float_to_fp4_e2m1(vals[k], tinv);
                const float ed = fabsf(vals[k]) - fabsf(kvalues_mxfp4[q & 0x7]) * ts;
                err = fmaf(ed, ed, err);
            }
            if (err < best_err) { best_err = err; fp8_code = (uint8_t)tc; subblock_scale = ts; }
        }
    } else {
        const int tc = first_code < 0 ? 0 : (first_code > 0x7e ? 0x7e : first_code);
        fp8_code = (uint8_t)tc;
        subblock_scale = ggml_cuda_ue4m3_to_fp32((uint8_t)tc);
    }
    out_scales[swz_off(r, ss, nsub)] = fp8_code;
    const float inv_scale = subblock_scale > 0.f ? 0.5f/subblock_scale : 0.f;
    uint8_t * od = &out_data[(size_t)r*(K/2) + ss*8];   // consecutive packing for cuBLASLt
    #pragma unroll
    for (int t=0;t<8;t++) {
        uint8_t n0 = ggml_cuda_float_to_fp4_e2m1(vals[2*t],   inv_scale);
        uint8_t n1 = ggml_cuda_float_to_fp4_e2m1(vals[2*t+1], inv_scale);
        od[t] = (n0 & 0xF) | ((n1 & 0xF)<<4);
    }
#else
    NO_DEVICE_CODE;
#endif
}

// ---------------- host glue ----------------

bool ggml_cuda_nvfp4_cublaslt_enabled() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("GGML_NVFP4_CUBLASLT"); v = (e && atoi(e)) ? 1 : 0; }
    return v == 1;
}

// Per-tensor weight global (ModelOpt weight_scale_2) of an UNFOLDED NVFP4 import,
// keyed by tensor NAME (not data pointer: a layer-offload/staging host swaps src0->data
// and src0->buffer per segment, but preserves src0->name).
// Folded into the GEMM alpha (alpha = A_global * W_global) so the scalar costs nothing
// instead of a full-size ggml_mul over every Linear output.
//
// A name that was never registered yields 1.0. That is byte-identical for a legacy
// FOLDED gguf (which has no globals at all), but it would be SILENTLY WRONG for an
// unfolded one — so the graph-level fallback MUL is only elided when
// ggml_cuda_nvfp4_weight_global_registered() says the scalar is really here
// (see ggml_cuda_nvfp4_weight_global_folded() in ggml-cuda.cu).
//
// Read concurrently: the layer-offload path runs GEMMs on worker threads, so every
// access is under g_wglobal_mtx (registration/clear only happen at load/swap, never
// concurrently with a graph).
static std::mutex                             g_wglobal_mtx;
static std::unordered_map<std::string, float> g_wglobal;

extern "C" void ggml_cuda_nvfp4_register_weight_global(const char * name, float g) {
    if (!name || !*name) return;
    std::lock_guard<std::mutex> lk(g_wglobal_mtx);
    g_wglobal[name] = g;
}

// Drop every registered weight global. Required before re-registering for a hot-swapped
// DiT variant (sd_ctx_swap_diffusion_model): this map is process-global, so swapping an
// UNFOLDED gguf out for a FOLDED one (or for a different unfolded one) must not leave the
// outgoing model's globals behind — a folded gguf already folds its global into the block
// scale, so a stale entry would scale it a second time.
void ggml_cuda_fp8_weight_scale_cache_clear(void);   // defined with the FP8 path below

extern "C" void ggml_cuda_nvfp4_clear_weight_globals(void) {
    {
        std::lock_guard<std::mutex> lk(g_wglobal_mtx);
        g_wglobal.clear();
    }
    // The FP8 e4m3 weight scale folds w_global in, so it is only valid for the set of globals
    // being dropped here. Same hook, same lifetime.
    ggml_cuda_fp8_weight_scale_cache_clear();
}

bool ggml_cuda_nvfp4_weight_global_registered(const char * name) {
    if (!name || !*name) return false;
    std::lock_guard<std::mutex> lk(g_wglobal_mtx);
    return g_wglobal.find(name) != g_wglobal.end();
}

static float nvfp4_weight_global_for(const char * name) {
    if (!name || !*name) return 1.0f;
    std::lock_guard<std::mutex> lk(g_wglobal_mtx);
    auto it = g_wglobal.find(name);
    return it == g_wglobal.end() ? 1.0f : it->second;
}

struct nvfp4_weight_repacked {
    uint8_t * data   = nullptr;   // consecutive E2M1   (in-place: offset 0 of src0->data)
    uint8_t * scales = nullptr;   // swizzled UE4M3      (in-place: offset data_bytes)
    bool      in_place = false;   // true => data/scales alias the ggml-owned src0 buffer
                                  //         (teardown must NOT cudaFree them)
};

static std::mutex                                   g_repack_mtx;
static std::unordered_map<const void*, nvfp4_weight_repacked> g_repack_cache;

// in-place repack default-ON; escape hatch GGML_NVFP4_CUBLASLT_INPLACE=0 forces the
// old out-of-place (duplicate-buffer) path for debugging / fallback.
static bool nvfp4_inplace_enabled() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("GGML_NVFP4_CUBLASLT_INPLACE"); v = (e && atoi(e)==0) ? 0 : 1; }
    return v == 1;
}

static thread_local cublasLtHandle_t g_lt = nullptr;
static cublasLtHandle_t get_lt() {
    if (!g_lt) { if (cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) return nullptr; }
    return g_lt;
}

// per-tensor weight repack (cached). Returns false on failure.
static bool get_repacked_weight(const ggml_tensor * src0, int N, int K, cudaStream_t stream,
                                nvfp4_weight_repacked & out) {
    const void * key = src0->data;
    // Hold the lock across the whole (one-time, warmup) repack: the in-place path overwrites
    // src0->data, so two threads repacking the SAME key concurrently would corrupt each other.
    // Cached entries (the hot per-step path) return early in the caller-visible fast path below.
    std::lock_guard<std::mutex> lk(g_repack_mtx);
    {
        auto it = g_repack_cache.find(key);
        if (it != g_repack_cache.end()) { out = it->second; return true; }
    }
    const int nsub = K/16;
    const size_t data_bytes  = (size_t)N*(K/2);
    const size_t rb_p = ((size_t)(N+127)/128)*128;
    const size_t cb_p = ((size_t)(nsub+3)/4)*4;
    const size_t scale_bytes = rb_p*cb_p;
    const size_t repack_bytes = data_bytes + scale_bytes;

    // The DiT runs 100% on cuBLASLt (no MMQ fallback), so once a weight is repacked its
    // original block_nvfp4 layout is never read again -> the repacked bytes can live in the
    // ggml-owned buffer, eliminating the duplicate. Repack is a re-layout of the SAME values,
    // so data+scales should fit in ggml_nbytes(src0); VERIFY at runtime per-weight.
    const size_t orig_bytes = ggml_nbytes(src0);
    const bool   fits = (repack_bytes <= orig_bytes);
    const bool   want_inplace = nvfp4_inplace_enabled() && fits;

    nvfp4_weight_repacked rp;
    const int threads = 256;
    const int total   = N*nsub;

    if (want_inplace) {
        // read-after-write hazard (repack permutes both nibbles & scales) forces a transient
        // scratch: original -> scratch, then scratch -> src0->data in-place. The scratch is
        // freed immediately (never accumulates), so net VRAM cost over baseline is ~0.
        uint8_t * scratch = nullptr;
        if (cudaMalloc(&scratch, repack_bytes) != cudaSuccess) return false; // OOM: caller falls back
        uint8_t * s_data   = scratch;
        uint8_t * s_scales = scratch + data_bytes;
        cudaMemsetAsync(s_scales, 0, scale_bytes, stream);
        repack_weight_kernel<<<(total+threads-1)/threads, threads, 0, stream>>>(
            (const block_nvfp4*)src0->data, s_data, s_scales, N, K);
        if (cudaPeekAtLastError() != cudaSuccess) { cudaFree(scratch); return false; }
        // copy repacked bytes back into the ORIGINAL ggml buffer (data@0, scales@data_bytes).
        cudaMemcpyAsync(src0->data, scratch, repack_bytes, cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);   // must finish before scratch is freed
        cudaFree(scratch);
        rp.data     = (uint8_t*)src0->data;
        rp.scales   = (uint8_t*)src0->data + data_bytes;
        rp.in_place = true;
    } else {
        // out-of-place fallback (env-forced, or repack doesn't fit the original buffer):
        // hold a duplicate persistent buffer for this weight.
        if (cudaMalloc(&rp.data, data_bytes)   != cudaSuccess) return false;
        if (cudaMalloc(&rp.scales, scale_bytes)!= cudaSuccess) { cudaFree(rp.data); return false; }
        cudaMemsetAsync(rp.scales, 0, scale_bytes, stream);
        repack_weight_kernel<<<(total+threads-1)/threads, threads, 0, stream>>>(
            (const block_nvfp4*)src0->data, rp.data, rp.scales, N, K);
        if (cudaPeekAtLastError() != cudaSuccess) { cudaFree(rp.data); cudaFree(rp.scales); return false; }
        cudaStreamSynchronize(stream);
        rp.in_place = false;
        if (getenv("GGML_NVFP4_CUBLASLT_TRACE"))
            fprintf(stderr, "[NVFP4_CUBLASLT] out-of-place weight N=%d K=%d (repack %zu > orig %zu, "
                            "or inplace disabled)\n", N, K, repack_bytes, orig_bytes);
    }
    g_repack_cache[key] = rp;   // lock held for the whole function (see top)
    out = rp;
    return true;
}

// ---------------------------------------------------------------------------
// NVFP4 activation-quant reuse cache (same design as the FP8 sibling below —
// fp8_act_quant_cache, ~line 950 — read that header for the full stale-safety argument).
//
// Krea2's KreaAttention calls wq/wk/wv/gate ->forward(ctx, x) on ONE ggml_tensor*, and
// KreaSwiGLU calls gate/up on another. So per block 8 FP4 GEMMs quantize only 4 DISTINCT
// activations: nvfp4_amax_kernel + quant_act_kernel each run twice as often as there are
// distinct results (measured 1134 ms at 1024^2, 4474 ms at 2048^2, purely duplicated work).
// Quantize once, reuse for any later GEMM in the SAME compute whose src1 is the same tensor.
//
// THREE things are cached, because all three feed the GEMM:
//   - d_data   : the packed E2M1 nibbles (M*(K/2) bytes)
//   - d_scales : the SWIZZLE_32_4_4 UE4M3 block scales (padded, memset-0 tail included)
//   - per_tensor: the HOST float from the two-level amax readback. It goes into `alpha`,
//                 so a hit that recomputed it would also have to redo the D2H sync; caching
//                 it makes the hit skip the amax kernel + sync entirely. Its value is a pure
//                 function of the same activation bytes, so replaying it is bit-identical.
//
// TRAP 1 (differs from the FP8 sibling): the FP8 budget check compares an ELEMENT count
// (n_act = M*K) against a BYTE budget. For e4m3 those coincide. For NVFP4 they do NOT — the
// packed payload is M*(K/2) bytes plus a padded scale plane — so copying that check verbatim
// would silently stop caching at roughly a quarter of the intended size (~1024^2 here).
// We count ACTUAL BYTES (a_data_bytes + a_scale_bytes) against the budget.
//
// TRAP 2: ggml_cuda_pool_alloc is a LIFO stack allocator (ggml-cuda.cu:682 asserts frees are
// the exact reverse of allocs), so a pool buffer can NEVER be held across other GEMM calls.
// Both cached buffers are OWNED cudaMalloc allocations (grow-only), exactly like the FP8 one.
//
// BIT-EXACTNESS: repack + quant are deterministic (atomicMax amax -> ue4m3 scale codes ->
// per-element E2M1 round), so a hit is byte-identical to a miss by construction. The key is
// deliberately stricter than needed: generation + src1 NODE pointer + data pointer + ne0/ne1
// + nb1 + type + device must ALL match. Inherited assumption (shared with the FP8 sibling):
// a graph must not clobber src1's bytes in place while src1 still has a pending consumer —
// that would already be a malformed graph, but it is the one way a hit could go stale.
//
// Default OFF (GGML_NVFP4_ACT_QUANT_CACHE=1 to enable) — mirrors how the FP8 sibling is
// gated, and keeps every existing render byte-untouched unless the flag is set.
struct nvfp4_act_quant_cache {
    uint64_t            gen        = (uint64_t)-1;   // generation filled at; -1 == empty
    const ggml_tensor * node       = nullptr;        // src1 node identity (unique in a graph)
    const void *        data       = nullptr;
    int64_t             ne0        = 0;
    int64_t             ne1        = 0;
    size_t              nb1        = 0;
    int                 type       = -1;
    int                 device     = -1;
    uint8_t *           d_data     = nullptr;        // owned packed-E2M1 buffer
    size_t              cap_data   = 0;
    uint8_t *           d_scales   = nullptr;        // owned swizzled-UE4M3 scale buffer
    size_t              cap_scales = 0;
    float               per_tensor = 0.f;            // host two-level global (feeds alpha)
};
static thread_local nvfp4_act_quant_cache g_nvfp4_act_cache;
// Bumped from ggml_cuda_fp8_act_cache_new_generation() (below), which
// ggml_backend_cuda_graph_compute() already calls once per graph compute. A hit requires the
// stored generation to equal the current one, so a new compute can never reuse a previous
// compute's buffer even if gallocr recycles a node/data address.
static std::atomic<uint64_t> g_nvfp4_act_cache_gen{0};

static bool ggml_cuda_nvfp4_act_cache_enabled() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_NVFP4_ACT_QUANT_CACHE"); v = (e && atoi(e)) ? 1 : 0; }
    return v == 1;
}
static size_t ggml_cuda_nvfp4_act_cache_budget_bytes() {
    static size_t b = 0; static int init = 0;
    if (!init) {
        const char * e = getenv("GGML_NVFP4_ACT_CACHE_MB");
        long mb = (e && *e) ? atol(e) : 128;
        if (mb < 0) mb = 0;
        b = (size_t)mb * 1024 * 1024;
        init = 1;
    }
    return b;
}

bool ggml_cuda_nvfp4_cublaslt_mul_mat(ggml_backend_cuda_context & ctx,
                                      const ggml_tensor * src0,
                                      const ggml_tensor * src1,
                                      ggml_tensor * dst) {
    // only the simple 2D linear-layer case (NVFP4 weight, f32 act, f32 out)
    static int dbgn = 0;
    const bool dbg = getenv("GGML_NVFP4_CUBLASLT_TRACE") && dbgn < 12;
    if (dbg) { dbgn++;
        fprintf(stderr,"[cublaslt-try] s0 t=%d ne=[%ld,%ld,%ld,%ld] cont=%d hostbuf=%d | s1 t=%d cont=%d | dst t=%d cont=%d\n",
          (int)src0->type,(long)src0->ne[0],(long)src0->ne[1],(long)src0->ne[2],(long)src0->ne[3],
          ggml_is_contiguous(src0), src0->buffer?ggml_backend_buffer_is_host(src0->buffer):-1,
          (int)src1->type,ggml_is_contiguous(src1),(int)dst->type,ggml_is_contiguous(dst)); }
    // Accept F32 OR F16 activations: the FP4 GEMM quantizes the activation to E2M1
    // anyway, so feeding the F16 residual stream (LTX_DIT_F16) keeps the matmul on
    // the fast tensor-core path instead of forcing a per-Linear F16->F32 cast.
    // Accept F32 OR F16 dst: cuBLASLt accumulates in F32 and can store F16 directly
    // (stage 2 — F16 Linear output so the residual/glue stays pure-F16 half-width).
    if (src0->type != GGML_TYPE_NVFP4)
        return false;
    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16)
        return false;
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16)
        return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1)
        return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst))
        return false;

    const int K = src0->ne[0];   // contraction
    const int N = src0->ne[1];   // out features
    const int M = src1->ne[1];   // tokens
    if (src1->ne[0] != K) return false;
    if (K % 64 != 0)      return false;   // need full 64-elem blocks
    if (dst->ne[0] != N || dst->ne[1] != M) return false;

    cudaStream_t stream = ctx.stream();
    cublasLtHandle_t lt = get_lt();
    if (!lt) return false;

    // 1) repack weight into TRANSIENT pool buffers, per-call (no cache, no in-place).
    // The prior cache-by-ptr + in-place path was unsafe: in-place corrupted the weight, and
    // out-of-place caching leaked one persistent buffer per (changing) offload-stream pointer
    // -> OOM. Repacking into the pool each call costs a cheap re-layout kernel (the activation
    // is already quantized per-call the same way) and works identically on resident & offload
    // weights with zero VRAM doubling.
    const size_t w_data_bytes  = (size_t)N*(K/2);
    const size_t w_rb_p = ((size_t)(N+127)/128)*128;
    const size_t w_cb_p = ((size_t)(K/16+3)/4)*4;
    const size_t w_scale_bytes = w_rb_p*w_cb_p;
    ggml_cuda_pool_alloc<uint8_t> w_data(ctx.pool(), w_data_bytes);
    ggml_cuda_pool_alloc<uint8_t> w_scales(ctx.pool(), w_scale_bytes);
    cudaMemsetAsync(w_scales.get(), 0, w_scale_bytes, stream);
    {
        const int threads = 256;
        const int total   = N*(K/16);
        repack_weight_kernel<<<(total+threads-1)/threads, threads, 0, stream>>>(
            (const block_nvfp4*)src0->data, w_data.get(), w_scales.get(), N, K);
        if (cudaPeekAtLastError() != cudaSuccess) return false;
    }

    // 2) quantize activation into cuBLASLt layout (pool scratch, or the owned reuse cache)
    const int nsub = K/16;
    const size_t a_data_bytes  = (size_t)M*(K/2);
    const size_t a_rb_p = ((size_t)(M+127)/128)*128;
    const size_t a_cb_p = ((size_t)(nsub+3)/4)*4;
    const size_t a_scale_bytes = a_rb_p*a_cb_p;

    // ------------------------------------------------------------------
    // 2a) Activation-quant reuse cache lookup (see nvfp4_act_quant_cache above).
    //     Resolved BEFORE any activation pool allocation so a hit skips them entirely and
    //     the LIFO pool order stays w_data -> w_scales -> [a_data] -> [a_scales] -> [amax] -> ws.
    // ------------------------------------------------------------------
    uint8_t * a_data_ptr   = nullptr;   // non-null => served by the owned cache buffers
    uint8_t * a_scales_ptr = nullptr;
    bool      act_reused   = false;     // true => the quantized bytes are already valid
    // Generation to PUBLISH on the entry, and only once the quant kernels for it have actually
    // been enqueued. Until then the entry stays marked empty, so any early return between the
    // miss and the quant cannot leave a later call reusing un-filled bytes.
    uint64_t  act_publish_gen = (uint64_t)-1;
    float     cached_per_tensor = 0.f;

    // While a CUDA graph is being captured, cudaMalloc / D2H copies / stream syncs are all
    // illegal, and a captured graph replays WITHOUT re-running any of this host code — so a
    // persistent pointer or a cached decision baked in at capture time would be replayed
    // blindly. Detect capture once and take the plain per-call path for everything.
    bool nv_capturing = false;
#ifdef USE_CUDA_GRAPH
    {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
            nv_capturing = true;
        }
    }
#endif

    // TRAP 1: budget the ACTUAL BYTES this activation occupies (packed nibbles + padded
    // swizzled scale plane), not an element count. See the struct header.
    const size_t a_cache_bytes = a_data_bytes + a_scale_bytes;
    bool act_cache_use = ggml_cuda_nvfp4_act_cache_enabled() && !nv_capturing &&
                         a_cache_bytes <= ggml_cuda_nvfp4_act_cache_budget_bytes();
    if (act_cache_use) {
        nvfp4_act_quant_cache & C = g_nvfp4_act_cache;
        const uint64_t cur = g_nvfp4_act_cache_gen.load(std::memory_order_relaxed);
        const bool hit = C.gen == cur && C.node == src1 && C.data == src1->data &&
                         C.ne0 == src1->ne[0] && C.ne1 == src1->ne[1] &&
                         C.nb1 == src1->nb[1] && C.type == (int)src1->type &&
                         C.device == ctx.device && C.d_data != nullptr &&
                         C.d_scales != nullptr && C.cap_data >= a_data_bytes &&
                         C.cap_scales >= a_scale_bytes;
        if (hit) {
            a_data_ptr        = C.d_data;
            a_scales_ptr      = C.d_scales;
            cached_per_tensor = C.per_tensor;
            act_reused        = true;
        } else {
            // MISS: (re)quantize into the OWNED persistent buffers (grow-only). cudaFree is
            // device-synchronizing, so any in-flight GEMM still reading the old buffer has
            // retired before it is released; growth only happens on the first calls.
            if (C.device != ctx.device && (C.d_data != nullptr || C.d_scales != nullptr)) {
                if (C.d_data)   { cudaFree(C.d_data);   C.d_data   = nullptr; C.cap_data   = 0; }
                if (C.d_scales) { cudaFree(C.d_scales); C.d_scales = nullptr; C.cap_scales = 0; }
            }
            if (C.cap_data < a_data_bytes) {
                if (C.d_data != nullptr) cudaFree(C.d_data);
                if (cudaMalloc((void**)&C.d_data, a_data_bytes) != cudaSuccess) { C.d_data = nullptr; C.cap_data = 0; }
                else                                                            C.cap_data = a_data_bytes;
            }
            if (C.cap_scales < a_scale_bytes) {
                if (C.d_scales != nullptr) cudaFree(C.d_scales);
                if (cudaMalloc((void**)&C.d_scales, a_scale_bytes) != cudaSuccess) { C.d_scales = nullptr; C.cap_scales = 0; }
                else                                                               C.cap_scales = a_scale_bytes;
            }
            if (C.d_data != nullptr && C.d_scales != nullptr) {
                a_data_ptr   = C.d_data;
                a_scales_ptr = C.d_scales;
                C.gen  = (uint64_t)-1;    // stays EMPTY until the quant is enqueued (below)
                C.node = src1;            C.data   = src1->data;
                C.ne0  = src1->ne[0];     C.ne1    = src1->ne[1];  C.nb1 = src1->nb[1];
                C.type = (int)src1->type; C.device = ctx.device;
                C.per_tensor = 0.f;
                act_publish_gen = cur;
            } else {
                C.gen = (uint64_t)-1; C.node = nullptr;   // alloc failed -> invalidate, use pool
                act_cache_use = false;
            }
        }
    }

    // Declared-but-unallocated when the cache serves the activation; the pool stays untouched
    // in that case and the destructors still unwind in reverse declaration order (LIFO-safe).
    ggml_cuda_pool_alloc<uint8_t> a_data(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> a_scales(ctx.pool());
    if (a_data_ptr == nullptr) {
        a_data_ptr   = a_data.alloc(a_data_bytes);
        a_scales_ptr = a_scales.alloc(a_scale_bytes);
    }
    if (!act_reused) {
        cudaMemsetAsync(a_scales_ptr, 0, a_scale_bytes, stream);
    }
    // TWO-LEVEL (comfy-faithful) one-shot: compute the per-tensor activation global
    // (amax / (6*448)) so block scales normalize into e4m3 range. Carried into the GEMM
    // alpha below. Gated GGML_NVFP4_QUANT_TWOLEVEL; supersedes NOREFINE when set.
    float a_per_tensor = 0.f;
    if (act_reused) {
        // Replaying the stored global is bit-identical: it is a pure function of the same
        // activation bytes, and skipping the amax kernel also skips its D2H stream sync.
        a_per_tensor = cached_per_tensor;
    } else {
        static int s_twolevel = -1;
        if (s_twolevel < 0) { const char* e = getenv("GGML_NVFP4_QUANT_TWOLEVEL"); s_twolevel = (e && atoi(e)) ? 1 : 0; }
        if (s_twolevel) {
            ggml_cuda_pool_alloc<unsigned int> amax_d(ctx.pool(), 1);
            cudaMemsetAsync(amax_d.get(), 0, sizeof(unsigned int), stream);
            const size_t n = (size_t)M*K;
            const int thr = 256;
            const int blk = (int)((n + thr - 1)/thr > 1024 ? 1024 : (n + thr - 1)/thr);
            if (src1->type == GGML_TYPE_F16) nvfp4_amax_kernel<half><<<blk, thr, 0, stream>>>((const half*)src1->data, amax_d.get(), n);
            else                             nvfp4_amax_kernel<float><<<blk, thr, 0, stream>>>((const float*)src1->data, amax_d.get(), n);
            unsigned int bits = 0;
            cudaMemcpyAsync(&bits, amax_d.get(), sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            float amax_t = 0.f; memcpy(&amax_t, &bits, sizeof(float));
            a_per_tensor = amax_t > 0.f ? amax_t/(6.0f*448.0f) : 0.f;
            // GGML_ACT_AMAX_REPORT=1 -- answer "would an F16 residual stream have overflowed?"
            // WITHOUT running F16. This amax is the largest absolute activation feeding this GEMM
            // and it is already on the host (the two-level quant needs it), so the probe is free.
            // Every large intermediate in the DiT is some Linear's input -- including the 16384-wide
            // SwiGLU product -- so this covers the tensors that would actually blow F16's 65504.
            // Reports headroom = 65504/amax; anything approaching 1.0 means F16 would clip.
            // Deliberately reports the RUNNING MAX, not per-call: the risk is content-dependent, so
            // what matters is the worst activation any prompt produces, not the average.
            if (amax_t > 0.f) {
                static int s_amax_report = -1;
                if (s_amax_report < 0) { const char* e = getenv("GGML_ACT_AMAX_REPORT"); s_amax_report = (e && atoi(e)) ? 1 : 0; }
                if (s_amax_report) {
                    static std::mutex        s_amax_mu;
                    static float             s_amax_hi = 0.f;
                    std::lock_guard<std::mutex> lk(s_amax_mu);
                    if (amax_t > s_amax_hi) {
                        s_amax_hi = amax_t;
                        const float f16_max  = 65504.0f;
                        const float headroom = f16_max / amax_t;
                        fprintf(stderr, "[act-amax] new max %.1f on '%s' [M=%d K=%d] -- f16 headroom %.1fx%s\n",
                                amax_t, src1->name[0] ? src1->name : "(unnamed)", (int)M, (int)K, headroom,
                                headroom < 4.0f ? "  <-- WARN: under 4x, an F16 stream is at risk" : "");
                    }
                }
            }
        }
    }
    if (!act_reused) {
        static int s_norefine = -1;
        if (s_norefine < 0) { const char* e = getenv("GGML_NVFP4_QUANT_NOREFINE"); s_norefine = (e && atoi(e)) ? 1 : 0; }
        const int threads = 256;
        const int total   = M*nsub;
        const int blocks  = (total+threads-1)/threads;
        const float pt = a_per_tensor;   // >0 => two-level branch in the kernel
        if (src1->type == GGML_TYPE_F16) {
            if (s_norefine || pt > 0.f) quant_act_kernel<half,false><<<blocks, threads, 0, stream>>>((const half*)src1->data, a_data_ptr, a_scales_ptr, M, K, pt);
            else                        quant_act_kernel<half,true ><<<blocks, threads, 0, stream>>>((const half*)src1->data, a_data_ptr, a_scales_ptr, M, K, pt);
        } else {
            if (s_norefine || pt > 0.f) quant_act_kernel<float,false><<<blocks, threads, 0, stream>>>((const float*)src1->data, a_data_ptr, a_scales_ptr, M, K, pt);
            else                        quant_act_kernel<float,true ><<<blocks, threads, 0, stream>>>((const float*)src1->data, a_data_ptr, a_scales_ptr, M, K, pt);
        }
        if (cudaPeekAtLastError() != cudaSuccess) return false;
        // Quant enqueued on `stream` -> the entry is now valid for any later GEMM in this same
        // compute (which necessarily runs after it on the same stream). Publishing here, not at
        // the miss, is what makes every early return above safe.
        if (act_publish_gen != (uint64_t)-1) {
            g_nvfp4_act_cache.per_tensor = a_per_tensor;
            g_nvfp4_act_cache.gen        = act_publish_gen;
        }
    }

    // 3) cuBLASLt FP4 GEMM: D[M,N] (row-major) = A_w[N,K] @ B_a[M,K]^T
    // cuBLAS column-major: m=N, n=M, k=K; A=weight (TN), B=activation.
    // alpha = A_per_tensor for the two-level activation quant (carries the per-tensor
    // global the kernel factored out of the stored block scales); 1.0 otherwise.
    const int m=N, n=M, k=K;
    // alpha carries BOTH per-tensor globals: the activation's (from the two-level act
    // quant) and the weight's (ModelOpt weight_scale_2, registered at load — see
    // g_wglobal). An UNFOLDED-import gguf stores well-conditioned block scales and
    // registers its weight global here, which is what lets the graph drop the per-Linear
    // full-size ggml_mul. A legacy FOLDED gguf registers nothing -> w_global = 1.0 ->
    // byte-identical to before.
    const float w_global = nvfp4_weight_global_for(src0->name);
    float alpha_h = (a_per_tensor > 0.f ? a_per_tensor : 1.0f) * w_global;
    static float beta_h = 0.0f;

    cublasLtMatmulDesc_t op = nullptr;
    if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return false;
    cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
    cublasOperation_t T=CUBLAS_OP_T, Nn=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &Nn, sizeof(Nn));
    void* wsp = (void*)w_scales.get(); void* asp = (void*)a_scales_ptr;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &wsp, sizeof(wsp));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &asp, sizeof(asp));
    cublasDataType_t st = CUDA_R_32F;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &st, sizeof(st));

    // output store type follows dst (F32 default; F16 for the dit_f16 residual stream)
    const cublasDataType_t out_dt = (dst->type == GGML_TYPE_F16) ? CUDA_R_16F : CUDA_R_32F;
    cublasLtMatrixLayout_t Ad=nullptr,Bd=nullptr,Cd=nullptr,Dd=nullptr;
    cublasLtMatrixLayoutCreate(&Ad, CUDA_R_4F_E2M1, k, m, k);
    cublasLtMatrixLayoutCreate(&Bd, CUDA_R_4F_E2M1, k, n, k);
    cublasLtMatrixLayoutCreate(&Cd, out_dt, m, n, m);
    cublasLtMatrixLayoutCreate(&Dd, out_dt, m, n, m);

    ggml_cuda_pool_alloc<uint8_t> ws(ctx.pool(), 32*1024*1024);
    size_t wsz = 32*1024*1024;

    // Per-shape ALGO cache (thread_local; g_lt is thread_local and the offload path runs
    // the GEMM on worker threads, so a per-thread cache needs no lock). The cuBLASLt
    // heuristic query is the expensive, host-serializing part of each call AND the source
    // of run-to-run non-determinism (it can return a different algo per call → two
    // identical configs diverge). The selected algo is a pure function of the problem
    // (m,n,k,out_dt) + layouts, so caching it and reusing with freshly-created (but
    // identical) descriptors/layouts — the standard cuBLASLt reuse idiom — removes the
    // query and pins one algo for determinism. Descriptors stay per-call (cheap; reusing
    // the desc objects + re-setting scale pointers tripped an illegal access). Escape
    // hatch GGML_NVFP4_CUBLASLT_NOCACHE forces a fresh heuristic every call.
    static int s_nocache = -1;
    if (s_nocache < 0) { const char* e = getenv("GGML_NVFP4_CUBLASLT_NOCACHE"); s_nocache = (e && atoi(e)) ? 1 : 0; }
    static thread_local std::map<std::tuple<int,int,int,int>, cublasLtMatmulAlgo_t> g_algo_cache;
    const auto key = std::make_tuple(m, n, k, (int)out_dt);

    cublasLtMatmulAlgo_t algo;
    bool have_algo = false;
    if (!s_nocache) {
        auto it = g_algo_cache.find(key);
        if (it != g_algo_cache.end()) { algo = it->second; have_algo = true; }
    }
    if (!have_algo) {
        cublasLtMatmulPreference_t pref=nullptr;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz));
        cublasLtMatmulHeuristicResult_t hr={}; int got=0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Dd, pref, 1, &hr, &got);
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (hs == CUBLAS_STATUS_SUCCESS && got > 0) {
            algo = hr.algo; have_algo = true;
            if (!s_nocache) g_algo_cache[key] = algo;
        }
    }

    bool ok = have_algo;
    if (ok) {
        cublasStatus_t ms = cublasLtMatmul(lt, op, &alpha_h, w_data.get(), Ad, a_data_ptr, Bd,
                                           &beta_h, dst->data, Cd, dst->data, Dd,
                                           &algo, ws.get(), wsz, stream);
        ok = (ms == CUBLAS_STATUS_SUCCESS);
    }

    if (ok) {
        static int n_handled = 0;
        if (n_handled++ == 0 || getenv("GGML_NVFP4_CUBLASLT_TRACE"))
            fprintf(stderr, "[NVFP4_CUBLASLT] handled mul_mat #%d  M=%d K=%d N=%d act=%s (cuBLASLt FP4 GEMM, algo-cached)\n",
                    n_handled, M, K, N,
                    act_reused ? "reused" : (act_cache_use ? "cached" : "pool"));
        if (getenv("GGML_NVFP4_NANCHECK") && dst->type == GGML_TYPE_F32) {
            float h[8] = {0};
            cudaMemcpyAsync(h, dst->data, sizeof(h), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            int bad = 0; float mx = 0.f;
            for (int i=0;i<8;i++){ if(!isfinite(h[i])) bad=1; mx=fmaxf(mx,fabsf(h[i])); }
            if (bad || mx > 1e4f)
                fprintf(stderr, "[NVFP4_NAN] M=%d K=%d N=%d  dst0=%g max8=%g %s\n",
                        M, K, N, h[0], mx, bad?"NONFINITE":"BIG");
        }
    }

    if (Ad) cublasLtMatrixLayoutDestroy(Ad);
    if (Bd) cublasLtMatrixLayoutDestroy(Bd);
    if (Cd) cublasLtMatrixLayoutDestroy(Cd);
    if (Dd) cublasLtMatrixLayoutDestroy(Dd);
    if (op) cublasLtMatmulDescDestroy(op);
    return ok;
}

// ============================ FP8 (e4m3) FFN path ============================
// Ported from the prod fork (ggml a965c1ea, the "worm fix") + dfa463bb (the F16-store clamp).
//
// WHY: the FP4 path quantizes the ACTIVATION to E2M1 as well as the weight. A flat,
// low-variance 16-element block sits near the FP4 `0 <-> 0.5*scale` decision boundary, so
// a tiny frame-to-frame drift flips the whole block between codes -> a pulsating
// blob/stipple ("worm") on flat coloured backgrounds under motion. The fix is the
// Q4_K-clean recipe: keep the 4-bit WEIGHT, but give the ACTIVATION 8 bits.
//
// HOW: cuBLASLt has no mixed FP8xFP4 GEMM (unsupported on sm120), so the NVFP4-stored
// weight is PROMOTED to e4m3 per call (decoded verbatim off its own FP4 grid -> no extra
// weight loss, and no extra VRAM: the weight stays FP4-STORED, the e4m3 copy is transient
// pool scratch) and the activation is quantized to e4m3. Both carry a per-tensor SCALAR_32F
// scale (amax/448) on the cuBLASLt A/B scale pointers; alpha = 1. Costs ~2x the FP4 GEMM
// instead of BF16's ~8x.
//
// Nothing here runs unless GGML_FP8_FFN=1 => the flag unset is byte-identical to today.

#define FP8_E4M3_MAX 448.0f

// per-tensor amax of the *reconstructed* NVFP4 weight (decoded to true float values,
// matching the cuBLASLt convention: kvalues_mxfp4 = 2x e2m1, ue4m3_to_fp32 = scale/2, the
// 2x cancel => stored bytes are the standard e2m1*e4m3 product; * the per-tensor wglobal).
//
// HAZARD (w_global): with GGML_FP8_LAYERS=transformer_blocks this path takes EVERY DiT
// Linear, so those weights no longer reach the FP4 GEMM's `alpha = A_global * W_global`
// fold — and the graph builder has already ELIDED the per-Linear ggml_mul on the strength of
// ggml_cuda_nvfp4_weight_global_folded(). The scalar therefore has to be applied HERE or
// every DiT Linear silently loses its per-tensor weight scale. It is folded into the
// promotion itself (both the amax and the decode multiply by `wglobal`), exactly as prod
// does, so the e4m3 weight bytes + their scalar scale already carry it and alpha stays 1.
static __global__ void fp8_w_amax_kernel(const block_nvfp4 * __restrict__ W,
                                         unsigned int * __restrict__ amax_bits,
                                         int N, int K, float wglobal) {
    const int nsub = K/16, nblk = K/64;
    float local = 0.f;
    for (long t = (long)blockIdx.x*blockDim.x + threadIdx.x; t < (long)N*nsub;
         t += (long)gridDim.x*blockDim.x) {
        const int r = (int)(t / nsub), ss = (int)(t % nsub);
        const int block = ss/4, s = ss%4;
        const block_nvfp4 & b = W[(long)r*nblk + block];
        const float sc = ggml_cuda_ue4m3_to_fp32(b.d[s]) * wglobal;
        const uint8_t * qs = &b.qs[s*8];
        #pragma unroll
        for (int e=0;e<16;e++) {
            const uint8_t nib = (e<8) ? (qs[e]&0xF) : (qs[e-8]>>4);
            local = fmaxf(local, fabsf(kvalues_mxfp4[nib & 7] * sc));
        }
    }
    __shared__ float sh[256];
    sh[threadIdx.x] = local; __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (threadIdx.x < st) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x+st]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(amax_bits, __float_as_uint(sh[0]));
}

// decode NVFP4 weight -> e4m3, row-major [N,K] (1 byte/elem). `scale` is the device-side
// per-tensor scalar (amax/448) produced by fp8_scale_from_amax; wglobal folded in (see above).
static __global__ void fp8_w_quant_kernel(const block_nvfp4 * __restrict__ W,
                                          uint8_t * __restrict__ out, int N, int K,
                                          float wglobal, const float * __restrict__ scale) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int nsub = K/16, nblk = K/64;
    const long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= (long)N*nsub) return;
    const int r = (int)(idx / nsub), ss = (int)(idx % nsub);
    const int block = ss/4, s = ss%4;
    const block_nvfp4 & b = W[(long)r*nblk + block];
    const float sc  = ggml_cuda_ue4m3_to_fp32(b.d[s]) * wglobal;
    const float inv = 1.0f / (*scale);
    const uint8_t * qs = &b.qs[s*8];
    uint8_t * od = &out[(long)r*K + (long)ss*16];   // row-major [N,K]
    #pragma unroll
    for (int e=0;e<16;e++) {
        const uint8_t nib = (e<8) ? (qs[e]&0xF) : (qs[e-8]>>4);
        float v = kvalues_mxfp4[nib & 7] * sc;
        if (nib & 8) v = -v;
        const __nv_fp8_e4m3 q(v * inv);
        od[e] = q.__x;
    }
#else
    NO_DEVICE_CODE;
#endif
}

template <typename act_t>
static __global__ void fp8_a_amax_kernel(const act_t * __restrict__ X,
                                         unsigned int * __restrict__ amax_bits, long n) {
    float local = 0.f;
    for (long i = (long)blockIdx.x*blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x*blockDim.x)
        local = fmaxf(local, fabsf(nvfp4_load_act(X[i])));
    __shared__ float sh[256];
    sh[threadIdx.x] = local; __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (threadIdx.x < st) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x+st]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(amax_bits, __float_as_uint(sh[0]));
}

// quantize activation -> e4m3, flat (src1 [K,M] contiguous == row-major [M,K]).
// Templated on the element type so the F16 residual stream (LTX_DIT_F16) feeds in with no
// per-Linear F16->F32 cast, exactly like the FP4 quant kernel above.
template <typename act_t>
static __global__ void fp8_a_quant_kernel(const act_t * __restrict__ X,
                                          uint8_t * __restrict__ out, long n,
                                          const float * __restrict__ scale) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const float inv = 1.0f / (*scale);
    for (long i = (long)blockIdx.x*blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x*blockDim.x) {
        const __nv_fp8_e4m3 q(nvfp4_load_act(X[i]) * inv);
        out[i] = q.__x;
    }
#else
    NO_DEVICE_CODE;
#endif
}

// finalize per-tensor SCALAR scale = amax / 448 (e4m3 max). amax==0 => 1 (all-zero tensor).
static __global__ void fp8_scale_from_amax(const unsigned int * __restrict__ amax_bits,
                                           float * __restrict__ scale_out) {
    const float a = __uint_as_float(*amax_bits);
    scale_out[0] = (a > 0.f) ? a * (1.0f / FP8_E4M3_MAX) : 1.0f;
}

static __global__ void fp8_set_scalar_kernel(float * __restrict__ p, float v) { p[0] = v; }

// ---------------------------------------------------------------------------
// Per-tensor e4m3 WEIGHT-SCALE cache (name -> the exact float fp8_scale_from_amax produced).
//
// fp8_w_amax_kernel reads the ENTIRE NVFP4 weight just to produce one scalar, and it does so on
// every GEMM call even though a weight's values are CONSTANT for the whole render — that is a
// second full pass over ~11 GiB of FP4 weight per DiT forward, purely to recompute a number
// that cannot have changed. Cache the scalar (4 bytes/weight, ~5 KiB total) and replay it with
// a 1-thread store; the promotion then reads the weight exactly once.
//
// BIT-IDENTICAL: the cached value is the literal float the amax pass computed (read back once,
// stored verbatim, written back verbatim), not a recomputation.
//
// Keyed by NAME, which is what makes it valid under layer-offload: offload swaps src0->data per
// segment but preserves the name, and the amax is a function of the weight VALUES, not of the
// address. `w_global` is folded into the amax, so the cache must die whenever the registered
// globals do — it is cleared from ggml_cuda_nvfp4_clear_weight_globals(), the existing
// model-load / variant-swap hook.
// Off-switch: GGML_FP8_WEIGHT_SCALE_CACHE=0 (then every call recomputes the amax as before).
static std::mutex                             g_fp8_wscale_mtx;
static std::unordered_map<std::string, float> g_fp8_wscale;

static bool ggml_cuda_fp8_wscale_cache_enabled() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_FP8_WEIGHT_SCALE_CACHE"); v = (e && atoi(e) == 0) ? 0 : 1; }
    return v == 1;
}

void ggml_cuda_fp8_weight_scale_cache_clear(void) {
    std::lock_guard<std::mutex> lk(g_fp8_wscale_mtx);
    g_fp8_wscale.clear();
}

bool ggml_cuda_fp8_ffn_enabled() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_FP8_FFN"); v = (e && atoi(e)) ? 1 : 0; }
    return v == 1;
}

// substring filter (GGML_FP8_LAYERS, default "ff.net"): matches the DiT FFN up/gate
// (ff.net.0.proj) + down-proj (ff.net.2) Linears; attention/other linears never match.
// Prod ships GGML_FP8_LAYERS=transformer_blocks, i.e. ALL 1344 DiT linears.
bool ggml_cuda_fp8_ffn_name_match(const char * name) {
    if (!name) return false;
    static std::string filt;
    static int init = 0;
    if (!init) { const char * e = getenv("GGML_FP8_LAYERS"); filt = (e && *e) ? e : "ff.net"; init = 1; }
    return strstr(name, filt.c_str()) != nullptr;
}

// MANDATORY F16-store clamp (ggml dfa463bb). The cuBLASLt FP8 GEMM accumulates in F32; a
// deep-block result > 65504 stores as +-inf into an F16 dst -> RoPE casts q/k F16->F32 ->
// inf*cos - inf*sin = NaN -> softmax's row-max over NaN is comparison-ORDER-dependent ->
// nondeterministic output / white frames. That was a live production bug 2026-07-04..07-17.
//
// DEFAULT ON, and it is a CORRECTNESS fix, not a tunable: for in-range values both paths
// round the same F32 to F16 round-to-nearest, so they are bit-identical unless |x| > 65504.
// It also has a second, structural benefit here — the GEMM writes an F32 pool temp, so a
// late cuBLASLt failure can never leave a PARTIALLY-WRITTEN F16 destination behind.
// This branch is not an edge case on this tree: LTX_DIT_F16 makes an F16 dst the norm.
// Escape hatch GGML_F8_NO_CLAMP_OUT=1 to A/B the old (overflowing) behaviour.
static bool ggml_cuda_f8_clamp_out_enabled() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_F8_NO_CLAMP_OUT"); v = (e && atoi(e)) ? 0 : 1; }
    return v == 1;
}

static __global__ void fp8_clamp_f32_to_f16(const float * __restrict__ in, half * __restrict__ out, size_t n) {
    for (size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x*blockDim.x) {
        const float v = fminf(fmaxf(in[i], -65504.0f), 65504.0f);   // IEEE fmin/fmax: NaN -> -65504 (harmless; F32 accum has no NaN)
        out[i] = __float2half_rn(v);
    }
}

// ---------------------------------------------------------------------------
// ALTERNATIVE clamp mode: GGML_F8_CLAMP_INPLACE=1 (default OFF).
//
// Lets cuBLASLt store F16 straight into dst (no F32 temp at all) and repairs the
// overflow AFTERWARDS, in place, on the same stream. This is VALUE-identical to the
// F32-temp clamp above, elementwise:
//   |x| <= 65504          -> both paths are __float2half_rn(x)                 (identical)
//   65504 < |x| < 65520   -> both round to +-65504 (F16 RN overflow threshold) (identical)
//   |x| >= 65520          -> direct store gives +-inf; this kernel rewrites it to +-65504,
//                            which is what fminf/fmaxf + __float2half_rn produce (identical)
//   NaN                   -> fmaxf(NaN,-65504) = -65504 -> this kernel maps NaN to -65504
// The ONE thing it does change is `out_dt` (CUDA_R_16F instead of CUDA_R_32F), which is an
// input to the cuBLASLt heuristic -> it may hand back a DIFFERENT algo, and a different algo
// can accumulate k in a different order. So this mode is NOT provably bit-identical to the
// approved output and stays opt-in; the default path (bounded F32 temp, below) keeps
// out_dt == CUDA_R_32F and reuses the very same algo.
static bool ggml_cuda_f8_clamp_inplace_enabled() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_F8_CLAMP_INPLACE"); v = (e && atoi(e)) ? 1 : 0; }
    return v == 1;
}

// +-inf -> +-65504 (0x7BFF), NaN -> -65504 (0xFBFF). Finite values are left untouched, so the
// common case is read-only traffic.
static __global__ void fp8_fix_f16_inplace(half * __restrict__ io, size_t n) {
    for (size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x*blockDim.x) {
        const unsigned short b = __half_as_ushort(io[i]);
        if ((b & 0x7C00u) == 0x7C00u) {                       // exponent all-ones => inf or NaN
            const bool is_nan = (b & 0x03FFu) != 0;
            io[i] = __ushort_as_half(is_nan ? (unsigned short)0xFBFFu
                                            : (unsigned short)((b & 0x8000u) | 0x7BFFu));
        }
    }
}

// ---------------------------------------------------------------------------
// BOUNDED transient buffers (the VRAM fix).
//
// At LTX-2.3 22B DiT shapes the two per-call transients are enormous:
//   F32 output temp   4*N*M  -> ff.net.0 (N = 4*3840 = 15360) at M ~ 22k tokens = ~1.29 GiB
//   e4m3 activation   M*K    -> ff.net.2 (K = 15360)          at M ~ 22k tokens = ~322 MiB
// The ggml CUDA VMM pool is a bump/stack allocator whose high-water mark is mapped physical
// VRAM (ggml-cuda.cu:660-685) — so the single largest concurrent transient sets peak VRAM for
// the whole render.
//
// Fix: slice the GEMM along the TOKEN axis (cuBLASLt `n`) so both buffers are bounded by
// GGML_F8_GEMM_CHUNK_MB (default 64 MiB). Column-major D is m x n with ld = m = N, so a token
// slice [c0, c0+nn) is a CONTIGUOUS run of dst starting at element c0*N; B (the activation) is
// k x n with ld = k, so its slice starts at byte c0*K. No transposes, no gather.
//
// NUMERICALLY TRANSPARENT, and this is load-bearing: the algo is still selected by the
// heuristic for the FULL (m, M, k, out_dt) problem — exactly the query the un-chunked path
// makes — and then REUSED for every slice. A cuBLASLt algo is a tile/stage/split-k CONFIG; the
// k-reduction order for a given output element is a property of that config, not of `n`.
// Slicing n only changes which CTA computes which output tile. Before the first sliced matmul
// the algo is validated against the slice layouts with cublasLtMatmulAlgoCheck(); if cuBLASLt
// rejects it (or wants more workspace) we fall back to the un-chunked full-size temp for that
// shape, so we never substitute a different algo behind the owner's back.
// GGML_F8_GEMM_CHUNK_MB=0 restores the old full-size behaviour exactly.
static size_t ggml_cuda_fp8_chunk_budget_bytes() {
    static size_t b = 0; static int init = 0;
    if (!init) {
        const char * e = getenv("GGML_F8_GEMM_CHUNK_MB");
        long mb = (e && *e) ? atol(e) : 64;
        if (mb < 0) mb = 0;
        b = (size_t)mb * 1024 * 1024;
        init = 1;
    }
    return b;
}

// cublasLtMatmulAlgoCheck verdict for (m, n_full, k, out_dt, n_chunk). thread_local because
// g_lt / the algo cache are thread_local (the offload path runs GEMMs on worker threads).
static thread_local std::map<std::tuple<int,int,int,int,int>, bool> g_fp8_chunk_ok;

// ---------------------------------------------------------------------------
// FP8 activation-quant reuse cache (ported from the prod fork, nvfp4-cublaslt.cu:1295-1358).
//
// Self-attn to_q/to_k/to_v — and cross-attn to_k/to_v — feed the SAME src1 activation to
// consecutive FP8 GEMMs, so its e4m3 quant (a full amax reduction + a full M*K quantize pass
// over the activation) is recomputed identically 2-3x per block. Quantize once, reuse the
// e4m3 buffer + its scalar scale for any later GEMM in the SAME compute whose src1 is the
// byte-for-byte same tensor.
//
// Stale-safety (must never reuse stale data, and must stay bit-identical when it does reuse):
//  - A GLOBAL atomic generation is bumped once per graph compute. Prod bumps it from its
//    host-side execute_graph(); THIS tree bumps it from ggml_backend_cuda_graph_compute()
//    (ggml-cuda.cu) instead — the backend's own graph entry point — so it fires for every
//    compute from every host path, including each ggml_backend_sched split. A hit requires the
//    stored generation to equal the current one, so a new compute can NEVER reuse a previous
//    compute's buffer even if gallocr recycles a node/data address.
//  - Within one compute every graph node has a unique, stable address, so keying on the src1
//    NODE pointer (plus data ptr / ne0 / ne1 / nb1 / type / device as belt-and-braces) means
//    two different logical activations cannot collide even when their ->data aliases across
//    non-overlapping lifetimes.
//  - The buffer is OWNED (cudaMalloc, grow-only), so neither gallocr nor the stream pool can
//    recycle it out from under a pending reuse.
//  - On ANY uncertainty (alloc failure, device change, oversize, stream capture) we MISS and
//    requantize. The quant is deterministic (atomicMax amax -> scalar scale -> per-element
//    e4m3 round), so a reused buffer is byte-identical to requantizing it.
//
// VRAM: the buffer is persistent, so it is only worth taking for activations that are actually
// SHARED. GGML_FP8_ACT_CACHE_MB (default 128) caps it — the DiT's shared q/k/v activation is
// M*hidden (~81 MiB at 22k tokens x 3840) and fits; ff.net.2's private M*15360 activation
// (~322 MiB) does not, and is left on the bounded/chunked pool path where it belongs.
// Off-switch: GGML_FP8_ACT_QUANT_CACHE=0.
struct fp8_act_quant_cache {
    uint64_t            gen     = (uint64_t)-1;   // generation filled at; -1 == empty
    const ggml_tensor * node    = nullptr;        // src1 node identity (unique within a graph)
    const void *        data    = nullptr;
    int64_t             ne0     = 0;
    int64_t             ne1     = 0;
    size_t              nb1     = 0;
    int                 type    = -1;
    int                 device  = -1;
    uint8_t *           d_fp8   = nullptr;        // owned persistent e4m3 buffer
    size_t              cap     = 0;              // capacity (bytes) of d_fp8
    float *             d_scale = nullptr;        // owned persistent scalar scale (1 float)
};
static thread_local fp8_act_quant_cache g_fp8_act_cache;
static std::atomic<uint64_t>            g_fp8_act_cache_gen{0};

static bool ggml_cuda_fp8_act_cache_enabled() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_FP8_ACT_QUANT_CACHE"); v = (e && atoi(e) == 0) ? 0 : 1; }
    return v == 1;
}
static size_t ggml_cuda_fp8_act_cache_budget_bytes() {
    static size_t b = 0; static int init = 0;
    if (!init) {
        const char * e = getenv("GGML_FP8_ACT_CACHE_MB");
        long mb = (e && *e) ? atol(e) : 128;
        if (mb < 0) mb = 0;
        b = (size_t)mb * 1024 * 1024;
        init = 1;
    }
    return b;
}

// Bump once per graph compute so cross-compute reuse can never happen. Cheap relaxed atomic.
// Bumps BOTH activation-quant caches: the FP8 one below and the NVFP4 one above
// (nvfp4_act_quant_cache). One call site (ggml_backend_cuda_graph_compute) already covers
// every host path and every ggml_backend_sched split, and an extra bump only ever costs a
// cache miss (a requant), never correctness — so the two caches share it.
void ggml_cuda_fp8_act_cache_new_generation(void) {
    g_fp8_act_cache_gen.fetch_add(1, std::memory_order_relaxed);
    g_nvfp4_act_cache_gen.fetch_add(1, std::memory_order_relaxed);
}

// NOT PORTED: prod's GGML_FP8_WEIGHT_QUANT_CACHE (nvfp4-cublaslt.cu:1385-1434). It is default
// OFF in prod too, and it is the wrong trade here: it turns the e4m3 weight from a pool
// transient (max ~56 MiB live at a time, shared with every other op) into up to
// GGML_FP8_WEIGHT_CACHE_MB of PERMANENT VRAM (the 22B DiT's NVFP4 Linears re-promoted to e4m3
// are ~11 GiB at full coverage — 2x their FP4-stored size). VRAM is the binding constraint on
// this box, so the weight requant stays per-call. If it is ever wanted, it must come with an
// eviction policy and a budget well under 1 GiB.

bool ggml_cuda_fp8_cublaslt_mul_mat(ggml_backend_cuda_context & ctx,
                                    const ggml_tensor * src0,
                                    const ggml_tensor * src1,
                                    ggml_tensor * dst) {
    // Bail list — MUST stay in sync with ggml_cuda_nvfp4_cublaslt_shapes_ok() in ggml-cuda.cu,
    // which promises supports_op() that this function runs for the nodes it advertises.
    // Every bail here is side-effect-free (dst untouched) so the caller falls cleanly
    // through to the FP4 / MMQ / dequant path.
    if (src0->type != GGML_TYPE_NVFP4) return false;
    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) return false;
    if (dst->type  != GGML_TYPE_F32 && dst->type  != GGML_TYPE_F16) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;

    const int K = src0->ne[0];   // contraction
    const int N = src0->ne[1];   // out features
    const int M = src1->ne[1];   // tokens
    if (src1->ne[0] != K) return false;
    if (K % 64 != 0)      return false;
    if (dst->ne[0] != N || dst->ne[1] != M) return false;

    cudaStream_t stream = ctx.stream();
    cublasLtHandle_t lt = get_lt();
    if (!lt) return false;

    // See the fp8_w_amax_kernel header: this scalar MUST be applied here, because the graph
    // builder elided the per-Linear ggml_mul on the promise that the GEMM folds it.
    // Unregistered name -> 1.0, which is correct for a legacy FOLDED gguf (it has no globals)
    // and is exactly what ggml_cuda_nvfp4_weight_global_registered() gates the elision on.
    const float w_global = nvfp4_weight_global_for(src0->name);

    // ------------------------------------------------------------------
    // Output store mode.
    //   use_f32_temp  : GEMM stores F32 into a temp, a kernel clamps+downconverts into the
    //                   F16 dst (the MANDATORY clamp; default).
    //   clamp_inplace : GEMM stores F16 straight into dst, a kernel repairs +-inf/NaN in place
    //                   (GGML_F8_CLAMP_INPLACE=1; value-identical but changes out_dt -> opt-in).
    // Either way the F16 destination is NEVER left unclamped: the repair kernel is enqueued on
    // the SAME stream immediately after the GEMM that produced the values, before any other
    // consumer can be enqueued, and a failure to enqueue it makes this function return false so
    // the caller's fallback rewrites dst in full.
    // ------------------------------------------------------------------
    const bool clamp_f16     = ggml_cuda_f8_clamp_out_enabled() && dst->type == GGML_TYPE_F16;
    const bool clamp_inplace = clamp_f16 && ggml_cuda_f8_clamp_inplace_enabled();
    const bool use_f32_temp  = clamp_f16 && !clamp_inplace;
    const cublasDataType_t out_dt = (dst->type == GGML_TYPE_F16 && !use_f32_temp) ? CUDA_R_16F : CUDA_R_32F;
    const size_t dst_esz = (dst->type == GGML_TYPE_F16) ? 2u : 4u;

    const size_t n_act = (size_t)M * (size_t)K;

    // ------------------------------------------------------------------
    // 0) Activation-quant reuse cache: resolve hit/miss BEFORE anything is allocated, because
    //    the chunking decision below needs to know whether the full-size e4m3 activation is
    //    already materialised in the owned cache buffer.
    // ------------------------------------------------------------------
    uint8_t * a_fp8_ptr   = nullptr;   // non-null => served by the owned cache buffer
    float   * a_scale_ptr = nullptr;
    bool      act_reused  = false;     // true => the e4m3 bytes are already valid, skip the quant
    // Generation to PUBLISH on the cache entry, and only once the quant kernels for it have
    // actually been enqueued. Until then the entry stays marked empty, so any early return
    // between here and the quant cannot leave a later call reusing un-filled bytes.
    uint64_t  act_publish_gen = (uint64_t)-1;

    // While a CUDA graph is being captured, cudaMalloc / D2H copies / stream syncs are all
    // illegal, and a captured graph replays WITHOUT re-running any of this host code — so a
    // persistent pointer or a cached decision baked in at capture time would be replayed
    // blindly. Detect capture once and take the plain per-call path for everything.
    bool capturing = false;
#ifdef USE_CUDA_GRAPH
    {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
            capturing = true;
        }
    }
#endif

    bool act_cache_use = ggml_cuda_fp8_act_cache_enabled() && !capturing &&
                         n_act <= ggml_cuda_fp8_act_cache_budget_bytes();
    if (act_cache_use) {
        fp8_act_quant_cache & C = g_fp8_act_cache;
        const uint64_t cur = g_fp8_act_cache_gen.load(std::memory_order_relaxed);
        const bool hit = C.gen == cur && C.node == src1 && C.data == src1->data &&
                         C.ne0 == src1->ne[0] && C.ne1 == src1->ne[1] &&
                         C.nb1 == src1->nb[1] && C.type == (int)src1->type &&
                         C.device == ctx.device && C.d_fp8 != nullptr &&
                         C.d_scale != nullptr && C.cap >= n_act;
        if (hit) {
            a_fp8_ptr   = C.d_fp8;
            a_scale_ptr = C.d_scale;
            act_reused  = true;
        } else {
            // MISS: (re)quantize into the OWNED persistent buffer (grow-only). cudaFree is
            // device-synchronizing, so any in-flight GEMM still reading the old buffer has
            // retired before it is released; growth only happens on the first calls.
            if (C.device != ctx.device && C.d_fp8 != nullptr) {
                cudaFree(C.d_fp8);   C.d_fp8   = nullptr; C.cap = 0;
                cudaFree(C.d_scale); C.d_scale = nullptr;
            }
            if (C.cap < n_act) {
                if (C.d_fp8 != nullptr) cudaFree(C.d_fp8);
                if (cudaMalloc((void**)&C.d_fp8, n_act) != cudaSuccess) { C.d_fp8 = nullptr; C.cap = 0; }
                else                                                     C.cap   = n_act;
            }
            if (C.d_scale == nullptr) {
                if (cudaMalloc((void**)&C.d_scale, sizeof(float)) != cudaSuccess) C.d_scale = nullptr;
            }
            if (C.d_fp8 != nullptr && C.d_scale != nullptr) {
                a_fp8_ptr   = C.d_fp8;
                a_scale_ptr = C.d_scale;
                C.gen  = (uint64_t)-1;    // stays EMPTY until the quant is enqueued (see below)
                C.node = src1;            C.data   = src1->data;
                C.ne0  = src1->ne[0];     C.ne1    = src1->ne[1];  C.nb1  = src1->nb[1];
                C.type = (int)src1->type; C.device = ctx.device;
                act_publish_gen = cur;
            } else {
                C.gen = (uint64_t)-1; C.node = nullptr;   // alloc failed -> invalidate, use pool
                act_cache_use = false;
            }
        }
    }

    // ------------------------------------------------------------------
    // 0b) Chunking decision — pure function of the shapes + env, so it is known before any
    //     allocation. `nc == M` means "one slice", i.e. byte-for-byte the old behaviour.
    // ------------------------------------------------------------------
    const size_t chunk_budget = ggml_cuda_fp8_chunk_budget_bytes();
    // The activation only needs slicing when it is NOT served by the cache and is oversized.
    bool   chunk_act = (a_fp8_ptr == nullptr) && chunk_budget > 0 && n_act > chunk_budget;
    size_t per_tok   = (use_f32_temp ? 4u * (size_t)N : 0u) + (chunk_act ? (size_t)K : 0u);
    int    nc        = M;
    if (chunk_budget > 0 && per_tok > 0) {
        size_t t = chunk_budget / per_tok;
        t &= ~(size_t)255;                     // multiple of 256 tokens: keeps every slice's B
        if (t < 256) t = 256;                  // offset (c0*K bytes, K%64==0) heavily aligned
        if (t < (size_t)M) nc = (int)t;
    }
    if (nc >= M) { nc = M; chunk_act = false; }

    // ------------------------------------------------------------------
    // ggml's CUDA pool is a STACK allocator (ggml-cuda.cu:682 asserts frees are the exact
    // reverse of allocs). Every pool buffer below is declared-and-allocated in one strictly
    // increasing order and the destructors unwind in reverse declaration order, so every early
    // return is LIFO-safe. Allocation order:
    //     scal -> w_fp8 -> a_fp8_pool -> d_f32 -> ws
    // ------------------------------------------------------------------

    // Single 32-byte scalar block instead of four 4-byte pool allocs. Besides saving three
    // bump-allocations per call it keeps every LATER pool pointer 32-byte aligned — the pool
    // hands back `base + used` with no rounding, so the old w_amax/w_scale pair left the e4m3
    // weight operand only 8-byte aligned, which cuBLASLt is entitled to reject.
    ggml_cuda_pool_alloc<unsigned int> scal(ctx.pool(), 8);
    unsigned int * w_amax_d  = scal.get() + 0;
    float        * w_scale_d = (float *)(scal.get() + 1);
    unsigned int * a_amax_d  = scal.get() + 2;
    if (a_scale_ptr == nullptr) a_scale_ptr = (float *)(scal.get() + 3);

    // ------------------------------------------------------------------
    // 1) cuBLASLt descriptor + FULL-shape layouts + algo selection.
    //    Built BEFORE the big allocations so that, if the pinned algo turns out not to accept
    //    the sliced layouts, we can still fall back to the full-size temp without having
    //    already committed a (too small) pool buffer.
    //
    //    D[M,N] = W_fp8[N,K] @ A_fp8[M,K]^T. Column-major m=N, n=M, k=K; A=weight (TN, e4m3),
    //    B=act (N, e4m3). Per-tensor SCALAR scales on A/B; alpha = 1 (w_global is already
    //    inside the weight's e4m3 bytes + scalar scale).
    // ------------------------------------------------------------------
    const int m=N, n=M, k=K;
    float alpha_h = 1.0f; static float beta_h = 0.0f;

    cublasLtMatmulDesc_t op = nullptr;
    if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return false;
    cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
    cublasOperation_t T=CUBLAS_OP_T, Nn=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &Nn, sizeof(Nn));
    void* wsp = (void*)w_scale_d; void* asp = (void*)a_scale_ptr;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &wsp, sizeof(wsp));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &asp, sizeof(asp));
    cublasDataType_t st = CUDA_R_32F;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &st, sizeof(st));

    cublasLtMatrixLayout_t Ad=nullptr,Bd=nullptr,Cd=nullptr,Dd=nullptr;   // full-shape layouts
    cublasLtMatrixLayout_t Bc=nullptr,Cc=nullptr,Dc=nullptr;              // slice layouts
    cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8F_E4M3, k, m, k);
    cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8F_E4M3, k, n, k);
    cublasLtMatrixLayoutCreate(&Cd, out_dt, m, n, m);
    cublasLtMatrixLayoutCreate(&Dd, out_dt, m, n, m);

    auto destroy_lt = [&]() {
        if (Bc) cublasLtMatrixLayoutDestroy(Bc);
        if (Cc) cublasLtMatrixLayoutDestroy(Cc);
        if (Dc) cublasLtMatrixLayoutDestroy(Dc);
        if (Ad) cublasLtMatrixLayoutDestroy(Ad);
        if (Bd) cublasLtMatrixLayoutDestroy(Bd);
        if (Cd) cublasLtMatrixLayoutDestroy(Cd);
        if (Dd) cublasLtMatrixLayoutDestroy(Dd);
        if (op) cublasLtMatmulDescDestroy(op);
    };

    const size_t wsz = 32*1024*1024;

    // Per-shape ALGO cache (thread_local, mirrors the FP4 path): removes the per-call
    // heuristic query, which is both the host-serializing cost and a source of run-to-run
    // nondeterminism (it can hand back a different algo per call). The query uses the FULL
    // (m, n=M, k, out_dt) problem — identical to the un-chunked path — so slicing cannot
    // change which algo runs.
    static int s_nocache = -1;
    if (s_nocache < 0) { const char* e = getenv("GGML_NVFP4_CUBLASLT_NOCACHE"); s_nocache = (e && atoi(e)) ? 1 : 0; }
    static thread_local std::map<std::tuple<int,int,int,int>, cublasLtMatmulAlgo_t> g_fp8_algo_cache;
    const auto key = std::make_tuple(m, n, k, (int)out_dt);

    cublasLtMatmulAlgo_t algo; bool have_algo = false;
    if (!s_nocache) { auto it = g_fp8_algo_cache.find(key); if (it != g_fp8_algo_cache.end()) { algo = it->second; have_algo = true; } }
    if (!have_algo) {
        cublasLtMatmulPreference_t pref=nullptr;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz));
        cublasLtMatmulHeuristicResult_t hr={}; int got=0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Dd, pref, 1, &hr, &got);
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (hs == CUBLAS_STATUS_SUCCESS && got > 0) {
            algo = hr.algo; have_algo = true;
            if (!s_nocache) g_fp8_algo_cache[key] = algo;
        }
    }
    if (!have_algo) { destroy_lt(); return false; }   // dst untouched

    // ------------------------------------------------------------------
    // 1b) Validate the PINNED algo against the slice layouts. If cuBLASLt will not run this
    //     exact algo at n = nc (or at the tail size), we refuse to silently substitute another
    //     one and fall back to the un-chunked full-size temp for this shape.
    // ------------------------------------------------------------------
    if (nc < M) {
        const auto ckey = std::make_tuple(m, n, k, (int)out_dt, nc);
        auto ck = g_fp8_chunk_ok.find(ckey);
        bool chunk_ok;
        if (ck != g_fp8_chunk_ok.end()) {
            chunk_ok = ck->second;
        } else {
            chunk_ok = true;
            const int tail = M % nc;
            const int probe_n[2] = { nc, tail };
            for (int p = 0; p < 2 && chunk_ok; ++p) {
                const int pn = probe_n[p];
                if (pn == 0 || pn == M) continue;
                cublasLtMatrixLayout_t pB=nullptr,pC=nullptr,pD=nullptr;
                cublasLtMatrixLayoutCreate(&pB, CUDA_R_8F_E4M3, k, pn, k);
                cublasLtMatrixLayoutCreate(&pC, out_dt, m, pn, m);
                cublasLtMatrixLayoutCreate(&pD, out_dt, m, pn, m);
                cublasLtMatmulHeuristicResult_t chk = {};
                const cublasStatus_t cs = cublasLtMatmulAlgoCheck(lt, op, Ad, pB, pC, pD, &algo, &chk);
                if (cs != CUBLAS_STATUS_SUCCESS || chk.state != CUBLAS_STATUS_SUCCESS || chk.workspaceSize > wsz) {
                    chunk_ok = false;
                }
                if (pB) cublasLtMatrixLayoutDestroy(pB);
                if (pC) cublasLtMatrixLayoutDestroy(pC);
                if (pD) cublasLtMatrixLayoutDestroy(pD);
            }
            g_fp8_chunk_ok[ckey] = chunk_ok;
            if (!chunk_ok && getenv("GGML_NVFP4_CUBLASLT_TRACE"))
                fprintf(stderr, "[FP8_FFN] slice-algo rejected for m=%d n=%d k=%d nc=%d -> full-size temp\n",
                        m, n, k, nc);
        }
        if (!chunk_ok) { nc = M; chunk_act = false; }
    }

    // ------------------------------------------------------------------
    // 2) Pool allocations (bounded by the slice size when chunking).
    // ------------------------------------------------------------------
    ggml_cuda_pool_alloc<uint8_t> w_fp8     (ctx.pool(), (size_t)N*(size_t)K);
    ggml_cuda_pool_alloc<uint8_t> a_fp8_pool(ctx.pool());
    ggml_cuda_pool_alloc<float>   d_f32     (ctx.pool());

    if (a_fp8_ptr == nullptr) {   // not served by the owned cache buffer
        a_fp8_ptr = a_fp8_pool.alloc(chunk_act ? (size_t)nc * (size_t)K : n_act);
    }
    float * d_f32_ptr = nullptr;
    if (use_f32_temp) {
        d_f32_ptr = d_f32.alloc((size_t)N * (size_t)nc);
    }

    // cuBLASLt REQUIRES a 256-byte-aligned matmul workspace; the ggml pool can hand back a
    // 128-byte-offset pointer depending on prior allocs. A misaligned workspace makes
    // cublasLtMatmul return CUBLAS_STATUS_INVALID_VALUE(7) -> silent fall-through to a path
    // that does NOT apply w_global -> wrong pixels. Over-allocate +256 and round up.
    ggml_cuda_pool_alloc<uint8_t> ws(ctx.pool(), wsz + 256);
    uint8_t * ws_ptr = (uint8_t *)(((uintptr_t)ws.get() + 255) & ~(uintptr_t)255);

    // ------------------------------------------------------------------
    // 3) weight -> e4m3 [N,K] (transient pool scratch; the weight stays FP4-STORED, so the
    //    steady-state VRAM delta of this path is ~zero).
    // ------------------------------------------------------------------
    {
        const int  threads = 256;
        const long total   = (long)N*(K/16);
        unsigned int grid = (unsigned int)((total + threads - 1)/threads);
        if (grid > 65535u) grid = 65535u;
        if (grid == 0)     grid = 1;

        const bool   wsc_cache = ggml_cuda_fp8_wscale_cache_enabled() && !capturing && src0->name[0] != '\0';
        bool         wsc_have  = false;
        float        wsc_val   = 0.0f;
        if (wsc_cache) {
            std::lock_guard<std::mutex> lk(g_fp8_wscale_mtx);
            auto it = g_fp8_wscale.find(src0->name);
            if (it != g_fp8_wscale.end()) { wsc_val = it->second; wsc_have = true; }
        }
        if (wsc_have) {
            // replay the byte-identical scalar; skips a full read of the NVFP4 weight
            fp8_set_scalar_kernel<<<1, 1, 0, stream>>>(w_scale_d, wsc_val);
        } else {
            cudaMemsetAsync(w_amax_d, 0, sizeof(unsigned int), stream);
            fp8_w_amax_kernel<<<grid, threads, 0, stream>>>((const block_nvfp4*)src0->data, w_amax_d, N, K, w_global);
            fp8_scale_from_amax<<<1, 1, 0, stream>>>(w_amax_d, w_scale_d);
            if (wsc_cache) {
                // one-off readback per weight (~1.3k over a render) so every later call replays
                // the SAME bits instead of re-deriving them.
                float h = 0.0f;
                if (cudaMemcpyAsync(&h, w_scale_d, sizeof(float), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
                    cudaStreamSynchronize(stream) == cudaSuccess) {
                    std::lock_guard<std::mutex> lk(g_fp8_wscale_mtx);
                    g_fp8_wscale[src0->name] = h;
                }
            }
        }
        fp8_w_quant_kernel<<<grid, threads, 0, stream>>>((const block_nvfp4*)src0->data, w_fp8.get(), N, K, w_global, w_scale_d);
        if (cudaPeekAtLastError() != cudaSuccess) { destroy_lt(); return false; }
    }

    // ------------------------------------------------------------------
    // 4) activation -> e4m3 [M,K] (src1 [K,M] contiguous == row-major [M,K]).
    //    The per-tensor amax is ALWAYS taken over the whole activation, chunked or not, so the
    //    scalar scale — and therefore every quantized byte — is identical either way. The
    //    quantize itself is elementwise, so slicing it is bit-identical by construction.
    //    A cache hit skips both passes outright (that is the 2-3x-per-block saving).
    // ------------------------------------------------------------------
    if (!act_reused) {
        const int threads = 256;
        unsigned int grid = (unsigned int)((n_act + threads - 1)/threads);
        if (grid > 1024u) grid = 1024u;
        if (grid == 0)    grid = 1;
        cudaMemsetAsync(a_amax_d, 0, sizeof(unsigned int), stream);
        if (src1->type == GGML_TYPE_F16) {
            fp8_a_amax_kernel<half> <<<grid, threads, 0, stream>>>((const half*)  src1->data, a_amax_d, (long)n_act);
        } else {
            fp8_a_amax_kernel<float><<<grid, threads, 0, stream>>>((const float*) src1->data, a_amax_d, (long)n_act);
        }
        fp8_scale_from_amax<<<1, 1, 0, stream>>>(a_amax_d, a_scale_ptr);
        // GGML_ACT_AMAX_REPORT=1 -- same probe as the NVFP4 path above, but this one costs a D2H
        // + sync, so it is STRICTLY env-gated: this path deliberately keeps everything on device.
        // This is the interesting one for LTX and Wan: they ship LTX_DIT_F16/WAN_DIT_F16 AND route
        // their FFN through here, so if an F16 stream is already clipping in production, the
        // non-finite amax shows up at THIS call site and nowhere else.
        {
            static int s_probe = -1;
            if (s_probe < 0) { const char * e = getenv("GGML_ACT_AMAX_REPORT"); s_probe = (e && atoi(e)) ? 1 : 0; }
            if (s_probe) {
                unsigned int abits = 0;
                if (cudaMemcpyAsync(&abits, a_amax_d, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
                    cudaStreamSynchronize(stream) == cudaSuccess) {
                    float a_amax_h = 0.f; memcpy(&a_amax_h, &abits, sizeof(float));
                    static std::mutex s_mu; static float s_hi = 0.f; static bool s_of = false;
                    std::lock_guard<std::mutex> lk(s_mu);
                    if (!std::isfinite(a_amax_h)) {
                        if (!s_of) { s_of = true;
                            fprintf(stderr, "[act-amax] *** NON-FINITE activation on '%s' [fp8 M=%d K=%d] -- the "
                                            "stream HAS overflowed (F16 max 65504); output is clipping\n",
                                    src1->name[0] ? src1->name : "(unnamed)", (int)M, (int)K); }
                    } else if (a_amax_h > s_hi) {
                        s_hi = a_amax_h;
                        const float headroom = 65504.0f / a_amax_h;
                        fprintf(stderr, "[act-amax] new max %.1f on '%s' [fp8 M=%d K=%d] -- f16 headroom %.1fx%s\n",
                                a_amax_h, src1->name[0] ? src1->name : "(unnamed)", (int)M, (int)K, headroom,
                                headroom < 4.0f ? "  <-- WARN: under 4x, an F16 stream is at risk" : "");
                    }
                }
            }
        }
        if (!chunk_act) {
            unsigned int qgrid = (unsigned int)((n_act + threads - 1)/threads);
            if (qgrid > 65535u) qgrid = 65535u;
            if (qgrid == 0)     qgrid = 1;
            if (src1->type == GGML_TYPE_F16) {
                fp8_a_quant_kernel<half> <<<qgrid, threads, 0, stream>>>((const half*)  src1->data, a_fp8_ptr, (long)n_act, a_scale_ptr);
            } else {
                fp8_a_quant_kernel<float><<<qgrid, threads, 0, stream>>>((const float*) src1->data, a_fp8_ptr, (long)n_act, a_scale_ptr);
            }
        }
        if (cudaPeekAtLastError() != cudaSuccess) { destroy_lt(); return false; }
        // Quant enqueued on `stream` -> the entry is now valid for any later GEMM in this same
        // compute (which necessarily runs after it on the same stream). Publishing here, not at
        // the miss, is what makes every early return above safe.
        if (act_publish_gen != (uint64_t)-1) g_fp8_act_cache.gen = act_publish_gen;
    }

    // cublasLtMatmul returns CUBLAS_STATUS_INVALID_VALUE (7) if a benign CUDA error is already
    // pending on the thread from an unrelated prior op (e.g. a RoPE/norm kernel between q and
    // k). Clear it so a legit fp8 matmul isn't spuriously rejected -> forced onto a fallback
    // that does not apply w_global. (Only clears already-consumed errors.)
    if (getenv("GGML_F8_CLEAR_ERR") == nullptr || atoi(getenv("GGML_F8_CLEAR_ERR")) != 0)
        (void)cudaGetLastError();

    // ------------------------------------------------------------------
    // 5) GEMM, one token slice at a time (a single iteration when nc == M).
    // ------------------------------------------------------------------
    bool ok = true;
    int  cur_slice_n = -1;
    for (int c0 = 0; c0 < M && ok; c0 += nc) {
        const int nn = (M - c0 < nc) ? (M - c0) : nc;

        cublasLtMatrixLayout_t Bl = Bd, Cl = Cd, Dl = Dd;
        if (nn != M) {
            if (nn != cur_slice_n) {
                if (Bc) { cublasLtMatrixLayoutDestroy(Bc); Bc = nullptr; }
                if (Cc) { cublasLtMatrixLayoutDestroy(Cc); Cc = nullptr; }
                if (Dc) { cublasLtMatrixLayoutDestroy(Dc); Dc = nullptr; }
                cublasLtMatrixLayoutCreate(&Bc, CUDA_R_8F_E4M3, k, nn, k);
                cublasLtMatrixLayoutCreate(&Cc, out_dt, m, nn, m);
                cublasLtMatrixLayoutCreate(&Dc, out_dt, m, nn, m);
                cur_slice_n = nn;
            }
            Bl = Bc; Cl = Cc; Dl = Dc;
        }

        const uint8_t * bptr;
        if (chunk_act) {
            // quantize just this token slice into the bounded buffer (same scale, same kernel)
            const int threads = 256;
            const size_t nsl = (size_t)nn * (size_t)K;
            unsigned int qgrid = (unsigned int)((nsl + threads - 1)/threads);
            if (qgrid > 65535u) qgrid = 65535u;
            if (qgrid == 0)     qgrid = 1;
            if (src1->type == GGML_TYPE_F16) {
                fp8_a_quant_kernel<half> <<<qgrid, threads, 0, stream>>>(((const half*) src1->data) + (size_t)c0*K, a_fp8_ptr, (long)nsl, a_scale_ptr);
            } else {
                fp8_a_quant_kernel<float><<<qgrid, threads, 0, stream>>>(((const float*)src1->data) + (size_t)c0*K, a_fp8_ptr, (long)nsl, a_scale_ptr);
            }
            if (cudaPeekAtLastError() != cudaSuccess) { ok = false; break; }
            bptr = a_fp8_ptr;
        } else {
            bptr = a_fp8_ptr + (size_t)c0 * (size_t)K;
        }

        // column-major D is m x n with ld = m = N, so slice [c0, c0+nn) is the contiguous run
        // of `dst` starting at element c0*N.
        void * gemm_out = use_f32_temp ? (void*)d_f32_ptr
                                       : (void*)((char*)dst->data + (size_t)c0*(size_t)N*dst_esz);

        const cublasStatus_t ms = cublasLtMatmul(lt, op, &alpha_h, w_fp8.get(), Ad, bptr, Bl,
                                                 &beta_h, gemm_out, Cl, gemm_out, Dl,
                                                 &algo, ws_ptr, wsz, stream);
        if (ms != CUBLAS_STATUS_SUCCESS) { ok = false; break; }

        const size_t nd = (size_t)nn * (size_t)N;
        const int thr = 256;
        unsigned int gr = (unsigned int)((nd + thr - 1) / thr);
        if (gr > 65535u) gr = 65535u;
        if (gr == 0)     gr = 1;
        if (use_f32_temp) {
            fp8_clamp_f32_to_f16<<<gr, thr, 0, stream>>>((const float*)d_f32_ptr,
                                                         ((half*)dst->data) + (size_t)c0*(size_t)N, nd);
            if (cudaPeekAtLastError() != cudaSuccess) { ok = false; break; }
        } else if (clamp_inplace) {
            fp8_fix_f16_inplace<<<gr, thr, 0, stream>>>(((half*)dst->data) + (size_t)c0*(size_t)N, nd);
            if (cudaPeekAtLastError() != cudaSuccess) { ok = false; break; }
        }
    }

    if (ok) {
        static int n_handled = 0;
        if (n_handled++ == 0 || getenv("GGML_NVFP4_CUBLASLT_TRACE"))
            fprintf(stderr, "[FP8_FFN] handled mul_mat #%d  M=%d K=%d N=%d nc=%d act=%s  name=%s (cuBLASLt FP8 GEMM)\n",
                    n_handled, M, K, N, nc,
                    act_reused ? "reused" : (act_cache_use ? "cached" : (chunk_act ? "sliced" : "pool")),
                    src0->name);
    }

    destroy_lt();
    return ok;
}
