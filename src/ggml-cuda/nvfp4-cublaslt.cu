// Phase-1 fast FP4 GEMM via cuBLASLt — see nvfp4-cublaslt.cuh.
#include <vector>
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
#include <mutex>
#include <unordered_map>
#include <map>
#include <tuple>
#include <string>
#include <cstring>

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

// In-place normalised size-16 Walsh-Hadamard transform (Sylvester, symmetric, /4 so
// H == H^T and H*H == I). NVIDIA's NVFP4 recipe uses a d=16 Hadamard matched to the
// micro-block: rotating each 16-channel activation block (and folding the SAME H into
// the weight offline) spreads the per-channel quant bias across the block so it cancels
// in the GEMM output instead of surviving as the structured drifting "blob". MUST be
// byte-for-byte the same transform as tools/import_ltx_nvfp4.py's had16(). Gated by
// GGML_NVFP4_ACT_HADAMARD (requires a matching Hadamard-folded gguf).
__device__ __forceinline__ void nvfp4_had16(float v[16]) {
#pragma unroll
    for (int h = 1; h < 16; h <<= 1) {
#pragma unroll
        for (int i = 0; i < 16; i += (h << 1)) {
#pragma unroll
            for (int j = i; j < i + h; j++) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b; v[j + h] = a - b;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 16; j++) v[j] *= 0.25f;
}

// Generic in-place normalised FWHT (Sylvester, symmetric) over N (power-of-2) elements,
// scaled by 1/sqrt(N) so H == H^T and H@H == I. nvfp4_had16 above is the N==16 special
// case (1/sqrt(16)==0.25). Used by the configurable block-Hadamard activation rotation
// (GGML_NVFP4_ACT_HADAMARD_BLOCK, default 256). MUST match the python H_N in
// tools/import_ltx_nvfp4_hadamard.py exactly (same butterfly order + single 1/sqrt(N) scale).
__device__ __forceinline__ void nvfp4_hadN(float * v, int N) {
    for (int h = 1; h < N; h <<= 1) {
        for (int i = 0; i < N; i += (h << 1)) {
            for (int j = i; j < i + h; j++) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b; v[j + h] = a - b;
            }
        }
    }
    const float s = 1.0f / sqrtf((float)N);   // IEEE sqrt (not approximate rsqrtf) to match python H_N
    for (int j = 0; j < N; j++) v[j] *= s;
}

// Cooperative block-Hadamard rotation kernel: one CUDA block rotates one B-element
// activation block (B = GGML_NVFP4_ACT_HADAMARD_BLOCK, power-of-2, B>16) in shared memory
// and writes the rotated F32 result to Y. The amax + quant kernels then run on Y with
// hadamard=false, i.e. the B-rotation is materialised once instead of redone per 16-sub
// (the FP4/cuBLASLt scale granularity stays per-16; only the rotation block grows). The B==16
// path stays fused in quant_act_kernel for byte-identity with the existing had16 gguf. The
// weight is folded offline with the SAME H_B, so x_rot @ W_rot == x @ W. Templated on the
// activation element type so the F16 residual stream feeds in with no extra cast.
template <typename act_t>
static __global__ void nvfp4_rotate_kernel(const act_t * __restrict__ X,
                                           float * __restrict__ Y, int B, size_t nblocks) {
    extern __shared__ float srot[];
    for (size_t blk = blockIdx.x; blk < nblocks; blk += gridDim.x) {
        const act_t * x = &X[blk * (size_t)B];
        float       * y = &Y[blk * (size_t)B];
        for (int i = threadIdx.x; i < B; i += blockDim.x) srot[i] = nvfp4_load_act(x[i]);
        __syncthreads();
        for (int h = 1; h < B; h <<= 1) {
            for (int p = threadIdx.x; p < (B >> 1); p += blockDim.x) {
                const int group = p / h, within = p % h;
                const int j = group * (h << 1) + within;       // butterfly partner = j+h, disjoint per stage
                const float a = srot[j], b = srot[j + h];
                srot[j] = a + b; srot[j + h] = a - b;
            }
            __syncthreads();
        }
        const float sc = 1.0f / sqrtf((float)B);   // IEEE sqrt to match python H_B (exact 1/16 for B=256)
        for (int i = threadIdx.x; i < B; i += blockDim.x) y[i] = srot[i] * sc;
        __syncthreads();   // before this block's shared mem is reused by the next grid-stride iter
    }
}

// FP4-quantize one activation element, optionally with stochastic rounding (dither).
// Stochastic rounding breaks the structured round-to-nearest bias that bands smooth /
// flat activation regions (the drifting "blob") into zero-mean grain; the per-element
// RNG is seeded from the element index + the value bits so it decorrelates across
// diffusion steps/frames yet stays reproducible. Gated by GGML_NVFP4_ACT_STOCHASTIC.
__device__ __forceinline__ uint8_t nvfp4_quant_elem(float val, float inv_scale,
                                                    unsigned int eidx, bool stoch) {
    if (stoch)
        return ggml_cuda_float_to_fp4_e2m1_stoch(val, inv_scale,
                   ggml_cuda_srand01(eidx, __float_as_uint(val)));
    return ggml_cuda_float_to_fp4_e2m1(val, inv_scale);
}

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
                                         unsigned int * __restrict__ out_bits, size_t n,
                                         bool hadamard) {
    float local = 0.f;
    if (hadamard) {
        // per-tensor amax must be measured on the SAME (rotated) values the block quant
        // sees, so the two-level scale stays self-consistent. 16-blocks are contiguous
        // (K % 16 == 0) so a flat 16-stride over M*K never straddles a row.
        const size_t nsub = n / 16;
        for (size_t b = (size_t)blockIdx.x*blockDim.x + threadIdx.x; b < nsub;
             b += (size_t)gridDim.x*blockDim.x) {
            const act_t * x = &X[b*16];
            float v[16];
            #pragma unroll
            for (int j=0;j<16;j++) v[j] = nvfp4_load_act(x[j]);
            nvfp4_had16(v);
            #pragma unroll
            for (int j=0;j<16;j++) local = fmaxf(local, fabsf(v[j]));
        }
    } else {
        for (size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x; i < n;
             i += (size_t)gridDim.x*blockDim.x) {
            local = fmaxf(local, fabsf(nvfp4_load_act(X[i])));
        }
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
                                        int M, int K, float per_tensor, bool stoch,
                                        bool hadamard, bool scale_ceil, bool refine_act) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int nsub = K/16;
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= M*nsub) return;
    const int r  = idx / nsub;
    const int ss = idx % nsub;
    const act_t * x = &X[(size_t)r*K + ss*16];
    const unsigned int eidx = (unsigned int)((size_t)r*K + (size_t)ss*16);  // SR seed base
    float vals[16];
    #pragma unroll
    for (int j=0;j<16;j++) vals[j]=nvfp4_load_act(x[j]);
    if (hadamard) nvfp4_had16(vals);   // size-16 rotation (matches the folded weight)
    float amax = 0.f;
    #pragma unroll
    for (int j=0;j<16;j++) amax = fmaxf(amax, fabsf(vals[j]));

    // comfy/ModelOpt TWO-LEVEL one-shot: the per-block scale (amax/6) is normalized by
    // the per-tensor global before being quantized to e4m3, so the e4m3 block-scale code
    // stays in its well-conditioned range (un-normalized amax/6 for low-magnitude blocks
    // lands in e4m3 subnormals -> coarse rounding -> the artifact the ±2 REFINE search was
    // papering over). The stored code is the NORMALIZED e4m3; the per_tensor factor is
    // carried by the cuBLASLt GEMM alpha (our weights already fold their own global into
    // the block scale, so alpha = A_per_tensor only). At per_tensor==1 this is byte-
    // identical to the single-level one-shot below.
    if (per_tensor > 0.f) {
        const float tgt = (amax/6.0f)/per_tensor;
        int tc = (int) ggml_cuda_fp32_to_ue4m3(tgt);
        tc = tc < 0 ? 0 : (tc > 0x7e ? 0x7e : tc);
        // "Four Over Six" scale round-UP: nearest e4m3 can land BELOW the target block scale,
        // making the block's peak element scale past E2M1's +-6 and clamp -> systematic
        // magnitude UNDERESTIMATE = the dimming blob. Bump to the next (larger) e4m3 code so
        // the scale is >= target -> no clip. Positive e4m3 byte order is monotonic in value.
        if (scale_ceil && tc < 0x7e && ggml_cuda_ue4m3_to_fp32((uint8_t)tc) < tgt) tc++;
        // Per-block scale REFINEMENT: search the +-2 neighbouring e4m3 scale codes and keep the
        // one with min L2 reconstruction error. Unlike scale-ceil it balances peak-clip against
        // small-value resolution PER BLOCK -> flat blocks (near the 0<->0.5*scale FP4 boundary,
        // the source of the drifting "worms") pick a finer scale, and a block can never collapse
        // to zero (that is max error, never chosen). Pure FP4, full GEMM speed. GGML_NVFP4_ACT_REFINE.
        if (refine_act) {
            float best_err = FLT_MAX; int best_tc = tc;
            #pragma unroll
            for (int off=-2; off<=2; off++) {
                const int cc = tc + off;
                if (cc < 0 || cc > 0x7e) continue;
                const float tot = per_tensor * ggml_cuda_ue4m3_to_fp32((uint8_t)cc);
                if (!(tot > 0.f)) continue;
                const float inv = 0.5f/tot;
                float err = 0.f;
                #pragma unroll
                for (int k=0;k<16;k++) {
                    const uint8_t q = ggml_cuda_float_to_fp4_e2m1(vals[k], inv);
                    const float ed = fabsf(vals[k]) - kvalues_mxfp4[q & 0x7] * tot;
                    err = fmaf(ed, ed, err);
                }
                if (err < best_err) { best_err = err; best_tc = cc; }
            }
            tc = best_tc;
        }
        const uint8_t code = (uint8_t)tc;
        out_scales[swz_off(r, ss, nsub)] = code;
        const float total = per_tensor * ggml_cuda_ue4m3_to_fp32(code);
        const float inv_scale = total > 0.f ? 0.5f/total : 0.f;
        uint8_t * od = &out_data[(size_t)r*(K/2) + ss*8];
        #pragma unroll
        for (int t=0;t<8;t++) {
            uint8_t n0 = nvfp4_quant_elem(vals[2*t],   inv_scale, eidx+2*t,   stoch);
            uint8_t n1 = nvfp4_quant_elem(vals[2*t+1], inv_scale, eidx+2*t+1, stoch);
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
        int tc = first_code < 0 ? 0 : (first_code > 0x7e ? 0x7e : first_code);
        // "Four Over Six" scale round-UP (see two-level branch): avoid peak-element clipping.
        if (scale_ceil && tc < 0x7e && ggml_cuda_ue4m3_to_fp32((uint8_t)tc) < amax/6.0f) tc++;
        fp8_code = (uint8_t)tc;
        subblock_scale = ggml_cuda_ue4m3_to_fp32((uint8_t)tc);
    }
    out_scales[swz_off(r, ss, nsub)] = fp8_code;
    const float inv_scale = subblock_scale > 0.f ? 0.5f/subblock_scale : 0.f;
    uint8_t * od = &out_data[(size_t)r*(K/2) + ss*8];   // consecutive packing for cuBLASLt
    #pragma unroll
    for (int t=0;t<8;t++) {
        uint8_t n0 = nvfp4_quant_elem(vals[2*t],   inv_scale, eidx+2*t,   stoch);
        uint8_t n1 = nvfp4_quant_elem(vals[2*t+1], inv_scale, eidx+2*t+1, stoch);
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

struct nvfp4_weight_repacked {
    uint8_t * data   = nullptr;   // consecutive E2M1   (in-place: offset 0 of src0->data)
    uint8_t * scales = nullptr;   // swizzled UE4M3      (in-place: offset data_bytes)
    bool      in_place = false;   // true => data/scales alias the ggml-owned src0 buffer
                                  //         (teardown must NOT cudaFree them)
};

static std::mutex                                   g_repack_mtx;
static std::unordered_map<const void*, nvfp4_weight_repacked> g_repack_cache;

// Per-tensor weight global (ModelOpt weight_scale_2), keyed by tensor NAME (not data
// pointer: layer-offload twin-swaps src0->data per segment but preserves src0->name).
// Folded into the GEMM alpha (alpha = A_global * W_global) so the stored per-block
// ue4m3 weight scales stay in their well-conditioned e4m3 range instead of underflowing
// into subnormals (the "patchy colour" artifact). Absent name => 1.0 (legacy/folded
// gguf is byte-identical, since it has no registered globals).
static std::mutex                          g_wglobal_mtx;
static std::unordered_map<std::string, float> g_wglobal;

extern "C" void ggml_cuda_nvfp4_register_weight_global(const char * name, float g) {
    if (!name || !*name) return;
    std::lock_guard<std::mutex> lk(g_wglobal_mtx);
    g_wglobal[name] = g;
}

static float nvfp4_weight_global_for(const char * name) {
    if (!name || !*name) return 1.0f;
    std::lock_guard<std::mutex> lk(g_wglobal_mtx);
    auto it = g_wglobal.find(name);
    return it == g_wglobal.end() ? 1.0f : it->second;
}

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

    // 2) quantize activation into cuBLASLt layout (pool scratch)
    const int nsub = K/16;
    const size_t a_data_bytes  = (size_t)M*(K/2);
    const size_t a_rb_p = ((size_t)(M+127)/128)*128;
    const size_t a_cb_p = ((size_t)(nsub+3)/4)*4;
    const size_t a_scale_bytes = a_rb_p*a_cb_p;
    ggml_cuda_pool_alloc<uint8_t> a_data(ctx.pool(), a_data_bytes);
    ggml_cuda_pool_alloc<uint8_t> a_scales(ctx.pool(), a_scale_bytes);
    cudaMemsetAsync(a_scales.get(), 0, a_scale_bytes, stream);
    // TWO-LEVEL (comfy-faithful) one-shot: compute the per-tensor activation global
    // (amax / (6*448)) so block scales normalize into e4m3 range. Carried into the GEMM
    // alpha below. Gated GGML_NVFP4_QUANT_TWOLEVEL; supersedes NOREFINE when set.
    // size-16 Hadamard rotation of the activation (matched to a Hadamard-folded weight
    // gguf). Off by default. Both the per-tensor amax and the block quant below apply the
    // SAME rotation so the two-level scale stays self-consistent.
    static int s_hadamard = -1;
    if (s_hadamard < 0) { const char* e = getenv("GGML_NVFP4_ACT_HADAMARD"); s_hadamard = (e && atoi(e)) ? 1 : 0; }
    const bool hadamard = s_hadamard != 0;
    // Configurable activation-Hadamard block size. Default 256 (NVIDIA NVFP4 recipe block);
    // the legacy had16 gguf still works via block=16. Must be a power-of-2 >= 16 (else clamped
    // to 16). PER-TENSOR: if K isn't divisible by the requested block, this tensor falls back to
    // 16 — the python import (NVFP4_HAD_BLOCK) applies the IDENTICAL rule so the folded weight
    // always matches the online rotation. Only consulted when hadamard is on. block==16 stays
    // fused in quant_act_kernel (in-kernel had16, byte-identical to the old path); block>16 is
    // materialised once by nvfp4_rotate_kernel into an F32 scratch, then quantized hadamard=false.
    static int s_had_block = -1;
    if (s_had_block < 0) {
        const char* e = getenv("GGML_NVFP4_ACT_HADAMARD_BLOCK");
        int b = e ? atoi(e) : 256;
        if (b < 16 || (b & (b - 1)) != 0) b = 16;
        s_had_block = b;
    }
    const int  had_block    = (hadamard && (K % s_had_block) == 0) ? s_had_block : 16;
    const bool rotate_blk   = hadamard && had_block > 16;     // pre-rotate into F32 scratch
    const bool inkernel_had = hadamard && had_block == 16;    // fused had16 inside the quant kernel
    // Pre-rotate the whole activation matrix by the block-B Hadamard (once), so amax + quant run
    // on the rotated values with no further in-kernel rotation. Only for block>16.
    ggml_cuda_pool_alloc<float> a_rot(ctx.pool());
    const float * rot_src = nullptr;
    if (rotate_blk) {
        a_rot.alloc((size_t)M * (size_t)K);
        const size_t nblocks = (size_t)M * (size_t)K / (size_t)had_block;
        const int    rthreads = had_block < 1024 ? had_block : 1024;
        unsigned int rgrid    = nblocks > 65535u ? 65535u : (unsigned int)nblocks;
        if (rgrid == 0) rgrid = 1;
        const size_t shmem = (size_t)had_block * sizeof(float);
        if (src1->type == GGML_TYPE_F16)
            nvfp4_rotate_kernel<half><<<rgrid, rthreads, shmem, stream>>>((const half*)src1->data, a_rot.get(), had_block, nblocks);
        else
            nvfp4_rotate_kernel<float><<<rgrid, rthreads, shmem, stream>>>((const float*)src1->data, a_rot.get(), had_block, nblocks);
        if (cudaPeekAtLastError() != cudaSuccess) return false;
        rot_src = a_rot.get();
    }
    // "Four Over Six" activation scale round-up — kills peak-element clip (dimming). Off by default.
    static int s_scaleceil = -1;
    if (s_scaleceil < 0) { const char* e = getenv("GGML_NVFP4_ACT_SCALE_CEIL"); s_scaleceil = (e && atoi(e)) ? 1 : 0; }
    const bool scale_ceil = s_scaleceil != 0;
    // Per-block scale refinement search (two-level path) — min-error scale per block. Off by default.
    static int s_refine = -1;
    if (s_refine < 0) { const char* e = getenv("GGML_NVFP4_ACT_REFINE"); s_refine = (e && atoi(e)) ? 1 : 0; }
    const bool refine_act = s_refine != 0;
    float a_per_tensor = 0.f;
    {
        static int s_twolevel = -1;
        if (s_twolevel < 0) { const char* e = getenv("GGML_NVFP4_QUANT_TWOLEVEL"); s_twolevel = (e && atoi(e)) ? 1 : 0; }
        if (s_twolevel) {
            ggml_cuda_pool_alloc<unsigned int> amax_d(ctx.pool(), 1);
            cudaMemsetAsync(amax_d.get(), 0, sizeof(unsigned int), stream);
            const size_t n = (size_t)M*K;
            const int thr = 256;
            const int blk = (int)((n + thr - 1)/thr > 1024 ? 1024 : (n + thr - 1)/thr);
            if (rotate_blk)                       nvfp4_amax_kernel<float><<<blk, thr, 0, stream>>>(rot_src, amax_d.get(), n, false);
            else if (src1->type == GGML_TYPE_F16) nvfp4_amax_kernel<half><<<blk, thr, 0, stream>>>((const half*)src1->data, amax_d.get(), n, inkernel_had);
            else                                  nvfp4_amax_kernel<float><<<blk, thr, 0, stream>>>((const float*)src1->data, amax_d.get(), n, inkernel_had);
            unsigned int bits = 0;
            cudaMemcpyAsync(&bits, amax_d.get(), sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            float amax_t = 0.f; memcpy(&amax_t, &bits, sizeof(float));
            a_per_tensor = amax_t > 0.f ? amax_t/(6.0f*448.0f) : 0.f;
        }
    }
    {
        static int s_norefine = -1;
        if (s_norefine < 0) { const char* e = getenv("GGML_NVFP4_QUANT_NOREFINE"); s_norefine = (e && atoi(e)) ? 1 : 0; }
        // Stochastic rounding of the FP4 activation codes (dither) — env-gated, off by default
        // so prod/flux2 stay byte-identical. Decorrelates the structured banding/blob into grain.
        static int s_stoch = -1;
        if (s_stoch < 0) { const char* e = getenv("GGML_NVFP4_ACT_STOCHASTIC"); s_stoch = (e && atoi(e)) ? 1 : 0; }
        const bool stoch = s_stoch != 0;
        const int threads = 256;
        const int total   = M*nsub;
        const int blocks  = (total+threads-1)/threads;
        const float pt = a_per_tensor;   // >0 => two-level branch in the kernel
        if (rotate_blk) {
            // activation already block-rotated into rot_src (F32); quantize per-16 with no further rotation.
            if (s_norefine || pt > 0.f) quant_act_kernel<float,false><<<blocks, threads, 0, stream>>>(rot_src, a_data.get(), a_scales.get(), M, K, pt, stoch, false, scale_ceil, refine_act);
            else                        quant_act_kernel<float,true ><<<blocks, threads, 0, stream>>>(rot_src, a_data.get(), a_scales.get(), M, K, pt, stoch, false, scale_ceil, refine_act);
        } else if (src1->type == GGML_TYPE_F16) {
            if (s_norefine || pt > 0.f) quant_act_kernel<half,false><<<blocks, threads, 0, stream>>>((const half*)src1->data, a_data.get(), a_scales.get(), M, K, pt, stoch, inkernel_had, scale_ceil, refine_act);
            else                        quant_act_kernel<half,true ><<<blocks, threads, 0, stream>>>((const half*)src1->data, a_data.get(), a_scales.get(), M, K, pt, stoch, inkernel_had, scale_ceil, refine_act);
        } else {
            if (s_norefine || pt > 0.f) quant_act_kernel<float,false><<<blocks, threads, 0, stream>>>((const float*)src1->data, a_data.get(), a_scales.get(), M, K, pt, stoch, inkernel_had, scale_ceil, refine_act);
            else                        quant_act_kernel<float,true ><<<blocks, threads, 0, stream>>>((const float*)src1->data, a_data.get(), a_scales.get(), M, K, pt, stoch, inkernel_had, scale_ceil, refine_act);
        }
        if (cudaPeekAtLastError() != cudaSuccess) return false;
    }

    // 3) cuBLASLt FP4 GEMM: D[M,N] (row-major) = A_w[N,K] @ B_a[M,K]^T
    // cuBLAS column-major: m=N, n=M, k=K; A=weight (TN), B=activation.
    // alpha = A_per_tensor for the two-level activation quant (carries the per-tensor
    // global the kernel factored out of the stored block scales); 1.0 otherwise. Weights
    // already fold their own global into the block scale, so alpha carries only A's.
    const int m=N, n=M, k=K;
    // alpha carries BOTH per-tensor globals: A's (from two-level act-quant) and W's
    // (ModelOpt weight_scale_2, registered at load — see g_wglobal). An UNFOLDED-import
    // gguf stores well-conditioned block scales + registers W's global here; a legacy
    // FOLDED gguf registers nothing -> w_global=1.0 -> byte-identical to before.
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
    void* wsp = (void*)w_scales.get(); void* asp = (void*)a_scales.get();
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
        cublasStatus_t ms = cublasLtMatmul(lt, op, &alpha_h, w_data.get(), Ad, a_data.get(), Bd,
                                           &beta_h, dst->data, Cd, dst->data, Dd,
                                           &algo, ws.get(), wsz, stream);
        ok = (ms == CUBLAS_STATUS_SUCCESS);
    }

    if (ok) {
        static int n_handled = 0;
        if (n_handled++ == 0 || getenv("GGML_NVFP4_CUBLASLT_TRACE"))
            fprintf(stderr, "[NVFP4_CUBLASLT] handled mul_mat #%d  M=%d K=%d N=%d (cuBLASLt FP4 GEMM, algo-cached)\n",
                    n_handled, M, K, N);
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
        // GGML_NVFP4_DBG: full-tensor scan to pin the FIRST GEMM that emits a non-finite or
        // blown-up output (the two-level-quant NaN origin). Reports the weight name + the
        // per-tensor activation global (amax = a_per_tensor*2688) so an outlier-driven
        // underflow shows up directly. Debug-only (copies all M*N); prints once then quiets.
        if (getenv("GGML_NVFP4_DBG")) {
            static int s_found = 0;
            if (!s_found) {
                const size_t ne = (size_t)M * (size_t)N;
                size_t nnan=0, ninf=0; float mx=0.f; size_t first=(size_t)-1;
                if (dst->type == GGML_TYPE_F32) {
                    std::vector<float> hb(ne);
                    cudaMemcpyAsync(hb.data(), dst->data, ne*sizeof(float), cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    for (size_t i=0;i<ne;i++){ float v=hb[i]; if(isnan(v)){nnan++; if(first==(size_t)-1)first=i;} else if(isinf(v)){ninf++; if(first==(size_t)-1)first=i;} mx=fmaxf(mx,fabsf(isfinite(v)?v:0.f)); }
                } else if (dst->type == GGML_TYPE_F16) {
                    std::vector<uint16_t> hb(ne);
                    cudaMemcpyAsync(hb.data(), dst->data, ne*sizeof(uint16_t), cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    for (size_t i=0;i<ne;i++){ uint16_t h=hb[i]; if((h&0x7C00)==0x7C00){ if(h&0x3FF) nnan++; else ninf++; if(first==(size_t)-1)first=i; } }
                }
                if (nnan||ninf||mx>1e4f) {
                    s_found = 1;
                    fprintf(stderr, "[NVFP4_DBG] FIRST-BAD name=%s dst=%s M=%d K=%d N=%d  a_per_tensor=%g (amax_act~%g) w_global=%g alpha=%g  nnan=%zu ninf=%zu max=%g firstbad@%zu\n",
                            src0->name?src0->name:"?", dst->type==GGML_TYPE_F16?"F16":"F32", M, K, N, a_per_tensor, a_per_tensor*2688.f, w_global, alpha_h, nnan, ninf, mx, first);
                }
            }
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
// The FFN linears carry the structured FP4-activation "worm"; promoting them to an
// 8-bit-activation GEMM (e4m3) removes it (Q4_K-clean = 4-bit weight + 8-bit act) at
// ~2x FP4 cost instead of BF16's ~8x. cuBLASLt only supports FP8xFP8 (mixed FP8xFP4 is
// unsupported on sm120 — verified by fp8probe.cu), so the stored NVFP4 weight is decoded
// to e4m3 verbatim (its FP4 grid, no extra weight loss) and the activation is quantized
// to e4m3. Both use a per-tensor (SCALAR_32F) scale = amax/448 carried on the cuBLASLt
// A/B scale pointers; alpha = 1. Nothing here runs unless GGML_FP8_FFN=1.

#define FP8_E4M3_MAX 448.0f

// per-tensor amax of the *reconstructed* NVFP4 weight (decoded to true float values,
// matching the cuBLASLt convention: kvalues_mxfp4 = 2x e2m1, ue4m3_to_fp32 = scale/2, the
// 2x cancel => stored bytes are the standard e2m1*e4m3 product; * the per-tensor wglobal).
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

// decode NVFP4 weight -> e4m3, row-major [N,K] (1 byte/elem). inv = 1/scale (read from dev).
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

bool ggml_cuda_fp8_ffn_enabled() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_FP8_FFN"); v = (e && atoi(e)) ? 1 : 0; }
    return v == 1;
}

// substring filter (GGML_FP8_LAYERS, default "ff.net"): matches the DiT FFN up/gate
// (ff.net.0.proj) + down-proj (ff.net.2) Linears; attention/other linears never match.
bool ggml_cuda_fp8_ffn_name_match(const char * name) {
    if (!name) return false;
    static std::string filt;
    static int init = 0;
    if (!init) { const char * e = getenv("GGML_FP8_LAYERS"); filt = (e && *e) ? e : "ff.net"; init = 1; }
    return strstr(name, filt.c_str()) != nullptr;
}

bool ggml_cuda_fp8_cublaslt_mul_mat(ggml_backend_cuda_context & ctx,
                                    const ggml_tensor * src0,
                                    const ggml_tensor * src1,
                                    ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_NVFP4) return false;   // weight source = the stored FP4 FFN weight
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

    const float w_global = nvfp4_weight_global_for(src0->name);

    // 1) weight -> e4m3 [N,K] (pool scratch, per-call: handles resident & offload weights)
    ggml_cuda_pool_alloc<uint8_t>      w_fp8(ctx.pool(), (size_t)N*K);
    ggml_cuda_pool_alloc<unsigned int> w_amax(ctx.pool(), 1);
    ggml_cuda_pool_alloc<float>        w_scale(ctx.pool(), 1);
    cudaMemsetAsync(w_amax.get(), 0, sizeof(unsigned int), stream);
    {
        const int threads = 256;
        const long total  = (long)N*(K/16);
        unsigned int grid = (unsigned int)((total + threads - 1)/threads);
        if (grid > 65535u) grid = 65535u; if (grid == 0) grid = 1;
        fp8_w_amax_kernel<<<grid, threads, 0, stream>>>((const block_nvfp4*)src0->data, w_amax.get(), N, K, w_global);
        fp8_scale_from_amax<<<1, 1, 0, stream>>>(w_amax.get(), w_scale.get());
        const unsigned int qgrid = (unsigned int)((total + threads - 1)/threads);
        fp8_w_quant_kernel<<<qgrid ? qgrid : 1, threads, 0, stream>>>((const block_nvfp4*)src0->data, w_fp8.get(), N, K, w_global, w_scale.get());
        if (cudaPeekAtLastError() != cudaSuccess) return false;
    }

    // 2) activation -> e4m3 [M,K] (flat; src1 [K,M] contiguous == row-major [M,K])
    ggml_cuda_pool_alloc<uint8_t>      a_fp8(ctx.pool(), (size_t)M*K);
    ggml_cuda_pool_alloc<unsigned int> a_amax(ctx.pool(), 1);
    ggml_cuda_pool_alloc<float>        a_scale(ctx.pool(), 1);
    cudaMemsetAsync(a_amax.get(), 0, sizeof(unsigned int), stream);
    {
        const long n = (long)M*K;
        const int  threads = 256;
        unsigned int grid = (unsigned int)((n + threads - 1)/threads);
        if (grid > 1024u) grid = 1024u; if (grid == 0) grid = 1;
        if (src1->type == GGML_TYPE_F16) fp8_a_amax_kernel<half ><<<grid, threads, 0, stream>>>((const half*)src1->data,  a_amax.get(), n);
        else                             fp8_a_amax_kernel<float><<<grid, threads, 0, stream>>>((const float*)src1->data, a_amax.get(), n);
        fp8_scale_from_amax<<<1, 1, 0, stream>>>(a_amax.get(), a_scale.get());
        unsigned int qgrid = (unsigned int)((n + threads - 1)/threads);
        if (qgrid > 65535u) qgrid = 65535u; if (qgrid == 0) qgrid = 1;
        if (src1->type == GGML_TYPE_F16) fp8_a_quant_kernel<half ><<<qgrid, threads, 0, stream>>>((const half*)src1->data,  a_fp8.get(), n, a_scale.get());
        else                             fp8_a_quant_kernel<float><<<qgrid, threads, 0, stream>>>((const float*)src1->data, a_fp8.get(), n, a_scale.get());
        if (cudaPeekAtLastError() != cudaSuccess) return false;
    }

    // 3) cuBLASLt FP8xFP8 GEMM: D[M,N] = W_fp8[N,K] @ A_fp8[M,K]^T. Column-major m=N,n=M,k=K;
    //    A=weight (TN, e4m3), B=act (N, e4m3). Per-tensor SCALAR scales on A/B pointers; alpha=1.
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
    void* wsp = (void*)w_scale.get(); void* asp = (void*)a_scale.get();
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &wsp, sizeof(wsp));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &asp, sizeof(asp));
    cublasDataType_t st = CUDA_R_32F;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &st, sizeof(st));

    const cublasDataType_t out_dt = (dst->type == GGML_TYPE_F16) ? CUDA_R_16F : CUDA_R_32F;
    cublasLtMatrixLayout_t Ad=nullptr,Bd=nullptr,Cd=nullptr,Dd=nullptr;
    cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8F_E4M3, k, m, k);
    cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8F_E4M3, k, n, k);
    cublasLtMatrixLayoutCreate(&Cd, out_dt, m, n, m);
    cublasLtMatrixLayoutCreate(&Dd, out_dt, m, n, m);

    ggml_cuda_pool_alloc<uint8_t> ws(ctx.pool(), 32*1024*1024);
    size_t wsz = 32*1024*1024;

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
        if (hs == CUBLAS_STATUS_SUCCESS && got > 0) { algo = hr.algo; have_algo = true; if (!s_nocache) g_fp8_algo_cache[key] = algo; }
    }

    bool ok = have_algo;
    if (ok) {
        cublasStatus_t ms = cublasLtMatmul(lt, op, &alpha_h, w_fp8.get(), Ad, a_fp8.get(), Bd,
                                           &beta_h, dst->data, Cd, dst->data, Dd,
                                           &algo, ws.get(), wsz, stream);
        ok = (ms == CUBLAS_STATUS_SUCCESS);
    }

    if (ok) {
        static int n_handled = 0;
        if (n_handled++ == 0 || getenv("GGML_NVFP4_CUBLASLT_TRACE"))
            fprintf(stderr, "[FP8_FFN] handled mul_mat #%d  M=%d K=%d N=%d  name=%s (cuBLASLt FP8 GEMM)\n",
                    n_handled, M, K, N, src0->name);
    }

    if (Ad) cublasLtMatrixLayoutDestroy(Ad);
    if (Bd) cublasLtMatrixLayoutDestroy(Bd);
    if (Cd) cublasLtMatrixLayoutDestroy(Cd);
    if (Dd) cublasLtMatrixLayoutDestroy(Dd);
    if (op) cublasLtMatmulDescDestroy(op);
    return ok;
}
