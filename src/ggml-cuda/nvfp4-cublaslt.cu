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
#include <set>
#include <tuple>
#include <string>
#include <cstring>
#include <atomic>

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

// Stochastic rounding of the per-16-block UE4M3 activation SCALE code (not the E2M1
// element). ggml_cuda_fp32_to_ue4m3 round-to-nearest bakes a per-block scale bias that
// is COHERENT across blocks and frames -> the drifting nvfp4 "grid". Instead, pick which
// of the two bracketing e4m3 scale codes to store at random, weighted by the fractional
// distance of the target between them: E[stored scale] == target (unbiased), so the
// coherent bias becomes a zero-mean ~sqrt(N) random walk that averages out. rnd in [0,1).
// t must be > 0 (the block target amax/scale_div[/per_tensor], same value fed to
// ggml_cuda_fp32_to_ue4m3); t==0 returns code 0 (==RTN). NOTE the grid: fp32_to_ue4m3 snaps t
// to the *standard* e4m3 value grid, while ue4m3_to_fp32 decodes standard/2 (ggml's UE /2
// convention, cancelled later by the 2x E2M1 LUT). So the code's grid value is 2*ue4m3_to_fp32;
// t is bracketed on THAT grid so E[stored scale] == t (unbiased) exactly like the RTN it replaces.
// Positive e4m3 byte order is monotonic in value, so the RTN code and its neighbour bracket t.
__device__ __forceinline__ uint8_t nvfp4_ue4m3_stoch(float t, float rnd) {
    int rtn = (int) ggml_cuda_fp32_to_ue4m3(t);
    rtn = rtn < 0 ? 0 : (rtn > 0x7e ? 0x7e : rtn);
    int c_lo, c_hi;
    if (2.0f * ggml_cuda_ue4m3_to_fp32((uint8_t)rtn) <= t) { c_lo = rtn;     c_hi = rtn + 1; }
    else                                                   { c_lo = rtn - 1; c_hi = rtn;     }
    c_lo = c_lo < 0 ? 0 : c_lo;
    c_hi = c_hi > 0x7e ? 0x7e : c_hi;
    const float v_lo = 2.0f * ggml_cuda_ue4m3_to_fp32((uint8_t)c_lo);   // standard-e4m3 grid values
    const float v_hi = 2.0f * ggml_cuda_ue4m3_to_fp32((uint8_t)c_hi);
    const float span = v_hi - v_lo;
    const float frac = span > 0.f ? (t - v_lo) / span : 0.f;   // 0 at/below range floor, clamps at top
    return (uint8_t)((rnd < frac) ? c_hi : c_lo);
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
                                        bool hadamard, bool scale_ceil, bool refine_act, float scale_div,
                                        bool scale_stoch) {
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
        const float tgt = (amax/scale_div)/per_tensor;
        // Stochastic block-scale rounding (GGML_NVFP4_ACT_SCALE_STOCH) SUPERSEDES the RTN +
        // ceil/refine deterministic strategies: it is the alternative that breaks the coherent
        // grid into a zero-mean random walk. Keyed on the (block,frame) index so it stays
        // reproducible for a fixed seed but decorrelates across frames (tgt drifts).
        int tc = scale_stoch
            ? (int) nvfp4_ue4m3_stoch(tgt, ggml_cuda_srand01((unsigned int)(r*nsub+ss), __float_as_uint(tgt)))
            : (int) ggml_cuda_fp32_to_ue4m3(tgt);
        tc = tc < 0 ? 0 : (tc > 0x7e ? 0x7e : tc);
        // "Four Over Six" scale round-UP: nearest e4m3 can land BELOW the target block scale,
        // making the block's peak element scale past E2M1's +-6 and clamp -> systematic
        // magnitude UNDERESTIMATE = the dimming blob. Bump to the next (larger) e4m3 code so
        // the scale is >= target -> no clip. Positive e4m3 byte order is monotonic in value.
        if (!scale_stoch && scale_ceil && tc < 0x7e && ggml_cuda_ue4m3_to_fp32((uint8_t)tc) < tgt) tc++;
        // Per-block scale REFINEMENT: search the +-2 neighbouring e4m3 scale codes and keep the
        // one with min L2 reconstruction error. Unlike scale-ceil it balances peak-clip against
        // small-value resolution PER BLOCK -> flat blocks (near the 0<->0.5*scale FP4 boundary,
        // the source of the drifting "worms") pick a finer scale, and a block can never collapse
        // to zero (that is max error, never chosen). Pure FP4, full GEMM speed. GGML_NVFP4_ACT_REFINE.
        // Skipped under scale_stoch: the min-error search is a deterministic scale strategy that
        // would collapse the random walk back to the biased RTN pick.
        if (!scale_stoch && refine_act) {
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

    uint8_t fp8_code; float subblock_scale;
    if (scale_stoch) {
        // Single-level stochastic block-scale rounding (GGML_NVFP4_ACT_SCALE_STOCH): supersedes
        // both the REFINE min-error search and the ceil, exactly as in the two-level branch. Same
        // per-(block,frame) RNG keying so it's reproducible for a fixed seed, coherent-bias-free.
        const float t = amax/scale_div;
        int tc = (int) nvfp4_ue4m3_stoch(t, ggml_cuda_srand01((unsigned int)(r*nsub+ss), __float_as_uint(t)));
        tc = tc < 0 ? 0 : (tc > 0x7e ? 0x7e : tc);
        fp8_code = (uint8_t)tc;
        subblock_scale = ggml_cuda_ue4m3_to_fp32((uint8_t)tc);
    } else {
    const int first_code = (int) ggml_cuda_fp32_to_ue4m3(amax/scale_div);
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
        if (scale_ceil && tc < 0x7e && ggml_cuda_ue4m3_to_fp32((uint8_t)tc) < amax/scale_div) tc++;
        fp8_code = (uint8_t)tc;
        subblock_scale = ggml_cuda_ue4m3_to_fp32((uint8_t)tc);
    }
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
    // "Four Over Six" (arXiv 2512.02010): target the block peak at E2M1's 4 (not 6) so the
    // round-to-nearest e4m3 block scale can't push the peak past 6 and clip -> removes the
    // systematic DOWNWARD magnitude bias that is worst in flat/low-variance regions and is the
    // DETERMINISTIC (coherent, temporally-accumulating) part of the nvfp4 grid. Env-gated,
    // default 6.0f => byte-identical to prod. Pairs with GGML_NVFP4_ACT_SCALE_CEIL.
    static int s_4o6 = -1;
    if (s_4o6 < 0) { const char* e = getenv("GGML_NVFP4_ACT_4OVER6"); s_4o6 = (e && atoi(e)) ? 1 : 0; }
    const float scale_div = s_4o6 ? 4.0f : 6.0f;
    // Per-block scale refinement search (two-level path) — min-error scale per block. Off by default.
    static int s_refine = -1;
    if (s_refine < 0) { const char* e = getenv("GGML_NVFP4_ACT_REFINE"); s_refine = (e && atoi(e)) ? 1 : 0; }
    const bool refine_act = s_refine != 0;
    // Stochastic rounding of the per-16-block UE4M3 activation SCALE code — env-gated, off by
    // default (byte-identical to prod). When on it supersedes scale_ceil/refine at both the two-
    // level and single-level quant sites: randomizes up/down between the two bracketing e4m3 scale
    // codes (weighted by fractional distance, unbiased) so the coherent grid becomes a ~sqrt(N)
    // random walk. Reproducible for a fixed seed (keyed on the (block,frame) index). GGML_NVFP4_ACT_SCALE_STOCH.
    static int s_scalestoch = -1;
    if (s_scalestoch < 0) { const char* e = getenv("GGML_NVFP4_ACT_SCALE_STOCH"); s_scalestoch = (e && atoi(e)) ? 1 : 0; }
    const bool scale_stoch = s_scalestoch != 0;
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
            if (s_norefine || pt > 0.f) quant_act_kernel<float,false><<<blocks, threads, 0, stream>>>(rot_src, a_data.get(), a_scales.get(), M, K, pt, stoch, false, scale_ceil, refine_act, scale_div, scale_stoch);
            else                        quant_act_kernel<float,true ><<<blocks, threads, 0, stream>>>(rot_src, a_data.get(), a_scales.get(), M, K, pt, stoch, false, scale_ceil, refine_act, scale_div, scale_stoch);
        } else if (src1->type == GGML_TYPE_F16) {
            if (s_norefine || pt > 0.f) quant_act_kernel<half,false><<<blocks, threads, 0, stream>>>((const half*)src1->data, a_data.get(), a_scales.get(), M, K, pt, stoch, inkernel_had, scale_ceil, refine_act, scale_div, scale_stoch);
            else                        quant_act_kernel<half,true ><<<blocks, threads, 0, stream>>>((const half*)src1->data, a_data.get(), a_scales.get(), M, K, pt, stoch, inkernel_had, scale_ceil, refine_act, scale_div, scale_stoch);
        } else {
            if (s_norefine || pt > 0.f) quant_act_kernel<float,false><<<blocks, threads, 0, stream>>>((const float*)src1->data, a_data.get(), a_scales.get(), M, K, pt, stoch, inkernel_had, scale_ceil, refine_act, scale_div, scale_stoch);
            else                        quant_act_kernel<float,true ><<<blocks, threads, 0, stream>>>((const float*)src1->data, a_data.get(), a_scales.get(), M, K, pt, stoch, inkernel_had, scale_ceil, refine_act, scale_div, scale_stoch);
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

// LOUD debug: exact NaN/Inf counts + finite max-abs (NOT fmaxf-amax, which SWALLOWS NaN).
// out[0]=nnan  out[1]=ninf  out[2]=finite-amax bits.  For GGML_F8_DBG value tracing.
template <typename act_t>
static __global__ void fp8_nanstats_kernel(const act_t * __restrict__ X, long n,
                                           unsigned int * __restrict__ out) {
    unsigned int lnan = 0, linf = 0; float lamax = 0.f;
    for (long i = (long)blockIdx.x*blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x*blockDim.x) {
        const float v = nvfp4_load_act(X[i]);
        if (isnan(v))       ++lnan;
        else if (isinf(v))  ++linf;
        else                lamax = fmaxf(lamax, fabsf(v));
    }
    atomicAdd(out + 0, lnan);
    atomicAdd(out + 1, linf);
    atomicMax(out + 2, __float_as_uint(lamax));
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

// Non-static wrapper exposing the per-tensor e4m3 quantization above to other TUs
// (the FP8 flash-attention kernel quantizes Q/K with it). Reuses the exact amax ->
// scale (amax/448) -> per-element __nv_fp8_e4m3(x/scale) math; no duplication. `X` is a
// contiguous [n]-element F16 or F32 buffer, `out` receives n e4m3 bytes, `d_scale`
// (1 float, caller-owned) receives the scalar scale, `d_amax` (1 uint) is scratch.
void ggml_cuda_fp8_quant_pertensor(const void * X, ggml_type xtype,
                                   uint8_t * out, float * d_scale, unsigned int * d_amax,
                                   long n, cudaStream_t stream) {
    cudaMemsetAsync(d_amax, 0, sizeof(unsigned int), stream);
    const int bs = 256;
    long gsl = (n + bs - 1) / bs;
    if (gsl > 1024) gsl = 1024;
    if (gsl < 1)    gsl = 1;
    const int gs = (int) gsl;
    if (xtype == GGML_TYPE_F16) {
        fp8_a_amax_kernel<half> <<<gs, bs, 0, stream>>>((const half *)  X, d_amax, n);
    } else {
        fp8_a_amax_kernel<float><<<gs, bs, 0, stream>>>((const float *) X, d_amax, n);
    }
    fp8_scale_from_amax<<<1, 1, 0, stream>>>(d_amax, d_scale);
    if (xtype == GGML_TYPE_F16) {
        fp8_a_quant_kernel<half> <<<gs, bs, 0, stream>>>((const half *)  X, out, n, d_scale);
    } else {
        fp8_a_quant_kernel<float><<<gs, bs, 0, stream>>>((const float *) X, out, n, d_scale);
    }
}

// ======================= MXFP8 (block-scaled e4m3) PROTOTYPE =======================
// GGML_F8_MXFP8=1 (native-e4m3 weight path only for now). Replaces the single PER-TENSOR
// SCALAR_32F activation/weight scale (amax/448) — which crushes per-block dynamic range and
// is the source of the "dotty in motion" artifact — with the Blackwell-native MXFP8 format:
// per-32-element e8m0 (UE8M0) block scales, fed to cuBLASLt via
// CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 on BOTH A (weight) and B (activation). The FP8
// tensor cores keep running (mma.block_scale) — same speed, but every 32-lane block now gets
// its own power-of-2 scale, so low-magnitude blocks stop collapsing into e4m3 subnormals.
//
// The block-scale array MUST be in the SAME SWIZZLE_32_4_4 layout the NVFP4 VEC16 path uses
// (swz_off), only with the column count = K/32 instead of K/16. This is the #1 thing to
// GPU-validate (a wrong swizzle -> garbage, not a crash). Everything here is additive: when
// GGML_F8_MXFP8 is unset the byte-for-byte prior per-tensor path runs unchanged.

// e8m0 shared-exponent for an e4m3 (E4M3 EMAX = 8) block. scale = 2^(floor(log2(amax)) - 8),
// so amax/scale lands at ~2^8 = 256 (<= 448 e4m3 max) with headroom. Encoded biased (+127),
// clamped to the e8m0 range [0,254]. NOTE: uses FLOOR (frexp) not round-to-nearest so the
// block peak can never scale PAST 448 and clip (the round-up variant would need a 448-guard).
static __device__ __forceinline__ uint8_t mxfp8_e8m0_from_amax_e4m3(float amax) {
    if (!(amax > 0.0f)) return 0;
    int e; (void)frexpf(amax, &e);      // amax = m * 2^e, m in [0.5,1) => floor(log2(amax)) = e-1
    int biased = (e - 1) - 8 + 127;     // shared_exp - E4M3_EMAX(8), then e8m0 bias(+127)
    biased = biased < 0 ? 0 : (biased > 254 ? 254 : biased);
    return (uint8_t)biased;
}

// decode one e4m3 byte to float (cuda_fp8.h provides operator float()).
static __device__ __forceinline__ float mxfp8_e4m3_decode(uint8_t b) {
    __nv_fp8_e4m3 t; t.__x = b; return (float)t;
}

// WEIGHT -> MXFP8. src0 is ALREADY e4m3 (ComfyUI dev-fp8), stored per-tensor; the true weight
// value is decode(byte)*wglobal. We re-scale each 32-block by its own e8m0 (folding wglobal in),
// so alpha stays 1.0 and each block re-enters e4m3's well-conditioned range. One thread per
// (row r in [0,N), 32-block b in [0,K/32)). Weight precision is unchanged (still e4m3 codes) —
// only the per-block SCALE changes, which is exactly the dynamic-range fix.
static __global__ void fp8_w_to_mxfp8_kernel(const uint8_t * __restrict__ Wsrc, float wglobal,
                                             uint8_t * __restrict__ Wout,     // e4m3 [N,K]
                                             uint8_t * __restrict__ Ssw,      // swizzled e8m0
                                             int N, int K) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int nblk = K/32;
    const long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= (long)N*nblk) return;
    const int r = (int)(idx / nblk), b = (int)(idx % nblk);
    const uint8_t * w = &Wsrc[(long)r*K + (long)b*32];
    float vals[32]; float amax = 0.f;
    #pragma unroll
    for (int j=0;j<32;j++) { const float v = mxfp8_e4m3_decode(w[j]) * wglobal; vals[j]=v; amax=fmaxf(amax,fabsf(v)); }
    const uint8_t code = mxfp8_e8m0_from_amax_e4m3(amax);
    Ssw[swz_off(r, b, nblk)] = code;
    const float inv = amax > 0.f ? 1.0f/ggml_cuda_e8m0_to_fp32(code) : 0.f;
    uint8_t * od = &Wout[(long)r*K + (long)b*32];
    #pragma unroll
    for (int j=0;j<32;j++) { const __nv_fp8_e4m3 q(vals[j]*inv); od[j] = q.__x; }
#else
    NO_DEVICE_CODE;
#endif
}

// ACTIVATION -> MXFP8. src1 [K,M] contiguous == row-major [M,K]; block along K (innermost).
// One thread per (row r in [0,M), 32-block b in [0,K/32)). This is THE fix: a per-block scale
// instead of one per-tensor amax/448, so motion-varying activations keep their per-block range.
template <typename act_t>
static __global__ void fp8_a_to_mxfp8_kernel(const act_t * __restrict__ X,
                                             uint8_t * __restrict__ Aout,   // e4m3 [M,K]
                                             uint8_t * __restrict__ Ssw,    // swizzled e8m0
                                             int M, int K) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int nblk = K/32;
    const long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= (long)M*nblk) return;
    const int r = (int)(idx / nblk), b = (int)(idx % nblk);
    const act_t * x = &X[(long)r*K + (long)b*32];
    float vals[32]; float amax = 0.f;
    #pragma unroll
    for (int j=0;j<32;j++) { const float v = nvfp4_load_act(x[j]); vals[j]=v; amax=fmaxf(amax,fabsf(v)); }
    const uint8_t code = mxfp8_e8m0_from_amax_e4m3(amax);
    Ssw[swz_off(r, b, nblk)] = code;
    const float inv = amax > 0.f ? 1.0f/ggml_cuda_e8m0_to_fp32(code) : 0.f;
    uint8_t * od = &Aout[(long)r*K + (long)b*32];
    #pragma unroll
    for (int j=0;j<32;j++) { const __nv_fp8_e4m3 q(vals[j]*inv); od[j] = q.__x; }
#else
    NO_DEVICE_CODE;
#endif
}

static bool ggml_cuda_f8_mxfp8_enabled() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_F8_MXFP8"); v = (e && atoi(e)) ? 1 : 0; }
    return v == 1;
}

bool ggml_cuda_fp8_ffn_enabled() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_FP8_FFN"); v = (e && atoi(e)) ? 1 : 0; }
    return v == 1;
}

// Native F8_E4M3 WEIGHT GEMM gate. Unlike GGML_FP8_FFN (which promotes an NVFP4 weight to
// e4m3 for a filtered subset of Linears), this routes a weight ALREADY stored as
// GGML_TYPE_F8_E4M3 (ComfyUI dev-fp8, repacked verbatim) straight to the cuBLASLt FP8xFP8
// GEMM. Default ON: an fp8 gguf should just work on Blackwell with no env flag; set
// GGML_CUDA_F8_GEMM=0 to force the dequant->f16/f32 cuBLAS fallback for A/B.
bool ggml_cuda_f8_gemm_enabled() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_CUDA_F8_GEMM"); v = (e && atoi(e) == 0) ? 0 : 1; }
    return v == 1;
}

// Set a single device float to a constant (used for the F8_E4M3 weight's unit scale: the
// stored e4m3 bytes ARE the weight values, so A_scale == 1.0).
static __global__ void fp8_set_scalar_kernel(float * __restrict__ p, float v) {
    if (blockIdx.x == 0 && threadIdx.x == 0) p[0] = v;
}

// GGML_F8_ZERO_SCRATCH (debug discriminator): zero the fp8 activation buffer, the cuBLASLt
// workspace, and the output before each e4m3 GEMM, and force a fresh activation requant (no
// cache reuse). If this makes the DiT NaN disappear, the NaN is a read of uninitialized /
// raced scratch (e.g. a cross-stream act-cache reuse), not a numerical overflow. Off by default.
static bool ggml_cuda_f8_zero_scratch_enabled() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_F8_ZERO_SCRATCH"); v = (e && atoi(e)) ? 1 : 0; }
    return v == 1;
}

// GGML_F8_CLAMP_OUT: keep the F16 residual stream (no VRAM doubling) but eliminate the
// F16-output-store overflow. The cuBLASLt FP8 GEMM already accumulates in F32; a deep-block
// result > 65504 (F16 max) becomes +-inf when stored straight to the F16 dst -> NaN cascade
// (the fp8-model white). With this on, the GEMM writes an F32 pool temp (one layer's output,
// ~no VRAM cost) and a tiny kernel clamps to +-65504 while converting F32->F16 into dst.
// Clamp is IDENTITY for in-range values; only the overflowing outliers are capped. Off by default.
static bool ggml_cuda_f8_clamp_out_enabled() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_F8_CLAMP_OUT"); v = (e && atoi(e)) ? 1 : 0; }
    return v == 1;
}

static __global__ void fp8_clamp_f32_to_f16(const float * __restrict__ in, half * __restrict__ out, size_t n) {
    for (size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x*blockDim.x) {
        const float v = fminf(fmaxf(in[i], -65504.0f), 65504.0f);   // IEEE fmin/fmax: NaN -> -65504 (harmless; F32 accum has no NaN)
        out[i] = __float2half_rn(v);
    }
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

// ---------------------------------------------------------------------------
// FP8 activation-quant reuse cache (fix #2: kill the redundant per-Linear
// activation requant). Self-attn q/k/v and cross-attn k/v feed the SAME src1
// activation to consecutive FP8 GEMMs, so the e4m3 quant of that activation is
// recomputed identically 2-3x per block. Quantize once, reuse the e4m3 buffer +
// its scalar scale for any later GEMM in the SAME compute whose src1 is the
// byte-for-byte same tensor.
//
// Stale-safety (must stay bit-identical AND never reuse stale data):
//  - A GLOBAL atomic generation is bumped once per graph compute
//    (ggml_cuda_fp8_act_cache_new_generation(), called from execute_graph). A
//    hit requires the cache's stored generation to equal the current one, so a
//    new compute can NEVER reuse a prior compute's buffer even if gallocr
//    recycles a node/data address.
//  - Within a single compute every ggml graph node has a unique, stable
//    address, so keying on the src1 NODE pointer (with data ptr / ne0 / ne1 /
//    nb1 / type as belt-and-suspenders) means two DIFFERENT logical
//    activations can never collide, even when their ->data aliases across
//    non-overlapping lifetimes.
//  - The e4m3 buffer is OWNED (cudaMalloc, grow-only) so neither gallocr nor
//    the stream pool can recycle it out from under a pending reuse.
//  - On ANY uncertainty (alloc failure, device/shape change) we MISS and
//    requant. The quant is deterministic (atomicMax amax -> scalar scale ->
//    per-elem e4m3 round), so a reused buffer is byte-identical to requanting.
// Off-switch: GGML_FP8_ACT_QUANT_CACHE=0 (default ON). Scope: FP8 activation
// quant only — weight requant and the FP4 path are untouched.
struct fp8_act_quant_cache {
    uint64_t           gen     = (uint64_t)-1;  // generation filled at; -1 == empty
    const ggml_tensor* node    = nullptr;       // src1 node identity (unique per graph)
    const void *       data    = nullptr;
    int64_t            ne0     = 0;
    int64_t            ne1     = 0;
    size_t             nb1     = 0;
    int                type    = -1;
    int                device  = -1;
    uint8_t *          d_fp8   = nullptr;        // owned persistent e4m3 buffer
    size_t             cap     = 0;              // capacity (bytes) of d_fp8
    float *            d_scale = nullptr;        // owned persistent scalar scale (1 float)
};
static thread_local fp8_act_quant_cache g_fp8_act_cache;
static std::atomic<uint64_t>             g_fp8_act_cache_gen{0};

static int ggml_cuda_fp8_act_cache_enabled() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_FP8_ACT_QUANT_CACHE"); v = (e && atoi(e) == 0) ? 0 : 1; }
    return v;
}

// Bump once per graph compute (host side, from execute_graph) so cross-compute
// reuse can never happen. Cheap relaxed atomic; safe to call on any backend.
extern "C" void ggml_cuda_fp8_act_cache_new_generation(void) {
    g_fp8_act_cache_gen.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// FP8 GEMM bias epilogue (fix #3): fold the Linear bias-add into the cuBLASLt
// epilogue (CUBLASLT_EPILOGUE_BIAS) instead of a separate op_add kernel. Only
// the BIAS is fused — Wan's FFN uses tanh-approx GELU while cuBLASLt's GELU
// epilogue is the erf gelu, so fusing GELU would silently change results; ffn.0's
// gelu stays a separate kernel. Env GGML_FP8_GEMM_EPILOGUE (default OFF). FP8 path
// only (the FP4 GEMM has no bias-epilogue algo). The bias is added in F32 compute
// before the store (cuBLASLt epilogue order) — same order as ggml's post-matmul
// ggml_add(bias), and for F16 output slightly MORE accurate (one fewer round).
static int ggml_cuda_fp8_gemm_epilogue_enabled() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_FP8_GEMM_EPILOGUE"); v = (e && atoi(e)) ? 1 : 0; }
    return v;
}

// (m,n,k,out_dt) shapes whose bias epilogue cuBLASLt couldn't serve — so later
// calls fail fast (no quant) and the caller falls back to a separate bias kernel.
static thread_local std::set<std::tuple<int,int,int,int>> g_fp8_epi_unsupported;

// F32 -> F16 bias downcast (tiny [N]; per-call, offload-safe, no ptr caching).
static __global__ void fp8_bias_f32_to_f16(const float * __restrict__ in, half * __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(in[i]);
}

// ---------------------------------------------------------------------------
// FP8 e4m3 WEIGHT cache (fix #1): the NVFP4 weight is re-quantized to e4m3 on
// every GEMM call, but weights are CONSTANT for the whole render. Cache the e4m3
// weight + its scalar scale, keyed by the weight's unique tensor NAME.
//
// Why NAME (not data ptr): the high/low-noise experts are SEPARATE runners with
// distinct name prefixes and every weight name is globally unique, so a name maps
// to ONE constant weight value for the whole render. Keying on the name (validated
// by N/K/nb1/type/device) makes false hits impossible even when the offload path
// recycles a buffer ADDRESS for a different weight — a different weight has a
// different name => a different entry. An offloaded weight re-homed to a new
// address still HITS by name and the cached e4m3 is correct (it is a deterministic
// function of the constant weight values, not the address), so resident AND
// offloaded weights both benefit. No generation invalidation: weights never change.
//
// Cross-stream safety: the GEMM can run on offload worker threads/streams, so each
// entry records a CUDA event after its quant and a consumer on another stream waits
// on it before reading the buffer (a near-no-op on the common single-stream path).
// The whole miss path is held under one mutex, so concurrent misses of the same
// weight serialize and the event is recorded before the entry is observable.
//
// VRAM budget GGML_FP8_WEIGHT_CACHE_MB (default 4096): cache until the budget is
// hit, then fall back to per-call pool requant (no eviction — weights are equally
// hot). Owned cudaMalloc buffers; leaked until process exit (first-experiment
// simplicity) — ggml_cuda_fp8_weight_cache_clear() frees them on demand.
// Env-gate GGML_FP8_WEIGHT_QUANT_CACHE (default OFF). Independent of #2/#3/#4.
struct fp8_weight_cache_entry {
    int         N = 0, K = 0;
    size_t      nb1 = 0;
    int         type = -1;
    int         device = -1;
    uint8_t *   d_fp8 = nullptr;
    float *     d_scale = nullptr;
    size_t      bytes = 0;
    cudaEvent_t ready = nullptr;
};
static std::mutex                                              g_fp8_wcache_mtx;
static std::unordered_map<std::string, fp8_weight_cache_entry> g_fp8_wcache;
static size_t                                                  g_fp8_wcache_bytes = 0;

static int ggml_cuda_fp8_weight_cache_enabled() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_FP8_WEIGHT_QUANT_CACHE"); v = (e && atoi(e)) ? 1 : 0; }
    return v;
}
static size_t ggml_cuda_fp8_weight_cache_budget_bytes() {
    static size_t b = 0; static int init = 0;
    if (!init) { const char * e = getenv("GGML_FP8_WEIGHT_CACHE_MB"); long mb = (e && atol(e) > 0) ? atol(e) : 4096; b = (size_t)mb * 1024 * 1024; init = 1; }
    return b;
}

// Profiling-only CUDA-event timing of the native FP8 GEMM. This deliberately
// synchronizes each call; use it to attribute work, never for headline timings.
static bool ggml_cuda_fp8_gemm_profile_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * e = getenv("GGML_FP8_GEMM_PROFILE");
        enabled = e && atoi(e) ? 1 : 0;
    }
    return enabled != 0;
}

extern "C" void ggml_cuda_fp8_weight_cache_clear(void) {
    std::lock_guard<std::mutex> lk(g_fp8_wcache_mtx);
    for (auto & kv : g_fp8_wcache) {
        if (kv.second.d_fp8)   cudaFree(kv.second.d_fp8);
        if (kv.second.d_scale) cudaFree(kv.second.d_scale);
        if (kv.second.ready)   cudaEventDestroy(kv.second.ready);
    }
    g_fp8_wcache.clear();
    g_fp8_wcache_bytes = 0;
}

bool ggml_cuda_fp8_cublaslt_mul_mat(ggml_backend_cuda_context & ctx,
                                    const ggml_tensor * src0,
                                    const ggml_tensor * src1,
                                    ggml_tensor * dst,
                                    const ggml_tensor * bias) {
    // weight source: either the stored FP4 FFN weight (promoted to e4m3 below), or a native
    // GGML_TYPE_F8_E4M3 weight (ComfyUI dev-fp8) that is ALREADY e4m3 -> used verbatim.
    const bool w_is_e4m3 = (src0->type == GGML_TYPE_F8_E4M3);
    // GGML_F8_DBG entry trace: every e4m3 GEMM that ENTERS this fn, with the exact operands the
    // guards below test. Cross-ref against [F8_LOUD] (which only prints post-matmul): a name here
    // but NOT in F8_LOUD => it bailed to the fallback path. Shapes/contig/bias reveal which guard.
    if (w_is_e4m3 && getenv("GGML_F8_DBG")) {
        static std::atomic<int> s_ent{0}; int e = s_ent.fetch_add(1);
        if (e < 20) fprintf(stderr,
            "[F8_ENTER] #%d %s  s0[%ld,%ld,%ld,%ld] s1[%ld,%ld,%ld,%ld] s1_t=%d dst_t=%d bias=%d contig(s0=%d s1=%d d=%d)\n",
            e, src0->name?src0->name:"?",
            (long)src0->ne[0],(long)src0->ne[1],(long)src0->ne[2],(long)src0->ne[3],
            (long)src1->ne[0],(long)src1->ne[1],(long)src1->ne[2],(long)src1->ne[3],
            (int)src1->type,(int)dst->type,(bias!=nullptr),
            (int)ggml_is_contiguous(src0),(int)ggml_is_contiguous(src1),(int)ggml_is_contiguous(dst));
    }
    if (src0->type != GGML_TYPE_NVFP4 && !w_is_e4m3) return false;
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

    // fix #3: optional bias fused into the cuBLASLt epilogue. When a bias is supplied
    // (from the FP8 mul_mat+bias graph matcher) we MUST apply it via the epilogue or
    // return false so the caller falls back to a separate bias add — never silently
    // drop it. cuBLASLt CUBLASLT_EPILOGUE_BIAS adds one value per output ROW (m = N
    // out-features) broadcast over tokens == the Linear bias [N]. (Default null bias =
    // the normal dispatch path, byte-identical to before.)
    const bool want_epi = (bias != nullptr) &&
                          ggml_cuda_fp8_gemm_epilogue_enabled() &&
                          bias->type == GGML_TYPE_F32 && ggml_is_contiguous(bias) &&
                          bias->ne[0] == N && bias->ne[1] == 1 && bias->ne[2] == 1 && bias->ne[3] == 1;
    if (bias != nullptr && !want_epi) return false;   // can't honor the requested bias here
    const cublasDataType_t epi_out_dt = (dst->type == GGML_TYPE_F16) ? CUDA_R_16F : CUDA_R_32F;
    const auto epi_key = std::make_tuple(N, M, K, (int)epi_out_dt);
    if (want_epi && g_fp8_epi_unsupported.count(epi_key)) return false;  // fail fast -> caller falls back

    cudaStream_t stream = ctx.stream();
    cublasLtHandle_t lt = get_lt();
    if (!lt) return false;

    const float w_global = nvfp4_weight_global_for(src0->name);

    // MXFP8 PROTOTYPE gate (GGML_F8_MXFP8): block-scaled e4m3 (per-32-block e8m0) instead of the
    // per-tensor SCALAR_32F scale. Native-e4m3 weight only for now (the reported-dotty path).
    // When off, everything below is byte-identical to the prior per-tensor path.
    const bool mxfp8 = ggml_cuda_f8_mxfp8_enabled() && w_is_e4m3;

    // 1) weight -> e4m3 [N,K]. Name-keyed persistent cache (fix #1) when enabled; else
    //    pool scratch per-call (byte-identical, handles resident & offload weights).
    uint8_t * w_fp8_ptr   = nullptr;
    float   * w_scale_ptr = nullptr;
    ggml_cuda_pool_alloc<uint8_t>      w_fp8(ctx.pool());      // pool fallback
    ggml_cuda_pool_alloc<float>        w_scale(ctx.pool());    // pool fallback
    ggml_cuda_pool_alloc<unsigned int> w_amax(ctx.pool());    // amax scratch (quant only)
    // MXFP8 swizzled weight block-scale buffer. Declared HERE (right after the weight pool
    // buffers) — NOT at function top — so its destructor unwinds in strict LIFO order relative
    // to w_fp8/a_fp8_pool/ws: the ggml CUDA pool is a stack allocator (ggml-cuda.cu:650 asserts
    // frees are reverse-of-alloc). mxfp8 allocs w_fp8 THEN mx_w_scale, so mx_w_scale must be
    // DECLARED after w_fp8 to be DESTROYED before it. Function-scope so it outlives the matmul.
    ggml_cuda_pool_alloc<uint8_t>      mx_w_scale(ctx.pool());

    // requant the NVFP4 weight -> e4m3 into (wdst, sdst). Deterministic => byte-identical
    // to a cached buffer, so a cache hit is bit-identical to requanting.
    auto run_weight_quant = [&](uint8_t * wdst, float * sdst) -> bool {
        w_amax.alloc(1);
        cudaMemsetAsync(w_amax.get(), 0, sizeof(unsigned int), stream);
        const int threads = 256;
        const long total  = (long)N*(K/16);
        unsigned int grid = (unsigned int)((total + threads - 1)/threads);
        if (grid > 65535u) grid = 65535u; if (grid == 0) grid = 1;
        fp8_w_amax_kernel<<<grid, threads, 0, stream>>>((const block_nvfp4*)src0->data, w_amax.get(), N, K, w_global);
        fp8_scale_from_amax<<<1, 1, 0, stream>>>(w_amax.get(), sdst);
        const unsigned int qgrid = (unsigned int)((total + threads - 1)/threads);
        fp8_w_quant_kernel<<<qgrid ? qgrid : 1, threads, 0, stream>>>((const block_nvfp4*)src0->data, wdst, N, K, w_global, sdst);
        return cudaPeekAtLastError() == cudaSuccess;
    };

    // Native e4m3 weight: the stored bytes are comfy's per-tensor-scaled e4m3 codes
    // (fp8 = weight/scale, scale = amax(|weight|)/448, filled to the +-448 range) -> feed
    // src0->data straight to cuBLASLt and apply the PER-TENSOR recovery scale as the A-scale.
    // The scale is carried in the gguf as a "<weight>.wglobal" sibling, registered at load into
    // g_wglobal (see stable-diffusion.cpp) and read back here via nvfp4_weight_global_for();
    // unregistered (e.g. an F16-stored Linear, or a raw-cast gguf) -> 1.0. Skips the NVFP4->e4m3
    // requant + weight cache entirely.
    if (mxfp8) {
        // MXFP8: re-scale the already-e4m3 weight to per-32-block e8m0 (folds wglobal in -> alpha=1).
        // TODO(perf): cache these swizzled buffers by weight name like the scalar weight cache;
        // the prototype re-quantizes per call (correct, just redundant work on constant weights).
        const int    nblk_w = K/32;
        const size_t w_rb_p = ((size_t)(N+127)/128)*128;
        const size_t w_cb_p = ((size_t)(nblk_w+3)/4)*4;
        w_fp8_ptr = w_fp8.alloc((size_t)N*K);
        uint8_t * wsc = mx_w_scale.alloc(w_rb_p*w_cb_p);
        cudaMemsetAsync(wsc, 0, w_rb_p*w_cb_p, stream);
        const int threads = 256; const long total = (long)N*nblk_w;
        unsigned int grid = (unsigned int)((total + threads - 1)/threads);
        if (grid > 65535u) grid = 65535u; if (grid == 0) grid = 1;
        fp8_w_to_mxfp8_kernel<<<grid, threads, 0, stream>>>((const uint8_t*)src0->data, w_global, w_fp8_ptr, wsc, N, K);
        if (cudaPeekAtLastError() != cudaSuccess) return false;
        w_scale_ptr = (float*)wsc;   // reinterpreted; the GEMM binds it as a void* block-scale ptr
    } else if (w_is_e4m3) {
        const float wg = nvfp4_weight_global_for(src0->name);
        w_fp8_ptr   = (uint8_t *) src0->data;
        w_scale_ptr = w_scale.alloc(1);
        fp8_set_scalar_kernel<<<1, 1, 0, stream>>>(w_scale_ptr, wg);
        // GGML_F8_DBG: one-shot per-name confirmation that the per-tensor recovery scale
        // (amax/448 ~ 1e-3) actually reached the GEMM. wg==1.0 => the wglobal did NOT resolve
        // for this src0->name (registration key mismatch) -> output over-scaled/white.
        static int s_f8dbg = -1;
        if (s_f8dbg < 0) { const char * e = getenv("GGML_F8_DBG"); s_f8dbg = (e && atoi(e)) ? 1 : 0; }
        if (s_f8dbg) {
            static std::atomic<int> s_n{0};
            const int k = s_n.fetch_add(1);
            if (k < 10)
                fprintf(stderr, "[F8_DBG] fp8-GEMM #%d  A_scale(wglobal)=%.8g  %s  src0->name='%s'  N=%d K=%d\n",
                        k, wg, (wg == 1.0f ? "**UNSCALED(1.0)**" : "scaled"), src0->name, N, K);
        }
    }

    const char * w_name   = src0->name;
    const bool w_cache_on = !w_is_e4m3 && ggml_cuda_fp8_weight_cache_enabled() && w_name && w_name[0] != '\0';
    if (w_cache_on) {
        std::lock_guard<std::mutex> lk(g_fp8_wcache_mtx);   // held across the (rare) miss requant
        auto it = g_fp8_wcache.find(w_name);
        const bool hit = it != g_fp8_wcache.end() && it->second.d_fp8 && it->second.d_scale &&
                         it->second.N == N && it->second.K == K && it->second.nb1 == src0->nb[1] &&
                         it->second.type == (int)src0->type && it->second.device == ctx.device;
        if (hit) {
            w_fp8_ptr   = it->second.d_fp8;
            w_scale_ptr = it->second.d_scale;
            if (it->second.ready) cudaStreamWaitEvent(stream, it->second.ready, 0);  // cross-stream safe
        } else {
            if (it != g_fp8_wcache.end()) {   // stale shape/device at this name (shouldn't happen) -> drop
                if (it->second.d_fp8)   cudaFree(it->second.d_fp8);
                if (it->second.d_scale) cudaFree(it->second.d_scale);
                if (it->second.ready)   cudaEventDestroy(it->second.ready);
                g_fp8_wcache_bytes -= it->second.bytes;
                g_fp8_wcache.erase(it);
            }
            const size_t need = (size_t)N*K + sizeof(float);
            if (g_fp8_wcache_bytes + need <= ggml_cuda_fp8_weight_cache_budget_bytes()) {
                fp8_weight_cache_entry e;
                e.N = N; e.K = K; e.nb1 = src0->nb[1]; e.type = (int)src0->type; e.device = ctx.device; e.bytes = need;
                if (cudaMalloc((void**)&e.d_fp8, (size_t)N*K) == cudaSuccess &&
                    cudaMalloc((void**)&e.d_scale, sizeof(float)) == cudaSuccess &&
                    cudaEventCreateWithFlags(&e.ready, cudaEventDisableTiming) == cudaSuccess) {
                    if (run_weight_quant(e.d_fp8, e.d_scale)) {
                        cudaEventRecord(e.ready, stream);       // record BEFORE the entry is observable
                        g_fp8_wcache_bytes += need;
                        g_fp8_wcache.emplace(std::string(w_name), e);
                        w_fp8_ptr = e.d_fp8; w_scale_ptr = e.d_scale;   // quant already done
                    } else {
                        cudaFree(e.d_fp8); cudaFree(e.d_scale); cudaEventDestroy(e.ready);
                        return false;
                    }
                } else {
                    if (e.d_fp8)   cudaFree(e.d_fp8);
                    if (e.d_scale) cudaFree(e.d_scale);
                    if (e.ready)   cudaEventDestroy(e.ready);
                    // alloc failed -> fall through to pool below
                }
            }
            // (budget full or alloc failed -> w_fp8_ptr stays null -> pool fallback)
        }
    }
    if (w_fp8_ptr == nullptr) {   // cache off / miss-not-cached / budget full
        w_fp8_ptr   = w_fp8.alloc((size_t)N*K);
        w_scale_ptr = w_scale.alloc(1);
        if (!run_weight_quant(w_fp8_ptr, w_scale_ptr)) return false;
    }

    // 2) activation -> e4m3 [M,K] (flat; src1 [K,M] contiguous == row-major [M,K]).
    //    Reuse-cache (fix #2): q/k/v (and cross k/v) share src1 -> quantize once,
    //    reuse the e4m3 buffer + scale. Stale-safe via per-compute generation +
    //    src1 node identity; falls back to a fresh requant on any miss.
    const long n_act = (long)M*K;
    uint8_t * a_fp8_ptr   = nullptr;
    float   * a_scale_ptr = nullptr;
    ggml_cuda_pool_alloc<uint8_t>      a_fp8_pool(ctx.pool());    // fallback (cache off / alloc fail)
    ggml_cuda_pool_alloc<float>        a_scale_pool(ctx.pool());
    ggml_cuda_pool_alloc<unsigned int> a_amax(ctx.pool());       // transient amax scratch (miss only)
    // MXFP8 swizzled activation block-scale buffer. Declared right after a_fp8_pool (see the
    // LIFO note on mx_w_scale above): mxfp8 allocs a_fp8_pool THEN mx_a_scale, so mx_a_scale
    // must be DESTROYED before a_fp8_pool -> DECLARED after it. Function-scope (used at matmul).
    ggml_cuda_pool_alloc<uint8_t>      mx_a_scale(ctx.pool());
    bool act_reused = false;

    if (!mxfp8 && ggml_cuda_fp8_act_cache_enabled()) {
        fp8_act_quant_cache & C = g_fp8_act_cache;
        const uint64_t cur = g_fp8_act_cache_gen.load(std::memory_order_relaxed);
        const bool hit = C.gen == cur && C.node == src1 && C.data == src1->data &&
                         C.ne0 == src1->ne[0] && C.ne1 == src1->ne[1] &&
                         C.nb1 == src1->nb[1] && C.type == (int)src1->type &&
                         C.device == ctx.device && C.d_fp8 != nullptr &&
                         C.d_scale != nullptr && C.cap >= (size_t)n_act;
        if (hit) {
            a_fp8_ptr   = C.d_fp8;
            a_scale_ptr = C.d_scale;
            act_reused  = true;
        } else {
            // MISS: (re)quantize into the OWNED persistent buffer (grow-only).
            if (C.device != ctx.device && C.d_fp8 != nullptr) {
                cudaFree(C.d_fp8);   C.d_fp8   = nullptr; C.cap = 0;
                cudaFree(C.d_scale); C.d_scale = nullptr;
            }
            if (C.cap < (size_t)n_act) {
                if (C.d_fp8 != nullptr) cudaFree(C.d_fp8);
                if (cudaMalloc((void**)&C.d_fp8, (size_t)n_act) != cudaSuccess) { C.d_fp8 = nullptr; C.cap = 0; }
                else                                                             C.cap   = (size_t)n_act;
            }
            if (C.d_scale == nullptr) {
                if (cudaMalloc((void**)&C.d_scale, sizeof(float)) != cudaSuccess) C.d_scale = nullptr;
            }
            if (C.d_fp8 != nullptr && C.d_scale != nullptr) {
                a_fp8_ptr   = C.d_fp8;
                a_scale_ptr = C.d_scale;
                C.gen  = cur;          C.node = src1;             C.data = src1->data;
                C.ne0  = src1->ne[0];  C.ne1  = src1->ne[1];      C.nb1  = src1->nb[1];
                C.type = (int)src1->type; C.device = ctx.device;
            } else {
                C.gen = (uint64_t)-1; C.node = nullptr;   // alloc failed -> invalidate, use pool
            }
        }
    }
    if (!mxfp8 && a_fp8_ptr == nullptr) {  // cache off, or owned-buffer alloc failed
        a_fp8_ptr   = a_fp8_pool.alloc((size_t)M*K);
        a_scale_ptr = a_scale_pool.alloc(1);
    }
    // GGML_F8_ZERO_SCRATCH: force a fresh requant into a zeroed activation buffer so nothing
    // stale/uninit/raced is fed to the GEMM (the uninitialized-buffer discriminator).
    const bool f8_zero = !mxfp8 && ggml_cuda_f8_zero_scratch_enabled();
    if (f8_zero) {
        act_reused = false;
        cudaMemsetAsync(a_fp8_ptr, 0, (size_t)M * K, stream);
    }
    if (!mxfp8 && !act_reused) {
        // --- Hadamard activation rotation (matches the nvfp4 path + the folded weight) ---
        // When GGML_NVFP4_ACT_HADAMARD is on, the FP8-diverted weight was folded by H offline
        // (all-folded "-had" gguf), so the activation MUST be rotated by the SAME block-B H
        // before the e4m3 quant: folded_W(=W@H) x rotated_A(=A@H) == W x A (H@H==I), identical
        // math to the FP4 path — but the finer e4m3 (~16x FP4) removes the residual FP4 grain.
        // Pre-rotate src1 into an F32 scratch ONCE with the SAME kernel/scale/butterfly the nvfp4
        // path uses (nvfp4_rotate_kernel == nvfp4_hadN, single 1/sqrt(B)); BOTH the per-tensor
        // amax AND the quant then consume the rotated scratch (never raw src1). PER-TENSOR block
        // rule is identical to the nvfp4 path: had_block = (K % B == 0) ? B : 16. When the flag is
        // off, fp8_rot_src stays null and the raw-src1 launches below are byte-identical to before.
        static int s_fp8_had = -1;
        if (s_fp8_had < 0) { const char* e = getenv("GGML_NVFP4_ACT_HADAMARD"); s_fp8_had = (e && atoi(e)) ? 1 : 0; }
        static int s_fp8_had_block = -1;
        if (s_fp8_had_block < 0) {
            const char* e = getenv("GGML_NVFP4_ACT_HADAMARD_BLOCK");
            int b = e ? atoi(e) : 256;
            if (b < 16 || (b & (b - 1)) != 0) b = 16;
            s_fp8_had_block = b;
        }
        const bool fp8_hadamard  = s_fp8_had != 0;
        const int  fp8_had_block = (fp8_hadamard && (K % s_fp8_had_block) == 0) ? s_fp8_had_block : 16;
        // LIFO pool ordering (ggml_cuda_pool is a stack allocator; ggml-cuda.cu:650 asserts on a
        // non-LIFO free): a_amax is FUNCTION-scope (freed at return) while the rotation scratch
        // a_rot is freed at the end of THIS block, so a_amax MUST be pushed BEFORE a_rot -> a_rot's
        // lifetime then nests strictly inside a_amax's (same nesting the nvfp4 path uses). Reversing
        // this (a_rot pushed first) freed a_rot while a_amax was still on top -> the crash.
        a_amax.alloc(1);
        cudaMemsetAsync(a_amax.get(), 0, sizeof(unsigned int), stream);

        ggml_cuda_pool_alloc<float> a_rot(ctx.pool());
        const float * fp8_rot_src = nullptr;
        if (fp8_hadamard) {
            a_rot.alloc((size_t)M * (size_t)K);
            const size_t nblocks  = (size_t)M * (size_t)K / (size_t)fp8_had_block;
            const int    rthreads = fp8_had_block < 1024 ? fp8_had_block : 1024;
            unsigned int rgrid    = nblocks > 65535u ? 65535u : (unsigned int)nblocks;
            if (rgrid == 0) rgrid = 1;
            const size_t shmem = (size_t)fp8_had_block * sizeof(float);
            if (src1->type == GGML_TYPE_F16)
                nvfp4_rotate_kernel<half> <<<rgrid, rthreads, shmem, stream>>>((const half*) src1->data, a_rot.get(), fp8_had_block, nblocks);
            else
                nvfp4_rotate_kernel<float><<<rgrid, rthreads, shmem, stream>>>((const float*)src1->data, a_rot.get(), fp8_had_block, nblocks);
            if (cudaPeekAtLastError() != cudaSuccess) return false;
            fp8_rot_src = a_rot.get();
        }

        const int  threads = 256;
        unsigned int grid = (unsigned int)((n_act + threads - 1)/threads);
        if (grid > 1024u) grid = 1024u; if (grid == 0) grid = 1;
        if (fp8_rot_src)                      fp8_a_amax_kernel<float><<<grid, threads, 0, stream>>>(fp8_rot_src,               a_amax.get(), n_act);
        else if (src1->type == GGML_TYPE_F16) fp8_a_amax_kernel<half ><<<grid, threads, 0, stream>>>((const half*)src1->data,  a_amax.get(), n_act);
        else                                  fp8_a_amax_kernel<float><<<grid, threads, 0, stream>>>((const float*)src1->data, a_amax.get(), n_act);
        fp8_scale_from_amax<<<1, 1, 0, stream>>>(a_amax.get(), a_scale_ptr);
        unsigned int qgrid = (unsigned int)((n_act + threads - 1)/threads);
        if (qgrid > 65535u) qgrid = 65535u; if (qgrid == 0) qgrid = 1;
        if (fp8_rot_src)                      fp8_a_quant_kernel<float><<<qgrid, threads, 0, stream>>>(fp8_rot_src,               a_fp8_ptr, n_act, a_scale_ptr);
        else if (src1->type == GGML_TYPE_F16) fp8_a_quant_kernel<half ><<<qgrid, threads, 0, stream>>>((const half*)src1->data,  a_fp8_ptr, n_act, a_scale_ptr);
        else                                  fp8_a_quant_kernel<float><<<qgrid, threads, 0, stream>>>((const float*)src1->data, a_fp8_ptr, n_act, a_scale_ptr);
        if (cudaPeekAtLastError() != cudaSuccess) return false;
    }

    // MXFP8 activation quant (block-scaled): per-32-block e8m0 into a swizzled scale buffer.
    // THE fix — replaces the single per-tensor amax/448 that crushed per-block dynamic range.
    if (mxfp8) {
        const int    nblk_a = K/32;
        const size_t a_rb_p = ((size_t)(M+127)/128)*128;
        const size_t a_cb_p = ((size_t)(nblk_a+3)/4)*4;
        a_fp8_ptr = a_fp8_pool.alloc((size_t)M*K);
        uint8_t * asc = mx_a_scale.alloc(a_rb_p*a_cb_p);
        cudaMemsetAsync(asc, 0, a_rb_p*a_cb_p, stream);
        const int threads = 256; const long total = (long)M*nblk_a;
        unsigned int grid = (unsigned int)((total + threads - 1)/threads);
        if (grid > 65535u) grid = 65535u; if (grid == 0) grid = 1;
        if (src1->type == GGML_TYPE_F16)
            fp8_a_to_mxfp8_kernel<half> <<<grid, threads, 0, stream>>>((const half*) src1->data, a_fp8_ptr, asc, M, K);
        else
            fp8_a_to_mxfp8_kernel<float><<<grid, threads, 0, stream>>>((const float*)src1->data, a_fp8_ptr, asc, M, K);
        if (cudaPeekAtLastError() != cudaSuccess) return false;
        a_scale_ptr = (float*)asc;   // reinterpreted; bound as a void* block-scale ptr below
    }

    // 3) cuBLASLt FP8xFP8 GEMM: D[M,N] = W_fp8[N,K] @ A_fp8[M,K]^T. Column-major m=N,n=M,k=K;
    //    A=weight (TN, e4m3), B=act (N, e4m3). Per-tensor SCALAR scales on A/B pointers; alpha=1.
    //    Under GGML_F8_MXFP8, A/B carry per-32-block UE8M0 scales (VEC32) instead.
    const int m=N, n=M, k=K;
    float alpha_h = 1.0f; static float beta_h = 0.0f;

    cublasLtMatmulDesc_t op = nullptr;
    if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return false;
    cublasLtMatmulMatrixScale_t sm = mxfp8 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
                                           : CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
    cublasOperation_t T=CUBLAS_OP_T, Nn=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &Nn, sizeof(Nn));
    void* wsp = (void*)w_scale_ptr; void* asp = (void*)a_scale_ptr;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &wsp, sizeof(wsp));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &asp, sizeof(asp));
    cublasDataType_t st = CUDA_R_32F;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &st, sizeof(st));

    // GGML_F8_CLAMP_OUT (native e4m3, plain F16 dst): accumulate into an F32 pool temp, then
    // clamp+downconvert to the F16 dst below — prevents the F16-store overflow -> inf without
    // widening the residual stream. Bias-epilogue GEMMs (never taken for e4m3) keep F16 store.
    const bool use_f32_temp = ggml_cuda_f8_clamp_out_enabled() && w_is_e4m3 &&
                              dst->type == GGML_TYPE_F16 && !want_epi;
    ggml_cuda_pool_alloc<float> d_f32(ctx.pool());
    void * gemm_out = dst->data;
    if (use_f32_temp) { gemm_out = d_f32.alloc((size_t)N * (size_t)M); }

    const cublasDataType_t out_dt = (dst->type == GGML_TYPE_F16 && !use_f32_temp) ? CUDA_R_16F : CUDA_R_32F;
    cublasLtMatrixLayout_t Ad=nullptr,Bd=nullptr,Cd=nullptr,Dd=nullptr;
    cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8F_E4M3, k, m, k);
    cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8F_E4M3, k, n, k);
    cublasLtMatrixLayoutCreate(&Cd, out_dt, m, n, m);
    cublasLtMatrixLayoutCreate(&Dd, out_dt, m, n, m);

    // fix #3: bias epilogue. bias dtype == output dtype (the canonical cuBLASLt case):
    // F32 bias passes straight through for F32 output; for F16 (dit_f16) output the F32
    // bias is downcast to F16 once into pool scratch. The bias is added in F32 compute
    // before the store, matching ggml's post-matmul bias order.
    ggml_cuda_pool_alloc<half> bias_f16(ctx.pool());
    if (want_epi) {
        void * bias_ptr = nullptr;
        if (dst->type == GGML_TYPE_F16) {
            half * b16 = bias_f16.alloc((size_t)N);
            fp8_bias_f32_to_f16<<<(unsigned)((N + 255)/256), 256, 0, stream>>>((const float*)bias->data, b16, N);
            if (cudaPeekAtLastError() != cudaSuccess) {
                if (Ad) cublasLtMatrixLayoutDestroy(Ad);
                if (Bd) cublasLtMatrixLayoutDestroy(Bd);
                if (Cd) cublasLtMatrixLayoutDestroy(Cd);
                if (Dd) cublasLtMatrixLayoutDestroy(Dd);
                if (op) cublasLtMatmulDescDestroy(op);
                return false;
            }
            bias_ptr = (void*)b16;
        } else {
            bias_ptr = (void*)bias->data;   // F32 bias + F32 output
        }
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &epi_out_dt, sizeof(epi_out_dt));
    }

    // cuBLASLt REQUIRES the matmul workspace to be 256-byte aligned; the ggml pool can hand back a
    // 128-byte-offset pointer (its state depends on prior allocs — e.g. the act-quant path shifts it
    // to 256 on a cache MISS but not on a HIT). A misaligned workspace makes cublasLtMatmul return
    // CUBLAS_STATUS_INVALID_VALUE(7) -> the caller falls to the wglobal-less dequant path -> NaN K/V.
    // Over-allocate +256 and round up so the workspace is always 256-aligned.
    size_t wsz = 32*1024*1024;
    ggml_cuda_pool_alloc<uint8_t> ws(ctx.pool(), wsz + 256);
    uint8_t * ws_ptr = (uint8_t *)(((uintptr_t)ws.get() + 255) & ~(uintptr_t)255);
    // GGML_F8_ZERO_SCRATCH: zero the cuBLASLt workspace + the output before the GEMM (the
    // uninitialized-scratch discriminator; pool memory is recycled and may carry stale NaNs).
    if (f8_zero) {
        cudaMemsetAsync(ws_ptr, 0, wsz, stream);
        cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream);
    }

    static int s_nocache = -1;
    if (s_nocache < 0) { const char* e = getenv("GGML_NVFP4_CUBLASLT_NOCACHE"); s_nocache = (e && atoi(e)) ? 1 : 0; }
    // key includes the epilogue flag (a bias GEMM may select a different algo) AND the mxfp8
    // flag (block-scaled VEC32 vs SCALAR selects a DIFFERENT algo family — must not cross-reuse).
    static thread_local std::map<std::tuple<int,int,int,int,int,int>, cublasLtMatmulAlgo_t> g_fp8_algo_cache;
    const auto key = std::make_tuple(m, n, k, (int)out_dt, want_epi ? 1 : 0, mxfp8 ? 1 : 0);

    cublasLtMatmulAlgo_t algo; bool have_algo = false;
    if (!s_nocache) { auto it = g_fp8_algo_cache.find(key); if (it != g_fp8_algo_cache.end()) { algo = it->second; have_algo = true; } }
    if (!have_algo) {
        cublasLtMatmulPreference_t pref=nullptr;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz));
        cublasLtMatmulHeuristicResult_t hr[1] = {};
        int got=0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Dd, pref, 1, hr, &got);
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (hs == CUBLAS_STATUS_SUCCESS && got > 0) {
            algo = hr[0].algo;
            have_algo = true;
            if (!s_nocache) g_fp8_algo_cache[key] = algo;
        }
    }

    // fix #3: if the bias epilogue has no algo for this shape, record it (fail fast next
    // time) and return false WITHOUT touching dst — the caller runs the plain GEMM + a
    // separate bias add. (Plain GEMMs keep their existing fall-through behaviour below.)
    if (want_epi && !have_algo) {
        g_fp8_epi_unsupported.insert(epi_key);
        if (Ad) cublasLtMatrixLayoutDestroy(Ad);
        if (Bd) cublasLtMatrixLayoutDestroy(Bd);
        if (Cd) cublasLtMatrixLayoutDestroy(Cd);
        if (Dd) cublasLtMatrixLayoutDestroy(Dd);
        if (op) cublasLtMatmulDescDestroy(op);
        return false;
    }

    // cublasLtMatmul returns CUBLAS_STATUS_INVALID_VALUE (7) if a benign CUDA error is already
    // pending on the thread from an unrelated prior op (e.g. a RoPE/norm kernel between q and k).
    // Clear it so a legit fp8 matmul isn't spuriously rejected -> forced onto the wglobal-less
    // dequant fallback -> NaN K/V. (Only clears already-consumed errors; doesn't mask our own.)
    if (getenv("GGML_F8_CLEAR_ERR") == nullptr || atoi(getenv("GGML_F8_CLEAR_ERR")) != 0)
        (void)cudaGetLastError();
    bool ok = have_algo;
    cublasStatus_t ms = CUBLAS_STATUS_SUCCESS;
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_end   = nullptr;
    const bool profile_gemm = ok && ggml_cuda_fp8_gemm_profile_enabled();
    if (profile_gemm) {
        cudaEventCreateWithFlags(&profile_start, cudaEventDefault);
        cudaEventCreateWithFlags(&profile_end, cudaEventDefault);
        cudaEventRecord(profile_start, stream);
    }
    if (ok) {
        ms = cublasLtMatmul(lt, op, &alpha_h, w_fp8_ptr, Ad, a_fp8_ptr, Bd,
                                           &beta_h, gemm_out, Cd, gemm_out, Dd,
                                           &algo, ws_ptr, wsz, stream);
        ok = (ms == CUBLAS_STATUS_SUCCESS);
    }
    if (profile_gemm) {
        cudaEventRecord(profile_end, stream);
        cudaEventSynchronize(profile_end);
        float ms_elapsed = 0.0f;
        cudaEventElapsedTime(&ms_elapsed, profile_start, profile_end);
        fprintf(stderr, "[FP8_GEMM_PROFILE] ms=%.3f M=%d K=%d N=%d name=%s\n",
                ms_elapsed, M, K, N, src0->name[0] ? src0->name : "?");
        cudaEventDestroy(profile_start);
        cudaEventDestroy(profile_end);
    }
    // GGML_F8_DBG: per-GEMM matmul outcome — pins whether a bailed GEMM lost the algo
    // (have_algo=0) or the matmul returned non-success (ms!=0). w_global==1.0 => wglobal miss.
    if (w_is_e4m3 && getenv("GGML_F8_DBG")) {
        static std::atomic<int> s_ms{0}; int mi = s_ms.fetch_add(1);
        if (mi < 20) fprintf(stderr, "[F8_MM] #%d %s  ms=%d ok=%d  wsAlign256=%d wsAlign16=%d W16=%d A16=%d  m=%d n=%d k=%d\n",
            mi, src0->name?src0->name:"?", (int)ms, (int)ok,
            (int)((uintptr_t)ws_ptr % 256), (int)((uintptr_t)ws_ptr % 16),
            (int)((uintptr_t)w_fp8_ptr % 16), (int)((uintptr_t)a_fp8_ptr % 16), m, n, k);
    }
    // GGML_F8_CLAMP_OUT: clamp the F32 temp to the F16 range and store F16 into the real dst.
    if (ok && use_f32_temp) {
        const size_t n = (size_t)N * (size_t)M;
        const int thr = 256;
        unsigned int gr = (unsigned int)((n + thr - 1) / thr);
        if (gr > 65535u) gr = 65535u; if (gr == 0) gr = 1;
        fp8_clamp_f32_to_f16<<<gr, thr, 0, stream>>>((const float*)d_f32.get(), (half*)dst->data, n);
        ok = (cudaPeekAtLastError() == cudaSuccess);
    }

    // GGML_F8_DBG: for the first ~10 native-e4m3 GEMMs, report the B(activation) scale
    // (should be amax_act/448, STANDARD e4m3 — NOT the two-level amax/2688) and the output
    // magnitude max|D|. max|D| ~ O(1-10) => GEMM is fine, white is DOWNSTREAM; max|D| huge
    // (1e3+) => an operand-scale bug. Reuses fp8_a_amax_kernel for a device-side max-abs.
    if (ok && w_is_e4m3) {
        static int s_f8dbg2 = -1;
        if (s_f8dbg2 < 0) { const char * e = getenv("GGML_F8_DBG"); s_f8dbg2 = (e && atoi(e)) ? 1 : 0; }
        if (s_f8dbg2) {
            static std::atomic<int> s_n2{0};
            const int idx = s_n2.fetch_add(1);
            if (idx < 10) {
                ggml_cuda_pool_alloc<unsigned int> dmax(ctx.pool(), 1);
                cudaMemsetAsync(dmax.get(), 0, sizeof(unsigned int), stream);
                const long ndd = (long)N * (long)M;
                const int  thr = 256;
                unsigned int gr = (unsigned int)((ndd + thr - 1) / thr);
                if (gr > 1024u) gr = 1024u; if (gr == 0) gr = 1;
                if (dst->type == GGML_TYPE_F16) fp8_a_amax_kernel<half> <<<gr, thr, 0, stream>>>((const half*) dst->data, dmax.get(), ndd);
                else                            fp8_a_amax_kernel<float><<<gr, thr, 0, stream>>>((const float*)dst->data, dmax.get(), ndd);
                unsigned int dbits = 0; float bscale = 0.f;
                cudaMemcpyAsync(&dbits,  dmax.get(),   sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(&bscale, a_scale_ptr,  sizeof(float),        cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                float maxD = 0.f; memcpy(&maxD, &dbits, sizeof(float));
                fprintf(stderr, "[F8_DBG] fp8-GEMM #%d  B_scale(act)=%.6g amax_act=%.6g  A_scale(w)=%.6g  max|D|=%.6g  N=%d K=%d M=%d  name=%s  dst=%s%s\n",
                        idx, bscale, bscale * 448.0f, nvfp4_weight_global_for(src0->name), maxD, N, K, M,
                        src0->name ? src0->name : "?", dst->type == GGML_TYPE_F16 ? "F16" : "F32",
                        act_reused ? " (act-reused)" : "");
                // LOUD: exact nnan/ninf/amax of the raw INPUT (src1) and the GEMM OUTPUT (dst).
                // Disambiguates "max|D|=0" (real zero vs all-NaN). IN clean + OUT nan => the GEMM
                // itself births the NaN; IN nan => it arrived from upstream.
                {
                    ggml_cuda_pool_alloc<unsigned int> ist(ctx.pool(), 3), ost(ctx.pool(), 3);
                    cudaMemsetAsync(ist.get(), 0, 3*sizeof(unsigned int), stream);
                    cudaMemsetAsync(ost.get(), 0, 3*sizeof(unsigned int), stream);
                    const long ni = (long)M * K, no = (long)N * M;
                    unsigned int gi = (unsigned int)((ni + 255)/256); if (gi > 1024u) gi = 1024u; if (!gi) gi = 1;
                    unsigned int go = (unsigned int)((no + 255)/256); if (go > 1024u) go = 1024u; if (!go) go = 1;
                    if (src1->type == GGML_TYPE_F16) fp8_nanstats_kernel<half> <<<gi,256,0,stream>>>((const half*) src1->data, ni, ist.get());
                    else                             fp8_nanstats_kernel<float><<<gi,256,0,stream>>>((const float*)src1->data, ni, ist.get());
                    if (dst->type  == GGML_TYPE_F16) fp8_nanstats_kernel<half> <<<go,256,0,stream>>>((const half*) dst->data,  no, ost.get());
                    else                             fp8_nanstats_kernel<float><<<go,256,0,stream>>>((const float*)dst->data,  no, ost.get());
                    unsigned int ih[3] = {0,0,0}, oh[3] = {0,0,0};
                    cudaMemcpyAsync(ih, ist.get(), 3*sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
                    cudaMemcpyAsync(oh, ost.get(), 3*sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    float iam = 0.f, oam = 0.f; memcpy(&iam, &ih[2], 4); memcpy(&oam, &oh[2], 4);
                    fprintf(stderr, "[F8_LOUD] #%d  IN{nnan=%u ninf=%u amax=%.5g}  OUT{nnan=%u ninf=%u amax=%.5g}  K=%d N=%d M=%d  %s\n",
                            idx, ih[0], ih[1], iam, oh[0], oh[1], oam, K, N, M, src0->name ? src0->name : "?");
                }
            }
        }
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
