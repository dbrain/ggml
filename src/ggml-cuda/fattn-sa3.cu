#include "fattn-sa3.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <mutex>

#include "fattn-lse-selfcheck.cuh"
#include "sageattention3/launch.h"

// This file is intentionally the only ggml-facing SA3 adapter.  The imported
// CUTLASS kernel remains upstream code; preprocessing, allocation and dispatch
// are native ggml/CUDA so enabling this feature never introduces Torch at runtime.
constexpr int kHeadDim = 128;
constexpr int kTokenBlock = 128;
constexpr int kQuantEltsPerThread = 16;

__global__ void sa3_cast_q_f32(const float * in, half * out, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(in[i]);
}

// GQA source-head map. The adapter always works on a contiguous run of `head_group` QUERY
// heads starting at `head_start`, and materialises one padded plane per query head. Q reads
// its own head (gqa_ratio == 1); K/V read the KV head that query head shares, which is
// (head_start + h) / gqa_ratio. Callers therefore pass the UNOFFSET tensor base — offsetting
// the pointer by a query-head index is exactly the out-of-bounds read that this replaces.
__device__ __forceinline__ int sa3_src_head(int h, int head_start, int gqa_ratio) {
    return (head_start + h) / gqa_ratio;
}

__global__ void sa3_cast_q_f32_pad(const float * in, half * out, int l, int lr, int heads,
                                   int head_start, int gqa_ratio) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) heads * lr * kHeadDim;
    if (i < n) {
        const int d = i % kHeadDim;
        const size_t token = i / kHeadDim;
        const int h = token / lr;
        const int t = token % lr;
        const int sh = sa3_src_head(h, head_start, gqa_ratio);
        out[i] = __float2half_rn(t < l ? in[((size_t) sh * l + t) * kHeadDim + d] : 0.0f);
    }
}

__global__ void sa3_pad_f16(const half * in, half * out, int l, int lr, int heads,
                            int head_start, int gqa_ratio) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) heads * lr * kHeadDim;
    if (i < n) {
        const int d = i % kHeadDim;
        const size_t token = i / kHeadDim;
        const int h = token / lr;
        const int t = token % lr;
        const int sh = sa3_src_head(h, head_start, gqa_ratio);
        out[i] = t < l ? in[((size_t) sh * l + t) * kHeadDim + d] : __float2half(0.0f);
    }
}

// `mean` is indexed by the OUTPUT (query) head h, so under GQA the same KV head's mean is
// recomputed gqa_ratio times. That redundancy is the deliberate cost of keeping the CUTLASS
// launch strictly h == h_k; see the h_h_k_ratio note in sa3_run().
__global__ void sa3_k_mean(const half * k, float * mean, int l, int head_start, int gqa_ratio) {
    const int d = blockIdx.x;
    const int h = blockIdx.y;
    const int sh = sa3_src_head(h, head_start, gqa_ratio);
    float sum = 0.0f;
    for (int t = threadIdx.x; t < l; t += blockDim.x) {
        sum += __half2float(k[((size_t) sh * l + t) * kHeadDim + d]);
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x / 2; n != 0; n /= 2) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (threadIdx.x == 0) mean[h * kHeadDim + d] = partial[0] / l;
}

__global__ void sa3_center_k(const half * in, const float * mean, half * out, size_t n, int l) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int d = i % kHeadDim;
        const int h = i / ((size_t) l * kHeadDim);
        out[i] = __float2half_rn(__half2float(in[i]) - mean[h * kHeadDim + d]);
    }
}

__global__ void sa3_center_k_pad(const half * in, const float * mean, half * out, int l, int lr, int heads,
                                 int head_start, int gqa_ratio) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) heads * lr * kHeadDim;
    if (i < n) {
        const int d = i % kHeadDim;
        const size_t token = i / kHeadDim;
        const int h = token / lr;
        const int t = token % lr;
        const int sh = sa3_src_head(h, head_start, gqa_ratio);
        out[i] = t < l ? __float2half_rn(__half2float(in[((size_t) sh * l + t) * kHeadDim + d]) - mean[h * kHeadDim + d]) : __float2half(0.0f);
    }
}

// One CTA computes a Q mean for one [head, 128-token block, channel].  The
// subsequent centering pass preserves the original BHSD physical layout.
//
// `valid_l` is the UNPADDED query length. `q` here is the already-padded scratch plane, so the
// tail block is backed by real zeros; dividing that block by the full kTokenBlock would shrink
// its mean toward zero and defeat the whole point of the block mean (narrowing Q's FP4 dynamic
// range). LTX never noticed: 32,640 / 128 = 255 exactly, so it has no tail block. Krea-2 does —
// 4,125 = 32*128 + 29 — and that tail block is the TEXT tokens.
__global__ void sa3_q_block_mean(const half * q, half * mean, int l, int blocks, int valid_l) {
    const int d = blockIdx.x;
    const int qb = blockIdx.y;
    const int h = blockIdx.z;
    const int t = qb * kTokenBlock + threadIdx.x;
    float x = t < l ? __half2float(q[((size_t) h * l + t) * kHeadDim + d]) : 0.0f;
    __shared__ float partial[kTokenBlock];
    partial[threadIdx.x] = x;
    __syncthreads();
    for (int n = kTokenBlock / 2; n != 0; n /= 2) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int valid = min(kTokenBlock, max(1, valid_l - qb * kTokenBlock));
        mean[((size_t) h * blocks + qb) * kHeadDim + d] = __float2half_rn(partial[0] / valid);
    }
}

__global__ void sa3_center_q(const half * in, const half * mean, half * out, size_t n, int l, int blocks) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int d = i % kHeadDim;
        const size_t token = i / kHeadDim;
        const int h = token / l;
        const int qb = (token % l) / kTokenBlock;
        out[i] = __float2half_rn(__half2float(in[i]) - __half2float(mean[((size_t) h * blocks + qb) * kHeadDim + d]));
    }
}

inline __device__ uint32_t sa3_fp32_to_e2m1(float2 * x) {
    uint32_t v;
    // The FP4 pack PTX (cvt.e2m1x2) is Blackwell-only (sm_100+); ptxas rejects it for
    // sm_86. Arch-gate so the SA3 sources COMPILE for sm86 (as a no-op) and one binary
    // builds for 86;120 — SA3 is only ever dispatched on Blackwell at runtime
    // (ggml_cuda_flash_attn_ext_sa3 is gated by GGML_LTX_SA3 + cc), so the sm86 stub is
    // never executed. (Same >= 1000 gate style as sageattention3/fp4_quantization_4d.cu.)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile(
        "{ .reg .b8 a,b,c,d;"
        "cvt.rn.satfinite.e2m1x2.f32 a,%2,%1;"
        "cvt.rn.satfinite.e2m1x2.f32 b,%4,%3;"
        "cvt.rn.satfinite.e2m1x2.f32 c,%6,%5;"
        "cvt.rn.satfinite.e2m1x2.f32 d,%8,%7;"
        "mov.b32 %0,{a,b,c,d}; }"
        : "=r"(v) : "f"(x[0].x), "f"(x[0].y), "f"(x[1].x), "f"(x[1].y),
          "f"(x[2].x), "f"(x[2].y), "f"(x[3].x), "f"(x[3].y));
#else
    (void) x;
    v = 0u;  // sm86 stub: SA3 never runs on non-Blackwell (runtime cc-gated)
#endif
    return v;
}

// Reconstruction error for one 16-value FP4 group at a proposed E4M3 scale. The final pack still
// uses the hardware cvt below; this scalar oracle is only for choosing the scale. At an exact
// E2M1 midpoint either neighbouring code has identical squared error, so its tie convention cannot
// change which scale minimizes MSE.
inline __device__ float sa3_e2m1_group_mse(const half2 * values, float scale) {
    constexpr float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float error = 0.0f;
    const float inv = scale == 0.0f ? 0.0f : 1.0f / scale;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float2 pair = __half22float2(values[i]);
        const float source[2] = {pair.x, pair.y};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float magnitude = fabsf(source[j] * inv);
            int best = 0;
            float best_distance = fabsf(magnitude - levels[0]);
            #pragma unroll
            for (int code = 1; code < 8; ++code) {
                const float distance = fabsf(magnitude - levels[code]);
                if (distance < best_distance) {
                    best = code;
                    best_distance = distance;
                }
            }
            const float reconstructed = copysignf(levels[best] * scale, source[j]);
            const float residual = source[j] - reconstructed;
            error = fmaf(residual, residual, error);
        }
    }
    return error;
}

inline __device__ uint8_t sa3_mse_refine_scale(const half2 * values, uint8_t baseline_bits, int radius) {
    if (radius <= 0) {
        return baseline_bits;
    }
    uint8_t best_bits = baseline_bits;
    const float baseline_scale = float(reinterpret_cast<const __nv_fp8_e4m3 &>(baseline_bits));
    float best_error = sa3_e2m1_group_mse(values, baseline_scale);
    // Baseline first and strict improvement preserve it on ties. Alternate lower/higher codes so
    // equal-distance neighbours have no systematic upward bias.
    #pragma unroll 1
    for (int distance = 1; distance <= radius; ++distance) {
        #pragma unroll
        for (int direction = -1; direction <= 1; direction += 2) {
            const int candidate = min(126, max(0, int(baseline_bits) + direction * distance));
            const uint8_t candidate_bits = static_cast<uint8_t>(candidate);
            const float candidate_scale =
                float(reinterpret_cast<const __nv_fp8_e4m3 &>(candidate_bits));
            const float candidate_error = sa3_e2m1_group_mse(values, candidate_scale);
            if (candidate_error < best_error) {
                best_error = candidate_error;
                best_bits = candidate_bits;
            }
        }
    }
    return best_bits;
}

template <bool Permute>
__global__ void sa3_quant_qk(const half * in, uint8_t * out, uint8_t * sf, int l, int lr,
                            bool clip_input, int mse_scale_radius) {
    const int tb = blockIdx.x, h = blockIdx.z;
    const int lane = threadIdx.x;
    const int token = tb * kTokenBlock + lane / (kHeadDim / kQuantEltsPerThread);
    int load_token = token;
    if constexpr (Permute) {
        const int local = lane / (kHeadDim / kQuantEltsPerThread);
        const int r = local % 32;
        load_token = tb * kTokenBlock + (local / 32) * 32 + (r / 8) * 2 + ((r % 8) / 2) * 8 + r % 2;
    }
    half2 values[8] = {};
    if (load_token < l) {
        const half * p = in + ((size_t) h * l + load_token) * kHeadDim + (lane % 8) * kQuantEltsPerThread;
        #pragma unroll
        for (int i = 0; i < 8; ++i) values[i] = reinterpret_cast<const half2 *>(p)[i];
    }
    if (clip_input) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float2 x = __half22float2(values[i]);
            x.x = fminf(fmaxf(x.x, -2688.0f), 2688.0f);
            x.y = fminf(fmaxf(x.y, -2688.0f), 2688.0f);
            values[i] = __floats2half2_rn(x.x, x.y);
        }
    }
    half2 mx = __habs2(values[0]);
    #pragma unroll
    for (int i = 1; i < 8; ++i) mx = __hmax2(mx, __habs2(values[i]));
    const float scale_raw = __half2float(__hmax(mx.x, mx.y)) / 6.0f;
    const __nv_fp8_e4m3 fp8_scale(scale_raw);
    uint8_t scale_bits = reinterpret_cast<const uint8_t &>(fp8_scale);
    scale_bits = sa3_mse_refine_scale(values, scale_bits, mse_scale_radius);
    const float scale = float(reinterpret_cast<const __nv_fp8_e4m3 &>(scale_bits));
    float2 x[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        x[i] = __half22float2(values[i]);
        const float inv = scale == 0.0f ? 0.0f : 1.0f / scale;
        x[i].x *= inv; x[i].y *= inv;
    }
    if (token < lr) {
        uint32_t * dst = reinterpret_cast<uint32_t *>(out + ((size_t) h * lr + token) * (kHeadDim / 2) + (lane % 8) * 8);
        dst[0] = sa3_fp32_to_e2m1(x);
        dst[1] = sa3_fp32_to_e2m1(x + 4);
        const int local_token = token % 64;
        const int col = lane % 8;
        sf[(size_t) h * lr * (kHeadDim / 16) + (token / 64) * 64 * (kHeadDim / 16)
           + (col / 4) * 256 + col % 4 + (local_token / 16) * 4 + (local_token % 16) * 16] = scale_bits;
    }
}

// V uses the transposed physical layout expected by the Blackwell PV MMA.
__global__ void sa3_quant_vt(const half * in, uint8_t * out, uint8_t * sf, int l, int lr,
                            bool clip_input, int mse_scale_radius) {
    const int tb = blockIdx.x, h = blockIdx.z, lane = threadIdx.x;
    const int d = lane / 8;
    const int token0_local = (lane % 8) * kQuantEltsPerThread;
    const int token0 = tb * kTokenBlock + token0_local;
    __shared__ half tile[kTokenBlock * kHeadDim];
    const int token = tb * kTokenBlock + lane / 8;
    half2 v[8] = {};
    if (token < l) {
        const half * p = in + ((size_t) h * l + token) * kHeadDim + (lane % 8) * kQuantEltsPerThread;
        #pragma unroll
        for (int i = 0; i < 8; ++i) reinterpret_cast<half2 *>(tile)[lane * 8 + i] = reinterpret_cast<const half2 *>(p)[i];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = make_half2(tile[(token0_local + 2*i) * kHeadDim + d], tile[(token0_local + 2*i + 1) * kHeadDim + d]);
    if (clip_input) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float2 x = __half22float2(v[i]);
            x.x = fminf(fmaxf(x.x, -2688.0f), 2688.0f);
            x.y = fminf(fmaxf(x.y, -2688.0f), 2688.0f);
            v[i] = __floats2half2_rn(x.x, x.y);
        }
    }
    half2 mx = __habs2(v[0]);
    #pragma unroll
    for (int i = 1; i < 8; ++i) mx = __hmax2(mx, __habs2(v[i]));
    const float raw = __half2float(__hmax(mx.x, mx.y)) / 6.0f;
    const __nv_fp8_e4m3 fp8_scale(raw);
    uint8_t bits = reinterpret_cast<const uint8_t &>(fp8_scale);
    bits = sa3_mse_refine_scale(v, bits, mse_scale_radius);
    const float scale = float(reinterpret_cast<const __nv_fp8_e4m3 &>(bits));
    float2 x[8]; const float inv = scale == 0.0f ? 0.0f : 1.0f / scale;
    #pragma unroll
    for (int i = 0; i < 8; ++i) { x[i] = __half22float2(v[i]); x[i].x *= inv; x[i].y *= inv; }
    uint32_t * p = reinterpret_cast<uint32_t *>(out + ((size_t) h * kHeadDim + d) * (lr / 2) + token0 / 2);
    p[0] = sa3_fp32_to_e2m1(x); p[1] = sa3_fp32_to_e2m1(x + 4);
    const int row = d % 64, col = tb * 8 + lane % 8;
    sf[(size_t) h * kHeadDim * (lr / 16) + (d / 64) * 64 * (lr / 16)
       + (col / 4) * 256 + col % 4 + (row / 16) * 4 + (row % 16) * 16] = bits;
}

__global__ void sa3_o_to_dst(const half * in, void * dst, int l, int lr, int heads, int head_start, int total_heads, bool f32) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) heads * l * kHeadDim;
    if (i < n) {
        const int d = i % kHeadDim; const size_t t = i / kHeadDim;
        const int h = t / l; const int token = t % l;
        const size_t out_i = ((size_t) token * total_heads + head_start + h) * kHeadDim + d;
        const half x = in[((size_t) h * lr + token) * kHeadDim + d];
        if (f32) static_cast<float *>(dst)[out_i] = __half2float(x);
        else static_cast<half *>(dst)[out_i] = x;
    }
}

// LSE fix-up and copy-out.
//
// SA3 smooths K by subtracting a per-(head, channel) mean before the QK GEMM and folds the
// q-block mean back in through delta_s, so what the kernel actually exponentiates is
//
//     S'_ij = q_i . (k_j - kmean_h) = S_ij - q_i . kmean_h
//
// -- a shift that is CONSTANT ALONG A QUERY ROW. The output O is invariant to it (softmax
// is shift-invariant), which is exactly why the kernel is correct today. The LSE is not:
// it comes out low by scale * (q_i . kmean_h). And kmean is computed over THIS call's keys,
// so in a segmented merge every segment carries a DIFFERENT shift and it does not cancel
// between them -- getting this wrong would silently mis-weight the merge, not crash.
//
// One warp per query row so the 128 head-dim values are read as one coalesced burst.
template <typename TQ>
__global__ void sa3_lse_to_dst(const float * __restrict__ lse_src, const TQ * __restrict__ q,
                               const float * __restrict__ kmean, float * __restrict__ dst,
                               int lq, int lqr, int head_start, float scale) {
    const int warps_per_block = blockDim.x / 32;
    const int lane = threadIdx.x % 32;
    const int row  = blockIdx.x * warps_per_block + threadIdx.x / 32;
    const int h    = blockIdx.y;
    if (row >= lq) return;
    const TQ    * qp = q     + ((size_t) h * lq + row) * kHeadDim;
    const float * mp = kmean + (size_t) h * kHeadDim;
    constexpr int per_lane = kHeadDim / 32;
    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < per_lane; ++i) {
        const int d = lane * per_lane + i;
        acc += (float) qp[d] * mp[d];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) {
        dst[(size_t) (head_start + h) * lq + row] = lse_src[(size_t) h * lqr + row] + scale * acc;
    }
}

// The shape/dtype contract. `allow_rect` relaxes ONLY the Lq == Lkv requirement and is set
// exclusively by the LSE entry point, so the production GGML_OP_FLASH_ATTN_EXT route keeps
// rejecting precisely what it rejected before (notably LTX's cross-attention).
// Minimum K/V sequence length SA3 will accept. See the scope note inside sa3_contract_ok().
//
// ⚠️ 768, NOT a rounder-looking 2048, and the difference is a MEASURED LTX regression. A 1280x704
// 97-frame LTX render dispatches its DiT attention at L=11440 but ALSO issues D=128, 32-head,
// mask-free, contiguous F16 attentions at **L=1024** — which satisfy every other clause here, so
// they were served by SA3 before this gate existed. A 2048 floor silently moved them to cuDNN and
// made LTX output non-identical for no reason. 768 clears every krea2 shape we mean to exclude
// (text tower L=29, Qwen3-VL vision tower L=580) while leaving LTX exactly as it was.
//
// Verify after ANY change here by counting "[sa3] reject" against a real LTX render: a reject at
// an L that LTX previously served is a regression, not a tightening.
static int sa3_min_lkv() {
    static const int v = [] {
        const char * e = getenv("GGML_LTX_SA3_MIN_LKV");
        const int parsed = e ? atoi(e) : 0;
        return parsed > 0 ? parsed : 768;
    }();
    return v;
}

static bool sa3_contract_ok(int cc, const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v,
                            const ggml_tensor * mask, ggml_type o_type, bool o_contiguous,
                            float max_bias, float softcap, bool allow_rect) {
    // ⚠️ `q->ne[2] == 32` used to stand here, and it was doing TWO jobs: matching the kernel's
    // (nonexistent) head requirement, and — accidentally — scoping SA3 to the LTX DiT. The
    // CUTLASS kernel never templated on head count, so the first job was imaginary; the second
    // was real. Widening this to GQA therefore has to REPLACE the accidental scope with a
    // deliberate one, or SA3 starts capturing every other D=128 attention in the process —
    // notably Krea-2's own text tower (20 heads, L~29), where padding 29 tokens to a 128-token
    // block and running the full quantise/delta/copy pipeline is a guaranteed net loss.
    // Hence the minimum-Lkv gate: SA3 only pays off once the O(L^2) attention it replaces
    // dominates its own O(L*H) preprocessing.
    const int min_lkv = sa3_min_lkv();
    return cc >= GGML_CUDA_CC_BLACKWELL && mask == nullptr &&
        q->ne[0] == kHeadDim && (allow_rect || q->ne[1] == k->ne[1]) && k->ne[1] == v->ne[1] &&
        k->ne[1] >= min_lkv &&
        q->ne[2] >= 1 && k->ne[2] >= 1 && k->ne[2] == v->ne[2] &&
        q->ne[2] % k->ne[2] == 0 && q->ne[3] == 1 &&
        k->ne[3] == 1 && v->ne[3] == 1 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 &&
        (q->type == GGML_TYPE_F16 || q->type == GGML_TYPE_F32) && (o_type == GGML_TYPE_F16 || o_type == GGML_TYPE_F32) &&
        max_bias == 0.0f && softcap == 0.0f && ggml_is_contiguous(q) && ggml_is_contiguous(k) &&
        ggml_is_contiguous(v) && o_contiguous;
}

// The single implementation behind both entry points.
//
//   o   -- [D, H, Lq, 1] BSHD, F32 or F16, contiguous (ggml's flash-attn dst layout).
//   lse -- null on the inference path; otherwise F32 [Lq, H, 1], query innermost, which is
//          the layout ggml_flash_attn_ext_lse_stats() views.
static void sa3_run(ggml_backend_cuda_context & ctx,
                    const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v,
                    void * o, bool o_f32, float * lse_dst, float scale) {
    const int total_h   = (int) q->ne[2];
    const int gqa_ratio = (int) (q->ne[2] / k->ne[2]);
    // Q and K/V lengths are independent: the relay's segment merge calls this with a full
    // query set against one slice of the keys. Each side is padded to its own multiple of
    // the 128-token block; the CUTLASS kernel already grids M over ceil(seqlen_q/kBlockM)
    // and masks the K tail at `unpadded_seqlen_k`, so only the adapter had to learn this.
    const int lq = q->ne[1], lk = k->ne[1];
    const int lqr = ((lq + 127) / 128) * 128, lkr = ((lk + 127) / 128) * 128;
    const int qblocks = lqr / 128, kblocks = lkr / 128;
    int head_group = total_h;
    if (const char * e = getenv("GGML_LTX_SA3_HEAD_GROUP")) {
        const int requested = atoi(e);
        if (requested > 0 && requested <= total_h && total_h % requested == 0) {
            head_group = requested;
        } else {
            // Silently falling back to total_h is a VRAM cliff, not a nicety: at 48 heads that
            // is ~280 MiB of scratch instead of ~47 MiB. Say so.
            static bool warned = false;
            if (!warned) {
                warned = true;
                fprintf(stderr, "[sa3] GGML_LTX_SA3_HEAD_GROUP=%d ignored (must divide %d); using %d\n",
                        requested, total_h, total_h);
            }
        }
    }
    const size_t elems_q        = (size_t) head_group * lq  * kHeadDim;   // unpadded Q/O
    const size_t elems_q_padded = (size_t) head_group * lqr * kHeadDim;
    const size_t elems_k_padded = (size_t) head_group * lkr * kHeadDim;
    const cudaStream_t stream = ctx.stream();
    // The locked singing repro enables the validated F16 storage path. Keep
    // the core runtime opt-in so other callers retain the original FP32 path.
    const char * delta_f16_env = getenv("GGML_LTX_SA3_DELTA_F16");
    const bool delta_f16 = delta_f16_env != nullptr && atoi(delta_f16_env) != 0;
    // Radius 1 is the validated common correctness default. Radius 0 remains
    // available as the byte-for-byte legacy quantizer for regression/bisect.
    int mse_scale_radius = 1;
    if (const char * e = getenv("GGML_LTX_SA3_MSE_SCALE_RADIUS")) {
        char * end = nullptr;
        const long requested = strtol(e, &end, 10);
        if (end != e && *end == '\0' && requested >= 0 && requested <= 8) {
            mse_scale_radius = static_cast<int>(requested);
        } else {
            static bool warned = false;
            if (!warned) {
                warned = true;
                fprintf(stderr,
                        "[sa3] GGML_LTX_SA3_MSE_SCALE_RADIUS=%s ignored (valid range 0..8); using 1\n", e);
            }
        }
    }
    const bool timing = getenv("GGML_LTX_SA3_TIMING") != nullptr;
    static cudaEvent_t timing_begin = nullptr, timing_mha = nullptr, timing_end = nullptr;
    static std::once_flag timing_once;
    if (timing) {
        std::call_once(timing_once, [] {
            CUDA_CHECK(cudaEventCreate(&timing_begin));
            CUDA_CHECK(cudaEventCreate(&timing_mha));
            CUDA_CHECK(cudaEventCreate(&timing_end));
        });
        CUDA_CHECK(cudaEventRecord(timing_begin, stream));
    }
    auto check_stage = [&](const char * stage) {
        (void) stage;
        CUDA_CHECK(cudaGetLastError());
    };
    // The preprocessing tensors have non-overlapping lifetimes.  Reuse one
    // padded BHSD scratch buffer for Q, K, V, then for the attention output;
    // at long chained resolutions this avoids four simultaneous 250+ MiB
    // buffers and keeps the native path within the VRAM budget.
    ggml_cuda_pool_alloc<half> scratch(ctx.pool()), q_mean(ctx.pool());
    ggml_cuda_pool_alloc<float> k_mean(ctx.pool());
    ggml_cuda_pool_alloc<half> delta_f16_buf(ctx.pool());
    ggml_cuda_pool_alloc<float> delta(ctx.pool()), lse(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> q4(ctx.pool()), k4(ctx.pool()), v4(ctx.pool()), sfq(ctx.pool()), sfk(ctx.pool()), sfv(ctx.pool());
    scratch.alloc(std::max(elems_q_padded, elems_k_padded));
    q_mean.alloc((size_t) head_group * qblocks * kHeadDim);
    k_mean.alloc((size_t) head_group * kHeadDim);
    if (delta_f16) delta_f16_buf.alloc((size_t) head_group * qblocks * lkr);
    else delta.alloc((size_t) head_group * qblocks * lkr);
    if (lse_dst != nullptr) lse.alloc((size_t) head_group * lqr);
    q4.alloc((size_t) head_group * lqr * kHeadDim / 2); k4.alloc((size_t) head_group * lkr * kHeadDim / 2); v4.alloc((size_t) head_group * kHeadDim * lkr / 2);
    sfq.alloc((size_t) head_group * lqr * kHeadDim / 16); sfk.alloc((size_t) head_group * lkr * kHeadDim / 16); sfv.alloc((size_t) head_group * kHeadDim * lkr / 16);
    for (int head_start = 0; head_start < total_h; head_start += head_group) {
        // ⚠️ The tensor bases are passed UNOFFSET and the per-head source index is resolved
        // inside each kernel by sa3_src_head(). Offsetting k/v by a QUERY-head index (what this
        // did before) is correct only when H_kv == H_q; under GQA it walks gqa_ratio times past
        // the end of K and V. Out-of-range TMA reads are clamped, not trapped, so that bug
        // produces wrong pixels at a plausible speed rather than a crash.
        const float * q_f32 = static_cast<const float *>(q->data);
        const half * k_f16 = static_cast<const half *>(k->data);
        const half * v_f16 = static_cast<const half *>(v->data);
        // Q alone may still be offset by a query-head index — that IS its own head index. Used
        // only by the LSE tail below, which indexes Q with a group-local head. K/V must never
        // be offset this way; that is what sa3_src_head() exists for.
        const size_t q_head_offset = (size_t) head_start * lq * kHeadDim;
        if (q->type == GGML_TYPE_F32) sa3_cast_q_f32_pad<<<(elems_q_padded + 255) / 256, 256, 0, stream>>>(q_f32, scratch.get(), lq, lqr, head_group, head_start, 1);
        else sa3_pad_f16<<<(elems_q_padded + 255) / 256, 256, 0, stream>>>((const half *) q->data, scratch.get(), lq, lqr, head_group, head_start, 1);
        check_stage("Q cast");
        sa3_k_mean<<<dim3(kHeadDim, head_group), 256, 0, stream>>>(k_f16, k_mean.get(), lk, head_start, gqa_ratio);
        sa3_q_block_mean<<<dim3(kHeadDim, qblocks, head_group), kTokenBlock, 0, stream>>>(scratch.get(), q_mean.get(), lqr, qblocks, lq);
        sa3_center_q<<<(elems_q_padded + 255) / 256, 256, 0, stream>>>(scratch.get(), q_mean.get(), scratch.get(), elems_q_padded, lqr, qblocks);
        check_stage("centering");
        sa3_quant_qk<false><<<dim3(qblocks, 1, head_group), 1024, 0, stream>>>(
            scratch.get(), q4.get(), sfq.get(), lqr, lqr, false, mse_scale_radius);
        check_stage("Q FP4 quantization");

        // Q is quantized, so its scratch storage can become centered K.
        sa3_center_k_pad<<<(elems_k_padded + 255) / 256, 256, 0, stream>>>(k_f16, k_mean.get(), scratch.get(), lk, lkr, head_group, head_start, gqa_ratio);
        // The long-resolution delta GEMM can otherwise select an atomic
        // reduction algorithm.  Those reductions are order-dependent on
        // Blackwell, so exclude them for this reproducibility-critical path.
        cublasAtomicsMode_t previous_atomics_mode;
        CUBLAS_CHECK(cublasGetAtomicsMode(ctx.cublas_handle(), &previous_atomics_mode));
        CUBLAS_CHECK(cublasSetAtomicsMode(ctx.cublas_handle(), CUBLAS_ATOMICS_NOT_ALLOWED));
        for (int head = 0; head < head_group; ++head) {
            const float one = 1.0f, zero = 0.0f;
            CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
            CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, lkr, qblocks, kHeadDim,
                &one, scratch.get() + (size_t) head * lkr * kHeadDim, CUDA_R_16F, kHeadDim,
                q_mean.get() + (size_t) head * qblocks * kHeadDim, CUDA_R_16F, kHeadDim,
                &zero,
                delta_f16 ? static_cast<void *>(delta_f16_buf.get() + (size_t) head * qblocks * lkr)
                          : static_cast<void *>(delta.get() + (size_t) head * qblocks * lkr),
                delta_f16 ? CUDA_R_16F : CUDA_R_32F, lkr, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        CUBLAS_CHECK(cublasSetAtomicsMode(ctx.cublas_handle(), previous_atomics_mode));
        check_stage("delta GEMM");
        sa3_quant_qk<true><<<dim3(kblocks, 1, head_group), 1024, 0, stream>>>(
            scratch.get(), k4.get(), sfk.get(), lkr, lkr, false, mse_scale_radius);
        check_stage("K FP4 quantization");
        // K is quantized, so the same scratch storage can become padded V.
        sa3_pad_f16<<<(elems_k_padded + 255) / 256, 256, 0, stream>>>(v_f16, scratch.get(), lk, lkr, head_group, head_start, gqa_ratio);
        sa3_quant_vt<<<dim3(kblocks, 1, head_group), 1024, 0, stream>>>(
            scratch.get(), v4.get(), sfv.get(), lkr, lkr, false, mse_scale_radius);
        check_stage("V FP4 quantization");
        Flash_fwd_params p = {};
        p.q_ptr=q4.get(); p.k_ptr=k4.get(); p.v_ptr=v4.get(); p.delta_s_ptr=delta_f16 ? static_cast<void *>(delta_f16_buf.get()) : static_cast<void *>(delta.get()); p.sfq_ptr=sfq.get(); p.sfk_ptr=sfk.get(); p.sfv_ptr=sfv.get(); p.o_ptr=scratch.get();
        // Null on the inference path: the epilogue's LSE store is predicated on this pointer,
        // so production SA3 executes exactly the instructions it did before.
        p.softmax_lse_ptr = lse_dst != nullptr ? static_cast<void *>(lse.get()) : nullptr;
        p.b=1; p.h=head_group; p.h_k=head_group; p.h_h_k_ratio=1; p.seqlen_q=lqr; p.seqlen_k=lkr; p.unpadded_seqlen_k=lk; p.seqlen_q_rounded=lqr; p.seqlen_k_rounded=lkr; p.d=kHeadDim; p.d_rounded=kHeadDim; p.head_divmod=cutlass::FastDivmod(head_group);
        // ⚠️ ARMED TRAP. GQA is handled entirely by REPLICATION during preprocessing, so every
        // plane handed to the kernel carries one KV head per QUERY head and the launch must stay
        // h == h_k. `p.h_h_k_ratio` is NOT a working knob: it is assigned here and in api.cu and
        // then read NOWHERE under sageattention3/ (grep it). Anyone who "enables GQA natively" by
        // setting h_k < h will get silently wrong pixels, because mainloop_tma_ws.h indexes
        // K/V/SFK/SFVt with bidh (a QUERY head) and launch.h:61 sizes shape_ds from h_k while
        // mDS is indexed by bidh — and out-of-range TMA is clamped, not trapped. Doing it
        // properly means patching those five sites together; until then, fail loudly here.
        GGML_ASSERT(p.h == p.h_k && p.h_h_k_ratio == 1 &&
                    "SA3: KV is replicated per query head; native GQA needs the mainloop/shape_ds fixes");
    // FP4 pointer arithmetic is in nibbles (cutlass::float_e2m1_t), whereas
    // the backing buffers above are byte-addressed.  The upstream adapter
    // therefore multiplies every Q/K/V stride by two.
    p.q_row_stride=kHeadDim; p.k_row_stride=kHeadDim; p.v_row_stride=lkr; p.q_head_stride=(int64_t) lqr*kHeadDim; p.k_head_stride=(int64_t) lkr*kHeadDim; p.v_head_stride=(int64_t) kHeadDim*lkr;
    p.q_batch_stride=(int64_t) head_group*lqr*kHeadDim; p.k_batch_stride=(int64_t) head_group*lkr*kHeadDim; p.v_batch_stride=(int64_t) head_group*kHeadDim*lkr;
    p.sfq_row_stride=kHeadDim/16; p.sfk_row_stride=kHeadDim/16; p.sfv_row_stride=lkr/16; p.sfq_head_stride=(int64_t) lqr*kHeadDim/16; p.sfk_head_stride=(int64_t) lkr*kHeadDim/16; p.sfv_head_stride=(int64_t) kHeadDim*lkr/16;
    p.sfq_batch_stride=(int64_t) head_group*lqr*kHeadDim/16; p.sfk_batch_stride=(int64_t) head_group*lkr*kHeadDim/16; p.sfv_batch_stride=(int64_t) head_group*kHeadDim*lkr/16;
    // delta_s is [head][q_block][lkr]; the kernel's LayoutDS derives that from shape alone
    // (seqlen_s picks the q-block count, seqlen_k the row length), so seqlen_s must be the
    // PADDED query length. ds_row_stride is informational -- the TMA never reads it.
    p.ds_row_stride=lkr; p.ds_head_stride=(int64_t) qblocks*lkr; p.ds_batch_stride=(int64_t) head_group*qblocks*lkr; p.o_row_stride=kHeadDim; p.o_head_stride=(int64_t) lqr*kHeadDim; p.o_batch_stride=(int64_t) head_group*lqr*kHeadDim;
    p.scale_softmax=scale; p.scale_softmax_log2=scale * M_LOG2E; p.is_causal=false; p.per_block_mean=true; p.seqlen_s=lqr; p.is_bf16=false; p.is_seqlens_k_cumulative=true;
    if (timing) CUDA_CHECK(cudaEventRecord(timing_mha, stream));
    if (delta_f16) run_mha_fwd_<cutlass::nv_float4_t<cutlass::float_e2m1_t>, kHeadDim, cutlass::half_t, cutlass::half_t>(p, stream);
    else run_mha_fwd_<cutlass::nv_float4_t<cutlass::float_e2m1_t>, kHeadDim, cutlass::half_t>(p, stream);
        sa3_o_to_dst<<<(elems_q + 255) / 256, 256, 0, stream>>>(scratch.get(), o, lq, lqr, head_group, head_start, total_h, o_f32);
        if (lse_dst != nullptr) {
            // 8 warps per block, one query row each.
            const dim3 grid((unsigned) ((lq + 7) / 8), (unsigned) head_group);
            if (q->type == GGML_TYPE_F32) {
                sa3_lse_to_dst<float><<<grid, 256, 0, stream>>>(lse.get(), q_f32 + q_head_offset, k_mean.get(), lse_dst, lq, lqr, head_start, scale);
            } else {
                sa3_lse_to_dst<half><<<grid, 256, 0, stream>>>(lse.get(), (const half *) q->data + q_head_offset, k_mean.get(), lse_dst, lq, lqr, head_start, scale);
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }
    if (timing) {
        CUDA_CHECK(cudaEventRecord(timing_end, stream));
        CUDA_CHECK(cudaEventSynchronize(timing_end));
        float total_ms = 0.0f, prep_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, timing_begin, timing_end));
        CUDA_CHECK(cudaEventElapsedTime(&prep_ms, timing_begin, timing_mha));
        fprintf(stderr, "[sa3-timing] Lq=%d Lkv=%d total=%.3fms prep=%.3fms mha+out=%.3fms\n", lq, lk, total_ms, prep_ms, total_ms - prep_ms);
    }
    // Separate counters: the inference path's "[sa3] dispatch" lines are how a render is
    // confirmed to be on SA3 at all, so LSE calls must not consume that budget.
    static int calls = 0, lse_calls = 0;
    if ((lse_dst != nullptr ? lse_calls++ : calls++) < 4) {
        fprintf(stderr, "[sa3%s] dispatch B=1 H=%d Hkv=%d group=%d Lq=%d Lkv=%d D=128 Q=%s delta=%s mse_scale_radius=%d\n",
                lse_dst != nullptr ? "-lse" : "", total_h, total_h / gqa_ratio, head_group, lq, lk,
                q->type == GGML_TYPE_F32 ? "f32" : "f16", delta_f16 ? "f16" : "f32",
                mse_scale_radius);
    }
    if (lse_dst != nullptr) {
        // Same brute-force F32 oracle the cuDNN stats were validated with. FP4 QK means the
        // residual here is quantization-sized, not 1e-3, so the tolerance is looser -- but
        // every failure mode worth worrying about (a missing or doubled 2^C offset is
        // ~7.9 nats, a log2 reading is ~30% of |LSE|) is still far outside it.
        ggml_cuda_attn_lse_selfcheck("sa3", q, k, lse_dst, total_h, lq, lk, kHeadDim, scale,
                                     /*tol=*/0.5f, stream);
    }
}

bool ggml_cuda_flash_attn_ext_sa3(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    float scale = 0.0f, max_bias = 0.0f, softcap = 0.0f;
    memcpy(&scale, dst->op_params, sizeof(scale));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(max_bias));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(softcap));
    if (!sa3_contract_ok(cc, q, k, v, dst->src[3], dst->type, ggml_is_contiguous(dst),
                         max_bias, softcap, /*allow_rect=*/false)) {
        static int rejected = 0;
        if (rejected++ < 3) {
            fprintf(stderr, "[sa3] reject cc=%d q=[%lld,%lld,%lld,%lld] qtype=%d kvtype=%d/%d dst=%d mask=%d contiguous=%d/%d/%d/%d bias=%g softcap=%g\n",
                    cc, (long long) q->ne[0], (long long) q->ne[1], (long long) q->ne[2], (long long) q->ne[3],
                    q->type, k->type, v->type, dst->type, dst->src[3] != nullptr,
                    ggml_is_contiguous(q), ggml_is_contiguous(k), ggml_is_contiguous(v), ggml_is_contiguous(dst), max_bias, softcap);
        }
        return false;
    }
    sa3_run(ctx, q, k, v, dst->data, dst->type == GGML_TYPE_F32, /*lse_dst=*/nullptr, scale);
    return true;
}

bool ggml_cuda_flash_attn_ext_lse_sa3(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // Opt-in on top of GGML_LTX_SA3, so simply turning SA3 on -- which is what production
    // does -- never moves the LSE op off cuDNN.
    // Read GGML_LTX_SA3 EVERY call, never cache it. The engine rewrites it per sampling
    // step when GGML_LTX_SA3_POLICY is set (stable-diffusion.cpp ~3020; prod uses "first",
    // i.e. cuDNN takes step 1 and SA3 the rest), which is why the main dispatcher also
    // getenvs per forward. Caching on first call latched the step-1 value of "0" and
    // disabled this path for entire renders with no reject line and no error -- it simply
    // never ran. GGML_LTX_SA3_LSE is our own gate and is safe to cache.
    static const int lse_gate = [] {
        const char * e = getenv("GGML_LTX_SA3_LSE");
        return e && atoi(e) ? 1 : 0;
    }();
    if (!lse_gate) return false;
    const char * sa3_now = getenv("GGML_LTX_SA3");
    if (!sa3_now || !atoi(sa3_now)) return false;

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    float scale = 0.0f, max_bias = 0.0f, softcap = 0.0f;
    memcpy(&scale, dst->op_params, sizeof(scale));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(max_bias));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(softcap));
    // dst is the flat F32 pack from ggml_flash_attn_ext_lse: O ([D,H,Lq,N] BSHD) first,
    // then the LSE ([Lq,H,N], query innermost).
    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || dst->src[4] != nullptr) return false;
    if (!sa3_contract_ok(cc, q, k, v, dst->src[3], GGML_TYPE_F32, /*o_contiguous=*/true,
                         max_bias, softcap, /*allow_rect=*/true)) {
        static int rejected = 0;
        if (rejected++ < 3) {
            fprintf(stderr, "[sa3-lse] reject cc=%d q=[%lld,%lld,%lld,%lld] k=[%lld,%lld,%lld,%lld] qtype=%d kvtype=%d/%d\n",
                    cc, (long long) q->ne[0], (long long) q->ne[1], (long long) q->ne[2], (long long) q->ne[3],
                    (long long) k->ne[0], (long long) k->ne[1], (long long) k->ne[2], (long long) k->ne[3],
                    q->type, k->type, v->type);
        }
        return false;
    }
    const size_t n_o = (size_t) v->ne[0] * q->ne[2] * q->ne[1] * q->ne[3];
    float * lse_dst  = (float *) dst->data + n_o;
    sa3_run(ctx, q, k, v, dst->data, /*o_f32=*/true, lse_dst, scale);
    return true;
}
