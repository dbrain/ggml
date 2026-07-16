#include "rope-pe.cuh"

// longcat-avatar fused RoPE from precomputed pe (cos/sin), interleaved (GPT-J)
// or non-interleaved (GPT-NeoX) per the `INTERLEAVED` template flag.
//
//   a (src0): pre-rope q/k, ggml ne = [d_head, n_head, L, N]   (post-norm)
//   pe(src1): rotation,     ggml ne = [2, 2, d_head/2, L]
//             laid out per pair j as [[cos_j, sin_j], [-sin_j, cos_j]]
//             (Rope::rope(): result[4j..4j+3] = cos, -sin, sin, cos), so
//             cos_j = pe[ne0=0, ne1=0, j, t], sin_j = pe[ne0=0, ne1=1, j, t].
//   dst:      ggml ne = [d_head, L, n_head*N]  (== apply_rope output)
//
// INTERLEAVED=true (GPT-J): the rotated pair is adjacent dims (2j, 2j+1):
//   out[2j]   = x[2j]*cos_j   - x[2j+1]*sin_j
//   out[2j+1] = x[2j+1]*cos_j + x[2j]*sin_j
// INTERLEAVED=false (NeoX): the pair is the first/second half of head_dim
//   (x[a], x[a+half]) and the output is written at the same split offsets:
//   out[a]      = x[a]*cos_a      - x[a+half]*sin_a
//   out[a+half] = x[a]*sin_a      + x[a+half]*cos_a
// The pe indexing is IDENTICAL in both cases (cos=pe[0,0,a,t], sin=pe[0,1,a,t]);
// only the x-read / dst-write element offsets differ. The LTX video self-attn
// folds the per-head pe into n_head=1 with L = video_tokens*num_heads, so this
// kernel sees that as plain (j,t) indexing too.
//
// One thread computes one output PAIR (2 dst elements). Grid covers
// (d_head/2) * L * (n_head*N) pairs.  No intermediate buffers.

// T = element storage type of src0 `a` and dst (float or half). pe is ALWAYS
// float. The rotation math runs in F32 regardless of T (load T->float, compute,
// store float->T), so the F16 path is bit-identical to F32-rope-then-round-to-F16
// (which is exactly what the F16-residual stream did downstream via ggml_cast /
// cuDNN's internal q cast). Keeping q/k F16 through RoPE drops the two 1237 MB
// F32 rope tensors from the DiT compute buffer (VACE 1280x704x65f: ~-1.85 GB).
static __device__ __forceinline__ float rp_ld(const float * p, int64_t i) { return p[i]; }
static __device__ __forceinline__ float rp_ld(const __half * p, int64_t i) { return __half2float(p[i]); }
static __device__ __forceinline__ void  rp_st(float * p, int64_t i, float v) { p[i] = v; }
static __device__ __forceinline__ void  rp_st(__half * p, int64_t i, float v) { p[i] = __float2half_rn(v); }

template <bool INTERLEAVED, typename T>
static __global__ void rope_pe_f32(
        const T     * __restrict__ a,
        const float * __restrict__ pe,
        T           * __restrict__ dst,
        const int64_t d_head,
        const int64_t n_head,
        const int64_t L,
        const int64_t N,
        // strides (in elements) of src0 `a` along [d_head, n_head, L, N]
        const int64_t a_s0,
        const int64_t a_s1,
        const int64_t a_s2,
        const int64_t a_s3) {
    const int64_t half = d_head / 2;
    const int64_t HN   = n_head * N;
    const int64_t npair = half * L * HN;

    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npair) {
        return;
    }

    // decode idx -> (j, t, h)  with layout matching dst contiguous [d_head, L, HN]
    // we iterate pairs as fastest=j, then t, then h
    const int64_t j = idx % half;
    const int64_t t = (idx / half) % L;
    const int64_t h = idx / (half * L);

    // head-row h = head + n_head * n  (N==1 in the avatar path; general anyway)
    const int64_t head = h % n_head;
    const int64_t n    = h / n_head;

    // input pair: interleaved -> dims (2j, 2j+1); non-interleaved -> (j, j+half)
    const int64_t d_e = INTERLEAVED ? (2 * j) : j;          // even/first elem dim
    const int64_t d_o = INTERLEAVED ? (2 * j + 1) : (j + half);  // odd/second elem dim
    const int64_t a_base = d_e * a_s0 + head * a_s1 + t * a_s2 + n * a_s3;
    const float x_e = rp_ld(a, a_base);
    const float x_o = rp_ld(a, a_base + (d_o - d_e) * a_s0);

    // pe is contiguous [2,2,half,L]: cos=pe[0,0,j,t]=4j+2*d_head*t, sin=pe[0,1,j,t]=2+4j+2*d_head*t
    const int64_t pe_base = 4 * j + (int64_t)2 * d_head * t;
    const float c = pe[pe_base];
    const float s = pe[pe_base + 2];

    // dst contiguous [d_head, L, HN]: element (i0, t, h) = i0 + d_head*t + d_head*L*h
    const int64_t d_base = d_e + d_head * t + d_head * L * h;
    // IEEE-exact match to the apply_rope chain (separate ggml_mul + ggml_add, NO
    // fused multiply-add). nvcc contracts `a*b + c*d` to fmaf by default, which
    // skips the intermediate product rounding and makes the op output differ from
    // the chain at ~1e-5 — small in isolation, but it compounds over 48 blocks x
    // 8 DMD steps and drifts the (chaotic) denoise trajectory off the 99 dB gate.
    // __fmul_rn / __fadd_rn / __fsub_rn are un-contractable round-to-nearest
    // primitives, so each product is rounded to F32 before the add — bit-identical
    // to the chain's mul-then-add. The chain stores -sin and does cos*x_e + (-sin)*x_o
    // for the even lane (negation is exact, so == x_e*c - x_o*s) and sin*x_e + cos*x_o
    // for the odd lane.
    const float pe_e = __fmul_rn(x_e, c);  // x_e * cos
    const float po_e = __fmul_rn(x_o, s);  // x_o * sin
    rp_st(dst, d_base,             __fsub_rn(pe_e, po_e));   // x_e*cos - x_o*sin
    const float pe_o = __fmul_rn(x_e, s);  // x_e * sin
    const float po_o = __fmul_rn(x_o, c);  // x_o * cos
    rp_st(dst, d_base + (d_o - d_e), __fadd_rn(pe_o, po_o));  // x_e*sin + x_o*cos
}

template <bool INTERLEAVED, typename T>
static __global__ void rope_pe_compact_f32(
        const T * __restrict__ a, const float * __restrict__ basis,
        const int32_t * __restrict__ token_axis_index, T * __restrict__ dst,
        const int64_t d_head, const int64_t n_head, const int64_t L, const int64_t N,
        const int64_t full_half,
        const int64_t a_s0, const int64_t a_s1, const int64_t a_s2, const int64_t a_s3) {
    const int64_t half = d_head / 2;
    const int64_t npair = half * L * n_head * N;
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npair) return;
    const int64_t j = idx % half;
    const int64_t t = (idx / half) % L;
    const int64_t h = idx / (half * L);
    const int64_t head = h % n_head;
    const int64_t n = h / n_head;
    // LTX folds [head, token] into this op's L dimension. The original head is
    // the low component and the original token is the high component.
    const int64_t rope_heads = full_half / half;
    const int64_t rope_head = t % rope_heads;
    const int64_t token = t / rope_heads;
    const int64_t global_pair = rope_head * half + j;
    // The legacy frequency grid begins with `pad` zero-frequency pairs when
    // full_half is not divisible by the three video axes (4096-wide LTX has
    // pad=2). They are identity rotations; the T/H/W cycle begins after them.
    const int64_t pad = full_half % 3;
    float c = 1.f, s = 0.f;
    if (global_pair >= pad) {
        const int axis = (int) ((global_pair - pad) % 3);
        const int32_t entry = token_axis_index[axis + 3 * token];
        // basis has ggml shape [2, 2, full_half, entries], so each entry
        // occupies 4 * full_half F32 values (not 2 * full_half).
        const int64_t basis_base = 4 * global_pair + 4 * full_half * (int64_t) entry;
        c = basis[basis_base];
        s = basis[basis_base + 2];
    }
    const int64_t d_e = INTERLEAVED ? 2 * j : j;
    const int64_t d_o = INTERLEAVED ? 2 * j + 1 : j + half;
    const int64_t a_base = d_e * a_s0 + head * a_s1 + t * a_s2 + n * a_s3;
    const float x_e = rp_ld(a, a_base);
    const float x_o = rp_ld(a, a_base + (d_o - d_e) * a_s0);
    const int64_t d_base = d_e + d_head * t + d_head * L * h;
    const float pe_e = __fmul_rn(x_e, c), po_e = __fmul_rn(x_o, s);
    rp_st(dst, d_base, __fsub_rn(pe_e, po_e));
    const float pe_o = __fmul_rn(x_e, s), po_o = __fmul_rn(x_o, c);
    rp_st(dst, d_base + (d_o - d_e), __fadd_rn(pe_o, po_o));
}

void ggml_cuda_op_rope_pe(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a  = dst->src[0];
    const ggml_tensor * pe = dst->src[1];
    const ggml_tensor * compact_index = dst->src[2];

    // a/dst may be F32 (prod default) or F16 (the *_DIT_F16 residual stream keeps
    // q/k F16 through RoPE so the two full-size F32 rope tensors leave the compute
    // buffer). pe is always F32. dst type must match a.
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(pe->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == a->type);

    const int64_t d_head = a->ne[0];
    const int64_t n_head = a->ne[1];
    const int64_t L      = a->ne[2];
    const int64_t N      = a->ne[3];

    // strides in elements (element size == the a/dst element size)
    const size_t  esz  = ggml_type_size(a->type);
    const int64_t a_s0 = a->nb[0] / esz;
    const int64_t a_s1 = a->nb[1] / esz;
    const int64_t a_s2 = a->nb[2] / esz;
    const int64_t a_s3 = a->nb[3] / esz;

    GGML_ASSERT(ggml_is_contiguous(pe));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const float * pe_d = (const float *) pe->data;

    cudaStream_t stream = ctx.stream();

    const int64_t half  = d_head / 2;
    const int64_t npair = half * L * n_head * N;
    const int     block = 256;
    const int64_t grid  = (npair + block - 1) / block;

    // op_params[0]: 1 = interleaved (GPT-J), 0 = non-interleaved (NeoX, LTX video).
    const int interleaved = ((const int32_t *) dst->op_params)[0];
    if (compact_index != nullptr) {
        GGML_ASSERT(compact_index->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(compact_index));
        const int64_t full_half = pe->ne[2];
        const int32_t * index_d = (const int32_t *) compact_index->data;
        if (a->type == GGML_TYPE_F16) {
            if (interleaved) rope_pe_compact_f32<true, __half><<<grid, block, 0, stream>>>((const __half *) a->data, pe_d, index_d, (__half *) dst->data, d_head, n_head, L, N, full_half, a_s0, a_s1, a_s2, a_s3);
            else             rope_pe_compact_f32<false, __half><<<grid, block, 0, stream>>>((const __half *) a->data, pe_d, index_d, (__half *) dst->data, d_head, n_head, L, N, full_half, a_s0, a_s1, a_s2, a_s3);
        } else {
            if (interleaved) rope_pe_compact_f32<true, float><<<grid, block, 0, stream>>>((const float *) a->data, pe_d, index_d, (float *) dst->data, d_head, n_head, L, N, full_half, a_s0, a_s1, a_s2, a_s3);
            else             rope_pe_compact_f32<false, float><<<grid, block, 0, stream>>>((const float *) a->data, pe_d, index_d, (float *) dst->data, d_head, n_head, L, N, full_half, a_s0, a_s1, a_s2, a_s3);
        }
        return;
    }
    if (a->type == GGML_TYPE_F16) {
        const __half * a_d = (const __half *) a->data;
        __half     * d_d = (__half *) dst->data;
        if (interleaved) {
            rope_pe_f32<true, __half><<<grid, block, 0, stream>>>(
                a_d, pe_d, d_d, d_head, n_head, L, N, a_s0, a_s1, a_s2, a_s3);
        } else {
            rope_pe_f32<false, __half><<<grid, block, 0, stream>>>(
                a_d, pe_d, d_d, d_head, n_head, L, N, a_s0, a_s1, a_s2, a_s3);
        }
    } else {
        const float * a_d = (const float *) a->data;
        float       * d_d = (float *) dst->data;
        if (interleaved) {
            rope_pe_f32<true, float><<<grid, block, 0, stream>>>(
                a_d, pe_d, d_d, d_head, n_head, L, N, a_s0, a_s1, a_s2, a_s3);
        } else {
            rope_pe_f32<false, float><<<grid, block, 0, stream>>>(
                a_d, pe_d, d_d, d_head, n_head, L, N, a_s0, a_s1, a_s2, a_s3);
        }
    }
}
