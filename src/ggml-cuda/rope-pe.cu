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

template <bool INTERLEAVED>
static __global__ void rope_pe_f32(
        const float * __restrict__ a,
        const float * __restrict__ pe,
        float       * __restrict__ dst,
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
    const float x_e = a[a_base];
    const float x_o = a[a_base + (d_o - d_e) * a_s0];

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
    dst[d_base]            = __fsub_rn(pe_e, po_e);   // x_e*cos - x_o*sin
    const float pe_o = __fmul_rn(x_e, s);  // x_e * sin
    const float po_o = __fmul_rn(x_o, c);  // x_o * cos
    dst[d_base + (d_o - d_e)] = __fadd_rn(pe_o, po_o);  // x_e*sin + x_o*cos
}

void ggml_cuda_op_rope_pe(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a  = dst->src[0];
    const ggml_tensor * pe = dst->src[1];

    GGML_ASSERT(a->type  == GGML_TYPE_F32);
    GGML_ASSERT(pe->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t d_head = a->ne[0];
    const int64_t n_head = a->ne[1];
    const int64_t L      = a->ne[2];
    const int64_t N      = a->ne[3];

    // strides in elements
    const int64_t a_s0 = a->nb[0] / sizeof(float);
    const int64_t a_s1 = a->nb[1] / sizeof(float);
    const int64_t a_s2 = a->nb[2] / sizeof(float);
    const int64_t a_s3 = a->nb[3] / sizeof(float);

    GGML_ASSERT(ggml_is_contiguous(pe));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const float * a_d  = (const float *) a->data;
    const float * pe_d = (const float *) pe->data;
    float       * d_d  = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t half  = d_head / 2;
    const int64_t npair = half * L * n_head * N;
    const int     block = 256;
    const int64_t grid  = (npair + block - 1) / block;

    // op_params[0]: 1 = interleaved (GPT-J), 0 = non-interleaved (NeoX, LTX video).
    const int interleaved = ((const int32_t *) dst->op_params)[0];
    if (interleaved) {
        rope_pe_f32<true><<<grid, block, 0, stream>>>(
            a_d, pe_d, d_d, d_head, n_head, L, N, a_s0, a_s1, a_s2, a_s3);
    } else {
        rope_pe_f32<false><<<grid, block, 0, stream>>>(
            a_d, pe_d, d_d, d_head, n_head, L, N, a_s0, a_s1, a_s2, a_s3);
    }
}
