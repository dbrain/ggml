#include "rope-pe.cuh"

// longcat-avatar fused interleaved (GPT-J) RoPE from precomputed pe (cos/sin).
//
//   a (src0): pre-rope q/k, ggml ne = [d_head, n_head, L, N]   (post-norm)
//   pe(src1): rotation,     ggml ne = [2, 2, d_head/2, L]
//             laid out per pair j as [[cos_j, sin_j], [-sin_j, cos_j]]
//             (Rope::rope(): result[4j..4j+3] = cos, -sin, sin, cos), so
//             cos_j = pe[ne0=0, ne1=0, j, t], sin_j = pe[ne0=0, ne1=1, j, t].
//   dst:      ggml ne = [d_head, L, n_head*N]  (== apply_rope rope_interleaved=true)
//
// Per pair j (out dims 2j, 2j+1), token t, head-row h in [0, n_head*N):
//   out[2j]   = x[2j]*cos_j - x[2j+1]*sin_j
//   out[2j+1] = x[2j+1]*cos_j + x[2j]*sin_j
//
// One thread computes one output PAIR (2 dst elements). Grid covers
// (d_head/2) * L * (n_head*N) pairs.  No intermediate buffers.

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

    // input pair (x_even, x_odd) at dims (2j, 2j+1) of `a`
    const int64_t a_base = (2 * j) * a_s0 + head * a_s1 + t * a_s2 + n * a_s3;
    const float x_e = a[a_base];
    const float x_o = a[a_base + a_s0];

    // pe is contiguous [2,2,half,L]: cos=pe[0,0,j,t]=4j+2*d_head*t, sin=pe[0,1,j,t]=2+4j+2*d_head*t
    const int64_t pe_base = 4 * j + (int64_t)2 * d_head * t;
    const float c = pe[pe_base];
    const float s = pe[pe_base + 2];

    // dst contiguous [d_head, L, HN]: element (i0, t, h) = i0 + d_head*t + d_head*L*h
    const int64_t d_base = (2 * j) + d_head * t + d_head * L * h;
    dst[d_base]     = x_e * c - x_o * s;
    dst[d_base + 1] = x_o * c + x_e * s;
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

    rope_pe_f32<<<grid, block, 0, stream>>>(
        a_d, pe_d, d_d, d_head, n_head, L, N, a_s0, a_s1, a_s2, a_s3);
}
