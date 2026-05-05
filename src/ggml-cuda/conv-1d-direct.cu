// Fused 1D convolution kernel for the qwen3-tts WavTokenizer vocoder.
//
// Direct implementation (no im2col temp). Each thread computes one output
// element via per-thread FMA accumulation. NOT yet using tensor cores —
// correctness-first; can swap to a wmma kernel later once shapes/layouts
// are validated end-to-end.
//
// Layout (matches ggml_conv_1d_direct in ggml.c):
//     w   [kernel, in_ch, out_ch]  F16, contiguous (= [out_ch, in_ch * kernel] flat)
//     x   [in_seq, in_ch, batch]   F32, contiguous
//     dst [out_seq, out_ch, batch] F32
// op_params: {s0, p_left, p_right, d0}.

#include "conv-1d-direct.cuh"

#define CONV1D_TILE_T 128

__global__ void conv1d_direct_naive_kernel(
    const __half * __restrict__ w,        // [out_ch, in_ch * kernel]
    const float  * __restrict__ x,        // [batch, in_ch, in_seq]
    float        * __restrict__ y,        // [batch, out_ch, out_seq]
    int in_seq, int out_seq,
    int in_ch,  int out_ch,
    int kernel, int s0, int p_left, int d0,
    int64_t x_stride_b, int64_t x_stride_c,
    int64_t y_stride_b, int64_t y_stride_c
) {
    const int oc      = blockIdx.y;
    const int batch_i = blockIdx.z;
    const int t       = blockIdx.x * CONV1D_TILE_T + threadIdx.x;

    // Cache the weight slice for this oc into shared memory once per block —
    // all 128 threads in the block share the same oc, so the (in_ch * kernel)
    // weights are reused 128 times. ALL threads must participate in the load
    // and reach the syncthreads barrier; otherwise threads that need a value
    // wait forever (or read uninitialized smem on newer arches).
    extern __shared__ __half wsm[];   // sized at launch: in_ch * kernel halves
    const int K = in_ch * kernel;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        wsm[i] = w[(int64_t)oc * K + i];
    }
    __syncthreads();

    if (oc >= out_ch || t >= out_seq) {
        return;
    }

    float acc = 0.0f;
    const float * xp_base = x + batch_i * x_stride_b;

    for (int ic = 0; ic < in_ch; ++ic) {
        const float * xp = xp_base + ic * x_stride_c;
        const __half * wp = wsm + ic * kernel;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            if (k >= kernel) break;
            const int in_t = t * s0 + k * d0 - p_left;
            if (in_t >= 0 && in_t < in_seq) {
                acc += __half2float(wp[k]) * xp[in_t];
            }
        }
    }

    y[batch_i * y_stride_b + (int64_t)oc * y_stride_c + t] = acc;
}

void ggml_cuda_op_conv_1d_direct(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0]; // weights [kernel, in_ch, out_ch]
    const ggml_tensor * b = dst->src[1]; // input   [in_seq, in_ch, batch]

    GGML_ASSERT(a->type == GGML_TYPE_F16);
    GGML_ASSERT(b->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int32_t * p = (const int32_t *) dst->op_params;
    const int s0      = p[0];
    const int p_left  = p[1];
    const int d0      = p[3];

    const int kernel = a->ne[0];
    const int in_ch  = a->ne[1];
    const int out_ch = a->ne[2];
    const int in_seq = b->ne[0];
    const int batch  = b->ne[2];
    const int out_seq = dst->ne[0];

    const int64_t x_stride_b = b->nb[2] / sizeof(float);
    const int64_t x_stride_c = b->nb[1] / sizeof(float);
    const int64_t y_stride_b = dst->nb[2] / sizeof(float);
    const int64_t y_stride_c = dst->nb[1] / sizeof(float);

    const __half * w_d = (const __half *) a->data;
    const float  * x_d = (const float *)  b->data;
    float        * y_d = (float *)        dst->data;

    cudaStream_t stream = ctx.stream();

    const int blocks_n = (out_seq + CONV1D_TILE_T - 1) / CONV1D_TILE_T;
    dim3 grid(blocks_n, out_ch, batch);
    dim3 block(CONV1D_TILE_T);

    const size_t shared_bytes = (size_t)in_ch * kernel * sizeof(__half);

    conv1d_direct_naive_kernel<<<grid, block, shared_bytes, stream>>>(
        w_d, x_d, y_d, in_seq, out_seq, in_ch, out_ch, kernel,
        s0, p_left, d0, x_stride_b, x_stride_c, y_stride_b, y_stride_c);
}
