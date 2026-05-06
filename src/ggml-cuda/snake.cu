#include "snake.cuh"
#include "convert.cuh"

#define CUDA_SNAKE_BLOCK_SIZE 256

// Snake activation: x + (1/exp(beta)) * sin(exp(alpha) * x)^2
// Vocoder cascade keeps activations as F16 to halve scheduler intermediates;
// math runs in F32 for numerical stability, only load/store endpoints vary.

template <typename Tx>
__device__ __forceinline__ float snake_load(const Tx * x, int64_t i);

template <>
__device__ __forceinline__ float snake_load<float>(const float * x, int64_t i) {
    return x[i];
}

template <>
__device__ __forceinline__ float snake_load<__half>(const __half * x, int64_t i) {
    return __half2float(x[i]);
}

template <typename Ty>
__device__ __forceinline__ void snake_store(Ty * y, int64_t i, float v);

template <>
__device__ __forceinline__ void snake_store<float>(float * y, int64_t i, float v) {
    y[i] = v;
}

template <>
__device__ __forceinline__ void snake_store<__half>(__half * y, int64_t i, float v) {
    y[i] = __float2half(v);
}

template <typename Tx, typename Ty>
static __global__ void snake_kernel(
    const Tx * x, const float * alpha, const float * beta, Ty * dst,
    const int64_t ne0, const int64_t ne1, const int64_t n
) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    // x shape: [ne0, ne1, ne2, ne3]; alpha/beta: [ne1] per-channel.
    const int64_t i1 = (i / ne0) % ne1;

    const float val = snake_load<Tx>(x, i);
    const float a = alpha[i1];
    const float b = beta[i1];

    const float ea = expf(a);
    const float eb = expf(b);

    // s² via plain multiply, not powf(s, 2) — CUDA's powf returns NaN for s<0
    // (it doesn't fast-path integer exponent).
    const float s = sinf(ea * val);
    snake_store<Ty>(dst, i, val + (1.0f / eb) * (s * s));
}

void ggml_cuda_op_snake(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1]; // alpha
    const ggml_tensor * src2 = dst->src[2]; // beta

    void * dst_d = dst->data;
    const float * src1_d = (const float *)src1->data;
    const float * src2_d = (const float *)src2->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_F32);

    const int64_t n = ggml_nelements(dst);
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t num_blocks = (n + CUDA_SNAKE_BLOCK_SIZE - 1) / CUDA_SNAKE_BLOCK_SIZE;

#define DISPATCH_SNAKE(TX, TY)                                                            \
    snake_kernel<TX, TY><<<num_blocks, CUDA_SNAKE_BLOCK_SIZE, 0, stream>>>(               \
        (const TX *) src0->data, src1_d, src2_d, (TY *) dst_d, ne0, ne1, n)

    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        DISPATCH_SNAKE(float, float);
    } else if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        DISPATCH_SNAKE(float, __half);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        DISPATCH_SNAKE(__half, float);
    } else {
        DISPATCH_SNAKE(__half, __half);
    }
#undef DISPATCH_SNAKE
}
