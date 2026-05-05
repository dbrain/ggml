#include "snake.cuh"
#include "convert.cuh"

// ============================================================================
// Upstream auto-fusion path (ggml-org): the snake compute pattern
//   add( x, mul( sqr(sin(mul(a, x))), inv_b ) )
// is detected in ggml-cuda.cu and fused here. a and inv_b are already the
// final per-channel values (any exp() is baked in by preceding graph ops).
//   y = x + sin^2(a * x) * inv_b
// x: [T, C] (T contiguous), a: [1, C], inv_b: [1, C]. F32/F16/BF16, F32 compute.
// ============================================================================

template <typename T>
static __global__ void snake_kernel(
        const T     * __restrict__ x,
        const float * __restrict__ a,
        const float * __restrict__ inv_b,
        T           * __restrict__ dst,
        const int    total,
        const uint3  T_len_fastdiv) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int c = (int) fastdiv((uint32_t) idx, T_len_fastdiv);

    const float xi = ggml_cuda_cast<float>(x[idx]);
    const float s  = sinf(a[c] * xi);
    dst[idx] = ggml_cuda_cast<T>(xi + s * s * inv_b[c]);
}

// Internal launcher with explicit x/a/inv_b/dst tensors.
static void launch_snake(ggml_backend_cuda_context & ctx,
                         const ggml_tensor * x,
                         const ggml_tensor * a,
                         const ggml_tensor * inv_b,
                         ggml_tensor *       dst) {
    const float * a_d     = (const float *)a->data;
    const float * inv_b_d = (const float *)inv_b->data;

    const int   T = (int)x->ne[0];
    const int   C = (int)x->ne[1];
    const int   total = T * C;
    const uint3 T_len_fastdiv = init_fastdiv_values((uint64_t) T);

    const int block_size = 256;
    const int grid_size  = (total + block_size - 1) / block_size;

    cudaStream_t stream = ctx.stream();

    switch (x->type) {
        case GGML_TYPE_F32: {
            snake_kernel<<<grid_size, block_size, 0, stream>>>(
                (const float *)x->data, a_d, inv_b_d, (float *)dst->data, total, T_len_fastdiv);
        } break;
        case GGML_TYPE_F16: {
            snake_kernel<<<grid_size, block_size, 0, stream>>>(
                (const half *)x->data, a_d, inv_b_d, (half *)dst->data, total, T_len_fastdiv);
        } break;
        case GGML_TYPE_BF16: {
            snake_kernel<<<grid_size, block_size, 0, stream>>>(
                (const nv_bfloat16 *)x->data, a_d, inv_b_d, (nv_bfloat16 *)dst->data, total, T_len_fastdiv);
        } break;
        default:
            GGML_ABORT("snake: unsupported type");
    }
}

// Fusion entry: caller supplies x/a/inv_b explicitly from the matched
// mul -> sin -> sqr -> mul -> add pattern. The dst is the trailing add output.
void ggml_cuda_op_snake_fused(ggml_backend_cuda_context & ctx,
                              const ggml_tensor * x,
                              const ggml_tensor * a,
                              const ggml_tensor * inv_b,
                              ggml_tensor *       dst) {
    launch_snake(ctx, x, a, inv_b, dst);
}

// ============================================================================
// Fork explicit-op path (dbrain): GGML_OP_SNAKE created via ggml_snake(x, a, b).
// Unlike the fusion above, this takes the RAW alpha/beta parameters and applies
// exp() internally (the contract the qwen3-tts vocoder was built against):
//   y = x + (1 / exp(beta)) * sin^2(exp(alpha) * x)
// alpha/beta are per-channel along ne1. F32 only.
// ============================================================================

#define CUDA_SNAKE_BLOCK_SIZE 256

static __global__ void snake_f32(const float * x, const float * alpha, const float * beta, float * dst,
                                 const int64_t ne0, const int64_t ne1, const int64_t n) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const int64_t i1 = (i / ne0) % ne1;

    const float val = x[i];
    const float a   = alpha[i1];
    const float b   = beta[i1];

    const float ea = expf(a);
    const float eb = expf(b);

    // s² via plain multiply, not powf(s, 2) — CUDA's powf is exp(y*log(x))
    // and returns NaN when s < 0 (it doesn't fast-path integer y).
    const float s = sinf(ea * val);
    dst[i] = val + (1.0f / eb) * (s * s);
}

void ggml_cuda_op_snake(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1]; // alpha
    const ggml_tensor * src2 = dst->src[2]; // beta

    void * dst_d = dst->data;
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    const float * src2_d = (const float *)src2->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    const int64_t n   = ggml_nelements(dst);
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int64_t num_blocks = (n + CUDA_SNAKE_BLOCK_SIZE - 1) / CUDA_SNAKE_BLOCK_SIZE;

    snake_f32<<<num_blocks, CUDA_SNAKE_BLOCK_SIZE, 0, stream>>>(src0_d, src1_d, src2_d, (float *)dst_d, ne0, ne1, n);
}
