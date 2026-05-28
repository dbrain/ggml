#include "scale_cast.cuh"

#define LC_SCALE_CAST_BLOCK_SIZE 256
#define LC_SCALE_CAST_MAX_GRID   0x7FFFFFFF

static __global__ void scale_cast_f32_to_f16_kernel(const float * __restrict__ x,
                                                     half * __restrict__       dst,
                                                     const float               scale,
                                                     const float               bias,
                                                     const int64_t             nelements) {
    int64_t tid    = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    int64_t stride = (int64_t) blockDim.x * (int64_t) gridDim.x;

    for (int64_t i = tid; i < nelements; i += stride) {
        // Bit-exact match for the unfused SCALE -> CAST(F16) chain:
        //   ggml_cuda_op_scale: dst[i] = scale * x[i] + bias  (single F32 mul+add, one rounding)
        //   ggml_cuda_op_cpy F32→F16: dst_f16[i] = (half) src_f32[i]  (F32→F16 cast)
        // Composed:   __float2half(scale * x[i] + bias).
        // No FMA contraction concern: when bias==0 (the kv_scale case), it's just
        //   __float2half(scale * x[i]) — one F32 mul + one F32→F16 cast, identical
        //   to the unfused chain.
        const float v = scale * x[i] + bias;
        dst[i]        = __float2half(v);
    }
}

void ggml_cuda_op_scale_cast_f16(ggml_backend_cuda_context & ctx,
                                 const ggml_tensor *         src_f32,
                                 ggml_tensor *               dst_f16,
                                 float                       scale,
                                 float                       bias) {
    GGML_ASSERT(src_f32->type == GGML_TYPE_F32);
    GGML_ASSERT(dst_f16->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(src_f32));
    GGML_ASSERT(ggml_is_contiguous(dst_f16));
    GGML_ASSERT(ggml_nelements(src_f32) == ggml_nelements(dst_f16));

    const int64_t nelements = ggml_nelements(src_f32);
    const float * x_d       = (const float *) src_f32->data;
    half *        dst_d     = (half *)        dst_f16->data;
    cudaStream_t  stream    = ctx.stream();

    const int64_t num_blocks_total = (nelements + LC_SCALE_CAST_BLOCK_SIZE - 1) / LC_SCALE_CAST_BLOCK_SIZE;
    const int     num_blocks       = (int) MIN((int64_t) LC_SCALE_CAST_MAX_GRID, num_blocks_total);
    scale_cast_f32_to_f16_kernel<<<num_blocks, LC_SCALE_CAST_BLOCK_SIZE, 0, stream>>>(
        x_d, dst_d, scale, bias, nelements);
}
