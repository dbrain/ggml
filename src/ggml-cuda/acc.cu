#include "acc.cuh"

template <typename T>
static __global__ void acc_kernel(
        const T * x, const T * y, T * dst, const int64_t ne,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s11, const int64_t s12, const int64_t s13, const int64_t offset) {
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= ne) {
        return;
    }

    int64_t src1_idx = i - offset;

    int64_t tmp = src1_idx;
    const int64_t i13 = tmp / s13;
    tmp -= i13 * s13;
    const int64_t i12 = tmp / s12;
    tmp -= i12 * s12;
    const int64_t i11 = tmp / s11;
    tmp -= i11 * s11;
    const int64_t i10 = tmp;

    if constexpr (std::is_same_v<T, float>) {
        float val = x[i];
        if (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
            val += y[((i13*ne12 + i12) * ne11 + i11) * ne10 + i10];
        }
        dst[i] = val;
    } else {
        // F16 path: do the add in F32 to avoid catastrophic cancellation.
        float val = __half2float(x[i]);
        if (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
            val += __half2float(y[((i13*ne12 + i12) * ne11 + i11) * ne10 + i10]);
        }
        dst[i] = __float2half(val);
    }
}

void ggml_cuda_op_acc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == src0->type);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);

    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(dst->nb[0] == ggml_element_size(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));

    const size_t elem = ggml_element_size(dst);
    const int64_t s1     = dst->op_params[0] / elem;
    const int64_t s2     = dst->op_params[1] / elem;
    const int64_t s3     = dst->op_params[2] / elem;
    const int64_t offset = dst->op_params[3] / elem;

    const int64_t n_elements = ggml_nelements(dst);
    const int num_blocks = (n_elements + CUDA_ACC_BLOCK_SIZE - 1) / CUDA_ACC_BLOCK_SIZE;

    if (src0->type == GGML_TYPE_F32) {
        acc_kernel<float><<<num_blocks, CUDA_ACC_BLOCK_SIZE, 0, stream>>>(
            (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
            n_elements, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], s1, s2, s3, offset);
    } else {
        acc_kernel<__half><<<num_blocks, CUDA_ACC_BLOCK_SIZE, 0, stream>>>(
            (const __half *) src0->data, (const __half *) src1->data, (__half *) dst->data,
            n_elements, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], s1, s2, s3, offset);
    }
}
