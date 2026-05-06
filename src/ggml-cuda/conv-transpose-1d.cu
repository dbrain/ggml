// CUDA conv_transpose_1d:
//   - F32 weights → naive triple-loop kernel (legacy path).
//   - F16 weights → smem-tiled wmma (tensor-core) kernel modeled on conv-1d-direct.cu.
//
// The qwen3-tts WavTokenizer vocoder stores its 5 conv_transpose_1d weight tensors
// (2 upsample + 4 dec block conv_t) as F16, so without an F16-capable kernel ggml
// routes the whole conv_transpose_1d to CPU and pays a per-call PCIe round-trip.
// The F16 wmma kernel keeps the conv on GPU.

#include "conv-transpose-1d.cuh"
#include <mma.h>

using namespace nvcuda;

// === Legacy F32 path =========================================================

static  __global__ void conv_transpose_1d_kernel(
        const int s0, const int p0, const int d0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const float * src0, const float * src1,  float * dst) {
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_index >= output_size) {
        return;
    }

    int out_index = global_index / dst_ne0;

    float accumulator = 0;

    for (int c = 0; c < src0_ne2; c++) {
        int idx = global_index % dst_ne0;

        int kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        int input_offset = src1_ne0 * c;

        for (int i = 0; i < src1_ne0; i++) {
            if (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                continue;
            }
            int weight_idx = idx - i*s0;

            float kernel_weight = src0[kernel_offset + weight_idx];
            float input_value =  src1[input_offset+i];

            accumulator += kernel_weight * input_value;
        }
    }
    dst[global_index] = accumulator;
    GGML_UNUSED_VARS(p0, d0, src0_ne3, src1_ne3, dst_ne3, src1_ne1, dst_ne1, src1_ne2, dst_ne2);
}

static void conv_transpose_1d_f32_f32_cuda(
        const int s0, const int p0, const int d0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const float * src0, const float * src1,  float * dst,
        cudaStream_t stream) {

    const int num_blocks = (output_size + CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE;
    conv_transpose_1d_kernel<<<num_blocks,CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 0, stream>>>(
        s0,p0,d0,output_size,
        src0_ne0, src0_ne1,  src0_ne2, src0_ne3,
        src1_ne0, src1_ne1,  src1_ne2, src1_ne3,
        dst_ne0,  dst_ne1,   dst_ne2,  dst_ne3,
        src0,src1, dst);
}

// === F16 wmma path ===========================================================
//
// Layout (matches ggml_conv_transpose_1d in ggml.c):
//     w   [kernel, out_ch, in_ch]   F16, contiguous.
//                                   Flat (innermost first): w_data[ic*(out_ch*kernel) + oc*kernel + kk]
//     x   [in_seq, in_ch, batch]    F32 or F16, contiguous.
//     dst [out_seq, out_ch, batch]  F32 or F16.
//
// X and Y dtypes are independent; the wmma c_frag accumulator stays F32 so
// only the load/store endpoints vary. Used to keep the cascade in F16 between
// transpose blocks (halves every dec_block intermediate).
//
// GEMM mapping per output tile [oc_base : oc_base+BM, t_base : t_base+BN):
//     C[oc, t] = sum over (ic, kk) of A[oc, ic*kernel+kk] * B[ic*kernel+kk, t]
//   A[oc, k_idx] = w[ic, oc, kk]                                       where (ic, kk) = (k_idx/kernel, k_idx%kernel)
//   B[k_idx, t]  = x[in_t, ic]  iff (t-kk) >= 0 && (t-kk) % s0 == 0 && in_t = (t-kk)/s0 in [0, in_seq); else 0.
//
// This is structurally identical to conv-1d-direct.cu's mma kernel, only the
// B-tile index formula inverts (the input/output time roles are swapped).
//
// Note the K-dimension is `in_ch * kernel` and only ~kernel/s0 of the kk
// positions per output column actually contribute — the rest become zeros in
// the B tile. Tensor cores process zeros at full rate, so this is no slower
// than a perfectly-packed kernel; it just leaves some throughput on the table.

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define BM 16
#define BN 64
#define BK 16
#define WARPS_PER_BLOCK 4

template <typename Tx>
__device__ __forceinline__ float ct1d_load_x(const Tx * x, int64_t idx);

template <>
__device__ __forceinline__ float ct1d_load_x<float>(const float * x, int64_t idx) {
    return x[idx];
}

template <>
__device__ __forceinline__ float ct1d_load_x<__half>(const __half * x, int64_t idx) {
    return __half2float(x[idx]);
}

template <typename Ty>
__device__ __forceinline__ void ct1d_store_y(Ty * y, int64_t idx, float v);

template <>
__device__ __forceinline__ void ct1d_store_y<float>(float * y, int64_t idx, float v) {
    y[idx] = v;
}

template <>
__device__ __forceinline__ void ct1d_store_y<__half>(__half * y, int64_t idx, float v) {
    y[idx] = __float2half(v);
}

template <typename Tx, typename Ty>
__global__ void conv_transpose_1d_mma_kernel(
    const __half * __restrict__ w,        // [in_ch, out_ch, kernel] flat
    const Tx     * __restrict__ x,        // [batch, in_ch, in_seq]
    Ty           * __restrict__ y,        // [batch, out_ch, out_seq]
    int in_seq, int out_seq,
    int in_ch,  int out_ch,
    int kernel, int s0,
    int64_t x_stride_b, int64_t x_stride_c,
    int64_t y_stride_b, int64_t y_stride_c
) {
    const int oc_tile = blockIdx.y;
    const int t_tile  = blockIdx.x;
    const int batch_i = blockIdx.z;
    const int warp_id = threadIdx.x / 32;
    const int oc_base = oc_tile * BM;
    const int t_base  = t_tile  * BN;
    const int K       = in_ch * kernel;

    __shared__ __half a_smem[BM * BK];
    __shared__ __half b_smem[BN * BK];
    __shared__ float  c_smem[BM * BN];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k_outer = 0; k_outer < K; k_outer += BK) {
        // Load A tile [BM x BK] row-major.
        for (int i = threadIdx.x; i < BM * BK; i += WARPS_PER_BLOCK * 32) {
            const int m       = i / BK;
            const int k_local = i % BK;
            const int oc      = oc_base + m;
            const int k_idx   = k_outer + k_local;
            __half v = __float2half(0.0f);
            if (oc < out_ch && k_idx < K) {
                const int ic = k_idx / kernel;
                const int kk = k_idx % kernel;
                v = w[(int64_t)ic * out_ch * kernel + (int64_t)oc * kernel + kk];
            }
            a_smem[i] = v;
        }

        // Load B tile (input via on-the-fly inverse index) into shared.
        // b_smem[n_local * BK + k_local] — col-major-of-the-tile so wmma's
        // matrix_b col_major load with ldb=BK reads the right element layout.
        for (int i = threadIdx.x; i < BN * BK; i += WARPS_PER_BLOCK * 32) {
            const int n_local = i / BK;
            const int k_local = i % BK;
            const int t       = t_base + n_local;
            const int k_idx   = k_outer + k_local;
            __half v = __float2half(0.0f);
            if (t < out_seq && k_idx < K) {
                const int ic = k_idx / kernel;
                const int kk = k_idx % kernel;
                const int t_minus_k = t - kk;
                if (t_minus_k >= 0 && (t_minus_k % s0) == 0) {
                    const int in_t = t_minus_k / s0;
                    if (in_t < in_seq) {
                        const float xv = ct1d_load_x<Tx>(x, batch_i * x_stride_b + ic * x_stride_c + in_t);
                        v = __float2half(xv);
                    }
                }
            }
            b_smem[i] = v;
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;

        wmma::load_matrix_sync(a_frag, a_smem, BK);
        wmma::load_matrix_sync(b_frag, b_smem + warp_id * WMMA_N * BK, BK);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(c_smem + warp_id * WMMA_N, c_frag, BN, wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < BM * BN; i += WARPS_PER_BLOCK * 32) {
        const int m  = i / BN;
        const int n  = i % BN;
        const int oc = oc_base + m;
        const int t  = t_base + n;
        if (oc < out_ch && t < out_seq) {
            ct1d_store_y<Ty>(y, batch_i * y_stride_b + (int64_t)oc * y_stride_c + t,
                             c_smem[m * BN + n]);
        }
    }
}

// === Dispatch ================================================================

void ggml_cuda_op_conv_transpose_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int32_t * opts = (const int32_t *) dst->op_params;
    const int s0 = opts[0];
    const int p0 = 0;//opts[3];
    const int d0 = 1;//opts[4];

    cudaStream_t stream = ctx.stream();

    if (src0->type == GGML_TYPE_F16) {
        const int kernel  = (int) src0->ne[0];
        const int out_ch  = (int) src0->ne[1];
        const int in_ch   = (int) src0->ne[2];
        const int in_seq  = (int) src1->ne[0];
        const int batch   = (int) src1->ne[2];
        const int out_seq = (int) dst->ne[0];

        const size_t x_elem = (src1->type == GGML_TYPE_F32) ? sizeof(float) : sizeof(__half);
        const size_t y_elem = ( dst->type == GGML_TYPE_F32) ? sizeof(float) : sizeof(__half);
        const int64_t x_stride_b = src1->nb[2] / x_elem;
        const int64_t x_stride_c = src1->nb[1] / x_elem;
        const int64_t y_stride_b =  dst->nb[2] / y_elem;
        const int64_t y_stride_c =  dst->nb[1] / y_elem;

        const __half * w_d = (const __half *) src0->data;

        const int blocks_n = (out_seq + BN - 1) / BN;
        const int blocks_m = (out_ch  + BM - 1) / BM;
        dim3 grid(blocks_n, blocks_m, batch);
        dim3 block(WARPS_PER_BLOCK * 32);

#define DISPATCH_CT1D(TX, TY)                                                             \
        conv_transpose_1d_mma_kernel<TX, TY><<<grid, block, 0, stream>>>(                 \
            w_d, (const TX *) src1->data, (TY *) dst->data,                               \
            in_seq, out_seq, in_ch, out_ch, kernel, s0,                                   \
            x_stride_b, x_stride_c, y_stride_b, y_stride_c)

        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            DISPATCH_CT1D(float, float);
        } else if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            DISPATCH_CT1D(float, __half);
        } else if (src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            DISPATCH_CT1D(__half, float);
        } else {
            DISPATCH_CT1D(__half, __half);
        }
#undef DISPATCH_CT1D

        GGML_UNUSED(p0);
        GGML_UNUSED(d0);
        return;
    }

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float *)       dst->data;

    const int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_cuda(s0, p0, d0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        dst->ne[0],  dst->ne[1],  dst->ne[2],  dst->ne[3],
        src0_d, src1_d, dst_d, stream);
}
