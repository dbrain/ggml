// Fused 1D convolution kernel using nvcuda::wmma (tensor cores).
//
// Bypasses ggml_conv_1d's im2col + cuBLAS gemm path. The im2col temp tensor
// (740 MB at the deepest WavTokenizer decoder layer) is replaced by on-the-fly
// index calculation inside the gemm's B-tile load. The cuBLAS gemm itself was
// confirmed (via a CUBLAS_GEMM_ALGO sweep across 0..23) to be already optimally
// tuned for our shapes — so the only remaining win is to do less total work,
// which means tensor-core mma + skipped im2col temp.
//
// Designed for the qwen3-tts WavTokenizer vocoder shapes:
//   - in_ch  ∈ {96, 192, 384, 768, 1024} (decoder block channels)
//   - out_ch ∈ {96, 192, 384, 768, 1024}
//   - kernel ∈ {1, 7}
//   - dilation ∈ {1, 3, 9}
//   - in_seq up to ~554k samples
//
// Layout (matches ggml_conv_1d_direct in ggml.c):
//     w   [kernel, in_ch, out_ch]  F16, contiguous (= [out_ch, in_ch * kernel] flat)
//     x   [in_seq, in_ch, batch]   F32 or F16, contiguous
//     dst [out_seq, out_ch, batch] F32 or F16
//
// X and Y can independently be F32 or F16 — the wmma c_frag accumulator stays
// F32 so precision is preserved through the gemm, only the load/store endpoints
// vary by dtype. Used to keep the vocoder cascade's intermediates F16 (halves
// every dec_block buffer; sched_cu drops ~50 % at chunk=30).
//
// Block tiling:
//     grid  = (ceil(out_seq / BN), ceil(out_ch / BM), batch)
//     block = WARPS_PER_BLOCK * 32 threads
// Each block computes a [BM x BN] output tile. Each warp handles a
// [WMMA_M x WMMA_N] sub-tile along N, all warps share the same M tile.

#include "conv-1d-direct.cuh"
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define BM 16
#define BN 64
#define BK 16
#define WARPS_PER_BLOCK 4

// Helper: load X element as F32 regardless of source dtype.
template <typename Tx>
__device__ __forceinline__ float load_x(const Tx * x, int64_t idx);

template <>
__device__ __forceinline__ float load_x<float>(const float * x, int64_t idx) {
    return x[idx];
}

template <>
__device__ __forceinline__ float load_x<__half>(const __half * x, int64_t idx) {
    return __half2float(x[idx]);
}

// Helper: store result as the target dtype.
template <typename Ty>
__device__ __forceinline__ void store_y(Ty * y, int64_t idx, float v);

template <>
__device__ __forceinline__ void store_y<float>(float * y, int64_t idx, float v) {
    y[idx] = v;
}

template <>
__device__ __forceinline__ void store_y<__half>(__half * y, int64_t idx, float v) {
    y[idx] = __float2half(v);
}

template <typename Tx, typename Ty>
__global__ void conv1d_mma_kernel(
    const __half * __restrict__ w,        // [out_ch, in_ch * kernel]
    const Tx     * __restrict__ x,        // [batch, in_ch, in_seq]
    Ty           * __restrict__ y,        // [batch, out_ch, out_seq]
    int in_seq, int out_seq,
    int in_ch,  int out_ch,
    int kernel, int s0, int p_left, int d0,
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

    // Shared memory:
    //   a_smem: [BM x BK] row-major (m * BK + k)
    //   b_smem: [BN x BK] col-major-of-the-tile (n * BK + k); each row is
    //           a contiguous BK chunk for one column n. wmma::matrix_b
    //           col_major load with ldb=BK reads exactly that.
    //   c_smem: [BM x BN] row-major (m * BN + n) — accumulator stage
    __shared__ __half a_smem[BM * BK];
    __shared__ __half b_smem[BN * BK];
    __shared__ float  c_smem[BM * BN];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k_outer = 0; k_outer < K; k_outer += BK) {
        // load A tile [BM x BK] row-major into shared
        for (int i = threadIdx.x; i < BM * BK; i += WARPS_PER_BLOCK * 32) {
            const int m       = i / BK;
            const int k_local = i % BK;
            const int oc      = oc_base + m;
            const int k_idx   = k_outer + k_local;
            __half v = __ushort_as_half(0);
            if (oc < out_ch && k_idx < K) {
                v = w[(int64_t)oc * K + k_idx];
            }
            a_smem[i] = v;
        }

        // load B tile (input via on-the-fly index), stored as
        // b_smem[n * BK + k] so wmma's col_major load with ldb=BK works.
        for (int i = threadIdx.x; i < BN * BK; i += WARPS_PER_BLOCK * 32) {
            const int n_local = i / BK;
            const int k_local = i % BK;
            const int t       = t_base + n_local;
            const int k_idx   = k_outer + k_local;
            __half v = __ushort_as_half(0);
            if (t < out_seq && k_idx < K) {
                const int ic   = k_idx / kernel;
                const int k    = k_idx % kernel;
                const int in_t = t * s0 + k * d0 - p_left;
                if (in_t >= 0 && in_t < in_seq) {
                    const float xv = load_x<Tx>(x, batch_i * x_stride_b + ic * x_stride_c + in_t);
                    v = __float2half(xv);
                }
            }
            b_smem[i] = v;
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;

        wmma::load_matrix_sync(a_frag, a_smem, BK);
        // warp_id-th 16-col chunk of B: starts at b_smem + (warp_id * WMMA_N) * BK
        wmma::load_matrix_sync(b_frag, b_smem + warp_id * WMMA_N * BK, BK);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // Each warp stores its [WMMA_M x WMMA_N] tile into c_smem (row-major
    // within the BM x BN block, so subsequent global write is straightforward).
    wmma::store_matrix_sync(c_smem + warp_id * WMMA_N, c_frag, BN, wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < BM * BN; i += WARPS_PER_BLOCK * 32) {
        const int m  = i / BN;
        const int n  = i % BN;
        const int oc = oc_base + m;
        const int t  = t_base + n;
        if (oc < out_ch && t < out_seq) {
            store_y<Ty>(y, batch_i * y_stride_b + (int64_t)oc * y_stride_c + t,
                        c_smem[m * BN + n]);
        }
    }
}

void ggml_cuda_op_conv_1d_direct(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0]; // weights [kernel, in_ch, out_ch]
    const ggml_tensor * b = dst->src[1]; // input   [in_seq, in_ch, batch]

    GGML_ASSERT(a->type == GGML_TYPE_F16);  // mma kernel currently F16-weights only
    GGML_ASSERT(b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int32_t * p = (const int32_t *) dst->op_params;
    const int s0      = p[0];
    const int p_left  = p[1];
    const int p_right = p[2];
    const int d0      = p[3];
    GGML_UNUSED(p_right);

    const int kernel = a->ne[0];
    const int in_ch  = a->ne[1];
    const int out_ch = a->ne[2];
    const int in_seq = b->ne[0];
    const int batch  = b->ne[2];
    const int out_seq = dst->ne[0];

    const size_t x_elem = (b->type == GGML_TYPE_F32) ? sizeof(float) : sizeof(__half);
    const size_t y_elem = (dst->type == GGML_TYPE_F32) ? sizeof(float) : sizeof(__half);
    const int64_t x_stride_b = b->nb[2] / x_elem;
    const int64_t x_stride_c = b->nb[1] / x_elem;
    const int64_t y_stride_b = dst->nb[2] / y_elem;
    const int64_t y_stride_c = dst->nb[1] / y_elem;

    cudaStream_t stream = ctx.stream();

    const int blocks_n = (out_seq + BN - 1) / BN;
    const int blocks_m = (out_ch  + BM - 1) / BM;
    dim3 grid(blocks_n, blocks_m, batch);
    dim3 block(WARPS_PER_BLOCK * 32);

    const __half * w_d = (const __half *) a->data;

#define DISPATCH_CONV1D(TX, TY)                                                            \
    conv1d_mma_kernel<TX, TY><<<grid, block, 0, stream>>>(                                  \
        w_d, (const TX *) b->data, (TY *) dst->data,                                        \
        in_seq, out_seq, in_ch, out_ch, kernel,                                             \
        s0, p_left, d0, x_stride_b, x_stride_c, y_stride_b, y_stride_c)

    if (b->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        DISPATCH_CONV1D(float, float);
    } else if (b->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        DISPATCH_CONV1D(float, __half);
    } else if (b->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        DISPATCH_CONV1D(__half, float);
    } else {
        DISPATCH_CONV1D(__half, __half);
    }
#undef DISPATCH_CONV1D
    CUDA_CHECK(cudaGetLastError());
}
