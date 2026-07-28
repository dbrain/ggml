#include "fattn-lse-selfcheck.cuh"

#include <algorithm>
#include <cuda_fp16.h>

// A backend documents its LSE loosely at best -- cuDNN says only "the log-sum-exp of the
// attention scores" (no base, no word on whether attn_scale is folded in), and SA3's row
// state carries a compile-time FP8xFP4 scale offset that cancels in O and therefore never
// had to be right. Every one of those is a silent factor in the segment merge, and a wrong
// LSE does not crash: it produces subtly mis-weighted video, the worst possible failure
// mode. So recompute a handful of rows by brute force in F32 and print the residual for
// BOTH candidate bases.
//
// One CTA per (head, query row): dot every key, track a running max and sum in the
// FlashAttention style, then reduce. O(rows * Lkv * D) -- microseconds for a few rows.
template <typename TQ>
static __global__ void attn_lse_bruteforce(
        const TQ * __restrict__ Q, const half * __restrict__ K,
        const float * __restrict__ lse, float * __restrict__ out_absdiff,
        int H, int Lq, int Lkv, int D, float scale, int row_stride) {
    const int h   = blockIdx.y;
    const int row = blockIdx.x * row_stride;
    if (row >= Lq) return;

    const TQ   * q = Q + ((long) h * Lq + row) * D;
    const half * k = K + (long) h * Lkv * D;

    float m = -INFINITY;
    float z = 0.0f;
    for (int j = threadIdx.x; j < Lkv; j += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += (float) q[d] * __half2float(k[(long) j * D + d]);
        }
        const float s = dot * scale;
        if (s > m) { z *= expf(m - s); m = s; }
        z += expf(s - m);
    }
    __shared__ float sm[256], sz[256];
    sm[threadIdx.x] = m; sz[threadIdx.x] = z;
    __syncthreads();
    for (int n = blockDim.x / 2; n > 0; n >>= 1) {
        if (threadIdx.x < n) {
            const float ma = sm[threadIdx.x], mb = sm[threadIdx.x + n];
            const float mx = fmaxf(ma, mb);
            sz[threadIdx.x] = sz[threadIdx.x] * expf(ma - mx) + sz[threadIdx.x + n] * expf(mb - mx);
            sm[threadIdx.x] = mx;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float ref = sm[0] + logf(sz[0]);
        const float got = lse[(long) h * Lq + row];
        // [0] = |got - ref| assuming natural log, [1] = |got*ln2 - ref| assuming log2,
        // [2] = |ref| for scale context. Max-reduced across rows via atomics on the bits.
        atomicMax((unsigned int *) out_absdiff + 0, __float_as_uint(fabsf(got - ref)));
        atomicMax((unsigned int *) out_absdiff + 1, __float_as_uint(fabsf(got * 0.6931472f - ref)));
        atomicMax((unsigned int *) out_absdiff + 2, __float_as_uint(fabsf(ref)));
    }
}

void ggml_cuda_attn_lse_selfcheck(const char * tag,
                                  const ggml_tensor * Q, const ggml_tensor * K, const float * lse,
                                  int64_t H, int64_t Lq, int64_t Lkv, int64_t D,
                                  float scale, float tol, cudaStream_t stream) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * e = getenv("GGML_CUDNN_ATTN_LSE_SELFCHECK");
        enabled = (e && atoi(e)) ? 1 : 0;
    }
    if (!enabled) return;
    static int done = 0;
    if (done >= 8) return;
    ++done;

    const int rows = 8;
    const int row_stride = (int) std::max<int64_t>(1, Lq / rows);
    float * d_res = nullptr;
    if (cudaMalloc((void **) &d_res, 3 * sizeof(float)) != cudaSuccess) return;
    cudaMemsetAsync(d_res, 0, 3 * sizeof(float), stream);
    const dim3 grid((unsigned) ((Lq + row_stride - 1) / row_stride), (unsigned) std::min<int64_t>(H, 4));
    if (Q->type == GGML_TYPE_F32) {
        attn_lse_bruteforce<float><<<grid, 256, 0, stream>>>(
            (const float *) Q->data, (const half *) K->data, lse, d_res,
            (int) H, (int) Lq, (int) Lkv, (int) D, scale, row_stride);
    } else {
        attn_lse_bruteforce<half><<<grid, 256, 0, stream>>>(
            (const half *) Q->data, (const half *) K->data, lse, d_res,
            (int) H, (int) Lq, (int) Lkv, (int) D, scale, row_stride);
    }
    if (cudaGetLastError() != cudaSuccess) { cudaFree(d_res); return; }
    float res[3] = {0, 0, 0};
    cudaMemcpyAsync(res, d_res, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_res);
    fprintf(stderr,
            "[%s-lse-selfcheck] Lq=%lld Lkv=%lld H=%lld D=%lld  max|LSE_ref|=%.4f  tol=%.3g  "
            "max|stats - ref| = %.3e (natural log)   %.3e (if stats were log2)  => %s\n",
            tag, (long long) Lq, (long long) Lkv, (long long) H, (long long) D,
            res[2], tol, res[0], res[1],
            res[0] < tol ? "NATURAL LOG, as assumed -- merge is safe"
                         : (res[1] < tol ? "*** LOG2 -- the merge needs stats * ln2 ***"
                                         : "*** NEITHER -- do NOT trust the merge ***"));
}
