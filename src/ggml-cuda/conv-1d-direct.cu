// Fused 1D convolution kernel for the qwen3-tts.cpp WavTokenizer vocoder.
//
// v3: tiled shared-memory load of input + weights, parametrised on KERNEL
// and DILATION at compile time so X_TILE = TILE_T + (KERNEL-1)*DILATION is
// known. Avoids the v1 cache-thrashing on (in_ch_stride = in_seq*4)-byte
// channel reads by streaming the input through shared mem in IN_CH_STEP
// chunks. Naive v1 was 5x slower than the default im2col + cuBLAS gemm path
// because every per-thread output went through ~672 L1 misses.
//
// Layout (matches ggml_conv_1d_direct in ggml.c):
//     w   [kernel, in_ch, out_ch]  F16 or F32, contiguous
//     x   [in_seq, in_ch, batch]   F32,        contiguous
//     dst [out_seq, out_ch, batch] F32

#include "conv-1d-direct.cuh"

#define CONV1D_TILE_T     128
#define CONV1D_IN_CH_STEP  16

template<int KERNEL, int DILATION, typename WT>
static __global__ void conv1d_direct_smem_kernel(
    const WT    * __restrict__ w,        // [out_ch, in_ch, KERNEL]
    const float * __restrict__ x,        // [batch, in_ch, in_seq]
    float       * __restrict__ y,        // [batch, out_ch, out_seq]
    int in_seq, int out_seq,
    int in_ch, int out_ch,
    int s0, int p0,
    int64_t x_stride_b, int64_t x_stride_c,
    int64_t y_stride_b, int64_t y_stride_c
) {
    constexpr int X_TILE = CONV1D_TILE_T + (KERNEL - 1) * DILATION;

    const int oc      = blockIdx.y;
    const int batch_i = blockIdx.z;
    const int t_base  = blockIdx.x * CONV1D_TILE_T;
    const int tid     = threadIdx.x;
    const int t       = t_base + tid;
    const bool valid  = (t < out_seq) && (oc < out_ch);

    float acc = 0.0f;

    __shared__ float xsm[CONV1D_IN_CH_STEP * X_TILE];
    __shared__ WT    wsm[CONV1D_IN_CH_STEP * KERNEL];

    for (int ic_base = 0; ic_base < in_ch; ic_base += CONV1D_IN_CH_STEP) {
        const int ic_avail = (ic_base + CONV1D_IN_CH_STEP <= in_ch)
                                ? CONV1D_IN_CH_STEP : (in_ch - ic_base);

        // Cooperative input-tile load. Each cell xsm[ic_local * X_TILE + t_local]
        // holds x[ic_base + ic_local, t_base * s0 + t_local - p0] (or 0 if out
        // of range). For our typical case (s0=1) consecutive t_local map to
        // contiguous global positions, which gives coalesced loads when the
        // 128-thread block sweeps ic_local-major.
        for (int i = tid; i < CONV1D_IN_CH_STEP * X_TILE; i += CONV1D_TILE_T) {
            const int ic_local = i / X_TILE;
            const int t_local  = i % X_TILE;
            float v = 0.0f;
            if (ic_local < ic_avail) {
                const int ic = ic_base + ic_local;
                const int in_t = t_base * s0 + t_local - p0;
                if (in_t >= 0 && in_t < in_seq) {
                    v = x[batch_i * x_stride_b + ic * x_stride_c + in_t];
                }
            }
            xsm[i] = v;
        }

        // Cooperative weight-tile load: this oc's slice for the current ic
        // chunk. Tiny (<= 16*7=112 elements F16); falls into broadcast cache.
        for (int i = tid; i < CONV1D_IN_CH_STEP * KERNEL; i += CONV1D_TILE_T) {
            const int ic_local = i / KERNEL;
            const int k        = i % KERNEL;
            if (ic_local < ic_avail) {
                const int ic = ic_base + ic_local;
                wsm[i] = w[(int64_t)oc * in_ch * KERNEL + (int64_t)ic * KERNEL + k];
            }
        }

        __syncthreads();

        if (valid) {
            #pragma unroll
            for (int ic_local = 0; ic_local < CONV1D_IN_CH_STEP; ++ic_local) {
                if (ic_local >= ic_avail) break;
                #pragma unroll
                for (int k = 0; k < KERNEL; ++k) {
                    const int t_idx = tid * s0 + k * DILATION;
                    const float xv = xsm[ic_local * X_TILE + t_idx];
                    float wv;
                    if constexpr (std::is_same<WT, __half>::value) {
                        wv = __half2float(wsm[ic_local * KERNEL + k]);
                    } else {
                        wv = wsm[ic_local * KERNEL + k];
                    }
                    acc += xv * wv;
                }
            }
        }

        __syncthreads();
    }

    if (valid) {
        y[batch_i * y_stride_b + (int64_t)oc * y_stride_c + t] = acc;
    }
}

// Fallback: any (kernel, dilation) combo we don't have a templated specialisation
// for. Falls through to per-thread compute with __ldg-cached global reads —
// strictly slower than the smem version but gets the job done correctly.
template<typename WT>
static __global__ void conv1d_direct_dynk_kernel(
    const WT    * __restrict__ w,
    const float * __restrict__ x,
    float       * __restrict__ y,
    int kernel,
    int in_seq, int out_seq,
    int in_ch, int out_ch,
    int s0, int p0, int d0,
    int64_t x_stride_b, int64_t x_stride_c,
    int64_t y_stride_b, int64_t y_stride_c
) {
    const int oc      = blockIdx.y;
    const int batch_i = blockIdx.z;
    const int t       = blockIdx.x * CONV1D_TILE_T + threadIdx.x;

    if (t >= out_seq || oc >= out_ch) return;

    const WT * wp_oc = w + (int64_t)oc * in_ch * kernel;

    float sum = 0.0f;
    for (int ic = 0; ic < in_ch; ++ic) {
        const float * xp = x + batch_i * x_stride_b + ic * x_stride_c;
        const WT    * wp = wp_oc + ic * kernel;
        for (int k = 0; k < kernel; ++k) {
            const int in_t = t * s0 + k * d0 - p0;
            if (in_t >= 0 && in_t < in_seq) {
                if constexpr (std::is_same<WT, __half>::value) {
                    sum += __half2float(__ldg(wp + k)) * __ldg(xp + in_t);
                } else {
                    sum += __ldg(wp + k) * __ldg(xp + in_t);
                }
            }
        }
    }

    y[batch_i * y_stride_b + (int64_t)oc * y_stride_c + t] = sum;
}

// dispatch helper — picks the best specialisation for (kernel, d0).
#define DISPATCH_KERNEL_DILATION(KERNEL, DILATION, WT)                                 \
    conv1d_direct_smem_kernel<KERNEL, DILATION, WT><<<grid, block, 0, stream>>>(       \
        (const WT *) a->data, x_d, y_d, in_seq, out_seq, in_ch, out_ch, s0, p0,        \
        x_stride_b, x_stride_c, y_stride_b, y_stride_c)

template<typename WT>
static void launch_conv1d_direct(
    const ggml_tensor * a, const float * x_d, float * y_d,
    int in_seq, int out_seq, int in_ch, int out_ch,
    int s0, int p0, int d0,
    int64_t x_stride_b, int64_t x_stride_c, int64_t y_stride_b, int64_t y_stride_c,
    int kernel, int batch, cudaStream_t stream
) {
    const int blocks_t = (out_seq + CONV1D_TILE_T - 1) / CONV1D_TILE_T;
    dim3 grid(blocks_t, out_ch, batch);
    dim3 block(CONV1D_TILE_T);

    // The qwen3-tts WavTokenizer hits exactly these (kernel, dilation) combos in
    // its residual blocks (kernel=7, d∈{1,3,9}) and decoder/upsample blocks
    // (kernel=7,d=1 and kernel=1,d=1). Templated specialisations only.
    if (kernel == 7 && d0 == 1) { DISPATCH_KERNEL_DILATION(7, 1, WT); return; }
    if (kernel == 7 && d0 == 3) { DISPATCH_KERNEL_DILATION(7, 3, WT); return; }
    if (kernel == 7 && d0 == 9) { DISPATCH_KERNEL_DILATION(7, 9, WT); return; }
    if (kernel == 1 && d0 == 1) { DISPATCH_KERNEL_DILATION(1, 1, WT); return; }
    if (kernel == 3 && d0 == 1) { DISPATCH_KERNEL_DILATION(3, 1, WT); return; }

    // Anything else: dynamic-kernel fallback (no smem tile, slower).
    conv1d_direct_dynk_kernel<WT><<<grid, block, 0, stream>>>(
        (const WT *) a->data, x_d, y_d, kernel,
        in_seq, out_seq, in_ch, out_ch, s0, p0, d0,
        x_stride_b, x_stride_c, y_stride_b, y_stride_c);
}

void ggml_cuda_op_conv_1d_direct(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0]; // weights [kernel, in_ch, out_ch]
    const ggml_tensor * b = dst->src[1]; // input   [in_seq, in_ch, batch]

    GGML_ASSERT(b->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(a));

    const int32_t * p = (const int32_t *) dst->op_params;
    const int s0 = p[0];
    const int p0 = p[1];
    const int d0 = p[2];

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

    const float * x_d = (const float *) b->data;
    float       * y_d = (float *)       dst->data;

    cudaStream_t stream = ctx.stream();

    if (a->type == GGML_TYPE_F16) {
        launch_conv1d_direct<__half>(
            a, x_d, y_d, in_seq, out_seq, in_ch, out_ch, s0, p0, d0,
            x_stride_b, x_stride_c, y_stride_b, y_stride_c, kernel, batch, stream);
    } else if (a->type == GGML_TYPE_F32) {
        launch_conv1d_direct<float>(
            a, x_d, y_d, in_seq, out_seq, in_ch, out_ch, s0, p0, d0,
            x_stride_b, x_stride_c, y_stride_b, y_stride_c, kernel, batch, stream);
    } else {
        GGML_ABORT("ggml_cuda_op_conv_1d_direct: unsupported weight dtype");
    }
}
