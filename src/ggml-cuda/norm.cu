#include "norm.cuh"
#include <cstdint>

// T: element type of x/dst AND the mul/add broadcast operands, in {float, half}.
// matting's F16 swin layer-norm casts weight/bias to match x (cast_like), so the
// mul/add operands share x's element type; the LongCat fused path is all-F32.
// Internal math is always FLOAT (variance/mean accumulation in float is numerically
// critical); the F32 (T=float) instantiation is byte-identical to the original
// because every (float)/(T) cast is a no-op for float.
template <int block_size, bool do_multiply = false, bool do_add = false, typename T = float>
static __global__ void norm_f32(const T *     x,
                                T *           dst,
                                const int     ncols,
                                const int64_t stride_row,
                                const int64_t stride_channel,
                                const int64_t stride_sample,
                                const float   eps,
                                const T *     mul                  = nullptr,
                                const int64_t mul_stride_row       = 0,
                                const int64_t mul_stride_channel   = 0,
                                const int64_t mul_stride_sample    = 0,
                                const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                const T *     add                  = nullptr,
                                const int64_t add_stride_row       = 0,
                                const int64_t add_stride_channel   = 0,
                                const int64_t add_stride_sample    = 0,
                                const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                const uint3   add_nsamples_packed  = make_uint3(0, 0, 0)) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float2 mean_var = make_float2(0.0f, 0.0f);

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = (float) x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        // LongCat lap-27: emit explicit IEEE round-to-nearest mul/add to match the
        // unfused {NORM, MUL, ADD} chain's rounding (each separate kernel writes its
        // FP32 result back to memory between ops, forcing a round). Without these,
        // -use_fast_math collapses the `*mul + add` tail into a single FMA, which has
        // fewer rounds and produces a numerically-different (drift ~1e-5/elt) result,
        // breaking the bit-exact gate. The intermediate `(x-mean)*inv_std` also gets
        // explicit __fmul_rn so its round matches the original NORM kernel's store.
        const float xc = (float) x[col] - mean;
        const float xn = __fmul_rn(xc, inv_std);
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            const float xn_mul = __fmul_rn(xn, (float) mul[mul_col]);
            dst[col]           = (T) __fadd_rn(xn_mul, (float) add[add_col]);
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = (T) __fmul_rn(xn, (float) mul[mul_col]);
        } else {
            dst[col] = (T) xn;
        }
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

// Optional pre-norm bias fold (do_prebias): adds a 1D F32 bias[col] to x BEFORE the
// sum-of-squares reduction, folding the Linear's bias-add that immediately precedes the
// RMSNorm (LTX DiT q_norm/k_norm). To stay BIT-EXACT with the unfused {ADD, RMS_NORM}
// chain — where the separate ADD stores (x+bias) as T (F16 under LTX_DIT_F16) and the
// RMS_NORM re-reads it as T — we round (x+bias) to T via `(float)(T)xb` (round_to_T,
// no-op for float; round-to-nearest-half precedent = mul_add_bcast's round_big) and use
// the SAME rounded value for both the variance accumulation and the final store. prebias
// (T_bias = F32) is full-width over the ne0 axis, indexed by `col` only and broadcast over
// all rows/channels/samples (the bias is [ne0,1,1,1]). Default do_prebias=false leaves the
// existing instantiations byte-identical.
// T_mul: element type of the mul/add (modulation) broadcast operands. Defaults to T, so all
// existing instantiations are unchanged. The fused AdaLN op (ggml_rms_modulate) sets T_mul
// independently of T (e.g. x=half, scale/shift=float) and reads them in their own type — no
// cast tensors. add_one: when true, the loaded mul value gets +1.0f (the intrinsic (1+scale)
// of AdaLN); default false keeps the plain rms_norm*mul+add behaviour byte-identical.
template <int block_size, bool do_multiply = false, bool do_add = false, typename T = float,
          bool do_prebias = false, typename T_bias = float, typename T_mul = T, bool add_one = false>
static __global__ void rms_norm_f32(const T *     x,
                                    T *           dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const T_mul * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const T_mul * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const T_bias * prebias             = nullptr) {
    // This kernel is launched with Programmatic Stream Serialization. It consumes
    // x/scale/shift immediately, so it must resolve the dependency from the prior
    // stream kernel before reading any of them. Triggering our own launch-completion
    // below only lets the *next* PDL-aware kernel overlap; it does not make this
    // kernel's inputs visible. Without this wait a preceding SA3/GEMM may still be
    // flushing its activation when the fused AdaLN starts its reduction.
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    // Reads x[col] (+ rounded prebias[col] when do_prebias) as the value used by BOTH the
    // reduction and the normalize loop, so the two passes see identical bytes — exactly
    // what the unfused chain does (separate ADD writes T to memory, RMS_NORM reads it back).
    auto load_xb = [&](int col) -> float {
        if constexpr (do_prebias) {
            const float xb = (float) x[col] + (float) prebias[col];
            return (float) (T) xb;  // round to T (no-op for float; round-to-half for half)
        } else {
            return (float) x[col];
        }
    };

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = load_xb(col);
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const float xb = load_xb(col);
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            const float m     = (float) mul[mul_col] + (add_one ? 1.0f : 0.0f);
            dst[col]          = (T)(scale * xb * m + (float) add[add_col]);
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = (T)(scale * xb * (float) mul[mul_col]);
        } else {
            dst[col] = (T)(scale * xb);
        }
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size, typename T = float>
static __global__ void l2_norm_f32(
        const T * x, T * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = (float) x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    ggml_cuda_pdl_lc();

    // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (T)(scale * (float) x[col]);
    }
}

template <typename T = float>
static void norm_f32_cuda(
        const T * x, T * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE, false, false, T><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024, false, false, T><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

// Fused LayerNorm + (optional MUL) + (optional ADD). x/dst and the MUL/ADD broadcast
// operands share element type T in {float, half}; broadcast strides are arbitrary
// (fastmodulo on each axis), mirroring rms_norm_mul_f32_cuda. At LongCat's modulate hot
// shape ncols=hidden_size=4096 -> the 1024-block path (T=float). matting F16 swin -> T=half.
template <typename T = float>
static void norm_mul_f32_cuda(const T *      x,
                              const T *      mul,
                              const T *      add,
                              T *            dst,
                              const int      ncols,
                              const int      nrows,
                              const int      nchannels,
                              const int      nsamples,
                              const int64_t  stride_row,
                              const int64_t  stride_channel,
                              const int64_t  stride_sample,
                              const int64_t  mul_stride_row,
                              const int64_t  mul_stride_channel,
                              const int64_t  mul_stride_sample,
                              const uint32_t mul_ncols,
                              const uint32_t mul_nrows,
                              const uint32_t mul_nchannels,
                              const uint32_t mul_nsamples,
                              const int64_t  add_stride_row,
                              const int64_t  add_stride_channel,
                              const int64_t  add_stride_sample,
                              const uint32_t add_ncols,
                              const uint32_t add_nrows,
                              const uint32_t add_nchannels,
                              const uint32_t add_nsamples,
                              const float    eps,
                              cudaStream_t   stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (mul == nullptr) {
        norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (add == nullptr) {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(WARP_SIZE, 1, 1);
            norm_f32<WARP_SIZE, true, false, T><<<blocks_num, block_dims, 0, stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
        } else {
            const dim3 block_dims(1024, 1, 1);
            norm_f32<1024, true, false, T><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
        }
    } else {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(WARP_SIZE, 1, 1);
            norm_f32<WARP_SIZE, true, true, T><<<blocks_num, block_dims, 0, stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed);
        } else {
            const dim3 block_dims(1024, 1, 1);
            norm_f32<1024, true, true, T><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed);
        }
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

template <typename T = float>
static void rms_norm_f32_cuda(
        const T * x, T * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<256, false, false, T>, launch_params,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        (const float *) nullptr);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<1024, false, false, T>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        (const float *) nullptr);
    }
}

// Pre-norm bias fold: RMS_NORM(x + prebias) with NO trailing mul/add. prebias is a 1D
// F32 bias[ncols] broadcast over rows/channels/samples (passed full-width over ne0,
// indexed by col). do_multiply/do_add stay false → identical reduction/normalize math
// to rms_norm_f32_cuda, only the per-element x is replaced by round_to_T(x + prebias).
template <typename T = float>
static void rms_norm_prebias_f32_cuda(
        const T * x, const float * prebias, T * dst, const int ncols, const int nrows, const int nchannels,
        const int nsamples, const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
        const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch((rms_norm_f32<256, false, false, T, true, float>), launch_params,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        (const T *) nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        (const T *) nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        prebias);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch((rms_norm_f32<1024, false, false, T, true, float>), launch_params,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        (const T *) nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        (const T *) nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        prebias);
    }
}

template <typename T = float>
static void rms_norm_mul_f32_cuda(const T *      x,
                                  const T *      mul,
                                  const T *      add,
                                  T *            dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (mul == nullptr) {
        rms_norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (add == nullptr) {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true, false, T>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            (const T *) nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
            (const float *) nullptr);
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true, false, T>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            (const T *) nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
            (const float *) nullptr);
        }
    } else {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims,block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true, true, T>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed,
                (const float *) nullptr);
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true, true, T>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed,
                (const float *) nullptr);
        }
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

template <typename T = float>
static void l2_norm_f32_cuda(
        const T * x, T * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE, T>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<1024, T>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    // Additive F16 support (Swin layer-norm runs F16). Internal math stays FLOAT;
    // F32->F32 is byte-identical.
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    if (src0->type == GGML_TYPE_F16) {
        norm_f32_cuda((const half *) src0->data, (half *) dst->data, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
    } else {
        norm_f32_cuda((const float *) src0->data, (float *) dst->data, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
    }
}

// Fused LayerNorm + MUL (mirror of ggml_cuda_op_rms_norm_fused).
// LongCat lap-27: shape/strides come from mul_tensor (not norm_src). modulate's MUL
// runs over a 4D RESHAPE view of NORM_out — shape and contiguous-strides match the
// 4D form, but the broadcast computation of `mul_src` depends on the per-frame
// (channel=T) axis exposed by the reshape. norm_src's data is shared with the MUL
// view (contiguous F32), so reading x via the 4D strides yields the same bytes.
void ggml_cuda_op_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    const ggml_tensor * norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const void *        src0_d  = norm_src->data;
    const void *        mul_d   = nullptr;
    const ggml_tensor * mul_src = nullptr;

    // Trace src[0]/src[1] through any RESHAPE/VIEW chains back to dst (the NORM node).
    // The non-traced side is the broadcast operand (scale1).
    auto traces_to = [](const ggml_tensor * t, const ggml_tensor * root) {
        while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW ||
                     t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_PERMUTE)) {
            t = t->src[0];
        }
        return t == root;
    };
    if (traces_to(mul_tensor->src[0], dst)) {
        mul_d   = mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (traces_to(mul_tensor->src[1], dst)) {
        mul_d   = mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    void *       dst_d  = mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    // Additive F16 support: x/dst and the mul broadcast operand share element type
    // (matting F16 swin casts the weight to match x via cast_like). F32->F32 is the
    // original byte-identical path. Internal reduction/math stays FLOAT.
    GGML_ASSERT(norm_src->type == GGML_TYPE_F32 || norm_src->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == norm_src->type);
    GGML_ASSERT(mul_tensor->type == norm_src->type);
    GGML_ASSERT(mul_src->type == norm_src->type);
    GGML_ASSERT(eps >= 0.0f);

    // Shape/strides from mul_tensor (4D after modulate's reshape; same shape as norm_src
    // when the reshape is a no-op). norm_src's data layout is contiguous so the 4D strides
    // also describe the underlying bytes correctly.
    const int64_t ne00 = mul_tensor->ne[0];
    const int64_t ne01 = mul_tensor->ne[1];
    const int64_t ne02 = mul_tensor->ne[2];
    const int64_t ne03 = mul_tensor->ne[3];

    const size_t ts0 = ggml_type_size(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == ts0);
    const int64_t s01 = mul_tensor->nb[1] / ts0;
    const int64_t s02 = mul_tensor->nb[2] / ts0;
    const int64_t s03 = mul_tensor->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    if (norm_src->type == GGML_TYPE_F16) {
        norm_mul_f32_cuda((const half *) src0_d, (const half *) mul_d, (const half *) nullptr, (half *) dst_d,
                          ne00, ne01, ne02, ne03,
                          s01, s02, s03,
                          mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream);
    } else {
        norm_mul_f32_cuda((const float *) src0_d, (const float *) mul_d, (const float *) nullptr, (float *) dst_d,
                          ne00, ne01, ne02, ne03,
                          s01, s02, s03,
                          mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream);
    }
}

// Fused LayerNorm + MUL + ADD (mirror of ggml_cuda_op_rms_norm_fused_add).
// LongCat lap-27: shape/strides come from mul_tensor (post-reshape 4D in modulate).
void ggml_cuda_op_norm_fused_add(ggml_backend_cuda_context & ctx,
                                 ggml_tensor *               dst,
                                 ggml_tensor *               mul_tensor,
                                 ggml_tensor *               add_tensor) {
    const ggml_tensor * norm_src = (ggml_tensor *) dst->src[0];
    float               eps      = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const void *        src0_d  = norm_src->data;
    const void *        mul_d   = nullptr;
    const ggml_tensor * mul_src = nullptr;

    auto traces_to = [](const ggml_tensor * t, const ggml_tensor * root) {
        while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW ||
                     t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_PERMUTE)) {
            t = t->src[0];
        }
        return t == root;
    };
    if (traces_to(mul_tensor->src[0], dst)) {
        mul_d   = mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (traces_to(mul_tensor->src[1], dst)) {
        mul_d   = mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const void *        add_d   = nullptr;
    const ggml_tensor * add_src = nullptr;

    if (traces_to(add_tensor->src[0], mul_tensor)) {
        add_d   = add_tensor->src[1]->data;
        add_src = add_tensor->src[1];
    } else if (traces_to(add_tensor->src[1], mul_tensor)) {
        add_d   = add_tensor->src[0]->data;
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    void *       dst_d  = add_tensor->data;
    cudaStream_t stream = ctx.stream();

    // Additive F16: x/dst, mul and add operands all share element type (matting F16
    // swin casts weight+bias to match x). F32->F32 byte-identical; math stays FLOAT.
    GGML_ASSERT(norm_src->type == GGML_TYPE_F32 || norm_src->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == norm_src->type);
    GGML_ASSERT(mul_tensor->type == norm_src->type);
    GGML_ASSERT(add_tensor->type == norm_src->type);
    GGML_ASSERT(mul_src->type == norm_src->type);
    GGML_ASSERT(add_src->type == norm_src->type);
    GGML_ASSERT(eps >= 0.0f);

    // Use mul_tensor's shape/strides (4D after the modulate reshape; norm_src's data
    // is contiguous and shares bytes with the reshape view, so these strides also
    // index the underlying x correctly).
    const int64_t ne00 = mul_tensor->ne[0];
    const int64_t ne01 = mul_tensor->ne[1];
    const int64_t ne02 = mul_tensor->ne[2];
    const int64_t ne03 = mul_tensor->ne[3];

    const size_t ts0 = ggml_type_size(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == ts0);
    const int64_t s01 = mul_tensor->nb[1] / ts0;
    const int64_t s02 = mul_tensor->nb[2] / ts0;
    const int64_t s03 = mul_tensor->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    if (norm_src->type == GGML_TYPE_F16) {
        norm_mul_f32_cuda((const half *) src0_d, (const half *) mul_d, (const half *) add_d, (half *) dst_d,
                          ne00, ne01, ne02, ne03,
                          s01, s02, s03,
                          mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream);
    } else {
        norm_mul_f32_cuda((const float *) src0_d, (const float *) mul_d, (const float *) add_d, (float *) dst_d,
                          ne00, ne01, ne02, ne03,
                          s01, s02, s03,
                          mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream);
    }
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    // Additive F16 support. Internal math stays FLOAT; F32->F32 is byte-identical.
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    if (src0->type == GGML_TYPE_F16) {
        rms_norm_f32_cuda((const half *) src0->data, (half *) dst->data, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
    } else {
        rms_norm_f32_cuda((const float *) src0->data, (float *) dst->data, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
    }
}

// Fused (pre-norm bias) + RMS_NORM. Folds a Linear's bias-add that immediately precedes
// the RMSNorm: the graph is ADD(matmul_out[ne0,tok], bias[ne0]) -> RMS_NORM(over ne0).
// `dst` is the RMS_NORM node; `add_node` is the preceding ADD. `x` is the matmul output
// (the ADD operand that traces to a compute node) and `prebias` is the 1D F32 bias (the
// other ADD operand). The trailing RMSNorm MUL (weight) is NOT folded here — under
// LTX_DIT_F16 x is F16 while the RMSNorm weight is F32, so it runs as a separate MUL
// (the launcher templates one element type T for x/dst). We only need the bias-add gone.
//
// Bit-exactness: the unfused chain is ADD: dst_add = (T)((float)x + bias) stored to
// memory, then RMS_NORM re-reads dst_add as T. The kernel reproduces that exactly by
// computing xb = (float)(T)((float)x + bias) (the do_prebias path's round_to_T) and using
// that SAME rounded xb for both the variance sum and the normalized store. F32 (T=float):
// (float)(float)(...) is a no-op, so this is byte-identical to feeding RMS_NORM the
// separately-added F32 buffer.
void ggml_cuda_op_rms_norm_fused_prebias(ggml_backend_cuda_context & ctx,
                                         ggml_tensor *               dst,
                                         ggml_tensor *               add_node,
                                         ggml_tensor *               x,
                                         ggml_tensor *               prebias) {
    float eps = 0.0f;
    memcpy(&eps, dst->op_params, sizeof(float));

    const void * src0_d   = x->data;
    const float * prebias_d = (const float *) prebias->data;
    void *       dst_d    = dst->data;
    cudaStream_t stream   = ctx.stream();

    // x/dst share element type (F32 byte-identical path, or F16 under LTX_DIT_F16). The
    // RMS_NORM dst type equals x's type (the ADD that produced x writes the same type the
    // RMS_NORM re-reads). prebias is the bias element type = F32.
    GGML_ASSERT(x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == x->type);
    GGML_ASSERT(prebias->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    GGML_UNUSED(add_node);

    const int64_t ne00 = x->ne[0];
    const int64_t ne01 = x->ne[1];
    const int64_t ne02 = x->ne[2];
    const int64_t ne03 = x->ne[3];

    const size_t ts0 = ggml_type_size(x->type);
    GGML_ASSERT(x->nb[0] == ts0);
    const int64_t s01 = x->nb[1] / ts0;
    const int64_t s02 = x->nb[2] / ts0;
    const int64_t s03 = x->nb[3] / ts0;

    // prebias is 1D [ne0,1,1,1], contiguous F32, full-width over ne0 (indexed by col).
    GGML_ASSERT(prebias->ne[0] == ne00);
    GGML_ASSERT(ggml_is_contiguous(prebias));

    if (x->type == GGML_TYPE_F16) {
        rms_norm_prebias_f32_cuda((const half *) src0_d, prebias_d, (half *) dst_d,
                                  ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
    } else {
        rms_norm_prebias_f32_cuda((const float *) src0_d, prebias_d, (float *) dst_d,
                                  ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
    }
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const void * src0_d = rms_norm_src->data;
    const void * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_d = mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    void * dst_d = mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    // Additive F16 support: x/dst share element type with the mul operand. F32 path
    // is byte-identical; reduction/math stays FLOAT.
    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32 || rms_norm_src->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == rms_norm_src->type);
    GGML_ASSERT(mul_tensor->type == rms_norm_src->type);
    GGML_ASSERT(mul_src->type == rms_norm_src->type);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    if (rms_norm_src->type == GGML_TYPE_F16) {
        rms_norm_mul_f32_cuda((const half *) src0_d, (const half *) mul_d, (const half *) nullptr, (half *) dst_d,
                              ne00, ne01, ne02, ne03,
                              /*s00*/ s01, s02, s03,
                              /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                              mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                              /*add_s00*/ 0, 0, 0,
                              0, 0, 0, 0,
                              eps, stream);
    } else {
        rms_norm_mul_f32_cuda((const float *) src0_d, (const float *) mul_d, (const float *) nullptr, (float *) dst_d,
                              ne00, ne01, ne02, ne03,
                              /*s00*/ s01, s02, s03,
                              /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                              mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                              /*add_s00*/ 0, 0, 0,
                              0, 0, 0, 0,
                              eps, stream);
    }
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const void *        src0_d  = rms_norm_src->data;
    const void *        mul_d   = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d   = mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d   = mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const void *        add_d   = nullptr;
    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_d   = add_tensor->src[1]->data;
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_d   = add_tensor->src[0]->data;
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    void *       dst_d  = add_tensor->data;
    cudaStream_t stream = ctx.stream();

    // Additive F16: x/dst, mul and add operands share element type. F32 path is
    // byte-identical; reduction/math stays FLOAT.
    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32 || rms_norm_src->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == rms_norm_src->type);
    GGML_ASSERT(mul_tensor->type == rms_norm_src->type);
    GGML_ASSERT(add_tensor->type == rms_norm_src->type);
    GGML_ASSERT(mul_src->type == rms_norm_src->type);
    GGML_ASSERT(add_src->type == rms_norm_src->type);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    if (rms_norm_src->type == GGML_TYPE_F16) {
        rms_norm_mul_f32_cuda((const half *) src0_d, (const half *) mul_d, (const half *) add_d, (half *) dst_d,
                              ne00,ne01, ne02, ne03,
                              /*s00*/ s01, s02, s03,
                              /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                              mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                              /*add_s00*/ add_s01, add_s02, add_s03,
                              add_ncols, add_nrows, add_nchannels, add_nsamples,
                              eps, stream);
    } else {
        rms_norm_mul_f32_cuda((const float *) src0_d, (const float *) mul_d, (const float *) add_d, (float *) dst_d,
                              ne00,ne01, ne02, ne03,
                              /*s00*/ s01, s02, s03,
                              /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                              mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                              /*add_s00*/ add_s01, add_s02, add_s03,
                              add_ncols, add_nrows, add_nchannels, add_nsamples,
                              eps, stream);
    }
}

// ---------------------------------------------------------------------------
// ggml_rms_modulate — fused AdaLN: out = rms_norm(x)*(1+scale)+shift.
// Mixed-type launcher: x/dst element type = T_x (half|float); scale/shift element
// type = T_mod (half|float), read WITHOUT cast. Always mul+add with add_one (the +1
// of (1+scale) lives in the kernel). Mirrors rms_norm_mul_f32_cuda's stride/fastdiv
// setup but specialises T_mul=T_mod and add_one=true. Math/reduction stay FLOAT.
template <typename T_x, typename T_mod>
static void rms_modulate_cuda(const T_x *    x,
                              const T_mod *  scale,
                              const T_mod *  shift,
                              T_x *          dst,
                              const int      ncols,
                              const int      nrows,
                              const int      nchannels,
                              const int      nsamples,
                              const int64_t  stride_row,
                              const int64_t  stride_channel,
                              const int64_t  stride_sample,
                              const int64_t  mul_stride_row,
                              const int64_t  mul_stride_channel,
                              const int64_t  mul_stride_sample,
                              const uint32_t mul_ncols,
                              const uint32_t mul_nrows,
                              const uint32_t mul_nchannels,
                              const uint32_t mul_nsamples,
                              const int64_t  add_stride_row,
                              const int64_t  add_stride_channel,
                              const int64_t  add_stride_sample,
                              const uint32_t add_ncols,
                              const uint32_t add_nrows,
                              const uint32_t add_nchannels,
                              const uint32_t add_nsamples,
                              const float    eps,
                              cudaStream_t   stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);

    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

    const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
    const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
    const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
    const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);

    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch((rms_norm_f32<256, true, true, T_x, false, float, T_mod, true>), launch_params,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps, scale, mul_stride_row, mul_stride_channel,
            mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, shift,
            add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
            add_nchannels_packed, add_nsamples_packed,
            (const float *) nullptr);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch((rms_norm_f32<1024, true, true, T_x, false, float, T_mod, true>), launch_params,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps, scale, mul_stride_row, mul_stride_channel,
            mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, shift,
            add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
            add_nchannels_packed, add_nsamples_packed,
            (const float *) nullptr);
    }
}

// ggml_rms_norm_channels — RMS-normalize over the CHANNEL dim (ne[3]) of a [W,H,T,C] activation,
// folding in per-channel gamma, in ONE coalesced pass. The Wan VAE decoder keeps activations
// [W,H,T,C] (channels = ne[3], stride S = ne0*ne1*ne2); stock ggml_rms_norm only reduces over
// ne[0], so the VAE path used permute(C->ne0)+cont, rms, mul(gamma), permute-back+cont == 2 conts
// + a separate mul per RMS_norm. This op reads [W,H,T,C] natively: one thread per spatial position
// p in [0,S) loops the C channels at stride S (adjacent threads -> adjacent p -> fully coalesced),
// reduces mean-square in FLOAT, then writes out[p+c*S] = x*rsqrt(ms+eps)*gamma[c]. x/dst F16 or F32
// (same type); gamma F32. Bit-for-bit formula-identical to rms_norm+mul (up to float reduction
// order). CUDA-only, mirrors ggml_rms_modulate.
template <typename T>
static __global__ void rms_norm_channels_kernel(const T * __restrict__ x, const float * __restrict__ gamma,
                                                T * __restrict__ dst, const int64_t S, const int C, const float eps) {
    const int64_t p = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= S) {
        return;
    }
    float ss = 0.0f;
    for (int c = 0; c < C; ++c) {
        const float v = (float) x[p + (int64_t) c * S];
        ss += v * v;
    }
    const float scale = rsqrtf(ss / (float) C + eps);
    for (int c = 0; c < C; ++c) {
        const int64_t idx = p + (int64_t) c * S;
        dst[idx] = (T) ((float) x[idx] * scale * gamma[c]);
    }
}

template <typename T>
static void rms_norm_channels_cuda(const T * x, const float * gamma, T * dst,
                                   const int64_t S, const int C, const float eps, cudaStream_t stream) {
    const int64_t block = 256;
    const int64_t grid  = (S + block - 1) / block;
    rms_norm_channels_kernel<T><<<(unsigned int) grid, (unsigned int) block, 0, stream>>>(x, gamma, dst, S, C, eps);
}

void ggml_cuda_op_rms_norm_channels(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];  // [W,H,T,C] contiguous activation, F16 or F32
    const ggml_tensor * w = dst->src[1];  // gamma [C], F32

    float eps = 0.0f;
    memcpy(&eps, dst->op_params, sizeof(float));

    GGML_ASSERT(x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == x->type);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(w->ne[0] == x->ne[3]);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t S = x->ne[0] * x->ne[1] * x->ne[2];
    const int     C = (int) x->ne[3];
    cudaStream_t  stream = ctx.stream();

    if (x->type == GGML_TYPE_F16) {
        rms_norm_channels_cuda<half>((const half *) x->data, (const float *) w->data, (half *) dst->data, S, C, eps, stream);
    } else {
        rms_norm_channels_cuda<float>((const float *) x->data, (const float *) w->data, (float *) dst->data, S, C, eps, stream);
    }
}

void ggml_cuda_op_rms_modulate(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x     = dst->src[0]; // activation (pre-rms), F16 or F32
    const ggml_tensor * scale = dst->src[1]; // modulation scale (mul), F16 or F32
    const ggml_tensor * shift = dst->src[2]; // modulation shift (add), same type as scale

    float eps = 0.0f;
    memcpy(&eps, dst->op_params, sizeof(float));

    const void * x_d     = x->data;
    const void * scale_d = scale->data;
    const void * shift_d = shift->data;
    void *       dst_d   = dst->data;
    cudaStream_t stream  = ctx.stream();

    // x/dst share type; scale/shift share type (MAY differ from x — the whole point: NO cast).
    GGML_ASSERT(x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == x->type);
    GGML_ASSERT(scale->type == GGML_TYPE_F32 || scale->type == GGML_TYPE_F16);
    GGML_ASSERT(shift->type == scale->type);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = x->ne[0];
    const int64_t ne01 = x->ne[1];
    const int64_t ne02 = x->ne[2];
    const int64_t ne03 = x->ne[3];

    const size_t ts0 = ggml_type_size(x->type);
    GGML_ASSERT(x->nb[0] == ts0);
    const int64_t s01 = x->nb[1] / ts0;
    const int64_t s02 = x->nb[2] / ts0;
    const int64_t s03 = x->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(scale->type);
    GGML_ASSERT(scale->nb[0] == ts_mul);
    const int64_t mul_s01 = scale->nb[1] / ts_mul;
    const int64_t mul_s02 = scale->nb[2] / ts_mul;
    const int64_t mul_s03 = scale->nb[3] / ts_mul;

    const int mul_ncols     = scale->ne[0];
    const int mul_nrows     = scale->ne[1];
    const int mul_nchannels = scale->ne[2];
    const int mul_nsamples  = scale->ne[3];

    const size_t ts_add = ggml_type_size(shift->type);
    GGML_ASSERT(shift->nb[0] == ts_add);
    const int64_t add_s01 = shift->nb[1] / ts_add;
    const int64_t add_s02 = shift->nb[2] / ts_add;
    const int64_t add_s03 = shift->nb[3] / ts_add;

    const int add_ncols     = shift->ne[0];
    const int add_nrows     = shift->ne[1];
    const int add_nchannels = shift->ne[2];
    const int add_nsamples  = shift->ne[3];

    // Dispatch the (x-type, mod-type) combos that actually occur: (half,float),(half,half),(float,float).
    if (x->type == GGML_TYPE_F16 && scale->type == GGML_TYPE_F32) {
        rms_modulate_cuda<half, float>((const half *) x_d, (const float *) scale_d, (const float *) shift_d, (half *) dst_d,
                                       ne00, ne01, ne02, ne03, s01, s02, s03,
                                       mul_s01, mul_s02, mul_s03, mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                                       add_s01, add_s02, add_s03, add_ncols, add_nrows, add_nchannels, add_nsamples,
                                       eps, stream);
    } else if (x->type == GGML_TYPE_F16 && scale->type == GGML_TYPE_F16) {
        rms_modulate_cuda<half, half>((const half *) x_d, (const half *) scale_d, (const half *) shift_d, (half *) dst_d,
                                      ne00, ne01, ne02, ne03, s01, s02, s03,
                                      mul_s01, mul_s02, mul_s03, mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                                      add_s01, add_s02, add_s03, add_ncols, add_nrows, add_nchannels, add_nsamples,
                                      eps, stream);
    } else if (x->type == GGML_TYPE_F32 && scale->type == GGML_TYPE_F32) {
        rms_modulate_cuda<float, float>((const float *) x_d, (const float *) scale_d, (const float *) shift_d, (float *) dst_d,
                                        ne00, ne01, ne02, ne03, s01, s02, s03,
                                        mul_s01, mul_s02, mul_s03, mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                                        add_s01, add_s02, add_s03, add_ncols, add_nrows, add_nchannels, add_nsamples,
                                        eps, stream);
    } else {
        // (x=F32, mod=F16) does not occur in the LTX modulation path; add a combo above if needed.
        GGML_ABORT("ggml_rms_modulate: unsupported (x,scale) type combo");
    }
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    // Additive F16 support. Internal math stays FLOAT; F32->F32 is byte-identical.
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    if (src0->type == GGML_TYPE_F16) {
        l2_norm_f32_cuda((const half *) src0->data, (half *) dst->data, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
    } else {
        l2_norm_f32_cuda((const float *) src0->data, (float *) dst->data, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
    }
}
