#include "fattn-sa3.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mutex>

#include "sageattention3/launch.h"

// This file is intentionally the only ggml-facing SA3 adapter.  The imported
// CUTLASS kernel remains upstream code; preprocessing, allocation and dispatch
// are native ggml/CUDA so enabling this feature never introduces Torch at runtime.
constexpr int kHeadDim = 128;
constexpr int kTokenBlock = 128;
constexpr int kQuantEltsPerThread = 16;

__global__ void sa3_cast_q_f32(const float * in, half * out, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(in[i]);
}

__global__ void sa3_cast_q_f32_pad(const float * in, half * out, int l, int lr, int heads) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) heads * lr * kHeadDim;
    if (i < n) {
        const int d = i % kHeadDim;
        const size_t token = i / kHeadDim;
        const int h = token / lr;
        const int t = token % lr;
        out[i] = __float2half_rn(t < l ? in[((size_t) h * l + t) * kHeadDim + d] : 0.0f);
    }
}

__global__ void sa3_pad_f16(const half * in, half * out, int l, int lr, int heads) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) heads * lr * kHeadDim;
    if (i < n) {
        const int d = i % kHeadDim;
        const size_t token = i / kHeadDim;
        const int h = token / lr;
        const int t = token % lr;
        out[i] = t < l ? in[((size_t) h * l + t) * kHeadDim + d] : __float2half(0.0f);
    }
}

__global__ void sa3_k_mean(const half * k, float * mean, int l) {
    const int d = blockIdx.x;
    const int h = blockIdx.y;
    float sum = 0.0f;
    for (int t = threadIdx.x; t < l; t += blockDim.x) {
        sum += __half2float(k[((size_t) h * l + t) * kHeadDim + d]);
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x / 2; n != 0; n /= 2) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (threadIdx.x == 0) mean[h * kHeadDim + d] = partial[0] / l;
}

__global__ void sa3_center_k(const half * in, const float * mean, half * out, size_t n, int l) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int d = i % kHeadDim;
        const int h = i / ((size_t) l * kHeadDim);
        out[i] = __float2half_rn(__half2float(in[i]) - mean[h * kHeadDim + d]);
    }
}

__global__ void sa3_center_k_pad(const half * in, const float * mean, half * out, int l, int lr, int heads) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) heads * lr * kHeadDim;
    if (i < n) {
        const int d = i % kHeadDim;
        const size_t token = i / kHeadDim;
        const int h = token / lr;
        const int t = token % lr;
        out[i] = t < l ? __float2half_rn(__half2float(in[((size_t) h * l + t) * kHeadDim + d]) - mean[h * kHeadDim + d]) : __float2half(0.0f);
    }
}

// One CTA computes a Q mean for one [head, 128-token block, channel].  The
// subsequent centering pass preserves the original BHSD physical layout.
__global__ void sa3_q_block_mean(const half * q, half * mean, int l, int blocks) {
    const int d = blockIdx.x;
    const int qb = blockIdx.y;
    const int h = blockIdx.z;
    const int t = qb * kTokenBlock + threadIdx.x;
    float x = t < l ? __half2float(q[((size_t) h * l + t) * kHeadDim + d]) : 0.0f;
    __shared__ float partial[kTokenBlock];
    partial[threadIdx.x] = x;
    __syncthreads();
    for (int n = kTokenBlock / 2; n != 0; n /= 2) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        mean[((size_t) h * blocks + qb) * kHeadDim + d] = __float2half_rn(partial[0] / kTokenBlock);
    }
}

__global__ void sa3_center_q(const half * in, const half * mean, half * out, size_t n, int l, int blocks) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int d = i % kHeadDim;
        const size_t token = i / kHeadDim;
        const int h = token / l;
        const int qb = (token % l) / kTokenBlock;
        out[i] = __float2half_rn(__half2float(in[i]) - __half2float(mean[((size_t) h * blocks + qb) * kHeadDim + d]));
    }
}

inline __device__ uint32_t sa3_fp32_to_e2m1(float2 * x) {
    uint32_t v;
    // The FP4 pack PTX (cvt.e2m1x2) is Blackwell-only (sm_100+); ptxas rejects it for
    // sm_86. Arch-gate so the SA3 sources COMPILE for sm86 (as a no-op) and one binary
    // builds for 86;120 — SA3 is only ever dispatched on Blackwell at runtime
    // (ggml_cuda_flash_attn_ext_sa3 is gated by GGML_LTX_SA3 + cc), so the sm86 stub is
    // never executed. (Same >= 1000 gate style as sageattention3/fp4_quantization_4d.cu.)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile(
        "{ .reg .b8 a,b,c,d;"
        "cvt.rn.satfinite.e2m1x2.f32 a,%2,%1;"
        "cvt.rn.satfinite.e2m1x2.f32 b,%4,%3;"
        "cvt.rn.satfinite.e2m1x2.f32 c,%6,%5;"
        "cvt.rn.satfinite.e2m1x2.f32 d,%8,%7;"
        "mov.b32 %0,{a,b,c,d}; }"
        : "=r"(v) : "f"(x[0].x), "f"(x[0].y), "f"(x[1].x), "f"(x[1].y),
          "f"(x[2].x), "f"(x[2].y), "f"(x[3].x), "f"(x[3].y));
#else
    (void) x;
    v = 0u;  // sm86 stub: SA3 never runs on non-Blackwell (runtime cc-gated)
#endif
    return v;
}

template <bool Permute>
__global__ void sa3_quant_qk(const half * in, uint8_t * out, uint8_t * sf, int l, int lr, bool clip_input) {
    const int tb = blockIdx.x, h = blockIdx.z;
    const int lane = threadIdx.x;
    const int token = tb * kTokenBlock + lane / (kHeadDim / kQuantEltsPerThread);
    int load_token = token;
    if constexpr (Permute) {
        const int local = lane / (kHeadDim / kQuantEltsPerThread);
        const int r = local % 32;
        load_token = tb * kTokenBlock + (local / 32) * 32 + (r / 8) * 2 + ((r % 8) / 2) * 8 + r % 2;
    }
    half2 values[8] = {};
    if (load_token < l) {
        const half * p = in + ((size_t) h * l + load_token) * kHeadDim + (lane % 8) * kQuantEltsPerThread;
        #pragma unroll
        for (int i = 0; i < 8; ++i) values[i] = reinterpret_cast<const half2 *>(p)[i];
    }
    if (clip_input) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float2 x = __half22float2(values[i]);
            x.x = fminf(fmaxf(x.x, -2688.0f), 2688.0f);
            x.y = fminf(fmaxf(x.y, -2688.0f), 2688.0f);
            values[i] = __floats2half2_rn(x.x, x.y);
        }
    }
    half2 mx = __habs2(values[0]);
    #pragma unroll
    for (int i = 1; i < 8; ++i) mx = __hmax2(mx, __habs2(values[i]));
    const float scale_raw = __half2float(__hmax(mx.x, mx.y)) / 6.0f;
    const __nv_fp8_e4m3 fp8_scale(scale_raw);
    uint8_t scale_bits = reinterpret_cast<const uint8_t &>(fp8_scale);
    const float scale = float(reinterpret_cast<const __nv_fp8_e4m3 &>(scale_bits));
    float2 x[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        x[i] = __half22float2(values[i]);
        const float inv = scale == 0.0f ? 0.0f : 1.0f / scale;
        x[i].x *= inv; x[i].y *= inv;
    }
    if (token < lr) {
        uint32_t * dst = reinterpret_cast<uint32_t *>(out + ((size_t) h * lr + token) * (kHeadDim / 2) + (lane % 8) * 8);
        dst[0] = sa3_fp32_to_e2m1(x);
        dst[1] = sa3_fp32_to_e2m1(x + 4);
        const int local_token = token % 64;
        const int col = lane % 8;
        sf[(size_t) h * lr * (kHeadDim / 16) + (token / 64) * 64 * (kHeadDim / 16)
           + (col / 4) * 256 + col % 4 + (local_token / 16) * 4 + (local_token % 16) * 16] = scale_bits;
    }
}

// V uses the transposed physical layout expected by the Blackwell PV MMA.
__global__ void sa3_quant_vt(const half * in, uint8_t * out, uint8_t * sf, int l, int lr, bool clip_input) {
    const int tb = blockIdx.x, h = blockIdx.z, lane = threadIdx.x;
    const int d = lane / 8;
    const int token0_local = (lane % 8) * kQuantEltsPerThread;
    const int token0 = tb * kTokenBlock + token0_local;
    __shared__ half tile[kTokenBlock * kHeadDim];
    const int token = tb * kTokenBlock + lane / 8;
    half2 v[8] = {};
    if (token < l) {
        const half * p = in + ((size_t) h * l + token) * kHeadDim + (lane % 8) * kQuantEltsPerThread;
        #pragma unroll
        for (int i = 0; i < 8; ++i) reinterpret_cast<half2 *>(tile)[lane * 8 + i] = reinterpret_cast<const half2 *>(p)[i];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = make_half2(tile[(token0_local + 2*i) * kHeadDim + d], tile[(token0_local + 2*i + 1) * kHeadDim + d]);
    if (clip_input) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float2 x = __half22float2(v[i]);
            x.x = fminf(fmaxf(x.x, -2688.0f), 2688.0f);
            x.y = fminf(fmaxf(x.y, -2688.0f), 2688.0f);
            v[i] = __floats2half2_rn(x.x, x.y);
        }
    }
    half2 mx = __habs2(v[0]);
    #pragma unroll
    for (int i = 1; i < 8; ++i) mx = __hmax2(mx, __habs2(v[i]));
    const float raw = __half2float(__hmax(mx.x, mx.y)) / 6.0f;
    const __nv_fp8_e4m3 fp8_scale(raw);
    uint8_t bits = reinterpret_cast<const uint8_t &>(fp8_scale);
    const float scale = float(reinterpret_cast<const __nv_fp8_e4m3 &>(bits));
    float2 x[8]; const float inv = scale == 0.0f ? 0.0f : 1.0f / scale;
    #pragma unroll
    for (int i = 0; i < 8; ++i) { x[i] = __half22float2(v[i]); x[i].x *= inv; x[i].y *= inv; }
    uint32_t * p = reinterpret_cast<uint32_t *>(out + ((size_t) h * kHeadDim + d) * (lr / 2) + token0 / 2);
    p[0] = sa3_fp32_to_e2m1(x); p[1] = sa3_fp32_to_e2m1(x + 4);
    const int row = d % 64, col = tb * 8 + lane % 8;
    sf[(size_t) h * kHeadDim * (lr / 16) + (d / 64) * 64 * (lr / 16)
       + (col / 4) * 256 + col % 4 + (row / 16) * 4 + (row % 16) * 16] = bits;
}

__global__ void sa3_o_to_dst(const half * in, void * dst, int l, int lr, int heads, int head_start, int total_heads, bool f32) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) heads * l * kHeadDim;
    if (i < n) {
        const int d = i % kHeadDim; const size_t t = i / kHeadDim;
        const int h = t / l; const int token = t % l;
        const size_t out_i = ((size_t) token * total_heads + head_start + h) * kHeadDim + d;
        const half x = in[((size_t) h * lr + token) * kHeadDim + d];
        if (f32) static_cast<float *>(dst)[out_i] = __half2float(x);
        else static_cast<half *>(dst)[out_i] = x;
    }
}

bool ggml_cuda_flash_attn_ext_sa3(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    float scale = 0.0f, max_bias = 0.0f, softcap = 0.0f;
    memcpy(&scale, dst->op_params, sizeof(scale));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(max_bias));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(softcap));
    const bool contract = cc >= GGML_CUDA_CC_BLACKWELL && dst->src[3] == nullptr &&
        q->ne[0] == kHeadDim && q->ne[1] == k->ne[1] && q->ne[1] == v->ne[1] &&
        q->ne[2] == 32 && k->ne[2] == 32 && v->ne[2] == 32 && q->ne[3] == 1 &&
        k->ne[3] == 1 && v->ne[3] == 1 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 &&
        (q->type == GGML_TYPE_F16 || q->type == GGML_TYPE_F32) && (dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32) &&
        max_bias == 0.0f && softcap == 0.0f && ggml_is_contiguous(q) && ggml_is_contiguous(k) &&
        ggml_is_contiguous(v) && ggml_is_contiguous(dst);
    if (!contract) {
        static int rejected = 0;
        if (rejected++ < 3) {
            fprintf(stderr, "[sa3] reject cc=%d q=[%lld,%lld,%lld,%lld] qtype=%d kvtype=%d/%d dst=%d mask=%d contiguous=%d/%d/%d/%d bias=%g softcap=%g\n",
                    cc, (long long) q->ne[0], (long long) q->ne[1], (long long) q->ne[2], (long long) q->ne[3],
                    q->type, k->type, v->type, dst->type, dst->src[3] != nullptr,
                    ggml_is_contiguous(q), ggml_is_contiguous(k), ggml_is_contiguous(v), ggml_is_contiguous(dst), max_bias, softcap);
        }
        return false;
    }

    const int l = q->ne[1], total_h = 32, lr = ((l + 127) / 128) * 128, blocks = lr / 128;
    int head_group = total_h;
    if (const char * e = getenv("GGML_LTX_SA3_HEAD_GROUP")) {
        const int requested = atoi(e);
        if (requested > 0 && requested <= total_h && total_h % requested == 0) head_group = requested;
    }
    const size_t elems = (size_t) head_group * l * kHeadDim;
    const size_t elems_padded = (size_t) head_group * lr * kHeadDim;
    const cudaStream_t stream = ctx.stream();
    // The locked singing repro enables the validated F16 storage path. Keep
    // the core runtime opt-in so other callers retain the original FP32 path.
    const char * delta_f16_env = getenv("GGML_LTX_SA3_DELTA_F16");
    const bool delta_f16 = delta_f16_env != nullptr && atoi(delta_f16_env) != 0;
    const bool timing = getenv("GGML_LTX_SA3_TIMING") != nullptr;
    static cudaEvent_t timing_begin = nullptr, timing_mha = nullptr, timing_end = nullptr;
    static std::once_flag timing_once;
    if (timing) {
        std::call_once(timing_once, [] {
            CUDA_CHECK(cudaEventCreate(&timing_begin));
            CUDA_CHECK(cudaEventCreate(&timing_mha));
            CUDA_CHECK(cudaEventCreate(&timing_end));
        });
        CUDA_CHECK(cudaEventRecord(timing_begin, stream));
    }
    auto check_stage = [&](const char * stage) {
        (void) stage;
        CUDA_CHECK(cudaGetLastError());
    };
    // The preprocessing tensors have non-overlapping lifetimes.  Reuse one
    // padded BHSD scratch buffer for Q, K, V, then for the attention output;
    // at long chained resolutions this avoids four simultaneous 250+ MiB
    // buffers and keeps the native path within the VRAM budget.
    ggml_cuda_pool_alloc<half> scratch(ctx.pool()), q_mean(ctx.pool());
    ggml_cuda_pool_alloc<float> k_mean(ctx.pool());
    ggml_cuda_pool_alloc<half> delta_f16_buf(ctx.pool());
    ggml_cuda_pool_alloc<float> delta(ctx.pool()), lse(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> q4(ctx.pool()), k4(ctx.pool()), v4(ctx.pool()), sfq(ctx.pool()), sfk(ctx.pool()), sfv(ctx.pool());
    scratch.alloc(elems_padded); q_mean.alloc((size_t) head_group * blocks * kHeadDim);
    k_mean.alloc((size_t) head_group * kHeadDim);
    if (delta_f16) delta_f16_buf.alloc((size_t) head_group * blocks * lr);
    else delta.alloc((size_t) head_group * blocks * lr);
    lse.alloc((size_t) head_group * lr);
    q4.alloc((size_t) head_group * lr * kHeadDim / 2); k4.alloc((size_t) head_group * lr * kHeadDim / 2); v4.alloc((size_t) head_group * kHeadDim * lr / 2);
    sfq.alloc((size_t) head_group * lr * kHeadDim / 16); sfk.alloc((size_t) head_group * lr * kHeadDim / 16); sfv.alloc((size_t) head_group * kHeadDim * lr / 16);
    for (int head_start = 0; head_start < total_h; head_start += head_group) {
        const size_t head_offset = (size_t) head_start * l * kHeadDim;
        const float * q_f32 = static_cast<const float *>(q->data) + head_offset;
        const half * k_f16 = static_cast<const half *>(k->data) + head_offset;
        const half * v_f16 = static_cast<const half *>(v->data) + head_offset;
        if (q->type == GGML_TYPE_F32) sa3_cast_q_f32_pad<<<(elems_padded + 255) / 256, 256, 0, stream>>>(q_f32, scratch.get(), l, lr, head_group);
        else sa3_pad_f16<<<(elems_padded + 255) / 256, 256, 0, stream>>>((const half *) q->data + head_offset, scratch.get(), l, lr, head_group);
        check_stage("Q cast");
        sa3_k_mean<<<dim3(kHeadDim, head_group), 256, 0, stream>>>(k_f16, k_mean.get(), l);
        sa3_q_block_mean<<<dim3(kHeadDim, blocks, head_group), kTokenBlock, 0, stream>>>(scratch.get(), q_mean.get(), lr, blocks);
        sa3_center_q<<<(elems_padded + 255) / 256, 256, 0, stream>>>(scratch.get(), q_mean.get(), scratch.get(), elems_padded, lr, blocks);
        check_stage("centering");
        sa3_quant_qk<false><<<dim3(blocks, 1, head_group), 1024, 0, stream>>>(scratch.get(), q4.get(), sfq.get(), lr, lr, false);
        check_stage("Q FP4 quantization");

        // Q is quantized, so its scratch storage can become centered K.
        sa3_center_k_pad<<<(elems_padded + 255) / 256, 256, 0, stream>>>(k_f16, k_mean.get(), scratch.get(), l, lr, head_group);
        // The long-resolution delta GEMM can otherwise select an atomic
        // reduction algorithm.  Those reductions are order-dependent on
        // Blackwell, so exclude them for this reproducibility-critical path.
        cublasAtomicsMode_t previous_atomics_mode;
        CUBLAS_CHECK(cublasGetAtomicsMode(ctx.cublas_handle(), &previous_atomics_mode));
        CUBLAS_CHECK(cublasSetAtomicsMode(ctx.cublas_handle(), CUBLAS_ATOMICS_NOT_ALLOWED));
        for (int head = 0; head < head_group; ++head) {
            const float one = 1.0f, zero = 0.0f;
            CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
            CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, lr, blocks, kHeadDim,
                &one, scratch.get() + (size_t) head * lr * kHeadDim, CUDA_R_16F, kHeadDim,
                q_mean.get() + (size_t) head * blocks * kHeadDim, CUDA_R_16F, kHeadDim,
                &zero,
                delta_f16 ? static_cast<void *>(delta_f16_buf.get() + (size_t) head * blocks * lr)
                          : static_cast<void *>(delta.get() + (size_t) head * blocks * lr),
                delta_f16 ? CUDA_R_16F : CUDA_R_32F, lr, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        CUBLAS_CHECK(cublasSetAtomicsMode(ctx.cublas_handle(), previous_atomics_mode));
        check_stage("delta GEMM");
        sa3_quant_qk<true><<<dim3(blocks, 1, head_group), 1024, 0, stream>>>(scratch.get(), k4.get(), sfk.get(), lr, lr, false);
        check_stage("K FP4 quantization");
        // K is quantized, so the same scratch storage can become padded V.
        sa3_pad_f16<<<(elems_padded + 255) / 256, 256, 0, stream>>>(v_f16, scratch.get(), l, lr, head_group);
        sa3_quant_vt<<<dim3(blocks, 1, head_group), 1024, 0, stream>>>(scratch.get(), v4.get(), sfv.get(), lr, lr, false);
        check_stage("V FP4 quantization");
        Flash_fwd_params p = {};
        p.q_ptr=q4.get(); p.k_ptr=k4.get(); p.v_ptr=v4.get(); p.delta_s_ptr=delta_f16 ? static_cast<void *>(delta_f16_buf.get()) : static_cast<void *>(delta.get()); p.sfq_ptr=sfq.get(); p.sfk_ptr=sfk.get(); p.sfv_ptr=sfv.get(); p.o_ptr=scratch.get(); p.softmax_lse_ptr=lse.get();
        p.b=1; p.h=head_group; p.h_k=head_group; p.h_h_k_ratio=1; p.seqlen_q=lr; p.seqlen_k=lr; p.unpadded_seqlen_k=l; p.seqlen_q_rounded=lr; p.seqlen_k_rounded=lr; p.d=kHeadDim; p.d_rounded=kHeadDim; p.head_divmod=cutlass::FastDivmod(head_group);
    // FP4 pointer arithmetic is in nibbles (cutlass::float_e2m1_t), whereas
    // the backing buffers above are byte-addressed.  The upstream adapter
    // therefore multiplies every Q/K/V stride by two.
    p.q_row_stride=kHeadDim; p.k_row_stride=kHeadDim; p.v_row_stride=lr; p.q_head_stride=(int64_t) lr*kHeadDim; p.k_head_stride=(int64_t) lr*kHeadDim; p.v_head_stride=(int64_t) kHeadDim*lr;
    p.q_batch_stride=(int64_t) head_group*lr*kHeadDim; p.k_batch_stride=p.q_batch_stride; p.v_batch_stride=(int64_t) head_group*kHeadDim*lr;
    p.sfq_row_stride=kHeadDim/16; p.sfk_row_stride=kHeadDim/16; p.sfv_row_stride=lr/16; p.sfq_head_stride=(int64_t) lr*kHeadDim/16; p.sfk_head_stride=p.sfq_head_stride; p.sfv_head_stride=(int64_t) kHeadDim*lr/16;
    p.sfq_batch_stride=(int64_t) head_group*lr*kHeadDim/16; p.sfk_batch_stride=p.sfq_batch_stride; p.sfv_batch_stride=(int64_t) head_group*kHeadDim*lr/16;
    p.ds_row_stride=lr; p.ds_head_stride=(int64_t) blocks*lr; p.ds_batch_stride=(int64_t) head_group*blocks*lr; p.o_row_stride=kHeadDim; p.o_head_stride=(int64_t) lr*kHeadDim; p.o_batch_stride=(int64_t) head_group*lr*kHeadDim;
    p.scale_softmax=scale; p.scale_softmax_log2=scale * M_LOG2E; p.is_causal=false; p.per_block_mean=true; p.seqlen_s=lr; p.is_bf16=false; p.is_seqlens_k_cumulative=true;
    if (timing) CUDA_CHECK(cudaEventRecord(timing_mha, stream));
    if (delta_f16) run_mha_fwd_<cutlass::nv_float4_t<cutlass::float_e2m1_t>, kHeadDim, cutlass::half_t, cutlass::half_t>(p, stream);
    else run_mha_fwd_<cutlass::nv_float4_t<cutlass::float_e2m1_t>, kHeadDim, cutlass::half_t>(p, stream);
        sa3_o_to_dst<<<(elems + 255) / 256, 256, 0, stream>>>(scratch.get(), dst->data, l, lr, head_group, head_start, total_h, dst->type == GGML_TYPE_F32);
        CUDA_CHECK(cudaGetLastError());
    }
    if (timing) {
        CUDA_CHECK(cudaEventRecord(timing_end, stream));
        CUDA_CHECK(cudaEventSynchronize(timing_end));
        float total_ms = 0.0f, prep_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, timing_begin, timing_end));
        CUDA_CHECK(cudaEventElapsedTime(&prep_ms, timing_begin, timing_mha));
        fprintf(stderr, "[sa3-timing] L=%d total=%.3fms prep=%.3fms mha+out=%.3fms\n", l, total_ms, prep_ms, total_ms - prep_ms);
    }
    static int calls = 0;
    if (calls++ < 4) fprintf(stderr, "[sa3] dispatch B=1 H=32 group=%d L=%d D=128 Q=%s delta=%s\n", head_group, l, q->type == GGML_TYPE_F32 ? "f32" : "f16", delta_f16 ? "f16" : "f32");
    return true;
}
