// Phase-1 fast FP4 GEMM via cuBLASLt — see nvfp4-cublaslt.cuh.
//
// Convention (proven in flux2.cpp/spike_cutlass_fp4/nvfp4_repack_golden.cu, cosine=1.0):
//   ggml block_nvfp4 decodes E2M1 nibbles via kvalues_mxfp4={0,1,2,3,4,6,8,12} (2x std
//   E2M1) and UE4M3 scales via /2 (ggml_ue4m3_to_fp32). The two 2x factors cancel, so
//   the stored bytes ARE the standard-e4m3 * standard-e2m1 values cuBLASLt expects.
//   => alpha = 1.0, weight bytes reused verbatim (only re-laid-out).
//
// Repacks (one-time per weight tensor, cached by device pointer):
//   - scales: contiguous block_nvfp4.d[4] -> cuBLASLt SWIZZLE_32_4_4 layout.
//   - nibbles: ggml (elem j low / j+8 high within each 16-sub) -> consecutive (2j,2j+1).
// Activation (src1 f32) is quantized per matmul into the same cuBLASLt layout.

#include "nvfp4-cublaslt.cuh"

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <mutex>
#include <unordered_map>

// SWIZZLE_32_4_4 offset over a (rows x col_length) UE4M3 scale grid (comfy float_utils)
__device__ __forceinline__ size_t swz_off(size_t row, size_t col, uint32_t col_len) {
    const uint32_t R=128, RC=32, CC=4;
    size_t rb = row/R, rem = row%R, d4 = rem/RC, d3 = rem%RC;
    size_t cbg = col/CC, d5 = col%CC;
    size_t cbg_cnt = (col_len + CC - 1)/CC;
    return ((rb*cbg_cnt + cbg)*RC + d3)*16 + d4*CC + d5;
}

// --- weight repack: one thread per (row r in [0,N), sub ss in [0,nsub)) ---
// reorders nibbles to consecutive (8 bytes) and swizzles the UE4M3 scale byte.
static __global__ void repack_weight_kernel(const block_nvfp4 * __restrict__ W,
                                            uint8_t * __restrict__ out_data,    // N*(K/2) bytes
                                            uint8_t * __restrict__ out_scales,  // swizzled
                                            int N, int K) {
    const int nsub = K/16, nblk = K/64;
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= N*nsub) return;
    const int r  = idx / nsub;
    const int ss = idx % nsub;
    const int block = ss/4, s = ss%4;
    const block_nvfp4 & b = W[(size_t)r*nblk + block];
    out_scales[swz_off(r, ss, nsub)] = b.d[s];                 // std-e4m3 byte, verbatim
    // ggml qs for this 16-sub: qs[s*8 + (local%8)], low=local<8, high=local>=8
    const uint8_t * qs = &b.qs[s*8];
    uint8_t * od = &out_data[(size_t)r*(K/2) + ss*8];
    #pragma unroll
    for (int t=0;t<8;t++) {
        int e0 = 2*t, e1 = 2*t+1;
        uint8_t n0 = (e0<8) ? (qs[e0]&0xF) : (qs[e0-8]>>4);
        uint8_t n1 = (e1<8) ? (qs[e1]&0xF) : (qs[e1-8]>>4);
        od[t] = n0 | (n1<<4);
    }
}

// --- activation quant: one thread per (row r in [0,M), sub ss in [0,nsub)) ---
// Mirrors quantize_mmq_nvfp4 EXACTLY (same ggml helpers + ±1/±2 scale-refinement
// search) so the cuBLASLt activation is bit-identical to the MMQ path; only the
// nibble *packing* differs (consecutive for cuBLASLt vs MMQ tile layout). This
// keeps cuBLASLt output as close to MMQ as the GEMM kernels themselves allow.
static __global__ void quant_act_kernel(const float * __restrict__ X,
                                        uint8_t * __restrict__ out_data,    // M*(K/2)
                                        uint8_t * __restrict__ out_scales,  // swizzled
                                        int M, int K) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int nsub = K/16;
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= M*nsub) return;
    const int r  = idx / nsub;
    const int ss = idx % nsub;
    const float * x = &X[(size_t)r*K + ss*16];
    float vals[16], amax = 0.f;
    #pragma unroll
    for (int j=0;j<16;j++) { vals[j]=x[j]; amax = fmaxf(amax, fabsf(x[j])); }

    static constexpr int test_offsets[5] = {0,-1,1,-2,2};
    const int first_code = (int) ggml_cuda_fp32_to_ue4m3(amax/6.0f);
    float best_err = FLT_MAX; uint8_t fp8_code = 0; float subblock_scale = 0.f;
    #pragma unroll
    for (int i=0;i<5;i++) {
        const int tc = first_code + test_offsets[i];
        if (tc < 0 || tc > 0x7e) continue;
        const float ts = ggml_cuda_ue4m3_to_fp32((uint8_t)tc);
        const float tinv = ts > 0.f ? 0.5f/ts : 0.f;
        float err = 0.f;
        #pragma unroll
        for (int k=0;k<16;k++) {
            const uint8_t q = ggml_cuda_float_to_fp4_e2m1(vals[k], tinv);
            const float ed = fabsf(vals[k]) - fabsf(kvalues_mxfp4[q & 0x7]) * ts;
            err = fmaf(ed, ed, err);
        }
        if (err < best_err) { best_err = err; fp8_code = (uint8_t)tc; subblock_scale = ts; }
    }
    out_scales[swz_off(r, ss, nsub)] = fp8_code;
    const float inv_scale = subblock_scale > 0.f ? 0.5f/subblock_scale : 0.f;
    uint8_t * od = &out_data[(size_t)r*(K/2) + ss*8];   // consecutive packing for cuBLASLt
    #pragma unroll
    for (int t=0;t<8;t++) {
        uint8_t n0 = ggml_cuda_float_to_fp4_e2m1(vals[2*t],   inv_scale);
        uint8_t n1 = ggml_cuda_float_to_fp4_e2m1(vals[2*t+1], inv_scale);
        od[t] = (n0 & 0xF) | ((n1 & 0xF)<<4);
    }
#else
    NO_DEVICE_CODE;
#endif
}

// ---------------- host glue ----------------

bool ggml_cuda_nvfp4_cublaslt_enabled() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("GGML_NVFP4_CUBLASLT"); v = (e && atoi(e)) ? 1 : 0; }
    return v == 1;
}

struct nvfp4_weight_repacked {
    uint8_t * data   = nullptr;   // consecutive E2M1   (in-place: offset 0 of src0->data)
    uint8_t * scales = nullptr;   // swizzled UE4M3      (in-place: offset data_bytes)
    bool      in_place = false;   // true => data/scales alias the ggml-owned src0 buffer
                                  //         (teardown must NOT cudaFree them)
};

static std::mutex                                   g_repack_mtx;
static std::unordered_map<const void*, nvfp4_weight_repacked> g_repack_cache;

// in-place repack default-ON; escape hatch GGML_NVFP4_CUBLASLT_INPLACE=0 forces the
// old out-of-place (duplicate-buffer) path for debugging / fallback.
static bool nvfp4_inplace_enabled() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("GGML_NVFP4_CUBLASLT_INPLACE"); v = (e && atoi(e)==0) ? 0 : 1; }
    return v == 1;
}

// TASK F: clear the pointer-keyed repack cache (see header). In-place entries
// (rp.in_place == true) aliased the repacked bytes into the ggml-owned src0 buffer, so
// there is nothing to cudaFree — that buffer is freed by its owner (free_params_buffer).
// Out-of-place fallback entries cudaMalloc'd duplicate data + scales buffers; free both
// so we don't leak across reloads. Hold the same mutex get_repacked_weight() uses.
void ggml_cuda_nvfp4_cublaslt_clear_repack_cache() {
    std::lock_guard<std::mutex> lk(g_repack_mtx);
    for (auto & kv : g_repack_cache) {
        nvfp4_weight_repacked & rp = kv.second;
        if (!rp.in_place) {
            // out-of-place fallback: data and scales are two separate cudaMalloc'd
            // buffers (see get_repacked_weight). Free both. In-place entries alias the
            // ggml-owned src0 buffer (freed by its owner) -> never cudaFree them.
            if (rp.data   != nullptr) cudaFree(rp.data);
            if (rp.scales != nullptr) cudaFree(rp.scales);
        }
    }
    g_repack_cache.clear();
}

// C-callable wrapper exported via ggml-cuda.h so host code (no CUDA headers) can clear
// the cache when it frees NVFP4 weight buffers.
extern "C" void ggml_cuda_nvfp4_clear_repack_cache(void) {
    ggml_cuda_nvfp4_cublaslt_clear_repack_cache();
}

static thread_local cublasLtHandle_t g_lt = nullptr;
static cublasLtHandle_t get_lt() {
    if (!g_lt) { if (cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) return nullptr; }
    return g_lt;
}

// per-tensor weight repack (cached). Returns false on failure.
static bool get_repacked_weight(const ggml_tensor * src0, int N, int K, cudaStream_t stream,
                                nvfp4_weight_repacked & out) {
    const void * key = src0->data;
    // Hold the lock across the whole (one-time, warmup) repack: the in-place path overwrites
    // src0->data, so two threads repacking the SAME key concurrently would corrupt each other.
    // Cached entries (the hot per-step path) return early in the caller-visible fast path below.
    std::lock_guard<std::mutex> lk(g_repack_mtx);
    {
        auto it = g_repack_cache.find(key);
        if (it != g_repack_cache.end()) { out = it->second; return true; }
    }
    const int nsub = K/16;
    const size_t data_bytes  = (size_t)N*(K/2);
    const size_t rb_p = ((size_t)(N+127)/128)*128;
    const size_t cb_p = ((size_t)(nsub+3)/4)*4;
    const size_t scale_bytes = rb_p*cb_p;
    const size_t repack_bytes = data_bytes + scale_bytes;

    // The DiT runs 100% on cuBLASLt (no MMQ fallback), so once a weight is repacked its
    // original block_nvfp4 layout is never read again -> the repacked bytes can live in the
    // ggml-owned buffer, eliminating the duplicate. Repack is a re-layout of the SAME values,
    // so data+scales should fit in ggml_nbytes(src0); VERIFY at runtime per-weight.
    const size_t orig_bytes = ggml_nbytes(src0);
    const bool   fits = (repack_bytes <= orig_bytes);
    const bool   want_inplace = nvfp4_inplace_enabled() && fits;

    nvfp4_weight_repacked rp;
    const int threads = 256;
    const int total   = N*nsub;

    if (want_inplace) {
        // read-after-write hazard (repack permutes both nibbles & scales) forces a transient
        // scratch: original -> scratch, then scratch -> src0->data in-place. The scratch is
        // freed immediately (never accumulates), so net VRAM cost over baseline is ~0.
        uint8_t * scratch = nullptr;
        if (cudaMalloc(&scratch, repack_bytes) != cudaSuccess) return false; // OOM: caller falls back
        uint8_t * s_data   = scratch;
        uint8_t * s_scales = scratch + data_bytes;
        cudaMemsetAsync(s_scales, 0, scale_bytes, stream);
        repack_weight_kernel<<<(total+threads-1)/threads, threads, 0, stream>>>(
            (const block_nvfp4*)src0->data, s_data, s_scales, N, K);
        if (cudaPeekAtLastError() != cudaSuccess) { cudaFree(scratch); return false; }
        // copy repacked bytes back into the ORIGINAL ggml buffer (data@0, scales@data_bytes).
        cudaMemcpyAsync(src0->data, scratch, repack_bytes, cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);   // must finish before scratch is freed
        cudaFree(scratch);
        rp.data     = (uint8_t*)src0->data;
        rp.scales   = (uint8_t*)src0->data + data_bytes;
        rp.in_place = true;
    } else {
        // out-of-place fallback (env-forced, or repack doesn't fit the original buffer):
        // hold a duplicate persistent buffer for this weight.
        if (cudaMalloc(&rp.data, data_bytes)   != cudaSuccess) return false;
        if (cudaMalloc(&rp.scales, scale_bytes)!= cudaSuccess) { cudaFree(rp.data); return false; }
        cudaMemsetAsync(rp.scales, 0, scale_bytes, stream);
        repack_weight_kernel<<<(total+threads-1)/threads, threads, 0, stream>>>(
            (const block_nvfp4*)src0->data, rp.data, rp.scales, N, K);
        if (cudaPeekAtLastError() != cudaSuccess) { cudaFree(rp.data); cudaFree(rp.scales); return false; }
        cudaStreamSynchronize(stream);
        rp.in_place = false;
        if (getenv("GGML_NVFP4_CUBLASLT_TRACE"))
            fprintf(stderr, "[NVFP4_CUBLASLT] out-of-place weight N=%d K=%d (repack %zu > orig %zu, "
                            "or inplace disabled)\n", N, K, repack_bytes, orig_bytes);
    }
    g_repack_cache[key] = rp;   // lock held for the whole function (see top)
    out = rp;
    return true;
}

bool ggml_cuda_nvfp4_cublaslt_mul_mat(ggml_backend_cuda_context & ctx,
                                      const ggml_tensor * src0,
                                      const ggml_tensor * src1,
                                      ggml_tensor * dst) {
    // only the simple 2D linear-layer case (NVFP4 weight, f32 act, f32 out)
    if (src0->type != GGML_TYPE_NVFP4 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
        return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1)
        return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst))
        return false;

    const int K = src0->ne[0];   // contraction
    const int N = src0->ne[1];   // out features
    const int M = src1->ne[1];   // tokens
    if (src1->ne[0] != K) return false;
    if (K % 64 != 0)      return false;   // need full 64-elem blocks
    if (dst->ne[0] != N || dst->ne[1] != M) return false;

    cudaStream_t stream = ctx.stream();
    cublasLtHandle_t lt = get_lt();
    if (!lt) return false;

    // 1) repacked weight (cached)
    nvfp4_weight_repacked W;
    if (!get_repacked_weight(src0, N, K, stream, W)) return false;

    // 2) quantize activation into cuBLASLt layout (pool scratch)
    const int nsub = K/16;
    const size_t a_data_bytes  = (size_t)M*(K/2);
    const size_t a_rb_p = ((size_t)(M+127)/128)*128;
    const size_t a_cb_p = ((size_t)(nsub+3)/4)*4;
    const size_t a_scale_bytes = a_rb_p*a_cb_p;
    ggml_cuda_pool_alloc<uint8_t> a_data(ctx.pool(), a_data_bytes);
    ggml_cuda_pool_alloc<uint8_t> a_scales(ctx.pool(), a_scale_bytes);
    cudaMemsetAsync(a_scales.get(), 0, a_scale_bytes, stream);
    {
        const int threads = 256;
        const int total   = M*nsub;
        quant_act_kernel<<<(total+threads-1)/threads, threads, 0, stream>>>(
            (const float*)src1->data, a_data.get(), a_scales.get(), M, K);
        if (cudaPeekAtLastError() != cudaSuccess) return false;
    }

    // 3) cuBLASLt FP4 GEMM: D[M,N] (row-major) = A_w[N,K] @ B_a[M,K]^T, alpha=1
    // cuBLAS column-major: m=N, n=M, k=K; A=weight (TN), B=activation.
    const int m=N, n=M, k=K;
    static float alpha_h = 1.0f, beta_h = 0.0f;

    cublasLtMatmulDesc_t op = nullptr;
    if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return false;
    cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
    cublasOperation_t T=CUBLAS_OP_T, Nn=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &Nn, sizeof(Nn));
    void* wsp = (void*)W.scales; void* asp = (void*)a_scales.get();
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &wsp, sizeof(wsp));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &asp, sizeof(asp));
    cublasDataType_t st = CUDA_R_32F;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &st, sizeof(st));

    cublasLtMatrixLayout_t Ad=nullptr,Bd=nullptr,Cd=nullptr,Dd=nullptr;
    cublasLtMatrixLayoutCreate(&Ad, CUDA_R_4F_E2M1, k, m, k);
    cublasLtMatrixLayoutCreate(&Bd, CUDA_R_4F_E2M1, k, n, k);
    cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32F, m, n, m);
    cublasLtMatrixLayoutCreate(&Dd, CUDA_R_32F, m, n, m);

    ggml_cuda_pool_alloc<uint8_t> ws(ctx.pool(), 32*1024*1024);
    size_t wsz = 32*1024*1024;
    cublasLtMatmulPreference_t pref=nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz));
    cublasLtMatmulHeuristicResult_t hr={}; int got=0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Dd, pref, 1, &hr, &got);

    bool ok = (hs == CUBLAS_STATUS_SUCCESS && got > 0);
    if (ok) {
        cublasStatus_t ms = cublasLtMatmul(lt, op, &alpha_h, W.data, Ad, a_data.get(), Bd,
                                           &beta_h, dst->data, Cd, dst->data, Dd,
                                           &hr.algo, ws.get(), wsz, stream);
        ok = (ms == CUBLAS_STATUS_SUCCESS);
    }

    if (ok) {
        static int n_handled = 0;
        if (n_handled++ == 0 || getenv("GGML_NVFP4_CUBLASLT_TRACE"))
            fprintf(stderr, "[NVFP4_CUBLASLT] handled mul_mat #%d  M=%d K=%d N=%d (cuBLASLt FP4 GEMM)\n",
                    n_handled, M, K, N);
    }

    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (Ad) cublasLtMatrixLayoutDestroy(Ad);
    if (Bd) cublasLtMatrixLayoutDestroy(Bd);
    if (Cd) cublasLtMatrixLayoutDestroy(Cd);
    if (Dd) cublasLtMatrixLayoutDestroy(Dd);
    if (op) cublasLtMatmulDescDestroy(op);
    return ok;
}
