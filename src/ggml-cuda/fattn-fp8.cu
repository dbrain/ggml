// Custom FP8 (e4m3) flash-attention for consumer Blackwell (sm120), v10.
// See fattn-fp8.cuh for the shape contract + env gate. Streaming online-softmax flash
// kernel: QK^T on the sm120 FP8 tensor cores (e4m3 m16n8k32 MMA in mma.cuh). Env-gated,
// OFF by default. GGML_FP8_ATTN=1 => F16 P*V (quality-safe); =2 => full-FP8 (e4m3) P*V too.
//
// GENERIC + REUSABLE: parameterized over batch N, heads H, query/key lengths Lq/Lkv
// (incl. cross-attn Lq!=Lkv) and head_dim D in {64,128}. Drops into ggml's FLASH_ATTN_EXT
// dispatch, so the same kernel accelerates LTX-2.3, Wan2.2, flux2 and the avatar rig.
//
// DATA PATH: K PRE-quantized to an e4m3 GLOBAL pool + V PRE-transposed to a pool (F16 for =1,
// e4m3 for =2), ONCE before the flash loop (per-head amax -> scale=amax/448). Q is NOT pooled:
// read once per query-tile and quantized to e4m3 in shared on the fly (Q operand is loaded ONCE
// into registers, not per k-step). The 4 warps of a block SHARE one cp.async-staged KV block
// (cooperative double buffer). K/V pools are key-padded to a Bc multiple so cp.async (no OOB
// check) never over-reads; out-of-range keys are masked to -inf in softmax.
//
// v10 (LSU lever): the ncu shows the kernel is INSTRUCTION-COUNT-bound on the LSU (top executed
// pipe 44.5%, and register-pipelining was neutral => not latency-bound). The per-k-step B-operand
// loads (K in QK, V in P*V) go through load_generic (2 scalar LDS/fragment). GGML_FP8_ATTN_LDM=1
// switches them to load_ldmatrix (1 LDSM/fragment => halves the LSU instruction count) -- this
// captures the main benefit of the reference's 16-wide-A orientation (per-step operand via
// ldmatrix) WITHOUT a full rewrite, since Q here is already loaded once (not per step). Correctness
// -safe: load_ldmatrix(tile<8,8,T>) fills the same get_i/get_j layout as load_generic (mma.cuh:1282
// compares them for speed, i.e. interchangeable). Default OFF => v9 (689 ms) byte-identical.

#include "fattn-fp8.cuh"
#include "mma.cuh"
#include "cp-async.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

using namespace ggml_cuda_mma;

#define FP8_ATTN_BR    16
#define FP8_ATTN_NW     4
#define FP8_ATTN_QROWS (FP8_ATTN_BR * FP8_ATTN_NW)

// float -> e4m3 raw byte (unsigned, for packing 4/int32 without signed-shift UB).
static __device__ __forceinline__ unsigned int f2e4m3(float v) {
    return (unsigned int)(unsigned char)__nv_fp8_e4m3(v).__x;
}

// B-operand load: ldmatrix (1 LDSM, fewer LSU instrs) or generic (scalar LDS); same tile layout.
template <bool USE_LDM, int I, int J, typename T>
static __device__ __forceinline__ void load_b(tile<I, J, T> & t, const T * p, int stride) {
    if constexpr (USE_LDM) load_ldmatrix(t, p, stride);
    else                   load_generic(t, p, stride);
}

// ---------------------------------------------------------------------------
// Pre-pass kernels (run ONCE per attention call, off the flash-loop critical path).
// ---------------------------------------------------------------------------

// Per-head e4m3 scale = amax(|X_head|)/448. One block reduces one head's [per_head_n] block.
template <typename T>
static __global__ void fp8_attn_amax_perhead(const T * __restrict__ X,
                                             float * __restrict__ scale_out, long per_head_n) {
    const long   head  = blockIdx.x;
    const T *    xh    = X + head * per_head_n;
    float        local = 0.0f;
    for (long i = threadIdx.x; i < per_head_n; i += blockDim.x) {
        local = fmaxf(local, fabsf((float) xh[i]));
    }
    __shared__ float sh[256];
    sh[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float a = sh[0];
        scale_out[head] = (a > 0.0f) ? a * (1.0f / 448.0f) : 1.0f;
    }
}

// Quantize X (F16/F32) -> e4m3 pool [HN][seq_pad][D] (per-head scale); pad rows [seq,seq_pad)=0.
template <typename T>
static __global__ void fp8_quant_pool(const T * __restrict__ X, uint8_t * __restrict__ out,
                                      const float * __restrict__ scale,
                                      long seq, long seq_pad, int D, long HN) {
    const long total = seq_pad * D * HN;
    for (long i = (long) blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long) gridDim.x * blockDim.x) {
        const long head = i / (seq_pad * D);
        const long rem  = i % (seq_pad * D);
        const long key  = rem / D;
        if (key < seq) {
            const float inv = 1.0f / scale[head];
            out[i] = (unsigned char) __nv_fp8_e4m3((float) X[head * seq * D + rem] * inv).__x;
        } else {
            out[i] = 0;
        }
    }
}

// Transpose+cast V [HN][Lkv][D] (d inner) -> Vt [HN][D][Lkv_pad] (key inner), F16 (=1 path).
template <typename T>
static __global__ void fp8_v_transpose(const T * __restrict__ V, half * __restrict__ Vt,
                                       int D, long Lkv, long Lkv_pad, long HN) {
    const long total = (long) D * Lkv_pad * HN;
    for (long i = (long) blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long) gridDim.x * blockDim.x) {
        const long key = i % Lkv_pad;
        long       r   = i / Lkv_pad;
        const int  d   = (int)(r % D);
        const long head = r / D;
        Vt[i] = (key < Lkv) ? (half)(float) V[(long) d + (long) D * key + (long) D * Lkv * head]
                            : __float2half(0.0f);
    }
}

// Transpose+quant V -> Vt e4m3 [HN][D][Lkv_pad] (per-head scale sV), for full-FP8 P*V (=2).
template <typename T>
static __global__ void fp8_v_transpose_quant(const T * __restrict__ V, uint8_t * __restrict__ Vt,
                                             const float * __restrict__ sV,
                                             int D, long Lkv, long Lkv_pad, long HN) {
    const long total = (long) D * Lkv_pad * HN;
    for (long i = (long) blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long) gridDim.x * blockDim.x) {
        const long key = i % Lkv_pad;
        long       r   = i / Lkv_pad;
        const int  d   = (int)(r % D);
        const long head = r / D;
        if (key < Lkv) {
            const float inv = 1.0f / sV[head];
            Vt[i] = (unsigned char) __nv_fp8_e4m3((float) V[(long) d + (long) D * key + (long) D * Lkv * head] * inv).__x;
        } else {
            Vt[i] = 0;
        }
    }
}

// Cooperatively cp.async one K block (e4m3) + one V block into shared. FP8_PV selects V element
// size (e4m3 1B vs F16 2B). Pure async copy, issued BEFORE compute so it overlaps the MMAs.
template <int D, bool FP8_PV, int BC>
static __device__ __forceinline__ void fp8_cp_async_kv(
        int * __restrict__ k_buf, uint8_t * __restrict__ v_buf,
        const int * __restrict__ k_head, const uint8_t * __restrict__ vt_head,
        long kv0, long Lkv_pad, int tid) {
    constexpr int DI = D / 4;
    const unsigned int k_base = ggml_cuda_cvta_generic_to_shared(k_buf);
    const unsigned int v_base = ggml_cuda_cvta_generic_to_shared(v_buf);
    #pragma unroll
    for (int c = tid; c < (BC * DI) / 4; c += FP8_ATTN_NW * 32) {
        const int i32 = c * 4;
        cp_async_cg_16<0>(k_base + (unsigned int)(i32 * 4), k_head + kv0 * DI + i32);
    }
    if constexpr (FP8_PV) {
        #pragma unroll
        for (int c = tid; c < (D * BC) / 16; c += FP8_ATTN_NW * 32) {
            const int byte = c * 16;
            const int d    = byte / BC;
            const int key  = byte % BC;
            cp_async_cg_16<0>(v_base + (unsigned int) byte, vt_head + (long) d * Lkv_pad + kv0 + key);
        }
    } else {
        const half * vh = (const half *) vt_head;
        #pragma unroll
        for (int c = tid; c < (D * BC) / 8; c += FP8_ATTN_NW * 32) {
            const int hh  = c * 8;
            const int d   = hh / BC;
            const int key = hh % BC;
            cp_async_cg_16<0>(v_base + (unsigned int)(hh * 2), vh + (long) d * Lkv_pad + kv0 + key);
        }
    }
}

// Flash kernel. Block == (32, NW): threadIdx.x = warp lane (0..31), threadIdx.y = warp.
template <int D, bool FP8_PV, int BC, bool USE_LDM>
static __global__ void __launch_bounds__(FP8_ATTN_NW * 32, (BC <= 32 ? 3 : 2)) fp8_attn_kernel(
        const void * __restrict__ q_ptr, int q_is_f16,
        const uint8_t * __restrict__ k_pool, const uint8_t * __restrict__ vt_pool,
        const float * __restrict__ sQ, const float * __restrict__ sK, const float * __restrict__ sV,
        void * __restrict__ dst, int dst_is_f16,
        int H, long Lq, long Lkv, long Lkv_pad, float softmax_scale) {
    constexpr int NKB    = D / 32;
    constexpr int NDT    = D / 8;
    constexpr int NST    = BC / 8;
    constexpr int NPV16  = BC / 16;
    constexpr int NPV32  = BC / 32;
    constexpr int DI     = D / 4;
    constexpr int VELEM  = FP8_PV ? 1 : 2;

    const int  warp = threadIdx.y;
    const int  lane = threadIdx.x;
    const int  tid  = threadIdx.y * 32 + threadIdx.x;
    const long q_base   = ((long) blockIdx.x * FP8_ATTN_NW + warp) * FP8_ATTN_BR;
    const long head_idx = (long) blockIdx.y + (long) H * blockIdx.z;

    const int     * k_head  = (const int *) k_pool + head_idx * (Lkv_pad * DI);
    const uint8_t * vt_head = vt_pool + head_idx * ((long) D * Lkv_pad * VELEM);
    const float qk_scale = sQ[head_idx] * sK[head_idx] * softmax_scale;

    __shared__ __align__(16) int     q_sh[FP8_ATTN_NW * FP8_ATTN_BR * DI];
    __shared__ __align__(16) int     k_sh[2][BC * DI];
    __shared__ __align__(16) uint8_t v_sh[2][D * BC * VELEM];
    __shared__ __align__(16) uint8_t p_sh[FP8_PV ? FP8_ATTN_NW * FP8_ATTN_BR * BC : 1];

    // Stage + quantize this warp's BR query rows to e4m3 in shared, ON THE FLY (no Q pool).
    int * q_sh_w = q_sh + warp * (FP8_ATTN_BR * DI);
    {
        const long  q_off = head_idx * Lq * D;
        const float invQ  = 1.0f / sQ[head_idx];
        for (int i = lane; i < FP8_ATTN_BR * DI; i += 32) {
            const int  r  = i / DI;
            const int  c  = i % DI;
            const long gr = q_base + r;
            unsigned int packed = 0;
            if (gr < Lq) {
                const long base = q_off + gr * (long) D + c * 4;
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const float qv = q_is_f16 ? __half2float(((const half *) q_ptr)[base + b])
                                              : ((const float *) q_ptr)[base + b];
                    packed |= f2e4m3(qv * invQ) << (b * 8);
                }
            }
            q_sh_w[i] = (int) packed;
        }
    }
    __syncwarp();

    tile<16, 8, int> qA[NKB];
    #pragma unroll
    for (int kb = 0; kb < NKB; ++kb) load_ldmatrix(qA[kb], q_sh_w + kb * 8, DI);

    float m_run[2] = { -INFINITY, -INFINITY };
    float l_run[2] = { 0.0f, 0.0f };
    tile<16, 8, float> oC[NDT];

    const long nb = (Lkv + BC - 1) / BC;

    fp8_cp_async_kv<D, FP8_PV, BC>(k_sh[0], v_sh[0], k_head, vt_head, 0, Lkv_pad, tid);
    cp_async_wait_all();
    __syncthreads();

    for (long b = 0; b < nb; ++b) {
        const int  cur = (int)(b & 1);
        const long kv0 = b * BC;

        if (b + 1 < nb) {
            fp8_cp_async_kv<D, FP8_PV, BC>(k_sh[cur ^ 1], v_sh[cur ^ 1], k_head, vt_head,
                                           (b + 1) * BC, Lkv_pad, tid);
        }

        const int * k_cur = k_sh[cur];

        // --- QK^T (FP8), register-pipelined B-loads (ldmatrix or generic per USE_LDM) ---
        tile<16, 8, float> sC[NST];
        {
            constexpr int NS = NKB * NST;   // kb-outer/nt-inner (preserves sC[nt] accum order)
            tile<8, 8, int> kB[2];
            load_b<USE_LDM>(kB[0], k_cur, DI);
            #pragma unroll
            for (int idx = 0; idx < NS; ++idx) {
                const int kb = idx / NST;
                const int nt = idx % NST;
                if (idx + 1 < NS) {
                    const int kb2 = (idx + 1) / NST;
                    const int nt2 = (idx + 1) % NST;
                    load_b<USE_LDM>(kB[(idx + 1) & 1], k_cur + (nt2 * 8) * DI + kb2 * 8, DI);
                }
                mma(sC[nt], qA[kb], kB[idx & 1]);
            }
        }

        #pragma unroll
        for (int nt = 0; nt < NST; ++nt) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int  keycol = 2 * (lane & 3) + (l & 1);
                const long gkey   = kv0 + (long) nt * 8 + keycol;
                sC[nt].x[l] = (gkey < Lkv) ? sC[nt].x[l] * qk_scale : -INFINITY;
            }
        }

        // --- online softmax ---
        float m_new[2] = { m_run[0], m_run[1] };
        #pragma unroll
        for (int nt = 0; nt < NST; ++nt) {
            m_new[0] = fmaxf(m_new[0], fmaxf(sC[nt].x[0], sC[nt].x[1]));
            m_new[1] = fmaxf(m_new[1], fmaxf(sC[nt].x[2], sC[nt].x[3]));
        }
        #pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            m_new[0] = fmaxf(m_new[0], __shfl_xor_sync(0xFFFFFFFF, m_new[0], off));
            m_new[1] = fmaxf(m_new[1], __shfl_xor_sync(0xFFFFFFFF, m_new[1], off));
        }

        float corr[2];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            corr[r]  = (m_new[r] == -INFINITY) ? 1.0f : expf(m_run[r] - m_new[r]);
            m_run[r] = m_new[r];
            l_run[r] *= corr[r];
        }
        #pragma unroll
        for (int dt = 0; dt < NDT; ++dt) {
            oC[dt].x[0] *= corr[0]; oC[dt].x[1] *= corr[0];
            oC[dt].x[2] *= corr[1]; oC[dt].x[3] *= corr[1];
        }

        float rs[2] = { 0.0f, 0.0f };
        #pragma unroll
        for (int nt = 0; nt < NST; ++nt) {
            sC[nt].x[0] = expf(sC[nt].x[0] - m_run[0]); rs[0] += sC[nt].x[0];
            sC[nt].x[1] = expf(sC[nt].x[1] - m_run[0]); rs[0] += sC[nt].x[1];
            sC[nt].x[2] = expf(sC[nt].x[2] - m_run[1]); rs[1] += sC[nt].x[2];
            sC[nt].x[3] = expf(sC[nt].x[3] - m_run[1]); rs[1] += sC[nt].x[3];
        }
        #pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            rs[0] += __shfl_xor_sync(0xFFFFFFFF, rs[0], off);
            rs[1] += __shfl_xor_sync(0xFFFFFFFF, rs[1], off);
        }
        l_run[0] += rs[0];
        l_run[1] += rs[1];

        // --- P*V ---
        if constexpr (FP8_PV) {
            uint8_t * p_sh_w = p_sh + warp * (FP8_ATTN_BR * BC);
            #pragma unroll
            for (int nt = 0; nt < NST; ++nt) {
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int q   = (l < 2) ? (lane >> 2) : (8 + (lane >> 2));
                    const int key = nt * 8 + 2 * (lane & 3) + (l & 1);
                    p_sh_w[q * BC + key] = (unsigned char) __nv_fp8_e4m3(sC[nt].x[l] * 448.0f).__x;
                }
            }
            __syncwarp();
            const int * v_cur = (const int *) v_sh[cur];
            #pragma unroll
            for (int kc = 0; kc < NPV32; ++kc) {
                tile<16, 8, int> pA;
                load_ldmatrix(pA, (const int *) p_sh_w + kc * 8, BC / 4);
                tile<8, 8, int> vB[2];
                load_b<USE_LDM>(vB[0], v_cur + kc * 8, BC / 4);
                #pragma unroll
                for (int dt = 0; dt < NDT; ++dt) {
                    if (dt + 1 < NDT) load_b<USE_LDM>(vB[(dt + 1) & 1], v_cur + ((dt + 1) * 8) * (BC / 4) + kc * 8, BC / 4);
                    mma(oC[dt], pA, vB[dt & 1]);
                }
            }
        } else {
            const half2 * v_cur_h2 = (const half2 *) v_sh[cur];
            #pragma unroll
            for (int pc = 0; pc < NPV16; ++pc) {
                tile<16, 8, half2> pA;
                pA.x[0] = __floats2half2_rn(sC[2 * pc + 0].x[0], sC[2 * pc + 0].x[1]);
                pA.x[1] = __floats2half2_rn(sC[2 * pc + 0].x[2], sC[2 * pc + 0].x[3]);
                pA.x[2] = __floats2half2_rn(sC[2 * pc + 1].x[0], sC[2 * pc + 1].x[1]);
                pA.x[3] = __floats2half2_rn(sC[2 * pc + 1].x[2], sC[2 * pc + 1].x[3]);
                tile<8, 8, half2> vB[2];
                load_b<USE_LDM>(vB[0], v_cur_h2 + pc * 8, BC / 2);
                #pragma unroll
                for (int dt = 0; dt < NDT; ++dt) {
                    if (dt + 1 < NDT) load_b<USE_LDM>(vB[(dt + 1) & 1], v_cur_h2 + (long)((dt + 1) * 8) * (BC / 2) + pc * 8, BC / 2);
                    mma(oC[dt], pA, vB[dt & 1]);
                }
            }
        }

        cp_async_wait_all();
        __syncthreads();
    }

    // Finalize O /= rowsum (=2 also rescales by sV/448) and scatter to dst (BSHD).
    const float o_scale = FP8_PV ? (sV[head_idx] * (1.0f / 448.0f)) : 1.0f;
    float inv_l[2];
    inv_l[0] = (l_run[0] > 0.0f) ? o_scale / l_run[0] : 0.0f;
    inv_l[1] = (l_run[1] > 0.0f) ? o_scale / l_run[1] : 0.0f;

    const long H_l = H;
    const long n   = blockIdx.z;
    const long h   = blockIdx.y;
    #pragma unroll
    for (int dt = 0; dt < NDT; ++dt) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const long qrow = q_base + ((l < 2) ? (lane >> 2) : (8 + (lane >> 2)));
            if (qrow >= Lq) continue;
            const int   d = dt * 8 + 2 * (lane & 3) + (l & 1);
            const float v = oC[dt].x[l] * inv_l[(l < 2) ? 0 : 1];
            const long  off = (((n * Lq + qrow) * H_l + h) * D) + d;
            if (dst_is_f16) ((half  *) dst)[off] = __float2half(v);
            else            ((float *) dst)[off] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// Host entry.
// ---------------------------------------------------------------------------

template <int D, bool FP8_PV, int BC, bool USE_LDM>
static void fp8_attn_launch(const void * q_ptr, int q_is_f16, const uint8_t * k_pool, const uint8_t * vt_pool,
                            const float * sQ, const float * sK, const float * sV, ggml_tensor * dst,
                            int H, long Lq, long Lkv, long Lkv_pad, long N, float softmax_scale,
                            cudaStream_t stream) {
    dim3 grid((unsigned)((Lq + FP8_ATTN_QROWS - 1) / FP8_ATTN_QROWS), (unsigned) H, (unsigned) N);
    dim3 block(32, FP8_ATTN_NW);
    fp8_attn_kernel<D, FP8_PV, BC, USE_LDM><<<grid, block, 0, stream>>>(
        q_ptr, q_is_f16, k_pool, vt_pool, sQ, sK, sV,
        dst->data, dst->type == GGML_TYPE_F16 ? 1 : 0, H, Lq, Lkv, Lkv_pad, softmax_scale);
}

void ggml_cuda_flash_attn_ext_fp8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const long D   = Q->ne[0];
    const long Lq  = Q->ne[1];
    const long H   = Q->ne[2];
    const long N   = Q->ne[3];
    const long Lkv = K->ne[1];

    GGML_ASSERT(D == 64 || D == 128);
    GGML_ASSERT(K->ne[0] == D && V->ne[0] == D && V->ne[1] == Lkv);
    GGML_ASSERT(K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(Q) && ggml_is_contiguous(K) && ggml_is_contiguous(V));
    GGML_ASSERT(Q->type == GGML_TYPE_F16 || Q->type == GGML_TYPE_F32);

    int mode = 1;
    { const char * e = getenv("GGML_FP8_ATTN"); if (e) { mode = atoi(e); if (mode < 1) mode = 1; } }
    const bool full_fp8 = (mode >= 2);

    int bc = 32;
    { const char * e = getenv("GGML_FP8_ATTN_BC"); if (e && atoi(e) == 64 && full_fp8) bc = 64; }

    // GGML_FP8_ATTN_LDM=1 => B-operand loads via ldmatrix (=2 only). Default 0 => v9 load_generic.
    bool use_ldm = false;
    { const char * e = getenv("GGML_FP8_ATTN_LDM"); if (e && atoi(e) && full_fp8) use_ldm = true; }

    float softmax_scale = 0.0f;
    memcpy(&softmax_scale, (const float *) dst->op_params + 0, sizeof(float));

    cudaStream_t stream = ctx.stream();
    const void * q_ptr  = Q->data;
    const int    q_f16  = (Q->type == GGML_TYPE_F16) ? 1 : 0;
    const long   HN      = H * N;
    const long   Lkv_pad = ((Lkv + bc - 1) / bc) * bc;
    const long   nkv     = D * Lkv_pad * HN;

    ggml_cuda_pool_alloc<float>   sQ(ctx.pool(), (size_t) HN);
    ggml_cuda_pool_alloc<float>   sK(ctx.pool(), (size_t) HN);
    ggml_cuda_pool_alloc<float>   sV(ctx.pool(), (size_t) HN);
    ggml_cuda_pool_alloc<uint8_t> k_pool(ctx.pool(),  (size_t) nkv);
    ggml_cuda_pool_alloc<uint8_t> vt_pool(ctx.pool(), (size_t)(nkv * (full_fp8 ? 1 : 2)));

    const int qb = 256;
    if (q_f16) fp8_attn_amax_perhead<half> <<<(unsigned) HN, 256, 0, stream>>>((const half *)  Q->data, sQ.get(), Lq * D);
    else       fp8_attn_amax_perhead<float><<<(unsigned) HN, 256, 0, stream>>>((const float *) Q->data, sQ.get(), Lq * D);
    fp8_attn_amax_perhead<half><<<(unsigned) HN, 256, 0, stream>>>((const half *) K->data, sK.get(), Lkv * D);
    fp8_quant_pool<half><<<1024, qb, 0, stream>>>((const half *) K->data, k_pool.get(), sK.get(), Lkv, Lkv_pad, (int) D, HN);

    if (full_fp8) {
        fp8_attn_amax_perhead<half><<<(unsigned) HN, 256, 0, stream>>>((const half *) V->data, sV.get(), Lkv * D);
        fp8_v_transpose_quant<half><<<1024, qb, 0, stream>>>((const half *) V->data, vt_pool.get(), sV.get(), (int) D, Lkv, Lkv_pad, HN);
        #define FP8_LAUNCH2(BCV, LDMV) \
            do { if (D == 128) fp8_attn_launch<128, true, BCV, LDMV>(q_ptr, q_f16, k_pool.get(), vt_pool.get(), sQ.get(), sK.get(), sV.get(), dst, (int) H, Lq, Lkv, Lkv_pad, N, softmax_scale, stream); \
                 else          fp8_attn_launch< 64, true, BCV, LDMV>(q_ptr, q_f16, k_pool.get(), vt_pool.get(), sQ.get(), sK.get(), sV.get(), dst, (int) H, Lq, Lkv, Lkv_pad, N, softmax_scale, stream); } while (0)
        if (bc == 64) { if (use_ldm) FP8_LAUNCH2(64, true); else FP8_LAUNCH2(64, false); }
        else          { if (use_ldm) FP8_LAUNCH2(32, true); else FP8_LAUNCH2(32, false); }
        #undef FP8_LAUNCH2
    } else {
        fp8_v_transpose<half><<<1024, qb, 0, stream>>>((const half *) V->data, (half *) vt_pool.get(), (int) D, Lkv, Lkv_pad, HN);
        if (D == 128) fp8_attn_launch<128, false, 32, false>(q_ptr, q_f16, k_pool.get(), vt_pool.get(), sQ.get(), sK.get(), sV.get(), dst, (int) H, Lq, Lkv, Lkv_pad, N, softmax_scale, stream);
        else          fp8_attn_launch< 64, false, 32, false>(q_ptr, q_f16, k_pool.get(), vt_pool.get(), sQ.get(), sK.get(), sV.get(), dst, (int) H, Lq, Lkv, Lkv_pad, N, softmax_scale, stream);
    }
}
