#include "im2col.cuh"

#include <cstring>  // memcmp (LONGCAT_IM2COL_VERIFY)

#define MAX_GRIDDIM_Y 65535
#define MAX_GRIDDIM_Z 65535

template <typename T>
static  __global__ void im2col_kernel(
        const float * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH,
        int64_t IC_IH_IW, int64_t IH_IW, int64_t N_OH, int64_t KH_KW, int64_t IC_KH_KW,
        int s0, int s1, int p0, int p1, int d0, int d1) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC_KH_KW) {
        return;
    }

    const int64_t iic = i / (KH_KW);
    const int64_t rem = i - iic * KH_KW;
    const int64_t ikh = rem / KW;
    const int64_t ikw = rem - ikh * KW;

    for (int64_t iow = blockIdx.y; iow < OW; iow += MAX_GRIDDIM_Y) {
        for (int64_t iz = blockIdx.z; iz < N_OH; iz += MAX_GRIDDIM_Z) {
            const int64_t  in = iz / OH;
            const int64_t  ioh = iz - in * OH;

            const int64_t iiw = iow * s0 + ikw * d0 - p0;
            const int64_t iih = ioh * s1 + ikh * d1 - p1;

            const int64_t offset_dst =
                ((in * OH + ioh) * OW + iow) * IC_KH_KW + iic * KH_KW + ikh * KW + ikw;

            if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                dst[offset_dst] = 0.0f;
            } else {
                const int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
                dst[offset_dst] = x[offset_src + iih * IW + iiw];
            }
        }
    }

    GGML_UNUSED(IC);
    GGML_UNUSED(KH);
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
template <typename T>
static void im2col_cuda(const float * x, T* dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {
    const int64_t IC_KH_KW = IC * KH * KW;
    const int64_t num_blocks = (IC_KH_KW + CUDA_IM2COL_BLOCK_SIZE - 1) / CUDA_IM2COL_BLOCK_SIZE;
    const int64_t N_OH = N * OH;
    const int64_t KH_KW = KW*KH;
    dim3 block_nums(num_blocks, MIN(OW, MAX_GRIDDIM_Y), MIN(N_OH, MAX_GRIDDIM_Z));
    im2col_kernel<<<block_nums, MIN(IC_KH_KW, CUDA_IM2COL_BLOCK_SIZE) , 0, stream>>>(x, dst, IC, IW, IH, OH, OW, KW, KH,
                                                                                     IC_IH_IW, IH_IW, N_OH, KH_KW, IC_KH_KW,
                                                                                     s0, s1, p0, p1, d0, d1);
}

static void im2col_cuda_f16(const float * x, half * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {

    im2col_cuda<half>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

static void im2col_cuda_f32(const float * x, float * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {

    im2col_cuda<float>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

void ggml_cuda_op_im2col(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t*)(dst->op_params))[5];

    const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

    const int64_t IC = src1->ne[is_2D ? 2 : 1];
    const int64_t IH = is_2D ? src1->ne[1] : 1;
    const int64_t IW =         src1->ne[0];

    const int64_t KH = is_2D ? src0->ne[1] : 1;
    const int64_t KW =         src0->ne[0];

    const int64_t OH = is_2D ? dst->ne[2] : 1;
    const int64_t OW =         dst->ne[1];

    const int64_t IC_IH_IW = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    const int64_t N        = src1->ne[is_2D ? 3 : 2];
    const int64_t IH_IW    = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    if(dst->type == GGML_TYPE_F16) {
        im2col_cuda_f16(src1_d, (half *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    } else {
        im2col_cuda_f32(src1_d, (float *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
template <typename T>
static  __global__ void im2col_3d_kernel(
        const float * src, T * dst,
        int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
        int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
        int64_t OH_OW, int64_t KD_KH_KW, int64_t ID_IH_IW, int64_t KH_KW, int64_t IH_IW, int64_t IC_ID_IH_IW,
        int64_t IC_KD_KH_KW, int64_t OW_KD_KH_KW, int64_t OD_OH_OW_IC_KD_KH_KW, int64_t OH_OW_IC_KD_KH_KW,
        int64_t OW_IC_KD_KH_KW, int64_t N_OD_OH, int64_t OD_OH,
        int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
        int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2,
        uint3 fd_kdkhkw, uint3 fd_khkw, uint3 fd_kw, uint3 fd_odoh, uint3 fd_oh) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC_KD_KH_KW) {
        return;
    }
    GGML_UNUSED(N); GGML_UNUSED(OC); GGML_UNUSED(OH_OW); GGML_UNUSED(OD); GGML_UNUSED(OW); GGML_UNUSED(KD); GGML_UNUSED(KH);
    GGML_UNUSED(ID_IH_IW); GGML_UNUSED(IH_IW); GGML_UNUSED(IC_ID_IH_IW); GGML_UNUSED(OW_KD_KH_KW);
    GGML_UNUSED(KD_KH_KW); GGML_UNUSED(KH_KW); GGML_UNUSED(KW);

    // Decompose i -> (iic, ikd, ikh, ikw) via fast (multiply-shift) division.
    // The 3060 has no hw int64 divide; the old int64 div/mod chain was the bottleneck
    // (im2col_3d was ALU-bound at ~7% of mem BW). fast_div_modulo is bit-exact here
    // (all indices/divisors fit u32), so the im2col output is unchanged. (lap-21)
    const uint2 dm_ic = fast_div_modulo((uint32_t) i,    fd_kdkhkw);   // iic, rem(kd*kh*kw)
    const uint2 dm_kd = fast_div_modulo(dm_ic.y,         fd_khkw);     // ikd, rem(kh*kw)
    const uint2 dm_kh = fast_div_modulo(dm_kd.y,         fd_kw);       // ikh, ikw
    const int64_t iic = dm_ic.x;
    const int64_t ikd = dm_kd.x;
    const int64_t ikh = dm_kh.x;
    const int64_t ikw = dm_kh.y;

    for (int64_t iow = blockIdx.y; iow < OW; iow += MAX_GRIDDIM_Y) {
        for (int64_t iz = blockIdx.z; iz < N_OD_OH; iz += MAX_GRIDDIM_Z) {
            const uint2 dm_n  = fast_div_modulo((uint32_t) iz, fd_odoh);  // in, rem(od*oh)
            const uint2 dm_d  = fast_div_modulo(dm_n.y,        fd_oh);    // iod, ioh
            const int64_t in  = dm_n.x;
            const int64_t iod = dm_d.x;
            const int64_t ioh = dm_d.y;

            const int64_t iiw = iow * s0 + ikw * d0 - p0;
            const int64_t iih = ioh * s1 + ikh * d1 - p1;
            const int64_t iid = iod * s2 + ikd * d2 - p2;

            const int64_t offset_dst = in*OD_OH_OW_IC_KD_KH_KW + iod*OH_OW_IC_KD_KH_KW + ioh*OW_IC_KD_KH_KW + iow*IC_KD_KH_KW + iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw;

            if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                dst[offset_dst] = 0.0f;
            } else {
                const int64_t offset_src = ((in * IC + iic) * stride_q) + (iid * stride_z) + (iih * stride_y) + (iiw * stride_x);
                dst[offset_dst] = src[offset_src];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// lap-23: shared-memory halo-tiled im2col_3d (fast path: stride=1, dilation=1,
// pad=0 — the Wan-VAE CausalConv3d shapes, which pre-pad the input externally
// so the im2col itself sees pad=0 ⇒ every tap is in-bounds, no boundary checks).
//
// The fastdiv kernel (above) is gather-read-latency bound: each output tap does
// an independent, cache-line-wasteful global load of the 3x3x3xICh neighborhood
// (~51 GB/s = 14% of peak). Here a block stages the reused input window for a
// (channel-block CB x all-OD x TOHxTOW) output tile into shared memory ONCE
// (coalesced global read), then threads write the im2col columns from smem.
// Input is re-read only ~1.4x (halo overlap) instead of the ~10x cache-line
// amplification of the scattered gather; writes stay coalesced (the contiguous
// dst column slab CB*KD*KH*KW is owned by consecutive threadIdx.x).
//
// Thread layout: blockDim.x = (CB*KD*KH*KW)/VEC (each thread owns VEC consecutive
// entries of one output's CB-channel column slab, coalesced), blockDim.y = P
// output positions processed concurrently (all sharing the staged smem). Each
// (tx,ty) loops the tile's OD*TOH*TOW outputs. VEC=2 (F16 only) packs the two
// consecutive F16 column entries into one 32-bit `half2` store — widening the
// per-warp write transaction 64B→128B, the lever past the ~155 GB/s scalar-store
// plateau. Falls back (host returns false) on any shape it can't cover.
// Bit-exact vs the fastdiv kernel (pure data movement, no arithmetic on values).
template <typename T, int VEC, bool HAS_PAD>
static __global__ void im2col_3d_tiled_kernel(
        const float * __restrict__ src, T * __restrict__ dst,
        int IC, int ID, int IH, int IW,
        int KD, int KH, int KW, int OD, int OH, int OW,
        int CB, int TOH, int TOW, int p0, int p1,
        int KH_KW, int KD_KH_KW,
        int64_t IC_KD_KH_KW, int64_t OW_IC_KD_KH_KW, int64_t OH_OW_IC_KD_KH_KW, int64_t OD_OH_OW_IC_KD_KH_KW,
        int num_cblocks, int n_oh_tiles, int n_ow_tiles,
        uint3 fd_kdkhkw, uint3 fd_khkw, uint3 fd_kw, uint3 fd_tow, uint3 fd_towtoh,
        uint3 fd_ncblocks, uint3 fd_nohtiles) {
    extern __shared__ char smem_raw[];
    float * smem = (float *) smem_raw;

    const int SH = TOH + KH - 1;          // staged input rows  (halo)
    const int SW = TOW + KW - 1;          // staged input cols  (halo)
    const int DHW = ID * SH * SW;         // smem stride per channel

    // Decode blockIdx.z -> (n, channel-block); blockIdx.y -> (?, oh-tile) [n folded in z]
    const int bz = blockIdx.z;
    const int in = fastdiv((uint32_t) bz, fd_ncblocks);
    const int cb = bz - in * num_cblocks;
    const int cb0 = cb * CB;
    const int CBeff = min(CB, IC - cb0);

    const int oh0 = blockIdx.y * TOH;
    const int ow0 = blockIdx.x * TOW;

    const int tid    = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;

    // ---- stage input window into smem (coalesced over the W/col dim) ----
    const int64_t src_chan_base = (int64_t)(in * IC + cb0) * ID * IH * IW;
    const int smem_elems = CBeff * DHW;
    for (int e = tid; e < smem_elems; e += nthreads) {
        int t = e;
        const int sc  = t % SW;  t /= SW;
        const int sr  = t % SH;  t /= SH;
        const int iid = t % ID;  t /= ID;
        const int icl = t;                       // < CBeff
        // pad=0 (Wan VAE pre-padded): the staged window starts at the output tile
        // origin and only the top/right halo can run off the input (zeroed). With
        // spatial padding (LTX VAE, p0/p1>0) the window starts p1/p0 before the
        // origin, so the bottom/left halo can also be out of bounds. HAS_PAD keeps
        // the pad=0 hot path byte-identical (the lower-bound checks compile out).
        int iih, iiw;
        bool in_bounds;
        if constexpr (HAS_PAD) {
            iih = oh0 + sr - p1;
            iiw = ow0 + sc - p0;
            in_bounds = (iih >= 0 && iih < IH && iiw >= 0 && iiw < IW);
        } else {
            iih = oh0 + sr;
            iiw = ow0 + sc;
            in_bounds = (iih < IH && iiw < IW);
        }
        float v = 0.0f;
        if (in_bounds) {
            v = src[src_chan_base + (int64_t)icl * ID * IH * IW + (int64_t)iid * IH * IW + (int64_t)iih * IW + iiw];
        }
        smem[(int64_t)icl * DHW + ((int64_t)iid * SH + sr) * SW + sc] = v;
    }
    __syncthreads();

    // ---- decode this thread's VEC consecutive column-slab entries ----
    const int c0 = threadIdx.x * VEC;            // block-local column start
    int s_icl[VEC], s_ikd[VEC], s_ikh[VEC], s_ikw[VEC];
    bool any_active = false;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
        const uint2 d_icl = fast_div_modulo((uint32_t)(c0 + j), fd_kdkhkw);
        const uint2 d_ikd = fast_div_modulo(d_icl.y,            fd_khkw);
        const uint2 d_ikh = fast_div_modulo(d_ikd.y,            fd_kw);
        s_icl[j] = d_icl.x; s_ikd[j] = d_ikd.x; s_ikh[j] = d_ikh.x; s_ikw[j] = d_ikh.y;
        any_active |= (d_icl.x < CBeff);
    }
    const int64_t col = (int64_t) cb0 * KD_KH_KW + c0;   // contiguous dst column index

    // ---- iterate the tile's output positions (each written from smem) ----
    const int n_out = OD * TOH * TOW;
    for (int o = threadIdx.y; o < n_out; o += blockDim.y) {
        const uint2 d_iod = fast_div_modulo((uint32_t) o, fd_towtoh);  // iod, rem(toh*tow)
        const uint2 d_loh = fast_div_modulo(d_iod.y,      fd_tow);     // loh, low
        const int iod = d_iod.x;
        const int loh = d_loh.x;
        const int low = d_loh.y;
        const int oh  = oh0 + loh;
        const int ow  = ow0 + low;
        if (!any_active || oh >= OH || ow >= OW) {
            continue;
        }
        const int64_t base = (int64_t) in * OD_OH_OW_IC_KD_KH_KW
                           + (int64_t) iod * OH_OW_IC_KD_KH_KW
                           + (int64_t) oh  * OW_IC_KD_KH_KW
                           + (int64_t) ow  * IC_KD_KH_KW
                           + col;
        // smem[icl][iid=iod+ikd][srow=loh+ikh][scol=low+ikw]
        float v[VEC];
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            v[j] = (s_icl[j] < CBeff)
                 ? smem[(int64_t)s_icl[j] * DHW + ((int64_t)(iod + s_ikd[j]) * SH + (loh + s_ikh[j])) * SW + (low + s_ikw[j])]
                 : 0.0f;
        }
        if constexpr (VEC == 2 && sizeof(T) == 2) {  // F16: one aligned half2 store
            __half2 h2 = __floats2half2_rn(v[0], v[1]);
            *reinterpret_cast<__half2 *>(&dst[base]) = h2;
        } else {
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                if (s_icl[j] < CBeff) dst[base + j] = v[j];
            }
        }
    }
    GGML_UNUSED(n_oh_tiles); GGML_UNUSED(n_ow_tiles);
    GGML_UNUSED(fd_nohtiles);
}

// Launch the tiled fast path. Returns false (no launch) for shapes it doesn't
// cover, so the caller falls back to the fastdiv kernel (correctness can't regress).
template <typename T>
static bool im2col_3d_tiled_try(const float * src, T * dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, cudaStream_t stream) {
    // Fast path: unit stride/dilation, zero DEPTH pad (the temporal/causal pad is
    // applied externally so OD=ID-KD+1). SPATIAL padding p0/p1 is handled in-kernel
    // (HAS_PAD): the Wan VAE pre-pads ⇒ p0=p1=0 (byte-identical hot path), the LTX
    // VAE passes p0=p1=1 ⇒ OH=IH, OW=IW. lap-D: cover that case instead of dumping
    // every LTX conv (37 s / VAE) onto the ALU-bound fastdiv gather.
    if (!(s0 == 1 && s1 == 1 && s2 == 1 && d0 == 1 && d1 == 1 && d2 == 1 && p2 == 0)) {
        return false;
    }
    const bool has_pad = (p0 != 0 || p1 != 0);
    // The spatial-pad extension is on by default; opt out (e.g. for an A/B against
    // the fastdiv fallback) with LONGCAT_IM2COL_TILE_PAD=0. pad=0 callers are never
    // affected by this gate.
    if (has_pad) {
        const char* g = getenv("LONGCAT_IM2COL_TILE_PAD");
        if (g != nullptr && g[0] == '0') return false;
    }
    if (getenv("LONGCAT_IM2COL_NOTILE")) {
        return false;
    }

    // Tunable tile (env LONGCAT_IM2COL_TILE="CB,TOH,TOW,P"); default measured-good
    // (sm_86 / RTX 3060, lap-23 sweep). VEC=2 packs two F16 column entries into a
    // half2 store (env LONGCAT_IM2COL_NOVEC2 forces the scalar store path).
    int CB = 8, TOH = 8, TOW = 8, P = 4;
    if (const char * t = getenv("LONGCAT_IM2COL_TILE")) {
        sscanf(t, "%d,%d,%d,%d", &CB, &TOH, &TOW, &P);
    }
    const int KD_KH_KW = (int)(KD * KH * KW);
    const int cols = CB * KD_KH_KW;              // CB-channel contiguous dst column slab
    // VEC=2 requires F16 dst, an even slab, and IC%CB==0 (every block full ⇒ the
    // half2 pair never straddles into an out-of-range channel).
    const int VEC = (sizeof(T) == 2 && (cols % 2) == 0 && (IC % CB) == 0 &&
                     !getenv("LONGCAT_IM2COL_NOVEC2")) ? 2 : 1;
    const int tpx = cols / VEC;                  // blockDim.x
    if (tpx > 1024 || (int64_t) tpx * P > 1024) {
        return false;
    }
    // smem stages the full input depth (ID) for a CB-channel x SHxSW spatial window.
    // The LTX VAE has large ID (e.g. 34) which busts the 48 KB opt-out-free limit at
    // the default 8x8 spatial tile; shrink the spatial tile (square-ish) until it
    // fits rather than dumping the conv on the slow fallback. CB/KD_KH_KW are fixed
    // (they set tpx/VEC), so shrinking TOH/TOW only trims smem + the P output loop.
    auto smem_for = [&](int toh, int tow) {
        const int sh = toh + (int) KH - 1, sw = tow + (int) KW - 1;
        return (size_t) CB * ID * sh * sw * sizeof(float);
    };
    // Scoped to the new spatial-pad path so the pad=0 hot path (Wan VAE / avatar)
    // keeps its exact prior tile + bail behavior.
    if (has_pad) {
        while ((TOH > 1 || TOW > 1) && smem_for(TOH, TOW) > 48 * 1024) {
            if (TOW >= TOH && TOW > 1) TOW--; else if (TOH > 1) TOH--; else break;
        }
    }
    const int SH = TOH + (int) KH - 1;
    const int SW = TOW + (int) KW - 1;
    const size_t smem_bytes = (size_t) CB * ID * SH * SW * sizeof(float);
    if (smem_bytes > 48 * 1024) {                // stay within the default opt-out-free limit
        return false;
    }

    const int num_cblocks = (int)((IC + CB - 1) / CB);
    const int n_oh_tiles  = (int)((OH + TOH - 1) / TOH);
    const int n_ow_tiles  = (int)((OW + TOW - 1) / TOW);
    const int64_t gz = N * num_cblocks;
    if (n_ow_tiles > MAX_GRIDDIM_Y || n_oh_tiles > MAX_GRIDDIM_Y || gz > MAX_GRIDDIM_Z) {
        return false;
    }

    const int64_t IC_KD_KH_KW          = IC * KD * KH * KW;
    const int64_t OW_IC_KD_KH_KW       = OW * IC_KD_KH_KW;
    const int64_t OH_OW_IC_KD_KH_KW    = OH * OW_IC_KD_KH_KW;
    const int64_t OD_OH_OW_IC_KD_KH_KW = OD * OH_OW_IC_KD_KH_KW;

    const uint3 fd_kdkhkw  = init_fastdiv_values((uint64_t) KD_KH_KW);
    const uint3 fd_khkw    = init_fastdiv_values((uint64_t)(KH * KW));
    const uint3 fd_kw      = init_fastdiv_values((uint64_t) KW);
    const uint3 fd_tow     = init_fastdiv_values((uint64_t) TOW);
    const uint3 fd_towtoh  = init_fastdiv_values((uint64_t)(TOW * TOH));
    const uint3 fd_ncblocks= init_fastdiv_values((uint64_t) num_cblocks);
    const uint3 fd_nohtiles= init_fastdiv_values((uint64_t) n_oh_tiles);

    dim3 grid(n_ow_tiles, n_oh_tiles, (unsigned) gz);
    dim3 block(tpx, P, 1);
#define LC_TILED_LAUNCH(VECN, HASPAD) \
    im2col_3d_tiled_kernel<T, VECN, HASPAD><<<grid, block, smem_bytes, stream>>>( \
        src, dst, (int)IC, (int)ID, (int)IH, (int)IW, \
        (int)KD, (int)KH, (int)KW, (int)OD, (int)OH, (int)OW, \
        CB, TOH, TOW, p0, p1, (int)(KH * KW), KD_KH_KW, \
        IC_KD_KH_KW, OW_IC_KD_KH_KW, OH_OW_IC_KD_KH_KW, OD_OH_OW_IC_KD_KH_KW, \
        num_cblocks, n_oh_tiles, n_ow_tiles, \
        fd_kdkhkw, fd_khkw, fd_kw, fd_tow, fd_towtoh, fd_ncblocks, fd_nohtiles)
    if (has_pad) {
        if (VEC == 2) { LC_TILED_LAUNCH(2, true); } else { LC_TILED_LAUNCH(1, true); }
    } else {
        if (VEC == 2) { LC_TILED_LAUNCH(2, false); } else { LC_TILED_LAUNCH(1, false); }
    }
#undef LC_TILED_LAUNCH
    return true;
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
template <typename T>
static void im2col_3d_cuda(const float * src, T* dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, cudaStream_t stream) {
    const int64_t OH_OW = OH*OW;
    const int64_t KD_KH_KW = KD*KH*KW;
    const int64_t ID_IH_IW = ID*IH*IW;
    const int64_t KH_KW = KH*KW;
    const int64_t IH_IW = IH*IW;
    const int64_t IC_KD_KH_KW = IC*KD*KH*KW;
    const int64_t OW_KD_KH_KW = OW*KD*KH*KW;
    const int64_t N_OD_OH = N*OD*OH;
    const int64_t OD_OH = OD*OH;
    const int64_t IC_ID_IH_IW = IC*ID*IH*IW;
    const int64_t OD_OH_OW_IC_KD_KH_KW = OD*OH*OW*IC*KD*KH*KW;
    const int64_t OH_OW_IC_KD_KH_KW = OH*OW*IC*KD*KH*KW;
    const int64_t OW_IC_KD_KH_KW = OW*IC*KD*KH*KW;
    const int64_t num_blocks = (IC_KD_KH_KW + CUDA_IM2COL_BLOCK_SIZE - 1) / CUDA_IM2COL_BLOCK_SIZE;
    dim3 block_nums(num_blocks, MIN(OW, MAX_GRIDDIM_Y), MIN(N_OD_OH, MAX_GRIDDIM_Z));
    // Precompute multiply-shift divisors for the kernel's index decomposition (loop-invariant).
    const uint3 fd_kdkhkw = init_fastdiv_values((uint64_t) KD_KH_KW);
    const uint3 fd_khkw   = init_fastdiv_values((uint64_t) KH_KW);
    const uint3 fd_kw     = init_fastdiv_values((uint64_t) KW);
    const uint3 fd_odoh   = init_fastdiv_values((uint64_t) OD_OH);
    const uint3 fd_oh     = init_fastdiv_values((uint64_t) OH);
    // [IM2COL_PROF] env-gated per-call shape/bandwidth probe (LONGCAT_IM2COL_PROF).
    // Times THIS im2col_3d launch and reports achieved write-bandwidth vs the GPU peak
    // so we can tell if the op is a memory-bound materialization at peak (=> only an
    // implicit-GEMM / direct conv beats it) or an improvable kernel. Adds a sync; off by default.
    // fastdiv kernel launch (the universal fallback path).
    auto launch_fastdiv = [&](T * out) {
        im2col_3d_kernel<<<block_nums, MIN(IC_KD_KH_KW, CUDA_IM2COL_BLOCK_SIZE) , 0, stream>>>(src, out, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                                                                                           OH_OW, KD_KH_KW, ID_IH_IW, KH_KW, IH_IW, IC_ID_IH_IW,
                                                                                           IC_KD_KH_KW, OW_KD_KH_KW, OD_OH_OW_IC_KD_KH_KW,
                                                                                           OH_OW_IC_KD_KH_KW, OW_IC_KD_KH_KW, N_OD_OH, OD_OH,
                                                                                           stride_q, stride_z, stride_y, stride_x,
                                                                                           s0, s1, s2, p0, p1, p2, d0, d1, d2,
                                                                                           fd_kdkhkw, fd_khkw, fd_kw, fd_odoh, fd_oh);
    };

    const bool lc_im2col_prof = getenv("LONGCAT_IM2COL_PROF") != nullptr;
    cudaEvent_t lc_ev0, lc_ev1;
    if (lc_im2col_prof) { cudaEventCreate(&lc_ev0); cudaEventCreate(&lc_ev1); cudaEventRecord(lc_ev0, stream); }

    // lap-23: try the smem-tiled fast path; fall back to fastdiv on uncovered shapes.
    const bool used_tiled = im2col_3d_tiled_try<T>(src, dst, N, IC, ID, IH, IW, KD, KH, KW, OD, OH, OW,
                                                   s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    if (!used_tiled) {
        launch_fastdiv(dst);
    }

    if (lc_im2col_prof) {
        cudaEventRecord(lc_ev1, stream); cudaEventSynchronize(lc_ev1);
        float lc_ms = 0.0f; cudaEventElapsedTime(&lc_ms, lc_ev0, lc_ev1);
        const double out_elems = (double)OD_OH_OW_IC_KD_KH_KW * (double)N;   // im2col output elements
        const double wbytes    = out_elems * (double)sizeof(T);             // write traffic (dominant)
        const double rbytes    = (double)IC_ID_IH_IW * (double)N * 4.0;     // input read once (F32)
        const double gbps_w    = lc_ms > 0 ? (wbytes / 1e9) / (lc_ms / 1e3) : 0.0;
        fprintf(stderr, "[IM2COL_PROF] %s N=%ld IC=%ld ID=%ld IH=%ld IW=%ld K=%ldx%ldx%ld OD=%ld OH=%ld OW=%ld | %.3f ms | wrote %.1f MiB (%.0f GB/s) | in %.1f MiB | expand x%ld\n",
                used_tiled ? "TILED" : "fastd",
                (long)N,(long)IC,(long)ID,(long)IH,(long)IW,(long)KD,(long)KH,(long)KW,(long)OD,(long)OH,(long)OW,
                lc_ms, wbytes/1048576.0, gbps_w, rbytes/1048576.0, (long)KD_KH_KW);
        cudaEventDestroy(lc_ev0); cudaEventDestroy(lc_ev1);
    }

    // [IM2COL_VERIFY] env-gated bit-exactness check: re-run fastdiv into a temp
    // buffer and byte-compare. im2col is pure data movement so the tiled path
    // must be byte-identical; this proves it per-call without a full render.
    if (used_tiled && getenv("LONGCAT_IM2COL_VERIFY")) {
        const int64_t out_elems = OD_OH_OW_IC_KD_KH_KW * N;
        const size_t  nbytes    = (size_t) out_elems * sizeof(T);
        T * ref = nullptr;
        if (cudaMalloc(&ref, nbytes) == cudaSuccess) {
            launch_fastdiv(ref);
            cudaStreamSynchronize(stream);
            std::vector<char> h_tiled(nbytes), h_ref(nbytes);
            cudaMemcpy(h_tiled.data(), dst, nbytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ref.data(),   ref, nbytes, cudaMemcpyDeviceToHost);
            int mism = memcmp(h_tiled.data(), h_ref.data(), nbytes);
            int64_t nmis = 0;
            if (mism != 0) {
                for (int64_t e = 0; e < out_elems; ++e) {
                    if (((const T *)h_tiled.data())[e] != ((const T *)h_ref.data())[e]) nmis++;
                }
            }
            fprintf(stderr, "[IM2COL_VERIFY] %s | %ld elems | %s (mismatched=%ld)\n",
                    mism == 0 ? "BIT-EXACT" : "MISMATCH",
                    (long) out_elems, mism == 0 ? "ok" : "DIFF", (long) nmis);
            cudaFree(ref);
        }
    }
}

static void im2col_3d_cuda_f16(const float * src, half * dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, cudaStream_t stream) {

    im2col_3d_cuda<half>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                         stride_q, stride_z, stride_y, stride_x,
                         s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

static void im2col_3d_cuda_f32(const float * src, float * dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, cudaStream_t stream) {

    im2col_3d_cuda<float>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                          stride_q, stride_z, stride_y, stride_x,
                          s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

void ggml_cuda_op_im2col_3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t s2 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[3];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[4];
    const int32_t p2 = ((const int32_t *)(dst->op_params))[5];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[6];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[7];
    const int32_t d2 = ((const int32_t *)(dst->op_params))[8];
    const int32_t IC = ((const int32_t *)(dst->op_params))[9];

    const int64_t N  = ne13 / IC;
    const int64_t ID = ne12;
    const int64_t IH = ne11;
    const int64_t IW = ne10;

    const int64_t OC = ne03 / IC;
    const int64_t KD = ne02;
    const int64_t KH = ne01;
    const int64_t KW = ne00;

    const int64_t OD = ne3 / N;
    const int64_t OH = ne2;
    const int64_t OW = ne1;

    const size_t  es       = ggml_element_size(src1);
    const int64_t stride_x = src1->nb[0] / es;
    const int64_t stride_y = src1->nb[1] / es;
    const int64_t stride_z = src1->nb[2] / es;
    const int64_t stride_q = src1->nb[3] / es;

    if(dst->type == GGML_TYPE_F16) {
        im2col_3d_cuda_f16(src1_d, (half *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    } else {
        im2col_3d_cuda_f32(src1_d, (float *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    }
}
