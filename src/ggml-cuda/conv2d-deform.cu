#include "conv2d-deform.cuh"

// Deformable conv2d (GGML_OP_CONV_2D_DEFORM), whcn layout.
//
// Faithful port of the CPU reference (ggml_compute_forward_conv_2d_deform_whcn
// in ggml-cpu/ops.cpp). The math is a fused deformable-im2col + GEMM:
//
//   dst[oc, n] = sum_k  W[oc, k] * col(n, k)
//
// where n indexes an output spatial location (dst_x, dst_y, batch), oc indexes
// an output channel, and k = ic*(kh*kw) + ky*kw + kx indexes the contraction
// (the deformable im2col column). col(n,k) is a single mask-modulated bilinear
// sample of input channel ic at the deformed location for tap (ky,kx).
//
// Fast path (conv2d_deform_gemm_kernel): a tiled register-blocked GEMM. Each
// block computes a BM(channels) x BN(pixels) output tile, streaming the
// contraction K in BK chunks through shared memory. The weight tile is loaded
// coalesced; the deformed column tile is gathered ONCE per (pixel,k) and reused
// across all BM output channels in the tile (the reuse the old
// one-block-per-pixel kernel lacked). Larger BM => fewer channel tiles => the
// expensive bilinear gather is recomputed by fewer blocks. Small smem footprint
// => high occupancy => hides the bilinear-gather latency.
//
// A simple cooperative-smem fallback (conv2d_deform_simple_kernel, one block
// per output pixel) is kept as a bit-exact reference but is not on the live path.
//
// Memory layouts (all match the CPU op):
//   src    : [src_w, src_h, c_in, N]            contiguous (whcn)
//   kernel : [knl_w, knl_h, c_in, c_out]        contiguous
//            column/kernel index ordering: ic*(kh*kw) + ky*kw + kx
//   offset : [dst_w, dst_h, 2*kw*kh, N]         (off_y at 2*i+0, off_x at 2*i+1)
//   mask   : [dst_w, dst_h,   kw*kh, N]         (may be null)
//   dst    : [dst_w, dst_h, c_out, N]           strides via dst->nb (may be permuted)
//
// offset/mask are typically non-contiguous ggml views, so they are indexed with
// their byte strides (nb), not assuming a packed layout.

static __device__ __forceinline__ float deform_bilinear(
        const float * __restrict__ src, int w, int h, float x, float y) {
    const int x0 = (int) floorf(x);
    const int y0 = (int) floorf(y);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float dx = x - (float) x0;
    const float dy = y - (float) y0;

    const float v00 = (x0 >= 0 && y0 >= 0) ? src[(int64_t) y0 * w + x0] : 0.0f;
    const float v01 = (x1 <  w && y0 >= 0) ? src[(int64_t) y0 * w + x1] : 0.0f;
    const float v10 = (x0 >= 0 && y1 <  h) ? src[(int64_t) y1 * w + x0] : 0.0f;
    const float v11 = (x1 <  w && y1 <  h) ? src[(int64_t) y1 * w + x1] : 0.0f;

    return (v00 * (1.0f - dx) + v01 * dx) * (1.0f - dy)
         + (v10 * (1.0f - dx) + v11 * dx) * dy;
}

// Gather one deformed im2col element col(pixel, k). Matches the CPU op exactly.
static __device__ __forceinline__ float deform_gather(
        const float * __restrict__ src_b,   // src + b*src_nb3
        const float * __restrict__ off_b,   // off + b*off_nb3 + dst_coord
        const float * __restrict__ msk_b,   // msk + b*msk_nb3 + dst_coord (or null)
        int dst_x, int dst_y, int k,
        int src_w, int src_h,
        int knl_w, int knl_wh,
        int stride_x, int stride_y, int pad_x, int pad_y,
        int64_t src_nb2, int64_t off_nb2, int64_t msk_nb2) {
    const int ic  = k / knl_wh;
    const int rem = k - ic * knl_wh;
    const int ky  = rem / knl_w;
    const int kx  = rem - ky * knl_w;
    const int knl_i = ky * knl_w + kx;

    const float off_x = off_b[(int64_t)(2 * knl_i + 1) * off_nb2];
    const float off_y = off_b[(int64_t)(2 * knl_i + 0) * off_nb2];

    const float sx = (float) (dst_x * stride_x + kx) - pad_x + off_x;
    const float sy = (float) (dst_y * stride_y + ky) - pad_y + off_y;

    if (sx <= -1.0f || sx >= src_w || sy <= -1.0f || sy >= src_h) {
        return 0.0f;
    }
    const float * src_ic = src_b + (int64_t) ic * src_nb2;
    const float weight = msk_b ? msk_b[(int64_t) knl_i * msk_nb2] : 1.0f;
    return weight * deform_bilinear(src_ic, src_w, src_h, sx, sy);
}

// ---------------------------------------------------------------------------
// Fast path: tiled register-blocked GEMM with fused deformable im2col.
//
// Tile: BM output channels x BN output pixels, contraction streamed in BK.
// Threads: (BN/TN) x (BM/TM); each thread owns a TM x TN register tile.
// ---------------------------------------------------------------------------
#define DEF_BM 128
#define DEF_BN 64
#define DEF_BK 16
#define DEF_TM 8
#define DEF_TN 4
#define DEF_NT ((DEF_BM / DEF_TM) * (DEF_BN / DEF_TN)) // = 16*16 = 256 threads

__global__ void __launch_bounds__(DEF_NT)
conv2d_deform_gemm_kernel(
        const float * __restrict__ src,
        const float * __restrict__ knl,
        const float * __restrict__ off,
        const float * __restrict__ msk,   // may be null
        float       * __restrict__ dst,
        const int src_w, const int src_h, const int c_in,
        const int knl_w, const int knl_h, const int c_out,
        const int dst_w, const int dst_h, const int batches,
        const int stride_x, const int stride_y,
        const int pad_x, const int pad_y,
        const int64_t src_nb2, const int64_t src_nb3,
        const int64_t off_nb2, const int64_t off_nb3,
        const int64_t msk_nb2, const int64_t msk_nb3,
        const int64_t dst_nb0, const int64_t dst_nb1,
        const int64_t dst_nb2, const int64_t dst_nb3) {

    const int knl_wh = knl_w * knl_h;
    const int knl_n  = c_in * knl_wh;
    const int64_t npix = (int64_t) batches * dst_h * dst_w;

    const int oc0 = blockIdx.y * DEF_BM;   // first output channel of this tile
    const int px0 = blockIdx.x * DEF_BN;   // first output pixel of this tile

    const int tx  = threadIdx.x % (DEF_BN / DEF_TN); // -> pixel sub-tile
    const int ty  = threadIdx.x / (DEF_BN / DEF_TN); // -> channel sub-tile
    const int tid = threadIdx.x;

    __shared__ float As[DEF_BK][DEF_BN]; // deformed column tile  (k, pixel)
    __shared__ float Bs[DEF_BK][DEF_BM]; // weight tile           (k, channel)

    float acc[DEF_TM][DEF_TN];
#pragma unroll
    for (int i = 0; i < DEF_TM; ++i)
#pragma unroll
        for (int j = 0; j < DEF_TN; ++j) acc[i][j] = 0.0f;

    // Precompute per-pixel addressing for the BN pixels this block owns, once.
    __shared__ int   pix_x[DEF_BN];
    __shared__ int   pix_y[DEF_BN];
    __shared__ int   pix_b[DEF_BN];
    __shared__ const float * pix_src[DEF_BN];
    __shared__ const float * pix_off[DEF_BN];
    __shared__ const float * pix_msk[DEF_BN];
    __shared__ bool  pix_valid[DEF_BN];

    for (int p = tid; p < DEF_BN; p += DEF_NT) {
        const int64_t pixel = (int64_t) px0 + p;
        if (pixel < npix) {
            const int dx = (int) ( pixel % dst_w );
            const int dy = (int) ((pixel / dst_w) % dst_h);
            const int b  = (int) ( pixel / ((int64_t) dst_w * dst_h) );
            const int dst_coord = dy * dst_w + dx;
            pix_x[p]   = dx;
            pix_y[p]   = dy;
            pix_b[p]   = b;
            pix_src[p] = src + (int64_t) b * src_nb3;
            pix_off[p] = off + (int64_t) b * off_nb3 + dst_coord;
            pix_msk[p] = msk ? msk + (int64_t) b * msk_nb3 + dst_coord : nullptr;
            pix_valid[p] = true;
        } else {
            pix_valid[p] = false;
        }
    }
    __syncthreads();

    // Stream the contraction dimension.
    for (int k0 = 0; k0 < knl_n; k0 += DEF_BK) {
        // --- load weight tile Bs[kk][m] : W[oc, k], oc = oc0+m, k = k0+kk ---
        // knl is contiguous [knl_n, c_out] (row = oc, length knl_n).
#pragma unroll
        for (int idx = tid; idx < DEF_BK * DEF_BM; idx += DEF_NT) {
            const int kk = idx % DEF_BK;
            const int m  = idx / DEF_BK;
            const int kg = k0 + kk;
            const int oc = oc0 + m;
            float w = 0.0f;
            if (kg < knl_n && oc < c_out) {
                w = knl[(int64_t) oc * knl_n + kg];
            }
            Bs[kk][m] = w;
        }

        // --- gather deformed column tile As[kk][n] : col(px0+n, k0+kk) ---
        // Map consecutive threads to consecutive PIXELS (n) at the same k so
        // neighbouring threads' bilinear reads hit neighbouring input addresses
        // (coalesced) instead of striding by src_nb2 across input channels.
#pragma unroll
        for (int idx = tid; idx < DEF_BK * DEF_BN; idx += DEF_NT) {
            const int n  = idx % DEF_BN;
            const int kk = idx / DEF_BN;
            const int kg = k0 + kk;
            float v = 0.0f;
            if (kg < knl_n && pix_valid[n]) {
                v = deform_gather(
                        pix_src[n], pix_off[n], pix_msk[n],
                        pix_x[n], pix_y[n], kg,
                        src_w, src_h, knl_w, knl_wh,
                        stride_x, stride_y, pad_x, pad_y,
                        src_nb2, off_nb2, msk_nb2);
            }
            As[kk][n] = v;
        }
        __syncthreads();

        // --- register-blocked multiply-accumulate ---
#pragma unroll
        for (int kk = 0; kk < DEF_BK; ++kk) {
            float a_reg[DEF_TN];
            float b_reg[DEF_TM];
#pragma unroll
            for (int j = 0; j < DEF_TN; ++j) a_reg[j] = As[kk][tx * DEF_TN + j];
#pragma unroll
            for (int i = 0; i < DEF_TM; ++i) b_reg[i] = Bs[kk][ty * DEF_TM + i];
#pragma unroll
            for (int i = 0; i < DEF_TM; ++i)
#pragma unroll
                for (int j = 0; j < DEF_TN; ++j)
                    acc[i][j] += b_reg[i] * a_reg[j];
        }
        __syncthreads();
    }

    // --- store ---
#pragma unroll
    for (int i = 0; i < DEF_TM; ++i) {
        const int oc = oc0 + ty * DEF_TM + i;
        if (oc >= c_out) continue;
#pragma unroll
        for (int j = 0; j < DEF_TN; ++j) {
            const int n = tx * DEF_TN + j;
            if (!pix_valid[n]) continue;
            float * dst_ptr = dst + (int64_t) pix_x[n] * dst_nb0
                                  + (int64_t) pix_y[n] * dst_nb1
                                  + (int64_t) oc       * dst_nb2
                                  + (int64_t) pix_b[n] * dst_nb3;
            *dst_ptr = acc[i][j];
        }
    }
}

// ---------------------------------------------------------------------------
// Fallback: one block per output pixel, cooperative smem column. Bit-exact
// reference path (not on the live dispatch path; kept for clarity / debugging).
// ---------------------------------------------------------------------------
extern __shared__ float s_col[];

__global__ void conv2d_deform_simple_kernel(
        const float * __restrict__ src,
        const float * __restrict__ knl,
        const float * __restrict__ off,
        const float * __restrict__ msk,   // may be null
        float       * __restrict__ dst,
        const int src_w, const int src_h, const int c_in,
        const int knl_w, const int knl_h, const int c_out,
        const int dst_w, const int dst_h, const int batches,
        const int stride_x, const int stride_y,
        const int pad_x, const int pad_y,
        const int64_t src_nb2, const int64_t src_nb3,
        const int64_t off_nb2, const int64_t off_nb3,
        const int64_t msk_nb2, const int64_t msk_nb3,
        const int64_t dst_nb0, const int64_t dst_nb1, const int64_t dst_nb2, const int64_t dst_nb3) {

    const int64_t pixel = (int64_t) blockIdx.x;
    const int64_t npix  = (int64_t) batches * dst_h * dst_w;
    if (pixel >= npix) {
        return;
    }

    const int dst_x = (int) ( pixel % dst_w );
    const int dst_y = (int) ((pixel / dst_w) % dst_h);
    const int b     = (int) ( pixel / ((int64_t) dst_w * dst_h) );

    const int knl_wh = knl_w * knl_h;
    const int knl_n  = c_in * knl_wh;
    const int dst_coord = dst_y * dst_w + dst_x;

    const float * src_b = src + (int64_t) b * src_nb3;
    const float * off_b = off + (int64_t) b * off_nb3 + dst_coord;
    const float * msk_b = msk ? msk + (int64_t) b * msk_nb3 + dst_coord : nullptr;

    for (int j = threadIdx.x; j < knl_n; j += blockDim.x) {
        s_col[j] = deform_gather(
                src_b, off_b, msk_b, dst_x, dst_y, j,
                src_w, src_h, knl_w, knl_wh,
                stride_x, stride_y, pad_x, pad_y,
                src_nb2, off_nb2, msk_nb2);
    }
    __syncthreads();

    for (int oc = threadIdx.x; oc < c_out; oc += blockDim.x) {
        const float * knl_oc = knl + (int64_t) oc * knl_n;
        float acc = 0.0f;
        for (int j = 0; j < knl_n; ++j) {
            acc += s_col[j] * knl_oc[j];
        }
        float * dst_ptr = dst + (int64_t) dst_x * dst_nb0 + (int64_t) dst_y * dst_nb1
                              + (int64_t) oc    * dst_nb2 + (int64_t) b     * dst_nb3;
        *dst_ptr = acc;
    }
}

void ggml_cuda_op_conv2d_deform(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    const ggml_tensor * offset = dst->src[2];
    const ggml_tensor * mask   = dst->src[3];

    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(offset->type == GGML_TYPE_F32);
    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F32);

    const int32_t * p          = (const int32_t *) dst->op_params;
    const int       stride_x   = p[0];
    const int       stride_y   = p[1];
    const int       pad_x      = p[2];
    const int       pad_y      = p[3];

    const int src_w   = input->ne[0];
    const int src_h   = input->ne[1];
    const int c_in    = input->ne[2];
    const int knl_w   = kernel->ne[0];
    const int knl_h   = kernel->ne[1];
    const int c_out   = kernel->ne[3];
    const int dst_w   = dst->ne[0];
    const int dst_h   = dst->ne[1];
    const int batches = dst->ne[3];

    GGML_ASSERT(kernel->ne[2] == c_in);
    GGML_ASSERT(offset->ne[2] == 2 * knl_w * knl_h);
    GGML_ASSERT(!mask || mask->ne[2] == knl_w * knl_h);

    const int64_t src_nb2 = input->nb[2]  / sizeof(float);
    const int64_t src_nb3 = input->nb[3]  / sizeof(float);
    const int64_t off_nb2 = offset->nb[2] / sizeof(float);
    const int64_t off_nb3 = offset->nb[3] / sizeof(float);
    const int64_t msk_nb2 = mask ? mask->nb[2] / sizeof(float) : 0;
    const int64_t msk_nb3 = mask ? mask->nb[3] / sizeof(float) : 0;
    const int64_t dst_nb0 = dst->nb[0] / sizeof(float);
    const int64_t dst_nb1 = dst->nb[1] / sizeof(float);
    const int64_t dst_nb2 = dst->nb[2] / sizeof(float);
    const int64_t dst_nb3 = dst->nb[3] / sizeof(float);

    const float * src_d = (const float *) input->data;
    const float * knl_d = (const float *) kernel->data;
    const float * off_d = (const float *) offset->data;
    const float * msk_d = mask ? (const float *) mask->data : nullptr;
    float *       dst_d = (float *) dst->data;

    cudaStream_t st = ctx.stream();

    const int64_t npix = (int64_t) batches * dst_h * dst_w;

    dim3 grid((unsigned)((npix  + DEF_BN - 1) / DEF_BN),
              (unsigned)((c_out + DEF_BM - 1) / DEF_BM));
    conv2d_deform_gemm_kernel<<<grid, DEF_NT, 0, st>>>(
        src_d, knl_d, off_d, msk_d, dst_d,
        src_w, src_h, c_in, knl_w, knl_h, c_out, dst_w, dst_h, batches,
        stride_x, stride_y, pad_x, pad_y,
        src_nb2, src_nb3, off_nb2, off_nb3, msk_nb2, msk_nb3,
        dst_nb0, dst_nb1, dst_nb2, dst_nb3);
}
