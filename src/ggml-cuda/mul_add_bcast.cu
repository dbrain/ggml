#include "mul_add_bcast.cuh"

// Fused: dst = x + y * gate, with gate broadcast across the MUL's bcast dim.
//
// Strategy: iterate over flat dst byte offsets (3D dst layout from ADD's
// output). The y/x tensors share the dst shape and byte layout (contiguous
// F32, the MUL's RESHAPE_3d view is a no-op on bytes). The gate is contiguous
// F32 with one dim collapsed to size 1; its byte offset for a given dst flat
// index requires decoding the 4D view coords (i_c, i_m, i_t, i_b) and
// skipping the i_m index (the broadcast dim).
//
// Hardcoded for the gate_add pattern: 4D MUL shape [d0, M, T, Nb] (d0=C,
// M=n_token/T, T=frames, Nb=batch), gate shape [d0, 1, T, Nb]. Detection in
// ggml-cuda.cu ensures we only fire here.

template <bool HAS_SHIFT>
__global__ void __launch_bounds__(256) mul_add_bcast_dim1_f32_kernel(
    const float * __restrict__ x,
    const float * __restrict__ y,
    const float * __restrict__ gate,
    const float * __restrict__ shift, // [d0,1,d2,Nb] bcast (same layout as gate); read only if HAS_SHIFT
    float       * __restrict__ dst,
    const int64_t n_elem,
    const int64_t d0,        // innermost dim (== ne00 of MUL)
    const int64_t d1,        // broadcast dim length on y/x (gate has 1 here)
    const int64_t d2,        // outer dim (== T)
    const int64_t row_stride, // == d0 (4-byte units): elements per (i_m, i_t, i_b) row
    const int64_t plane_d1,   // == d0 * d1: elements per (i_t, i_b) plane in y/x flat
    const int64_t plane_d2,   // == d0 * d2: gate's "i_t row" stride * d2 — see decode below
    const int64_t gate_d0d2)  // == d0 * d2: elements per (i_b) plane in gate flat
{
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_elem) {
        return;
    }

    // Decode flat index t into (i_b, i_t, i_m, i_c) for the 4D MUL view:
    //   t = i_b * d0*d1*d2 + i_t * d0*d1 + i_m * d0 + i_c
    // (ggml column-major-like: dim 0 = innermost contiguous).
    int64_t       rem = t;
    const int64_t i_c = rem % d0; rem /= d0;
    const int64_t i_m = rem % d1; rem /= d1;
    const int64_t i_t = rem % d2; rem /= d2;
    const int64_t i_b = rem;
    (void) i_m;

    // gate flat index in [d0, 1, d2, Nb]:
    //   g = i_b * d0*d2 + i_t * d0 + i_c
    const int64_t g_idx = i_b * gate_d0d2 + i_t * d0 + i_c;
    (void) row_stride; (void) plane_d1; (void) plane_d2;

    const float xv = x[t];
    const float yv = y[t];
    const float gv = gate[g_idx];

    // __fmul_rn + __fadd_rn → two independent IEEE-roundings, matching the
    // unfused MUL+ADD chain exactly. -use_fast_math compiles `x + y*g` to FMA
    // (one rounding) which would drift by ULP and fail PSNR 99.
    float r = __fadd_rn(xv, __fmul_rn(yv, gv));
    if (HAS_SHIFT) {
        // flux AdaLN: fold the trailing broadcast `+ shift` (same [d0,1,d2,Nb]
        // layout as gate). The extra __fadd_rn matches the standalone bcast-ADD
        // it replaces (single rounding) → bit-exact vs the unfused 3-op chain.
        r = __fadd_rn(r, shift[g_idx]);
    }
    dst[t] = r;
}

// ---------------------------------------------------------------------------
// Same-shape (no-broadcast) fused multiply-add: dst = x + y*g (+ shift), all
// operands contiguous F32 of identical element count. NAVA's per-token AdaLN
// modulation (x + x*scale + shift, batch N=1) is this shape — the broadcast
// detector above never fires (gate is full [d0,L,1], not [d0,1,...]), so the
// chain ran as 3 separate full-size kernels (~17% of the DiT). This fuses it
// to one pass. Bit-exact: __fmul_rn + __fadd_rn = the same independent
// roundings as the unfused MUL/ADD chain (no FMA contraction).
template <bool HAS_SHIFT>
__global__ void __launch_bounds__(256) fused_madd_same_f32_kernel(
    const float * __restrict__ x,
    const float * __restrict__ y,
    const float * __restrict__ g,
    const float * __restrict__ shift,  // same layout as x; read only if HAS_SHIFT
    float       * __restrict__ dst,
    const int64_t n_elem) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_elem) {
        return;
    }
    float r = __fadd_rn(x[t], __fmul_rn(y[t], g[t]));
    if (HAS_SHIFT) {
        r = __fadd_rn(r, shift[t]);
    }
    dst[t] = r;
}

// Same-shape fused multiply-add with a STRIDED gate (dst = x + y*g [+ shift]).
// LTX-2.3 align_token_modulation permutes the gate [d0,1,L] -> [d0,L,1] (moving the
// size-1 dim) so `g` matches the MUL's [d0,L,1] shape but ggml_is_contiguous(g) is
// false -> the flat kernel above can't be triggered for it. `g` is read via its own
// 4D strides (decoded from the contiguous dst/iteration shape ne0..ne2); x/y/dst/shift
// stay flat-contiguous. For the LTX permute the gate data IS contiguous (the permute
// only reorders a size-1 dim), so g[g_idx] reads exactly what g[t] would on a
// contiguous gate -> bit-identical to the flat path. Bit-exact (__fmul_rn + __fadd_rn).
template <bool HAS_SHIFT>
__global__ void __launch_bounds__(256) fused_madd_same_strided_g_f32_kernel(
    const float * __restrict__ x,
    const float * __restrict__ y,
    const float * __restrict__ g,      // strided gate (may be non-contiguous)
    const float * __restrict__ shift,  // flat, same layout as x; read only if HAS_SHIFT
    float       * __restrict__ dst,
    const int64_t n_elem,
    const int64_t ne0,
    const int64_t ne1,
    const int64_t ne2,
    const int64_t gs0,   // g strides in ELEMENTS (g->nb / sizeof(float))
    const int64_t gs1,
    const int64_t gs2,
    const int64_t gs3) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_elem) {
        return;
    }
    int64_t       rem = t;
    const int64_t i0 = rem % ne0; rem /= ne0;
    const int64_t i1 = rem % ne1; rem /= ne1;
    const int64_t i2 = rem % ne2; rem /= ne2;
    const int64_t i3 = rem;
    const int64_t g_idx = i0 * gs0 + i1 * gs1 + i2 * gs2 + i3 * gs3;
    float r = __fadd_rn(x[t], __fmul_rn(y[t], g[g_idx]));
    if (HAS_SHIFT) {
        r = __fadd_rn(r, shift[t]);
    }
    dst[t] = r;
}

// x = residual side of the ADD; y,g = the two MUL operands (dst = x + y*g).
// x/y/dst/shift contiguous F32, ggml_nelements all == add_n's. `g` may be a strided
// (permuted) view of the same logical shape — read via its strides. shift optional.
void ggml_cuda_op_fused_madd_same(ggml_backend_cuda_context & ctx,
                                  ggml_tensor *               add_n,
                                  const ggml_tensor *         x,
                                  const ggml_tensor *         y,
                                  const ggml_tensor *         g,
                                  const ggml_tensor *         shift) {
    const int64_t n_elem = ggml_nelements(add_n);
    GGML_ASSERT(ggml_nelements(x) == n_elem);
    GGML_ASSERT(ggml_nelements(y) == n_elem);
    GGML_ASSERT(ggml_nelements(g) == n_elem);
    if (shift) {
        GGML_ASSERT(ggml_nelements(shift) == n_elem);
    }

    // Use each tensor's own ->data, which already includes any view offset
    // (ggml-backend sets a view's data = view_src->data + view_offs at alloc).
    // The previous code resolved to view_src->data, which DROPPED the offset and
    // read from element 0 of the base buffer — fine when operands are full tensors
    // (NAVA avatar AdaLN, offset 0) but corrupt when they are offset views into a
    // larger tensor (LTX-2.3 ltxav modulation slices scale/shift) → fuzzy mush.
    // dst already used add_n->data directly; match that for the inputs.
    const float * x_d     = (const float *) x->data;
    const float * y_d     = (const float *) y->data;
    const float * g_d     = (const float *) g->data;
    const float * shift_d = shift ? (const float *) shift->data : nullptr;
    float       * dst_d   = (float *) add_n->data;

    const int     block_size = 256;
    const int64_t num_blocks = (n_elem + block_size - 1) / block_size;
    GGML_ASSERT(num_blocks <= (1LL << 31) - 1);

    if (ggml_is_contiguous(g)) {
        auto kern = shift_d ? fused_madd_same_f32_kernel<true> : fused_madd_same_f32_kernel<false>;
        kern<<<(int) num_blocks, block_size, 0, ctx.stream()>>>(x_d, y_d, g_d, shift_d, dst_d, n_elem);
    } else {
        // Strided gate (LTX align_token_modulation permute). Decode the flat index over
        // the contiguous iteration shape (== add_n / mul_n shape) and gather g via its
        // own strides. x/y/dst/shift stay flat-contiguous.
        GGML_ASSERT(ggml_are_same_shape(g, add_n));
        const int64_t gs0 = g->nb[0] / sizeof(float);
        const int64_t gs1 = g->nb[1] / sizeof(float);
        const int64_t gs2 = g->nb[2] / sizeof(float);
        const int64_t gs3 = g->nb[3] / sizeof(float);
        auto kern = shift_d ? fused_madd_same_strided_g_f32_kernel<true>
                            : fused_madd_same_strided_g_f32_kernel<false>;
        kern<<<(int) num_blocks, block_size, 0, ctx.stream()>>>(
            x_d, y_d, g_d, shift_d, dst_d, n_elem,
            add_n->ne[0], add_n->ne[1], add_n->ne[2],
            gs0, gs1, gs2, gs3);
    }
}

void ggml_cuda_op_mul_add_bcast(ggml_backend_cuda_context & ctx,
                                ggml_tensor *               mul_n,
                                ggml_tensor *               add_n,
                                const ggml_tensor *         x,
                                const ggml_tensor *         y_view,
                                const ggml_tensor *         gate,
                                const ggml_tensor *         shift) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(y_view->type == GGML_TYPE_F32);
    GGML_ASSERT(gate->type == GGML_TYPE_F32);
    GGML_ASSERT(add_n->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(gate));
    GGML_ASSERT(ggml_is_contiguous(add_n));
    if (shift) {
        GGML_ASSERT(shift->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(shift));
    }

    // 4D shape from the MUL node ([d0, d1, d2, Nb]). gate is [d0, 1, d2, Nb].
    const int64_t d0 = mul_n->ne[0];
    const int64_t d1 = mul_n->ne[1];
    const int64_t d2 = mul_n->ne[2];
    const int64_t Nb = mul_n->ne[3];
    const int64_t n_elem = d0 * d1 * d2 * Nb;

    GGML_ASSERT(gate->ne[0] == d0);
    GGML_ASSERT(gate->ne[1] == 1);
    GGML_ASSERT(gate->ne[2] == d2);
    GGML_ASSERT(gate->ne[3] == Nb);

    if (shift) {
        GGML_ASSERT(shift->ne[0] == d0);
        GGML_ASSERT(shift->ne[1] == 1);
        GGML_ASSERT(shift->ne[2] == d2);
        GGML_ASSERT(shift->ne[3] == Nb);
    }

    // x and ADD's output share total element count with the MUL.
    GGML_ASSERT(ggml_nelements(add_n) == n_elem);
    GGML_ASSERT(ggml_nelements(x)     == n_elem);

    // The MUL's underlying y buffer: we follow the view chain from y_view to
    // the producing tensor (the RESHAPE_4d view is a no-op on bytes, same
    // contiguous storage as the original y).
    const ggml_tensor * y_buf = y_view;
    while (y_buf->view_src != nullptr) {
        y_buf = y_buf->view_src;
    }
    GGML_ASSERT(y_buf->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(y_buf));
    GGML_ASSERT(ggml_nelements(y_buf) == n_elem);

    const float * x_d     = (const float *) x->data;
    const float * y_d     = (const float *) y_buf->data;
    const float * gate_d  = (const float *) gate->data;
    const float * shift_d = shift ? (const float *) shift->data : nullptr;
    float       * dst_d   = (float *)       add_n->data;

    const int     block_size = 256;
    const int64_t num_blocks = (n_elem + block_size - 1) / block_size;
    GGML_ASSERT(num_blocks <= (1LL << 31) - 1);

    auto kern = shift_d ? mul_add_bcast_dim1_f32_kernel<true>
                        : mul_add_bcast_dim1_f32_kernel<false>;
    kern<<<(int)num_blocks, block_size, 0, ctx.stream()>>>(
        x_d, y_d, gate_d, shift_d, dst_d,
        n_elem, d0, d1, d2,
        /*row_stride=*/ d0,
        /*plane_d1  =*/ d0 * d1,
        /*plane_d2  =*/ d0 * d2,
        /*gate_d0d2 =*/ d0 * d2);
}
