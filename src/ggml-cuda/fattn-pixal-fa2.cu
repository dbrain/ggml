#include "fattn-pixal-fa2.cuh"

#ifdef GGML_PIXAL_FA2
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"
#include <cstdlib>
#include <cstdio>

static cudaError_t pixal_fa2_forward(void * q, void * k, void * v, void * o,
                                      int qlen, int klen, int qheads, int kvheads,
                                      int batch, bool causal, float scale, cudaStream_t stream,
                                      int64_t qrow, int64_t qhead, int64_t qbatch,
                                      int64_t krow, int64_t khead, int64_t kbatch) {
    float * lse = nullptr;
    cudaError_t err = cudaMallocAsync((void **) &lse, (size_t) batch * qheads * qlen * sizeof(float), stream);
    if (err != cudaSuccess) return err;
    flash::Flash_fwd_params p{};
    // GGML contiguous [D, T, H] maps to FlashAttention logical [T, H, D]:
    // D is adjacent, T advances by D, H advances by a complete sequence.
    p.q_ptr = q; p.k_ptr = k; p.v_ptr = v; p.o_ptr = o;
    p.q_batch_stride = qbatch; p.k_batch_stride = kbatch; p.v_batch_stride = kbatch; p.o_batch_stride = (int64_t)qheads * qlen * 128;
    p.q_row_stride = qrow; p.k_row_stride = krow; p.v_row_stride = krow;
    p.q_head_stride = qhead; p.k_head_stride = khead; p.v_head_stride = khead;
    // ggml_flash_attn_ext output is [D, H, T], unlike its [D, T, H] inputs.
    p.o_row_stride = (int64_t) qheads * 128;
    p.o_head_stride = 128;
    p.h = qheads; p.h_k = kvheads; p.h_h_k_ratio = qheads / kvheads;
    p.b = batch; p.seqlen_q = qlen; p.seqlen_k = klen; p.d = 128;
    p.seqlen_q_rounded = ((qlen + 127) / 128) * 128;
    p.seqlen_k_rounded = ((klen + 127) / 128) * 128; p.d_rounded = 128;
    p.scale_softmax = scale;
    p.scale_softmax_log2 = p.scale_softmax * 1.4426950408889634f;
    p.softmax_lse_ptr = lse; p.p_dropout = 1.f; p.p_dropout_in_uint8_t = 255; p.rp_dropout = 1.f;
    p.scale_softmax_rp_dropout = p.scale_softmax;
    // FA2's causal template still reads window_size_right when constructing
    // the diagonal mask.  The public API canonicalizes causal to (-1, 0);
    // using -1 here incorrectly excludes every query's diagonal key.
    p.window_size_left = -1; p.window_size_right = causal ? 0 : -1;
    p.is_bf16 = true; p.is_causal = causal; p.is_seqlens_k_cumulative = true; p.num_splits = 0;
    p.philox_args = at::PhiloxCudaState(0, 0);
    if (causal) flash::run_mha_fwd_hdim128<cutlass::bfloat16_t, true>(p, stream);
    else        flash::run_mha_fwd_hdim128<cutlass::bfloat16_t, false>(p, stream);
    err = cudaGetLastError();
    const cudaError_t free_err = cudaFreeAsync(lse, stream);
    return err != cudaSuccess ? err : free_err;
}

bool ggml_cuda_flash_attn_ext_pixal_fa2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2];
    const bool pixal_self = getenv("PIXAL3D_FA2") != nullptr;
    const int qwen_mode = ggml_get_op_params_i32(dst, 4);
    const bool qwen_prefill = qwen_mode == 1, qwen_decode_swap = qwen_mode == 2;
    if (qwen_decode_swap && std::getenv("RIG_QWEN_FA2_SWAP_DEBUG")) {
        std::fprintf(stderr,
            "[qwen-fa2-swap] dst=%d q=%d k=%d v=%d q=[%lld,%lld,%lld,%lld] k=[%lld,%lld,%lld,%lld] contig=%d/%d/%d/%d\\n",
            (int) dst->type, (int) q->type, (int) k->type, (int) v->type,
            (long long) q->ne[0], (long long) q->ne[1], (long long) q->ne[2], (long long) q->ne[3],
            (long long) k->ne[0], (long long) k->ne[1], (long long) k->ne[2], (long long) k->ne[3],
            ggml_is_contiguous(q), ggml_is_contiguous(k), ggml_is_contiguous(v), ggml_is_contiguous(dst));
    }
    if ((!pixal_self && !qwen_prefill && !qwen_decode_swap) || dst->src[3] != nullptr ||
        dst->type != GGML_TYPE_BF16 || q->type != GGML_TYPE_BF16 ||
        k->type != GGML_TYPE_BF16 || v->type != GGML_TYPE_BF16 ||
        q->ne[0] != 128 || k->ne[0] != 128 || v->ne[0] != 128 ||
        k->ne[1] != v->ne[1] ||
        q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3] ||
        !ggml_is_contiguous(dst)) {
        return false;
    }
    const bool qwen = qwen_prefill && q->ne[2] == 16 && k->ne[2] == 8 && v->ne[2] == 8;
    const bool qwen_swap = qwen_decode_swap && q->ne[1] == 2 && q->ne[2] == 8 && k->ne[2] == 8 && v->ne[2] == 8;
    // Pixal self-attention remains batch-one.  The operation-local Qwen
    // causal-GQA contract additionally supports the real B-row expanded
    // prefix used by Transformers beam search.
    if (!qwen && !qwen_swap && !(pixal_self && q->ne[3] == 1 && q->ne[2] == k->ne[2] && q->ne[2] == v->ne[2])) return false;
    // During cached Qwen decode the logical single query is the *last*
    // position in a longer K/V sequence.  FA2's causal Q=1 convention is
    // kernel-version sensitive; retain causal prefill but permit an explicit
    // parity probe of the all-visible decode contract.
    const bool causal = qwen && !(q->ne[1] == 1 && std::getenv("RIG_QWEN_FA2_DECODE_NONCAUSAL"));
    CUDA_CHECK(pixal_fa2_forward(q->data, k->data, v->data, dst->data,
                                 (int) q->ne[1], (int) k->ne[1], (int) q->ne[2], (int) k->ne[2],
                                 (int) q->ne[3], causal, ggml_get_op_params_f32(dst, 0), ctx.stream(),
                                 q->nb[1] / 2, q->nb[2] / 2, q->nb[3] / 2,
                                 k->nb[1] / 2, k->nb[2] / 2, k->nb[3] / 2));
    return true;
}
#else
bool ggml_cuda_flash_attn_ext_pixal_fa2(ggml_backend_cuda_context &, ggml_tensor *) { return false; }
#endif
