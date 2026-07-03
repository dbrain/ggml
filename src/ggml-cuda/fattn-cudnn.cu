// cuDNN fused SDPA flash-attention variant — see fattn-cudnn.cuh.
//
// Lifts the proven cudnn-frontend SDPA forward graph from
// flux2.cpp/spike_cutlass_fp4/attn_golden.cu (cosine 1.000000 on the flux2-klein
// DiT shape [B=1,H=24,S=4608,D=128], fp16, no mask, scale=1/sqrt(D)). The graph is
// built once and cached by (B,H,L,D,dtype); execution is 64x/render so build cost
// amortizes. Output is written straight into ggml's F32 dst with BSHD strides so no
// extra permute/convert kernel is needed.

#include "fattn-cudnn.cuh"
#include "fattn-fp8.cuh"

#ifdef GGML_CUDNN

#include "convert.cuh"

#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace fe = cudnn_frontend;

#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4

bool ggml_cuda_cudnn_available() { return true; }

// Permute+convert cuDNN's BHSD F16 output -> ggml's BSHD F32 dst.
// in : [N,H,L,D] contiguous (BHSD), half. out: ggml dst ne=[D,H,L,N], i.e. memory
// D inner, H next, L next, N outer (BSHD). out[n,l,h,d] = in[n,h,l,d].
static __global__ void cudnn_o_bhsd_half_to_bshd_f32(
        const half * __restrict__ in, float * __restrict__ out,
        int N, int H, int L, int D) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long tot = (long)N * H * L * D;
    if (idx >= tot) return;
    // decode BHSD index
    int d = idx % D;
    long t = idx / D;
    int l = t % L; t /= L;
    int h = t % H; t /= H;
    int n = t;
    // BSHD output offset: ((n*L + l)*H + h)*D + d
    long o = (((long)n * L + l) * H + h) * D + d;
    out[o] = __half2float(in[idx]);
}

// F16-dst variant: keep the cuDNN output F16 (permute only, NO upcast) so the LTX_DIT_F16
// residual stream stays F16 across the attn->proj boundary (drops the bhsd->F32 cast and
// keeps to_out on the F16 activation-quant path). Used when the flash node's dst is F16.
static __global__ void cudnn_o_bhsd_half_to_bshd_f16(
        const half * __restrict__ in, half * __restrict__ out,
        int N, int H, int L, int D) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long tot = (long)N * H * L * D;
    if (idx >= tot) return;
    int d = idx % D;
    long t = idx / D;
    int l = t % L; t /= L;
    int h = t % H; t /= H;
    int n = t;
    long o = (((long)n * L + l) * H + h) * D + d;
    out[o] = in[idx];
}

// bf16-IO variants (WAN_ATTN_BF16): the SDPA scratch O is bf16; permute BHSD->BSHD and
// convert to the ggml dst dtype (F32 upcast, or F16 for the residual stream).
static __global__ void cudnn_o_bhsd_bf16_to_bshd_f32(
        const nv_bfloat16 * __restrict__ in, float * __restrict__ out,
        int N, int H, int L, int D) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long tot = (long)N * H * L * D;
    if (idx >= tot) return;
    int d = idx % D;
    long t = idx / D;
    int l = t % L; t /= L;
    int h = t % H; t /= H;
    int n = t;
    long o = (((long)n * L + l) * H + h) * D + d;
    out[o] = __bfloat162float(in[idx]);
}
static __global__ void cudnn_o_bhsd_bf16_to_bshd_f16(
        const nv_bfloat16 * __restrict__ in, half * __restrict__ out,
        int N, int H, int L, int D) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long tot = (long)N * H * L * D;
    if (idx >= tot) return;
    int d = idx % D;
    long t = idx / D;
    int l = t % L; t /= L;
    int h = t % H; t /= H;
    int n = t;
    long o = (((long)n * L + l) * H + h) * D + d;
    out[o] = __float2half(__bfloat162float(in[idx]));
}

// One cuDNN handle per thread (the backend runs the graph on a single worker thread).
static thread_local cudnnHandle_t g_cudnn_handle = nullptr;
static cudnnHandle_t get_cudnn_handle() {
    if (!g_cudnn_handle) {
        if (cudnnCreate(&g_cudnn_handle) != CUDNN_STATUS_SUCCESS) {
            GGML_ABORT("cudnnCreate failed");
        }
    }
    return g_cudnn_handle;
}

struct cudnn_sdpa_plan {
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t workspace = 0;
};

struct sdpa_key {
    int64_t B, H, Lq, Lkv, D;   // Lq==Lkv for self-attn; Lq!=Lkv for cross-attn (LTX-AV a2v/v2a)
    int     io_half;   // 1 == fp16 io, 0 == bf16 (reserved)
    bool operator==(const sdpa_key & o) const {
        return B == o.B && H == o.H && Lq == o.Lq && Lkv == o.Lkv && D == o.D && io_half == o.io_half;
    }
};
struct sdpa_key_hash {
    size_t operator()(const sdpa_key & k) const {
        size_t h = 1469598103934665603ull;
        for (int64_t v : {k.B, k.H, k.Lq, k.Lkv, k.D, (int64_t)k.io_half}) {
            h ^= (size_t)v; h *= 1099511628211ull;
        }
        return h;
    }
};

static std::mutex g_plan_mtx;
static std::unordered_map<sdpa_key, cudnn_sdpa_plan, sdpa_key_hash> g_plan_cache;

static cudnn_sdpa_plan & get_or_build_plan(cudnnHandle_t handle, const sdpa_key & key, float scale) {
    std::lock_guard<std::mutex> lk(g_plan_mtx);
    auto it = g_plan_cache.find(key);
    if (it != g_plan_cache.end()) {
        return it->second;
    }

    const int64_t B = key.B, H = key.H, Lq = key.Lq, Lkv = key.Lkv, D = key.D;

    // io_half==1 => fp16 IO (prod). io_half==0 => bf16 IO (WAN_ATTN_BF16): run the SDPA in
    // bf16 to avoid the documented FP16 repeated-key attention divergence (the reference's
    // format). Intermediate/compute stay F32, so only the Q/K/V/O storage dtype changes.
    const fe::DataType_t io_dt = key.io_half ? fe::DataType_t::HALF : fe::DataType_t::BFLOAT16;
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_dt)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    // Q/K/V: BHSD contiguous strides (N outer, then H, then L, then D innermost). Q uses the
    // query seqlen Lq; K/V use the key/value seqlen Lkv (Lq==Lkv self-attn, Lq!=Lkv cross-attn).
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q").set_uid(Q_UID)
                 .set_dim({B, H, Lq, D}).set_stride({H * Lq * D, Lq * D, D, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K").set_uid(K_UID)
                 .set_dim({B, H, Lkv, D}).set_stride({H * Lkv * D, Lkv * D, D, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V").set_uid(V_UID)
                 .set_dim({B, H, Lkv, D}).set_stride({H * Lkv * D, Lkv * D, D, 1}));

    auto sdpa_opts = fe::graph::SDPA_attributes().set_name("flash_attention")
                         .set_generate_stats(false)   // inference: no LSE stats
                         .set_attn_scale(scale);
    // no causal mask, no padding mask -> full bidirectional attention

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_opts);
    (void) Stats;
    // O dims [B,H,L,D], standard BHSD F16 into a scratch buffer; a follow-up kernel
    // permutes+converts to ggml's BSHD F32 dst. (Non-standard strided FP32 output is
    // not reliably honored by the SDPA engine, so keep the graph output canonical.)
    O->set_output(true).set_data_type(io_dt)
        .set_dim({B, H, Lq, D}).set_stride({H * Lq * D, Lq * D, D, 1}).set_uid(O_UID);

    if (!graph->validate().is_good())                                 { GGML_ABORT("cudnn sdpa validate failed"); }
    if (!graph->build_operation_graph(handle).is_good())              { GGML_ABORT("cudnn sdpa build_operation_graph failed"); }
    if (!graph->create_execution_plans({fe::HeurMode_t::A}).is_good()){ GGML_ABORT("cudnn sdpa create_execution_plans failed"); }
    if (!graph->check_support(handle).is_good())                      { GGML_ABORT("cudnn sdpa check_support failed (no SDPA engine for this arch?)"); }
    if (!graph->build_plans(handle).is_good())                        { GGML_ABORT("cudnn sdpa build_plans failed"); }

    int64_t ws = 0;
    if (!graph->get_workspace_size(ws).is_good()) { GGML_ABORT("cudnn sdpa get_workspace_size failed"); }

    cudnn_sdpa_plan plan;
    plan.graph     = graph;
    plan.workspace = ws;
    auto [ins, ok] = g_plan_cache.emplace(key, std::move(plan));
    GGML_UNUSED(ok);
    return ins->second;
}

void ggml_cuda_flash_attn_ext_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // ne = [D, L, H, N]. Q has the query seqlen (Lq); K/V have the key/value seqlen (Lkv).
    // For LTX-AV self-attn Lq==Lkv; the a2v/v2a cross-attn has Lq!=Lkv (and no mask), which
    // cuDNN SDPA handles natively (the flux2 borrow only ever saw the self-attn case).
    const int64_t D   = Q->ne[0];
    const int64_t Lq  = Q->ne[1];
    const int64_t H   = Q->ne[2];
    const int64_t N   = Q->ne[3];
    const int64_t Lkv = K->ne[1];

    // io_half==1 => fp16 K/V/Q IO (prod). io_half==0 => bf16 (WAN_ATTN_BF16 casts K/V/Q to
    // bf16 in build_kqv, and the selection in fattn.cu routes bf16 D-in-{64,128} here). K/V
    // must share dtype.
    GGML_ASSERT(K->type == V->type && (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_BF16));
    const int io_half = (K->type == GGML_TYPE_F16) ? 1 : 0;
    GGML_ASSERT(K->ne[0] == D && V->ne[0] == D && V->ne[1] == Lkv);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    float scale = 0.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    // Custom FP8 (e4m3) flash-attention (v2): env-gated, D in {64,128}, mask-free, self- or
    // cross-attn (Lq may != Lkv). Everything else falls through to the cuDNN F16 SDPA below.
    if (getenv("GGML_FP8_ATTN") && (D == 64 || D == 128) && dst->src[3] == nullptr) {
        ggml_cuda_flash_attn_ext_fp8(ctx, dst);
        return;
    }

    if (getenv("GGML_CUDNN_TRACE")) {
        static int n = 0;
        if (n++ < 2) {
            fprintf(stderr, "[cudnn] Q ne=[%ld,%ld,%ld,%ld] nb=[%zu,%zu,%zu,%zu] t=%d\n",
                    (long)Q->ne[0],(long)Q->ne[1],(long)Q->ne[2],(long)Q->ne[3], Q->nb[0],Q->nb[1],Q->nb[2],Q->nb[3], Q->type);
            fprintf(stderr, "[cudnn] K ne=[%ld,%ld,%ld,%ld] nb=[%zu,%zu,%zu,%zu] t=%d\n",
                    (long)K->ne[0],(long)K->ne[1],(long)K->ne[2],(long)K->ne[3], K->nb[0],K->nb[1],K->nb[2],K->nb[3], K->type);
            fprintf(stderr, "[cudnn] V ne=[%ld,%ld,%ld,%ld] nb=[%zu,%zu,%zu,%zu] t=%d\n",
                    (long)V->ne[0],(long)V->ne[1],(long)V->ne[2],(long)V->ne[3], V->nb[0],V->nb[1],V->nb[2],V->nb[3], V->type);
            fprintf(stderr, "[cudnn] O ne=[%ld,%ld,%ld,%ld] nb=[%zu,%zu,%zu,%zu] t=%d scale=%.6f\n",
                    (long)dst->ne[0],(long)dst->ne[1],(long)dst->ne[2],(long)dst->ne[3], dst->nb[0],dst->nb[1],dst->nb[2],dst->nb[3], dst->type, scale);
        }
    }

    cudaStream_t  stream = ctx.stream();
    cudnnHandle_t handle = get_cudnn_handle();
    if (cudnnSetStream(handle, stream) != CUDNN_STATUS_SUCCESS) {
        GGML_ABORT("cudnnSetStream failed");
    }

    // Q IO dtype must match the K/V IO dtype the graph was built for. fp16 path: Q may arrive
    // F32 (cast to F16 here) or F16. bf16 path (io_half==0): build_kqv casts Q to bf16, so Q is
    // already bf16 — pass through (robustly cast F32->bf16 if some caller left it F32).
    const void * q_ptr = Q->data;
    ggml_cuda_pool_alloc<half>        q_f16(ctx.pool());
    ggml_cuda_pool_alloc<nv_bfloat16> q_bf16(ctx.pool());
    if (io_half) {
        if (Q->type == GGML_TYPE_F32) {
            const int64_t nelem = ggml_nelements(Q);
            q_f16.alloc(nelem);
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
            // Q is contiguous (BHSD); a straight element-wise cast preserves layout.
            GGML_ASSERT(ggml_is_contiguous(Q));
            to_fp16((const float *) Q->data, q_f16.get(), nelem, stream);
            q_ptr = q_f16.get();
        } else {
            GGML_ASSERT(Q->type == GGML_TYPE_F16);
        }
    } else {
        if (Q->type == GGML_TYPE_F32) {
            const int64_t nelem = ggml_nelements(Q);
            q_bf16.alloc(nelem);
            to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(Q));
            to_bf16((const float *) Q->data, q_bf16.get(), nelem, stream);
            q_ptr = q_bf16.get();
        } else {
            GGML_ASSERT(Q->type == GGML_TYPE_BF16);
        }
    }

    sdpa_key key{N, H, Lq, Lkv, D, io_half};
    cudnn_sdpa_plan & plan = get_or_build_plan(handle, key, scale);

    ggml_cuda_pool_alloc<uint8_t> ws(ctx.pool());
    void * ws_ptr = nullptr;
    if (plan.workspace > 0) {
        ws.alloc((size_t) plan.workspace);
        ws_ptr = ws.get();
    }

    // SDPA writes O as BHSD (io dtype) into a scratch buffer (O seqlen = Lq).
    ggml_cuda_pool_alloc<half>        o_bhsd_h(ctx.pool());
    ggml_cuda_pool_alloc<nv_bfloat16> o_bhsd_b(ctx.pool());
    void * o_ptr = nullptr;
    if (io_half) { o_bhsd_h.alloc((size_t)(N * H * Lq * D)); o_ptr = o_bhsd_h.get(); }
    else         { o_bhsd_b.alloc((size_t)(N * H * Lq * D)); o_ptr = o_bhsd_b.get(); }

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> vpack = {
        {Q_UID, const_cast<void *>(q_ptr)},
        {K_UID, K->data},
        {V_UID, V->data},
        {O_UID, o_ptr},
    };

    if (!plan.graph->execute(handle, vpack, ws_ptr).is_good()) {
        GGML_ABORT("cudnn sdpa execute failed");
    }

    // Permute BHSD (io dtype) -> BSHD dst (O seqlen = Lq). F32 dst = upcast (prod default);
    // F16 dst = keep the residual stream F16 through attn->proj.
    const long tot = (long)N * H * Lq * D;
    const int  bs  = 256;
    const int  gs  = (int)((tot + bs - 1) / bs);
    if (io_half) {
        if (dst->type == GGML_TYPE_F16) {
            cudnn_o_bhsd_half_to_bshd_f16<<<gs, bs, 0, stream>>>(
                o_bhsd_h.get(), (half *) dst->data, (int)N, (int)H, (int)Lq, (int)D);
        } else {
            cudnn_o_bhsd_half_to_bshd_f32<<<gs, bs, 0, stream>>>(
                o_bhsd_h.get(), (float *) dst->data, (int)N, (int)H, (int)Lq, (int)D);
        }
    } else {
        if (dst->type == GGML_TYPE_F16) {
            cudnn_o_bhsd_bf16_to_bshd_f16<<<gs, bs, 0, stream>>>(
                o_bhsd_b.get(), (half *) dst->data, (int)N, (int)H, (int)Lq, (int)D);
        } else {
            cudnn_o_bhsd_bf16_to_bshd_f32<<<gs, bs, 0, stream>>>(
                o_bhsd_b.get(), (float *) dst->data, (int)N, (int)H, (int)Lq, (int)D);
        }
    }
}

#else  // !GGML_CUDNN

bool ggml_cuda_cudnn_available() { return false; }

void ggml_cuda_flash_attn_ext_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("ggml-cuda built without GGML_CUDNN");
}

#endif // GGML_CUDNN
