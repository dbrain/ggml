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
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fe = cudnn_frontend;

#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4
#define SEQ_LEN_Q_UID 5
#define SEQ_LEN_KV_UID 6

bool ggml_cuda_cudnn_available() { return true; }

// Optional pre-build workspace cap for cuDNN SDPA engine selection. This filters
// candidate engine configs before build_plans(), unlike checking get_workspace_size
// afterward, so it can prevent cuDNN from selecting/building a plan that triggers
// large context-level reservations.
static int64_t sdpa_ws_cap() {
    static int64_t c = -1;
    if (c < 0) {
        const char * e = getenv("GGML_CUDNN_ATTN_WS_MB");
        c = (e && atoll(e) > 0) ? ((int64_t) atoll(e) << 20) : 0;
    }
    return c;
}

static int sdpa_timing_limit() {
    static int limit = -1;
    if (limit < 0) {
        const char * e = getenv("GGML_CUDNN_ATTN_TIMING");
        limit = e ? std::max(0, atoi(e)) : 0;
    }
    return limit;
}

static int sdpa_timing_skip() {
    static int skip = -1;
    if (skip < 0) {
        const char * e = getenv("GGML_CUDNN_ATTN_TIMING_SKIP");
        skip = e ? std::max(0, atoi(e)) : 0;
    }
    return skip;
}

static int64_t sdpa_timing_min_lq() {
    static int64_t min_lq = -1;
    if (min_lq < 0) {
        const char * e = getenv("GGML_CUDNN_ATTN_TIMING_MIN_LQ");
        min_lq = e ? std::max<int64_t>(0, atoll(e)) : 0;
    }
    return min_lq;
}

static int64_t sdpa_ncu_hot_min_lq() {
    static int64_t min_lq = -1;
    if (min_lq < 0) {
        const char * e = getenv("GGML_CUDNN_NCU_HOT_MIN_LQ");
        min_lq = e ? std::max<int64_t>(0, atoll(e)) : 0;
    }
    return min_lq;
}

static bool sdpa_build_all_plans() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * e = getenv("GGML_CUDNN_ATTN_BUILD_ALL_PLANS");
        enabled = e && atoi(e) ? 1 : 0;
    }
    return enabled != 0;
}

// Permute+convert cuDNN's BHSD F16 output -> ggml's BSHD F32 dst.
// in : [N,H,L,D] contiguous (BHSD), half. out: ggml dst ne=[D,H,L,N], i.e. memory
// D inner, H next, L next, N outer (BSHD). out[n,l,h,d] = in[n,h,l,d].
static __global__ void cudnn_o_bhsd_half_to_bshd_f32(
        const half * __restrict__ in, float * __restrict__ out,
        int N, int H, int L_out, int L_in, int D) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long tot = (long)N * H * L_out * D;
    if (idx >= tot) return;
    // decode BHSD index
    int d = idx % D;
    long t = idx / D;
    int l = t % L_out; t /= L_out;
    int h = t % H; t /= H;
    int n = t;
    long in_o = (((long)n * H + h) * L_in + l) * D + d;
    // BSHD output offset: ((n*L + l)*H + h)*D + d
    long o = (((long)n * L_out + l) * H + h) * D + d;
    out[o] = __half2float(in[in_o]);
}

// F16-dst variant: keep the cuDNN output F16 (permute only, NO upcast) so the LTX_DIT_F16
// residual stream stays F16 across the attn->proj boundary (drops the bhsd->F32 cast and
// keeps to_out on the F16 activation-quant path). Used when the flash node's dst is F16.
static __global__ void cudnn_o_bhsd_half_to_bshd_f16(
        const half * __restrict__ in, half * __restrict__ out,
        int N, int H, int L_out, int L_in, int D) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long tot = (long)N * H * L_out * D;
    if (idx >= tot) return;
    int d = idx % D;
    long t = idx / D;
    int l = t % L_out; t /= L_out;
    int h = t % H; t /= H;
    int n = t;
    long in_o = (((long)n * H + h) * L_in + l) * D + d;
    long o = (((long)n * L_out + l) * H + h) * D + d;
    out[o] = in[in_o];
}

// bf16-IO variants (WAN_ATTN_BF16): the SDPA scratch O is bf16; permute BHSD->BSHD and
// convert to the ggml dst dtype (F32 upcast, or F16 for the residual stream).
static __global__ void cudnn_o_bhsd_bf16_to_bshd_f32(
        const nv_bfloat16 * __restrict__ in, float * __restrict__ out,
        int N, int H, int L_out, int L_in, int D) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long tot = (long)N * H * L_out * D;
    if (idx >= tot) return;
    int d = idx % D;
    long t = idx / D;
    int l = t % L_out; t /= L_out;
    int h = t % H; t /= H;
    int n = t;
    long in_o = (((long)n * H + h) * L_in + l) * D + d;
    long o = (((long)n * L_out + l) * H + h) * D + d;
    out[o] = __bfloat162float(in[in_o]);
}
static __global__ void cudnn_o_bhsd_bf16_to_bshd_f16(
        const nv_bfloat16 * __restrict__ in, half * __restrict__ out,
        int N, int H, int L_out, int L_in, int D) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long tot = (long)N * H * L_out * D;
    if (idx >= tot) return;
    int d = idx % D;
    long t = idx / D;
    int l = t % L_out; t /= L_out;
    int h = t % H; t /= H;
    int n = t;
    long in_o = (((long)n * H + h) * L_in + l) * D + d;
    long o = (((long)n * L_out + l) * H + h) * D + d;
    out[o] = __float2half(__bfloat162float(in[in_o]));
}

template <typename T>
static __global__ void bhsd_pad_seq_kernel(
        const T * __restrict__ in, T * __restrict__ out,
        int N, int H, int L_in, int L_out, int D) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long tot = (long)N * H * L_in * D;
    if (idx >= tot) return;
    int d = idx % D;
    long t = idx / D;
    int l = t % L_in; t /= L_in;
    int h = t % H; t /= H;
    int n = t;
    out[(((long)n * H + h) * L_out + l) * D + d] = in[idx];
}

static __global__ void fill_seq_len_kernel(int32_t * out, int n, int value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = value;
    }
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
    int plan_index = -1;
};

struct sdpa_key {
    int64_t B, H, Lq, Lkv, D;   // Lq==Lkv for self-attn; Lq!=Lkv for cross-attn (LTX-AV a2v/v2a)
    int     io_half;   // 1 == fp16 io, 0 == bf16 (reserved)
    int     padding_mask;
    bool operator==(const sdpa_key & o) const {
        return B == o.B && H == o.H && Lq == o.Lq && Lkv == o.Lkv && D == o.D &&
               io_half == o.io_half && padding_mask == o.padding_mask;
    }
};
struct sdpa_key_hash {
    size_t operator()(const sdpa_key & k) const {
        size_t h = 1469598103934665603ull;
        for (int64_t v : {k.B, k.H, k.Lq, k.Lkv, k.D, (int64_t)k.io_half, (int64_t)k.padding_mask}) {
            h ^= (size_t)v; h *= 1099511628211ull;
        }
        return h;
    }
};

static int sdpa_plan_index_for(const sdpa_key & key) {
    static int requested = -2;
    static int64_t min_lq = -1;
    if (requested == -2) {
        const char * e = getenv("GGML_CUDNN_ATTN_PLAN_INDEX");
        requested = e ? atoi(e) : -1;
        const char * min_e = getenv("GGML_CUDNN_ATTN_PLAN_MIN_LQ");
        min_lq = min_e ? std::max<int64_t>(0, atoll(min_e)) : 0;
    }
    return requested >= 0 && key.Lq >= min_lq ? requested : -1;
}

static std::mutex g_plan_mtx;
static std::unordered_map<sdpa_key, cudnn_sdpa_plan, sdpa_key_hash> g_plan_cache;

static bool cudnn_sdpa_vram_trace() {
    return getenv("LONGCAT_VRAM_BREAKDOWN") || getenv("GGML_CUDNN_TRACE");
}

static bool cudnn_sdpa_exec_trace() {
    return getenv("GGML_CUDNN_EXEC_TRACE") || getenv("GGML_CUDNN_ATTN_EXEC_TRACE");
}

static bool cudnn_op_trace() {
    auto enabled = [](const char * name) {
        const char * value = getenv(name);
        return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0;
    };
    return enabled("GGML_CUDNN_OP_TRACE") || enabled("GGML_CUDNN_TRACE");
}

static const std::vector<int64_t> & sdpa_buckets() {
    static std::vector<int64_t> buckets = [] {
        std::vector<int64_t> out;
        const char * e = getenv("GGML_CUDNN_ATTN_BUCKETS");
        if (e != nullptr && e[0] != '\0') {
            const char * p = e;
            while (*p) {
                char * end = nullptr;
                long long v = strtoll(p, &end, 10);
                if (v > 0) {
                    out.push_back((int64_t)v);
                }
                if (end == p || *end == '\0') {
                    break;
                }
                p = (*end == ',') ? end + 1 : end;
            }
        } else if (getenv("GGML_CUDNN_ATTN_BUCKET") != nullptr) {
            // LTX continuation defaults from measured kernel-library fanout:
            // small cross-attn (127/152), text/self 1024, base 8160, hires refine 32640/38760.
            out = {160, 1024, 8160, 38760};
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }();
    return buckets;
}

// Round a real seq len UP to the smallest bucket B >= len. buckets is sorted
// ascending, so the first B with len <= B is the tightest cover: this catches BOTH
// sub-min lengths (127/152 -> 160) AND in-between lengths (9690/32640 -> 38760), not
// just exact-bucket hits. Applied independently to Lq and Lkv (cross-attn has Lq!=Lkv;
// both dims fork cuDNN kernel variants, so both must land on a bucket). When bucketing
// is OFF (empty list) the loop is a no-op and len passes through unchanged (byte-identical).
static int64_t sdpa_bucket_len(int64_t len) {
    const std::vector<int64_t> & bk = sdpa_buckets();
    for (int64_t b : bk) {
        if (len <= b) {
            return b;
        }
    }
    // len exceeds the largest bucket: there is no bucket to pad UP to (padding DOWN would
    // drop real tokens), so this shape runs un-bucketed with its own kernel set. Warn once
    // per distinct uncovered length so a still-fanned-out shape is visible in the trace ->
    // add a covering value via GGML_CUDNN_ATTN_BUCKETS to collapse it. (Silent when bucketing
    // is OFF: bk is empty, so we never reach here for the default byte-identical path.)
    if (!bk.empty()) {
        static std::mutex warn_mtx;
        static std::unordered_set<int64_t> warned;
        std::lock_guard<std::mutex> lk(warn_mtx);
        if (warned.insert(len).second) {
            fprintf(stderr,
                    "[cudnn-sdpa-bucket] seq len %lld exceeds largest bucket %lld -- running "
                    "un-bucketed (own kernel set); add a covering value to GGML_CUDNN_ATTN_BUCKETS.\n",
                    (long long) len, (long long) bk.back());
        }
    }
    return len;
}

static uint64_t next_cudnn_sdpa_op_id() {
    static std::atomic<uint64_t> id{1};
    return id.fetch_add(1, std::memory_order_relaxed);
}

struct nvtx_range {
    explicit nvtx_range(const std::string & label) { nvtxRangePushA(label.c_str()); }
    ~nvtx_range() { nvtxRangePop(); }
};

// Drop every cached SDPA plan. Some cuDNN-internal device reservations appear to
// be handle-owned rather than graph-owned, so ggml_cuda_cudnn_sdpa_release_handle
// is the stronger boundary reset used by the public API.
void ggml_cuda_cudnn_sdpa_release_plans() {
    std::lock_guard<std::mutex> lk(g_plan_mtx);
    g_plan_cache.clear();
}

void ggml_cuda_cudnn_sdpa_release_handle() {
    ggml_cuda_cudnn_sdpa_release_plans();
    if (g_cudnn_handle) {
        if (cudnnDestroy(g_cudnn_handle) != CUDNN_STATUS_SUCCESS) {
            GGML_ABORT("cudnnDestroy failed");
        }
        g_cudnn_handle = nullptr;
    }
}

static cudnn_sdpa_plan & get_or_build_plan(cudnnHandle_t handle, const sdpa_key & key, float scale) {
    std::lock_guard<std::mutex> lk(g_plan_mtx);
    auto it = g_plan_cache.find(key);
    if (it != g_plan_cache.end()) {
        return it->second;
    }

    const int64_t B = key.B, H = key.H, Lq = key.Lq, Lkv = key.Lkv, D = key.D;
    size_t free_before = 0;
    const bool trace_vram = cudnn_sdpa_vram_trace();
    if (trace_vram) {
        size_t total_before = 0;
        cudaMemGetInfo(&free_before, &total_before);
    }

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
    if (key.padding_mask) {
        auto seq_q = graph->tensor(fe::graph::Tensor_attributes()
                .set_name("seq_q").set_uid(SEQ_LEN_Q_UID)
                .set_dim({B, 1, 1, 1}).set_stride({1, 1, 1, 1})
                .set_data_type(fe::DataType_t::INT32));
        auto seq_kv = graph->tensor(fe::graph::Tensor_attributes()
                .set_name("seq_kv").set_uid(SEQ_LEN_KV_UID)
                .set_dim({B, 1, 1, 1}).set_stride({1, 1, 1, 1})
                .set_data_type(fe::DataType_t::INT32));
        sdpa_opts.set_padding_mask(true).set_seq_len_q(seq_q).set_seq_len_kv(seq_kv);
    }

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
    if (sdpa_ws_cap() > 0) {
        graph->deselect_workspace_greater_than(sdpa_ws_cap());
    }
    if (!graph->check_support(handle).is_good())                      { GGML_ABORT("cudnn sdpa check_support failed (no SDPA engine for this arch?)"); }
    const int plan_index = sdpa_plan_index_for(key);
    const auto build_policy = sdpa_build_all_plans() || plan_index >= 0
        ? fe::BuildPlanPolicy_t::ALL
        : fe::BuildPlanPolicy_t::HEURISTICS_CHOICE;
    if (!graph->build_plans(handle, build_policy).is_good())          { GGML_ABORT("cudnn sdpa build_plans failed"); }

    int64_t ws = 0;
    if (!graph->get_workspace_size(ws).is_good()) { GGML_ABORT("cudnn sdpa get_workspace_size failed"); }

    cudnn_sdpa_plan plan;
    plan.graph     = graph;
    plan.workspace = ws;
    plan.plan_index = plan_index;
    auto [ins, ok] = g_plan_cache.emplace(key, std::move(plan));
    GGML_UNUSED(ok);
    if (trace_vram) {
        size_t free_after = 0, total_after = 0;
        cudaMemGetInfo(&free_after, &total_after);
        fprintf(stderr,
                "[cudnn-sdpa-plan] build #%zu B=%lld H=%lld Lq=%lld Lkv=%lld D=%lld io=%s ws=%lld MB "
                "padmask=%d free %.1f -> %.1f MB (delta %+.1f MB, used %.1f MB)\n",
                g_plan_cache.size(),
                (long long) B, (long long) H, (long long) Lq, (long long) Lkv, (long long) D,
                key.io_half ? "f16" : "bf16", (long long) (ws >> 20),
                key.padding_mask,
                free_before / 1048576.0, free_after / 1048576.0,
                ((double) free_after - (double) free_before) / 1048576.0,
                (total_after - free_after) / 1048576.0);
    }
    return ins->second;
}

// GGML_F8_DBG: device max-abs of an attention operand, to confirm whether Q (unprotected by
// kv_scale) exceeds F16 max (65504) before cuDNN casts it to F16 — the suspected NaN source.
static __global__ void f8dbg_amax_f16_k(const half * __restrict__ x, size_t n, unsigned int * __restrict__ out) {
    float local = 0.f;
    for (size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x; i < n; i += (size_t)gridDim.x*blockDim.x)
        local = fmaxf(local, fabsf(__half2float(x[i])));
    __shared__ float s[256]; s[threadIdx.x]=local; __syncthreads();
    for (int st=blockDim.x/2; st>0; st>>=1){ if(threadIdx.x<st) s[threadIdx.x]=fmaxf(s[threadIdx.x],s[threadIdx.x+st]); __syncthreads(); }
    if (threadIdx.x==0) atomicMax(out, __float_as_uint(s[0]));
}
static __global__ void f8dbg_amax_f32_k(const float * __restrict__ x, size_t n, unsigned int * __restrict__ out) {
    float local = 0.f;
    for (size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x; i < n; i += (size_t)gridDim.x*blockDim.x)
        local = fmaxf(local, fabsf(x[i]));
    __shared__ float s[256]; s[threadIdx.x]=local; __syncthreads();
    for (int st=blockDim.x/2; st>0; st>>=1){ if(threadIdx.x<st) s[threadIdx.x]=fmaxf(s[threadIdx.x],s[threadIdx.x+st]); __syncthreads(); }
    if (threadIdx.x==0) atomicMax(out, __float_as_uint(s[0]));
}
static float f8dbg_amax_dev(const void * d, bool is_f16, size_t n, cudaStream_t stream) {
    static unsigned int * dm = nullptr;
    if (dm == nullptr && cudaMalloc((void**)&dm, sizeof(unsigned int)) != cudaSuccess) return -1.f;
    cudaMemsetAsync(dm, 0, sizeof(unsigned int), stream);
    const int thr = 256; unsigned int gr = (unsigned int)((n + thr - 1)/thr); if (gr > 1024u) gr = 1024u; if (gr == 0) gr = 1;
    if (is_f16) f8dbg_amax_f16_k<<<gr, thr, 0, stream>>>((const half*)d, n, dm);
    else        f8dbg_amax_f32_k<<<gr, thr, 0, stream>>>((const float*)d, n, dm);
    unsigned int bits = 0; cudaMemcpyAsync(&bits, dm, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); float f; memcpy(&f, &bits, sizeof(float)); return f;
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

    static int sdpa_timing_count = 0;
    static int sdpa_timing_recorded = 0;
    const int sdpa_timing_id = sdpa_timing_count++;
    const bool time_sdpa_window = sdpa_timing_id >= sdpa_timing_skip()
        && sdpa_timing_id < sdpa_timing_skip() + sdpa_timing_limit();
    const int64_t timing_min_lq = sdpa_timing_min_lq();
    const bool time_sdpa_shape = (timing_min_lq > 0
            ? sdpa_timing_recorded < sdpa_timing_limit()
            : time_sdpa_window)
        && Lq >= timing_min_lq;
    cudaEvent_t timing_events[4] = {};
    if (time_sdpa_shape) {
        for (cudaEvent_t & event : timing_events) {
            CUDA_CHECK(cudaEventCreate(&event));
        }
        CUDA_CHECK(cudaEventRecord(timing_events[0], stream));
    }

    // GGML_F8_DBG: report max|Q|/|K|/|V| for the first few SDPA calls. Q>65504 confirms the
    // F16-cast overflow (Q is unprotected by kv_scale) -> the DiT-NaN root cause.
    {
        static int s_dbg = -1;
        if (s_dbg < 0) { const char * e = getenv("GGML_F8_DBG"); s_dbg = (e && atoi(e)) ? 1 : 0; }
        if (s_dbg) {
            static int s_n = 0;
            if (s_n++ < 8) {
                const float aq = f8dbg_amax_dev(Q->data, Q->type == GGML_TYPE_F16, ggml_nelements(Q), stream);
                const float ak = f8dbg_amax_dev(K->data, K->type == GGML_TYPE_F16, ggml_nelements(K), stream);
                const float av = f8dbg_amax_dev(V->data, V->type == GGML_TYPE_F16, ggml_nelements(V), stream);
                fprintf(stderr, "[F8_DBG] cudnn-SDPA #%d  max|Q|=%.6g(%s) max|K|=%.6g max|V|=%.6g  F16max=65504  %s  D=%ld Lq=%ld Lkv=%ld H=%ld\n",
                        s_n - 1, aq, Q->type == GGML_TYPE_F16 ? "f16" : "f32", ak, av,
                        aq > 65504.f ? "*** Q>F16MAX -> cast overflows to inf ***" : "(Q fits F16)",
                        (long)D, (long)Lq, (long)Lkv, (long)H);
            }
        }
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
    if (time_sdpa_shape) {
        CUDA_CHECK(cudaEventRecord(timing_events[1], stream));
    }

    const int64_t Lq_plan  = sdpa_bucket_len(Lq);
    const int64_t Lkv_plan = sdpa_bucket_len(Lkv);
    const bool use_mask    = !sdpa_buckets().empty();
    const bool pad_q       = Lq_plan != Lq;
    const bool pad_kv      = Lkv_plan != Lkv;

    sdpa_key key{N, H, Lq_plan, Lkv_plan, D, io_half, use_mask ? 1 : 0};
    cudnn_sdpa_plan & plan = get_or_build_plan(handle, key, scale);

    ggml_cuda_pool_alloc<uint8_t> ws(ctx.pool());
    void * ws_ptr = nullptr;
    if (plan.workspace > 0) {
        ws.alloc((size_t) plan.workspace);
        ws_ptr = ws.get();
    }

    ggml_cuda_pool_alloc<half>        q_pad_h(ctx.pool());
    ggml_cuda_pool_alloc<half>        k_pad_h(ctx.pool());
    ggml_cuda_pool_alloc<half>        v_pad_h(ctx.pool());
    ggml_cuda_pool_alloc<nv_bfloat16> q_pad_b(ctx.pool());
    ggml_cuda_pool_alloc<nv_bfloat16> k_pad_b(ctx.pool());
    ggml_cuda_pool_alloc<nv_bfloat16> v_pad_b(ctx.pool());
    const void * q_exec = q_ptr;
    const void * k_exec = K->data;
    const void * v_exec = V->data;
    if (use_mask && (pad_q || pad_kv)) {
        const int bs = 256;
        if (io_half) {
            if (pad_q) {
                q_pad_h.alloc((size_t)(N * H * Lq_plan * D));
                cudaMemsetAsync(q_pad_h.get(), 0, (size_t)(N * H * Lq_plan * D) * sizeof(half), stream);
                bhsd_pad_seq_kernel<half><<<(int)(((long)N * H * Lq * D + bs - 1) / bs), bs, 0, stream>>>((const half *)q_ptr, q_pad_h.get(), (int)N, (int)H, (int)Lq, (int)Lq_plan, (int)D);
                q_exec = q_pad_h.get();
            }
            if (pad_kv) {
                k_pad_h.alloc((size_t)(N * H * Lkv_plan * D));
                v_pad_h.alloc((size_t)(N * H * Lkv_plan * D));
                cudaMemsetAsync(k_pad_h.get(), 0, (size_t)(N * H * Lkv_plan * D) * sizeof(half), stream);
                cudaMemsetAsync(v_pad_h.get(), 0, (size_t)(N * H * Lkv_plan * D) * sizeof(half), stream);
                bhsd_pad_seq_kernel<half><<<(int)(((long)N * H * Lkv * D + bs - 1) / bs), bs, 0, stream>>>((const half *)K->data, k_pad_h.get(), (int)N, (int)H, (int)Lkv, (int)Lkv_plan, (int)D);
                bhsd_pad_seq_kernel<half><<<(int)(((long)N * H * Lkv * D + bs - 1) / bs), bs, 0, stream>>>((const half *)V->data, v_pad_h.get(), (int)N, (int)H, (int)Lkv, (int)Lkv_plan, (int)D);
                k_exec = k_pad_h.get();
                v_exec = v_pad_h.get();
            }
        } else {
            if (pad_q) {
                q_pad_b.alloc((size_t)(N * H * Lq_plan * D));
                cudaMemsetAsync(q_pad_b.get(), 0, (size_t)(N * H * Lq_plan * D) * sizeof(nv_bfloat16), stream);
                bhsd_pad_seq_kernel<nv_bfloat16><<<(int)(((long)N * H * Lq * D + bs - 1) / bs), bs, 0, stream>>>((const nv_bfloat16 *)q_ptr, q_pad_b.get(), (int)N, (int)H, (int)Lq, (int)Lq_plan, (int)D);
                q_exec = q_pad_b.get();
            }
            if (pad_kv) {
                k_pad_b.alloc((size_t)(N * H * Lkv_plan * D));
                v_pad_b.alloc((size_t)(N * H * Lkv_plan * D));
                cudaMemsetAsync(k_pad_b.get(), 0, (size_t)(N * H * Lkv_plan * D) * sizeof(nv_bfloat16), stream);
                cudaMemsetAsync(v_pad_b.get(), 0, (size_t)(N * H * Lkv_plan * D) * sizeof(nv_bfloat16), stream);
                bhsd_pad_seq_kernel<nv_bfloat16><<<(int)(((long)N * H * Lkv * D + bs - 1) / bs), bs, 0, stream>>>((const nv_bfloat16 *)K->data, k_pad_b.get(), (int)N, (int)H, (int)Lkv, (int)Lkv_plan, (int)D);
                bhsd_pad_seq_kernel<nv_bfloat16><<<(int)(((long)N * H * Lkv * D + bs - 1) / bs), bs, 0, stream>>>((const nv_bfloat16 *)V->data, v_pad_b.get(), (int)N, (int)H, (int)Lkv, (int)Lkv_plan, (int)D);
                k_exec = k_pad_b.get();
                v_exec = v_pad_b.get();
            }
        }
    }

    // SDPA writes O as BHSD (io dtype) into a scratch buffer (O seqlen = Lq_plan).
    ggml_cuda_pool_alloc<half>        o_bhsd_h(ctx.pool());
    ggml_cuda_pool_alloc<nv_bfloat16> o_bhsd_b(ctx.pool());
    void * o_ptr = nullptr;
    if (io_half) { o_bhsd_h.alloc((size_t)(N * H * Lq_plan * D)); o_ptr = o_bhsd_h.get(); }
    else         { o_bhsd_b.alloc((size_t)(N * H * Lq_plan * D)); o_ptr = o_bhsd_b.get(); }

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> vpack = {
        {Q_UID, const_cast<void *>(q_exec)},
        {K_UID, const_cast<void *>(k_exec)},
        {V_UID, const_cast<void *>(v_exec)},
        {O_UID, o_ptr},
    };
    ggml_cuda_pool_alloc<int32_t> seq_q(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> seq_kv(ctx.pool());
    if (use_mask) {
        seq_q.alloc((size_t)N);
        seq_kv.alloc((size_t)N);
        const int bs = 256;
        fill_seq_len_kernel<<<(int)((N + bs - 1) / bs), bs, 0, stream>>>(seq_q.get(), (int)N, (int)Lq);
        fill_seq_len_kernel<<<(int)((N + bs - 1) / bs), bs, 0, stream>>>(seq_kv.get(), (int)N, (int)Lkv);
        vpack[SEQ_LEN_Q_UID]  = seq_q.get();
        vpack[SEQ_LEN_KV_UID] = seq_kv.get();
    }

    const uint64_t op_id = next_cudnn_sdpa_op_id();
    char op_label[224];
    const bool ncu_hot = sdpa_ncu_hot_min_lq() > 0 && Lq >= sdpa_ncu_hot_min_lq();
    if (ncu_hot) {
        snprintf(op_label, sizeof(op_label), "cudnn-hot-sdpa");
    } else {
        snprintf(op_label, sizeof(op_label),
                 "cudnn-op id=%llu kind=sdpa B=%lld H=%lld Lq=%lld/%lld Lkv=%lld/%lld D=%lld io=%s bucket=%d ws=%lldMB",
                 (unsigned long long)op_id, (long long)N, (long long)H, (long long)Lq, (long long)Lq_plan,
                 (long long)Lkv, (long long)Lkv_plan, (long long)D, io_half ? "f16" : "bf16",
                 (int)use_mask, (long long)(plan.workspace >> 20));
    }
    nvtx_range op_range(op_label);
    if (cudnn_op_trace()) {
        fprintf(stderr,
                "[cudnn-op-begin] id=%llu kind=sdpa B=%lld H=%lld Lq=%lld planLq=%lld Lkv=%lld planLkv=%lld D=%lld io=%s bucket=%d ws=%lldMB t_us=%lld\n",
                (unsigned long long)op_id, (long long)N, (long long)H, (long long)Lq, (long long)Lq_plan,
                (long long)Lkv, (long long)Lkv_plan, (long long)D, io_half ? "f16" : "bf16",
                (int)use_mask, (long long)(plan.workspace >> 20),
                (long long)ggml_time_us());
    }

    size_t free_before_exec = 0;
    const bool trace_exec = cudnn_sdpa_exec_trace();
    if (trace_exec) {
        size_t total_before_exec = 0;
        cudaStreamSynchronize(stream);
        cudaMemGetInfo(&free_before_exec, &total_before_exec);
    }
    const auto execute_status = plan.plan_index >= 0
        ? plan.graph->execute_plan_at_index(handle, vpack, ws_ptr, plan.plan_index)
        : plan.graph->execute(handle, vpack, ws_ptr);
    if (!execute_status.is_good()) {
        GGML_ABORT("cudnn sdpa execute failed");
    }
    if (time_sdpa_shape) {
        CUDA_CHECK(cudaEventRecord(timing_events[2], stream));
    }
    if (cudnn_op_trace()) {
        fprintf(stderr, "[cudnn-op-end] id=%llu kind=sdpa t_us=%lld\n",
                (unsigned long long)op_id, (long long)ggml_time_us());
    }
    if (trace_exec) {
        cudaStreamSynchronize(stream);
        size_t free_after_exec = 0, total_after_exec = 0;
        cudaMemGetInfo(&free_after_exec, &total_after_exec);
        fprintf(stderr,
                "[cudnn-sdpa-exec] B=%lld H=%lld Lq=%lld Lkv=%lld D=%lld io=%s ws=%lld MB "
                "planLq=%lld planLkv=%lld bucket=%d free %.1f -> %.1f MB (delta %+.1f MB, used %.1f MB)\n",
                (long long)N, (long long)H, (long long)Lq, (long long)Lkv, (long long)D,
                io_half ? "f16" : "bf16", (long long)(plan.workspace >> 20),
                (long long)Lq_plan, (long long)Lkv_plan, (int)use_mask,
                free_before_exec / 1048576.0, free_after_exec / 1048576.0,
                ((double)free_after_exec - (double)free_before_exec) / 1048576.0,
                (total_after_exec - free_after_exec) / 1048576.0);
    }

    // Permute BHSD (io dtype) -> BSHD dst (O seqlen = Lq). F32 dst = upcast (prod default);
    // F16 dst = keep the residual stream F16 through attn->proj.
    const long tot = (long)N * H * Lq * D;
    const int  bs  = 256;
    const int  gs  = (int)((tot + bs - 1) / bs);
    if (io_half) {
        if (dst->type == GGML_TYPE_F16) {
            cudnn_o_bhsd_half_to_bshd_f16<<<gs, bs, 0, stream>>>(
                o_bhsd_h.get(), (half *) dst->data, (int)N, (int)H, (int)Lq, (int)Lq_plan, (int)D);
        } else {
            cudnn_o_bhsd_half_to_bshd_f32<<<gs, bs, 0, stream>>>(
                o_bhsd_h.get(), (float *) dst->data, (int)N, (int)H, (int)Lq, (int)Lq_plan, (int)D);
        }
    } else {
        if (dst->type == GGML_TYPE_F16) {
            cudnn_o_bhsd_bf16_to_bshd_f16<<<gs, bs, 0, stream>>>(
                o_bhsd_b.get(), (half *) dst->data, (int)N, (int)H, (int)Lq, (int)Lq_plan, (int)D);
        } else {
            cudnn_o_bhsd_bf16_to_bshd_f32<<<gs, bs, 0, stream>>>(
                o_bhsd_b.get(), (float *) dst->data, (int)N, (int)H, (int)Lq, (int)Lq_plan, (int)D);
        }
    }
    if (time_sdpa_shape) {
        CUDA_CHECK(cudaEventRecord(timing_events[3], stream));
        CUDA_CHECK(cudaEventSynchronize(timing_events[3]));
        float q_cast_ms = 0.0f, sdpa_ms = 0.0f, output_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&q_cast_ms, timing_events[0], timing_events[1]));
        CUDA_CHECK(cudaEventElapsedTime(&sdpa_ms, timing_events[1], timing_events[2]));
        CUDA_CHECK(cudaEventElapsedTime(&output_ms, timing_events[2], timing_events[3]));
        fprintf(stderr,
                "[cudnn-sdpa-timing] B=%lld H=%lld Lq=%lld Lkv=%lld D=%lld q_cast=%.3fms sdpa=%.3fms output=%.3fms total=%.3fms\n",
                (long long) N, (long long) H, (long long) Lq, (long long) Lkv, (long long) D,
                q_cast_ms, sdpa_ms, output_ms, q_cast_ms + sdpa_ms + output_ms);
        for (cudaEvent_t event : timing_events) {
            CUDA_CHECK(cudaEventDestroy(event));
        }
        ++sdpa_timing_recorded;
    }
}

#else  // !GGML_CUDNN

bool ggml_cuda_cudnn_available() { return false; }

void ggml_cuda_cudnn_sdpa_release_plans() {}
void ggml_cuda_cudnn_sdpa_release_handle() {}

void ggml_cuda_flash_attn_ext_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("ggml-cuda built without GGML_CUDNN");
}

#endif // GGML_CUDNN
