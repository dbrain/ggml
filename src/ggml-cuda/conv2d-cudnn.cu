// cuDNN implicit-GEMM conv2d — see conv2d-cudnn.cuh.
//
// Lifts the proven cudnn-frontend conv-fprop graph from
// flux2.cpp/spike_cutlass_fp4/conv_golden.cu (3x3 s1 p1, NHWC fp16, cosine 1.0 on the
// flux2-klein VAE decoder shapes). Graph is built once per (N,C,H,W,Cout,KH,KW,stride,
// pad,dil) shape-key and cached; weight is reordered KCRS->KRSC f16 once per weight
// (cached by device ptr). Activations are transposed/cast NCHW-f32 <-> NHWC-f16 around
// the call (transpose cost measured; cuDNN is ~3.6x so there is headroom).

#include "conv2d-cudnn.cuh"

#ifdef GGML_CUDNN

#include <cuda_fp16.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;

#define X_UID 1
#define W_UID 2
#define Y_UID 3

bool ggml_cuda_conv2d_cudnn_available() { return true; }

static int64_t conv2d_ws_cap() {
    static int64_t c = -1;
    if (c < 0) {
        const char * e = getenv("GGML_CUDNN_CONV_WS_MB");
        c = (e && atoll(e) > 0) ? ((int64_t) atoll(e) << 20) : 0;
    }
    return c;
}

static bool conv2d_filter_engines() {
    static int on = -1;
    if (on < 0) {
        const char * e = getenv("GGML_CUDNN_CONV_FILTER_ENGINES");
        on = (e && atoi(e) != 0) ? 1 : 0;
    }
    return on != 0;
}

static bool cudnn_conv2d_vram_trace() {
    return getenv("LONGCAT_VRAM_BREAKDOWN") || getenv("GGML_CUDNN_TRACE") || getenv("GGML_CONV2D_DBG");
}

// ---- layout-conversion kernels ----------------------------------------------------

// Weight reorder: ggml KCRS (ne=[KW,KH,IC,OC], memory ((oc*IC+ic)*KH+ky)*KW+kx)
// -> KRSC f16 (memory ((oc*KH+ky)*KW+kx)*IC+ic). Source may be f16 or f32.
template <typename T>
static __global__ void kcrs_to_krsc_f16(const T * __restrict__ in, half * __restrict__ out,
                                        int OC, int IC, int KH, int KW) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long tot = (long)OC * IC * KH * KW;
    if (idx >= tot) return;
    // decode source KCRS index
    int kx = idx % KW;
    long t = idx / KW;
    int ky = t % KH; t /= KH;
    int ic = t % IC; t /= IC;
    int oc = t;
    long o = (((long)oc * KH + ky) * KW + kx) * IC + ic;   // KRSC
    out[o] = __float2half((float)in[idx]);
}

// Tiled shared-memory transpose between NCHW(f32) and NHWC(f16). Per batch n, treat the
// data as a [C x HW] (NCHW) <-> [HW x C] (NHWC) matrix transpose. 32x32 tiles give
// coalesced reads AND writes on both sides (the +1 pad kills bank conflicts).
#define CV_TILE 32
#define CV_BR   8   // rows handled per thread (blockDim.y == CV_TILE/CV_BR ... we use 8 threads-y)

// NCHW f32 -> NHWC f16: src is [C, HW] row-major (per n), dst is [HW, C] row-major.
static __global__ void nchw_f32_to_nhwc_f16_tiled(const float * __restrict__ in, half * __restrict__ out,
                                                  int N, int C, int HW) {
    __shared__ float tile[CV_TILE][CV_TILE + 1];
    int n = blockIdx.z;
    const float * inb  = in  + (long)n * C * HW;   // [C, HW]
    half        * outb = out + (long)n * HW * C;   // [HW, C]

    // load tile: rows = C (y), cols = HW (x) -> coalesced read along HW
    int c0  = blockIdx.y * CV_TILE;
    int hw0 = blockIdx.x * CV_TILE;
    for (int j = 0; j < CV_TILE; j += CV_BR) {
        int c  = c0 + threadIdx.y + j;
        int hw = hw0 + threadIdx.x;
        if (c < C && hw < HW) tile[threadIdx.y + j][threadIdx.x] = inb[(long)c * HW + hw];
    }
    __syncthreads();
    // store transposed: out[hw, c] ; coalesced write along C (now the x dim)
    int hwT = hw0 + threadIdx.y;     // becomes row
    int cT  = c0  + threadIdx.x;     // becomes col (contiguous in out)
    for (int j = 0; j < CV_TILE; j += CV_BR) {
        int hw = hwT + j;
        if (cT < C && hw < HW) outb[(long)hw * C + cT] = __float2half(tile[threadIdx.x][threadIdx.y + j]);
    }
}

// NHWC f16 -> NCHW f32: src is [HW, C] row-major (per n), dst is [C, HW] row-major.
static __global__ void nhwc_f16_to_nchw_f32_tiled(const half * __restrict__ in, float * __restrict__ out,
                                                  int N, int C, int HW) {
    __shared__ float tile[CV_TILE][CV_TILE + 1];
    int n = blockIdx.z;
    const half * inb  = in  + (long)n * HW * C;    // [HW, C]
    float      * outb = out + (long)n * C * HW;    // [C, HW]

    // load tile: rows = HW (y), cols = C (x) -> coalesced read along C
    int hw0 = blockIdx.y * CV_TILE;
    int c0  = blockIdx.x * CV_TILE;
    for (int j = 0; j < CV_TILE; j += CV_BR) {
        int hw = hw0 + threadIdx.y + j;
        int c  = c0 + threadIdx.x;
        if (hw < HW && c < C) tile[threadIdx.y + j][threadIdx.x] = __half2float(inb[(long)hw * C + c]);
    }
    __syncthreads();
    // store transposed: out[c, hw] ; coalesced write along HW
    int cT  = c0  + threadIdx.y;     // becomes row
    int hwT = hw0 + threadIdx.x;     // becomes col (contiguous in out)
    for (int j = 0; j < CV_TILE; j += CV_BR) {
        int c = cT + j;
        if (c < C && hwT < HW) outb[(long)c * HW + hwT] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// ---- per-thread cuDNN handle -------------------------------------------------------

static thread_local cudnnHandle_t g_cudnn_conv_handle = nullptr;
static cudnnHandle_t get_conv_handle() {
    if (!g_cudnn_conv_handle) {
        if (cudnnCreate(&g_cudnn_conv_handle) != CUDNN_STATUS_SUCCESS) {
            GGML_ABORT("cudnnCreate (conv) failed");
        }
    }
    return g_cudnn_conv_handle;
}

// ---- graph plan cache --------------------------------------------------------------

struct conv_plan {
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t workspace = 0;
    bool supported = false;
    std::string fail;
};

struct conv_key {
    int64_t N, C, H, W, Cout, KH, KW, SY, SX, PY, PX, DY, DX;
    bool operator==(const conv_key & o) const {
        return N==o.N && C==o.C && H==o.H && W==o.W && Cout==o.Cout &&
               KH==o.KH && KW==o.KW && SY==o.SY && SX==o.SX &&
               PY==o.PY && PX==o.PX && DY==o.DY && DX==o.DX;
    }
};
struct conv_key_hash {
    size_t operator()(const conv_key & k) const {
        size_t h = 1469598103934665603ull;
        for (int64_t v : {k.N,k.C,k.H,k.W,k.Cout,k.KH,k.KW,k.SY,k.SX,k.PY,k.PX,k.DY,k.DX}) {
            h ^= (size_t)v; h *= 1099511628211ull;
        }
        return h;
    }
};

static std::mutex g_conv_mtx;
static std::unordered_map<conv_key, conv_plan, conv_key_hash> g_conv_cache;

// Weight cache: reordered KRSC f16 buffer per source weight device ptr.
struct weight_buf {
    half * d = nullptr;
    size_t n = 0;
};
static std::unordered_map<const void *, weight_buf> g_weight_cache;

static conv_plan build_conv_plan_once(cudnnHandle_t handle, const conv_key & k, bool apply_filter) {
    const int64_t N=k.N, C=k.C, H=k.H, W=k.W, Cout=k.Cout, R=k.KH, S=k.KW;
    const int64_t OH = (H + 2*k.PY - k.DY*(R-1) - 1)/k.SY + 1;
    const int64_t OW = (W + 2*k.PX - k.DX*(S-1) - 1)/k.SX + 1;

    conv_plan plan;
    size_t free_before = 0;
    const bool trace_vram = cudnn_conv2d_vram_trace();
    if (trace_vram) {
        size_t total_before = 0;
        cudaMemGetInfo(&free_before, &total_before);
    }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    // X: [N,C,H,W] dims, NHWC strides
    auto X = graph->tensor(fe::graph::Tensor_attributes().set_name("X").set_uid(X_UID)
                 .set_dim({N, C, H, W})
                 .set_stride({H*W*C, 1, W*C, C}));
    // W: [K,C,R,S] dims, KRSC strides
    auto Wt = graph->tensor(fe::graph::Tensor_attributes().set_name("W").set_uid(W_UID)
                 .set_dim({Cout, C, R, S})
                 .set_stride({R*S*C, 1, S*C, C}));

    auto conv_opts = fe::graph::Conv_fprop_attributes().set_name("conv")
                         .set_padding({k.PY, k.PX}).set_stride({k.SY, k.SX}).set_dilation({k.DY, k.DX});
    auto Y = graph->conv_fprop(X, Wt, conv_opts);
    Y->set_output(true).set_dim({N, Cout, OH, OW})
     .set_stride({OH*OW*Cout, 1, OW*Cout, Cout}).set_uid(Y_UID);

    const bool v_ok  = graph->validate().is_good();
    const bool b_ok  = v_ok && graph->build_operation_graph(handle).is_good();
    const bool p_ok  = b_ok && graph->create_execution_plans({fe::HeurMode_t::A}).is_good();
    if (p_ok && conv2d_ws_cap() > 0) {
        graph->deselect_workspace_greater_than(conv2d_ws_cap());
    }
    if (p_ok && apply_filter) {
        graph->deselect_numeric_notes({fe::NumericalNote_t::WINOGRAD,
                                       fe::NumericalNote_t::WINOGRAD_TILE_4x4,
                                       fe::NumericalNote_t::WINOGRAD_TILE_6x6,
                                       fe::NumericalNote_t::WINOGRAD_TILE_13x13,
                                       fe::NumericalNote_t::REDUCED_PRECISION_REDUCTION});
    }
    const bool s_ok  = p_ok && graph->check_support(handle).is_good();
    const bool bp_ok = s_ok && graph->build_plans(handle).is_good();

    int64_t ws = -1;
    if (bp_ok && graph->get_workspace_size(ws).is_good() && (conv2d_ws_cap() == 0 || ws <= conv2d_ws_cap())) {
        plan.graph = graph;
        plan.workspace = ws;
        plan.supported = true;
    }
    if (!plan.supported) {
        char buf[224];
        snprintf(buf, sizeof(buf), "validate=%d opgraph=%d plans=%d support=%d build=%d ws=%lldMB(cap=%lldMB) filter=%d",
                 v_ok, b_ok, p_ok, s_ok, bp_ok, (long long)(ws >> 20), (long long)(conv2d_ws_cap() >> 20), (int)apply_filter);
        plan.fail = buf;
    }
    if (getenv("GGML_CUDNN_TRACE") || getenv("GGML_CONV2D_DBG")) {
        fprintf(stderr, "[cudnn-conv2d] plan N=%lld C=%lld %lldx%lld->OC=%lld k=%lldx%lld s=%lld,%lld "
                "filter=%d : validate=%d opgraph=%d plans=%d support=%d build=%d ws=%lldMB(cap=%lldMB) -> %s\n",
                (long long)N, (long long)C, (long long)H, (long long)W, (long long)Cout,
                (long long)R, (long long)S, (long long)k.SY, (long long)k.SX, (int)apply_filter,
                v_ok, b_ok, p_ok, s_ok, bp_ok, (long long)(ws >> 20), (long long)(conv2d_ws_cap() >> 20),
                plan.supported ? "CUDNN" : "FALLBACK(direct)");
    }
    if (trace_vram) {
        size_t free_after = 0, total_after = 0;
        cudaMemGetInfo(&free_after, &total_after);
        fprintf(stderr,
                "[cudnn-conv2d-plan] build N=%lld C=%lld %lldx%lld->OC=%lld k=%lldx%lld "
                "filter=%d supported=%d ws=%lld MB free %.1f -> %.1f MB (delta %+.1f MB, used %.1f MB)\n",
                (long long)N, (long long)C, (long long)H, (long long)W, (long long)Cout,
                (long long)R, (long long)S, (int)apply_filter, (int)plan.supported, (long long)(ws >> 20),
                free_before / 1048576.0, free_after / 1048576.0,
                ((double)free_after - (double)free_before) / 1048576.0,
                (total_after - free_after) / 1048576.0);
    }
    return plan;
}

static conv_plan & get_or_build_conv_plan(cudnnHandle_t handle, const conv_key & k) {
    auto it = g_conv_cache.find(k);
    if (it != g_conv_cache.end()) return it->second;

    const bool filter = conv2d_filter_engines();
    conv_plan plan = build_conv_plan_once(handle, k, filter);
    if (!plan.supported && filter) {
        fprintf(stderr, "[cudnn-conv2d] GGML_CUDNN_CONV_FILTER_ENGINES removed all engines for "
                "C=%lld->OC=%lld %lldx%lld (%s) -> retrying unfiltered\n",
                (long long)k.C, (long long)k.Cout, (long long)k.H, (long long)k.W, plan.fail.c_str());
        plan = build_conv_plan_once(handle, k, false);
    }

    auto [ins, ok] = g_conv_cache.emplace(k, std::move(plan));
    GGML_UNUSED(ok);
    return ins->second;
}

void ggml_cuda_cudnn_conv2d_release_handle() {
    std::lock_guard<std::mutex> lk(g_conv_mtx);
    g_conv_cache.clear();
    if (g_cudnn_conv_handle) {
        if (cudnnDestroy(g_cudnn_conv_handle) != CUDNN_STATUS_SUCCESS) {
            GGML_ABORT("cudnnDestroy (conv2d) failed");
        }
        g_cudnn_conv_handle = nullptr;
    }
}

// Reorder + cache the KRSC f16 weight for this source weight tensor.
static half * get_or_build_weight(const ggml_tensor * kernel, cudaStream_t stream) {
    auto it = g_weight_cache.find(kernel->data);
    if (it != g_weight_cache.end()) return it->second.d;

    const int KW = kernel->ne[0], KH = kernel->ne[1], IC = kernel->ne[2], OC = kernel->ne[3];
    const long n = (long)OC * IC * KH * KW;
    half * d = nullptr;
    if (cudaMalloc(&d, n * sizeof(half)) != cudaSuccess) GGML_ABORT("cudnn conv weight cudaMalloc failed");

    const int bs = 256;
    const int gs = (int)((n + bs - 1) / bs);
    if (kernel->type == GGML_TYPE_F16) {
        kcrs_to_krsc_f16<half><<<gs, bs, 0, stream>>>((const half *)kernel->data, d, OC, IC, KH, KW);
    } else {
        kcrs_to_krsc_f16<float><<<gs, bs, 0, stream>>>((const float *)kernel->data, d, OC, IC, KH, KW);
    }
    g_weight_cache[kernel->data] = {d, (size_t)n};
    return d;
}

bool ggml_cuda_op_conv2d_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // Honor GGML_CUDNN_CONV3D as well as GGML_CUDNN_CONV: both envs route VAE convs through
    // GGML_OP_CONV_2D (set_conv2d_direct_enabled), and the 3D dispatch (conv3d-cudnn.cu) +
    // supports_op already gate on the pair. Without CONV3D here, a GGML_OP_CONV_2D emitted
    // under GGML_CUDNN_CONV3D (e.g. explicit vae_conv_direct) fell through to the naive
    // conv2d_kernel instead of this validated cuDNN implicit-GEMM path.
    if (!getenv("GGML_CUDNN_CONV") && !getenv("GGML_CUDNN_CONV3D")) return false;

    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    const int32_t * p = (const int32_t *) dst->op_params;
    const int SX=p[0], SY=p[1], PX=p[2], PY=p[3], DX=p[4], DY=p[5];
    if (p[6] != 0) return false;   // cwhn unsupported

    if (!ggml_is_contiguous(kernel) || !ggml_is_contiguous(input)) return false;
    if (kernel->type != GGML_TYPE_F16 && kernel->type != GGML_TYPE_F32) return false;
    if (input->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;

    const int IW = input->ne[0], IH = input->ne[1], IC = input->ne[2], N = input->ne[3];
    const int OW = dst->ne[0], OH = dst->ne[1], OC = kernel->ne[3];
    const int KW = kernel->ne[0], KH = kernel->ne[1];

    if (kernel->ne[2] != IC) return false;
    if (input->ne[3] != dst->ne[3]) return false;
    if (dst->ne[2] != OC) return false;

    cudaStream_t  stream = ctx.stream();
    cudnnHandle_t handle = get_conv_handle();
    if (cudnnSetStream(handle, stream) != CUDNN_STATUS_SUCCESS) GGML_ABORT("cudnnSetStream (conv) failed");

    static int traced = 0;
    if (getenv("GGML_CUDNN_TRACE") && traced++ < 8) {
        fprintf(stderr, "[cudnn-conv] N=%d IC=%d %dx%d -> OC=%d %dx%d  k=%dx%d s=%d,%d p=%d,%d d=%d,%d wt=%d\n",
                N, IC, IH, IW, OC, OH, OW, KH, KW, SY, SX, PY, PX, DY, DX, kernel->type);
    }

    // Lock covers the plan cache, weight cache and execute. The flux2 VAE runs on a
    // single worker thread, but keep it correct under concurrent callers.
    std::lock_guard<std::mutex> lk(g_conv_mtx);

    conv_key key{N, IC, IH, IW, OC, KH, KW, SY, SX, PY, PX, DY, DX};
    conv_plan & plan = get_or_build_conv_plan(handle, key);
    if (!plan.supported) {
        return false;
    }

    // weight (cached, reordered once)
    half * w_krsc = get_or_build_weight(kernel, stream);

    // X: NCHW f32 -> NHWC f16 (pool temp), tiled transpose (coalesced both sides)
    ggml_cuda_pool_alloc<half> x_nhwc(ctx.pool(), (size_t)N * IC * IH * IW);
    {
        const int HW = IH * IW;
        dim3 blk(CV_TILE, CV_BR);
        dim3 grd((HW + CV_TILE - 1) / CV_TILE, (IC + CV_TILE - 1) / CV_TILE, N);
        nchw_f32_to_nhwc_f16_tiled<<<grd, blk, 0, stream>>>((const float *)input->data, x_nhwc.get(), N, IC, HW);
    }

    // Y scratch: NHWC f16
    ggml_cuda_pool_alloc<half> y_nhwc(ctx.pool(), (size_t)N * OC * OH * OW);

    ggml_cuda_pool_alloc<uint8_t> ws(ctx.pool());
    void * ws_ptr = nullptr;
    if (plan.workspace > 0) { ws.alloc((size_t)plan.workspace); ws_ptr = ws.get(); }

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> vpack = {
        {X_UID, x_nhwc.get()},
        {W_UID, w_krsc},
        {Y_UID, y_nhwc.get()},
    };
    size_t free_before_exec = 0;
    const bool trace_exec = cudnn_conv2d_vram_trace();
    if (trace_exec) {
        size_t total_before_exec = 0;
        cudaStreamSynchronize(stream);
        cudaMemGetInfo(&free_before_exec, &total_before_exec);
    }
    if (!plan.graph->execute(handle, vpack, ws_ptr).is_good()) GGML_ABORT("cudnn conv execute failed");
    if (trace_exec) {
        cudaStreamSynchronize(stream);
        size_t free_after_exec = 0, total_after_exec = 0;
        cudaMemGetInfo(&free_after_exec, &total_after_exec);
        fprintf(stderr,
                "[cudnn-conv2d-exec] N=%d C=%d %dx%d->OC=%d k=%dx%d ws=%lld MB free %.1f -> %.1f MB "
                "(delta %+.1f MB, used %.1f MB)\n",
                N, IC, IH, IW, OC, KH, KW, (long long)(plan.workspace >> 20),
                free_before_exec / 1048576.0, free_after_exec / 1048576.0,
                ((double)free_after_exec - (double)free_before_exec) / 1048576.0,
                (total_after_exec - free_after_exec) / 1048576.0);
    }

    // Y: NHWC f16 -> NCHW f32 directly into ggml dst, tiled transpose
    {
        const int HW = OH * OW;
        dim3 blk(CV_TILE, CV_BR);
        dim3 grd((OC + CV_TILE - 1) / CV_TILE, (HW + CV_TILE - 1) / CV_TILE, N);
        nhwc_f16_to_nchw_f32_tiled<<<grd, blk, 0, stream>>>(y_nhwc.get(), (float *)dst->data, N, OC, HW);
    }

    return true;
}

#else  // !GGML_CUDNN

bool ggml_cuda_conv2d_cudnn_available() { return false; }

void ggml_cuda_cudnn_conv2d_release_handle() {}

bool ggml_cuda_op_conv2d_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
}

#endif // GGML_CUDNN
