// cuDNN implicit-GEMM conv3d — see conv3d-cudnn.cuh.
//
// Lifts the proven cudnn-frontend 3D conv-fprop graph from
// flux2.cpp/spike_cutlass_fp4/conv3d_golden.cu (3x3x3 s1, NDHWC fp16, depth-pad0 /
// spatial-pad1, cosine 1.0 on the LTX-2.3 22B VideoVAE decoder shapes). The LTX VAE
// otherwise decodes via im2col_3d (materialize the IC*27 column blowup to HBM) + GEMM;
// cuDNN's implicit 3D GEMM skips the materialization.
//
// Structurally identical to conv2d-cudnn.cu: the input/output transposes are the same
// [C,SPATIAL]<->[SPATIAL,C] tiled shared-mem transpose with SPATIAL = D*H*W (NCDHW<->NDHWC),
// the weight is reordered KCRS-3d->KRSC-3d f16 (cached per ptr), and the cuDNN plan is
// cached per shape. Differences vs 2D: 5 conv dims, a 3D weight reorder, and a workspace cap.

#include "conv3d-cudnn.cuh"

#ifdef GGML_CUDNN

#include <cuda_fp16.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace fe = cudnn_frontend;

#define X_UID 1
#define W_UID 2
#define Y_UID 3

// Bound the per-call cuDNN workspace so a pathological shape cannot blow the VRAM budget.
// The LTX decoder ladder measured <=103 MB at high temporal-context (conv3d_golden.cu); a
// 1 GB cap leaves ~10x headroom while still rejecting anything absurd (-> caller falls back).
static const int64_t CONV3D_WS_CAP = (int64_t)1 << 30;

bool ggml_cuda_conv3d_cudnn_available() { return true; }

// ---- layout-conversion kernels (DHW collapsed to one spatial dim) ------------------
// Identical math to conv2d-cudnn.cu with HW := D*H*W. 32x32 tiles, +1 pad kills bank
// conflicts, coalesced reads AND writes on both sides.
#define CV_TILE 32
#define CV_BR   8

// Weight reorder: ggml KCRS-3d (ne=[KW,KH,KD,c*oc], memory
// ((((oc*IC+ic)*KD+kd)*KH+kh)*KW+kw)) -> KRSC-3d f16 (memory
// ((((oc*KD+kd)*KH+kh)*KW+kw)*IC+ic)). Source may be f16 or f32.
template <typename T>
static __global__ void kcrs3d_to_krsc3d_f16(const T * __restrict__ in, half * __restrict__ out,
                                            int OC, int IC, int KD, int KH, int KW) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long tot = (long)OC * IC * KD * KH * KW;
    if (idx >= tot) return;
    // decode source KCRS-3d index (kw fastest)
    int kw = idx % KW;
    long t = idx / KW;
    int kh = t % KH; t /= KH;
    int kd = t % KD; t /= KD;
    int ic = t % IC; t /= IC;
    int oc = t;
    long o = ((((long)oc * KD + kd) * KH + kh) * KW + kw) * IC + ic;   // KRSC-3d
    out[o] = __float2half((float)in[idx]);
}

// NCDHW f32 -> NDHWC f16: per batch n, [C, S] (NCDHW) -> [S, C] (NDHWC), S = D*H*W.
static __global__ void ncdhw_f32_to_ndhwc_f16_tiled(const float * __restrict__ in, half * __restrict__ out,
                                                    int N, int C, int S) {
    __shared__ float tile[CV_TILE][CV_TILE + 1];
    int n = blockIdx.z;
    const float * inb  = in  + (long)n * C * S;   // [C, S]
    half        * outb = out + (long)n * S * C;   // [S, C]

    int c0 = blockIdx.y * CV_TILE;
    int s0 = blockIdx.x * CV_TILE;
    for (int j = 0; j < CV_TILE; j += CV_BR) {
        int c = c0 + threadIdx.y + j;
        int s = s0 + threadIdx.x;
        if (c < C && s < S) tile[threadIdx.y + j][threadIdx.x] = inb[(long)c * S + s];
    }
    __syncthreads();
    int sT = s0 + threadIdx.y;   // becomes row
    int cT = c0 + threadIdx.x;   // becomes col (contiguous in out)
    for (int j = 0; j < CV_TILE; j += CV_BR) {
        int s = sT + j;
        if (cT < C && s < S) outb[(long)s * C + cT] = __float2half(tile[threadIdx.x][threadIdx.y + j]);
    }
}

// NDHWC f16 -> NCDHW f32: per batch n, [S, C] (NDHWC) -> [C, S] (NCDHW), S = OD*OH*OW.
static __global__ void ndhwc_f16_to_ncdhw_f32_tiled(const half * __restrict__ in, float * __restrict__ out,
                                                    int N, int C, int S) {
    __shared__ float tile[CV_TILE][CV_TILE + 1];
    int n = blockIdx.z;
    const half * inb  = in  + (long)n * S * C;    // [S, C]
    float      * outb = out + (long)n * C * S;    // [C, S]

    int s0 = blockIdx.y * CV_TILE;
    int c0 = blockIdx.x * CV_TILE;
    for (int j = 0; j < CV_TILE; j += CV_BR) {
        int s = s0 + threadIdx.y + j;
        int c = c0 + threadIdx.x;
        if (s < S && c < C) tile[threadIdx.y + j][threadIdx.x] = __half2float(inb[(long)s * C + c]);
    }
    __syncthreads();
    int cT = c0 + threadIdx.y;   // becomes row
    int sT = s0 + threadIdx.x;   // becomes col (contiguous in out)
    for (int j = 0; j < CV_TILE; j += CV_BR) {
        int c = cT + j;
        if (c < C && sT < S) outb[(long)c * S + sT] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// ---- per-thread cuDNN handle -------------------------------------------------------

static thread_local cudnnHandle_t g_cudnn_conv3d_handle = nullptr;
static cudnnHandle_t get_conv3d_handle() {
    if (!g_cudnn_conv3d_handle) {
        if (cudnnCreate(&g_cudnn_conv3d_handle) != CUDNN_STATUS_SUCCESS) {
            GGML_ABORT("cudnnCreate (conv3d) failed");
        }
    }
    return g_cudnn_conv3d_handle;
}

// ---- graph plan cache --------------------------------------------------------------

struct conv3d_plan {
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t workspace = 0;
    bool    supported = false;   // false => caller must fall back
};

struct conv3d_key {
    int64_t N, C, D, H, W, Cout, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW;
    bool operator==(const conv3d_key & o) const {
        return N==o.N && C==o.C && D==o.D && H==o.H && W==o.W && Cout==o.Cout &&
               KD==o.KD && KH==o.KH && KW==o.KW && SD==o.SD && SH==o.SH && SW==o.SW &&
               PD==o.PD && PH==o.PH && PW==o.PW && DD==o.DD && DH==o.DH && DW==o.DW;
    }
};
struct conv3d_key_hash {
    size_t operator()(const conv3d_key & k) const {
        size_t h = 1469598103934665603ull;
        for (int64_t v : {k.N,k.C,k.D,k.H,k.W,k.Cout,k.KD,k.KH,k.KW,
                          k.SD,k.SH,k.SW,k.PD,k.PH,k.PW,k.DD,k.DH,k.DW}) {
            h ^= (size_t)v; h *= 1099511628211ull;
        }
        return h;
    }
};

static std::mutex g_conv3d_mtx;
static std::unordered_map<conv3d_key, conv3d_plan, conv3d_key_hash> g_conv3d_cache;

struct weight3d_buf { half * d = nullptr; size_t n = 0; };
static std::unordered_map<const void *, weight3d_buf> g_weight3d_cache;

static conv3d_plan & get_or_build_conv3d_plan(cudnnHandle_t handle, const conv3d_key & k) {
    auto it = g_conv3d_cache.find(k);
    if (it != g_conv3d_cache.end()) return it->second;

    const int64_t N=k.N, C=k.C, D=k.D, H=k.H, W=k.W, Cout=k.Cout;
    const int64_t KD=k.KD, KH=k.KH, KW=k.KW;
    const int64_t OD = (D + 2*k.PD - k.DD*(KD-1) - 1)/k.SD + 1;
    const int64_t OH = (H + 2*k.PH - k.DH*(KH-1) - 1)/k.SH + 1;
    const int64_t OW = (W + 2*k.PW - k.DW*(KW-1) - 1)/k.SW + 1;

    conv3d_plan plan;

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    // X: dims [N,C,D,H,W], NDHWC strides
    auto X = graph->tensor(fe::graph::Tensor_attributes().set_name("X").set_uid(X_UID)
                 .set_dim({N, C, D, H, W})
                 .set_stride({(int64_t)D*H*W*C, 1, (int64_t)H*W*C, (int64_t)W*C, (int64_t)C}));
    // W: dims [Cout,C,KD,KH,KW], KRSC-3d strides
    auto Wt = graph->tensor(fe::graph::Tensor_attributes().set_name("W").set_uid(W_UID)
                 .set_dim({Cout, C, KD, KH, KW})
                 .set_stride({(int64_t)KD*KH*KW*C, 1, (int64_t)KH*KW*C, (int64_t)KW*C, (int64_t)C}));

    auto conv_opts = fe::graph::Conv_fprop_attributes().set_name("conv3d")
                         .set_padding({k.PD, k.PH, k.PW})
                         .set_stride({k.SD, k.SH, k.SW})
                         .set_dilation({k.DD, k.DH, k.DW});
    auto Y = graph->conv_fprop(X, Wt, conv_opts);
    Y->set_output(true).set_dim({N, Cout, OD, OH, OW})
     .set_stride({(int64_t)OD*OH*OW*Cout, 1, (int64_t)OH*OW*Cout, (int64_t)OW*Cout, (int64_t)Cout})
     .set_uid(Y_UID);

    // Anything cuDNN can't do for this shape/arch -> mark unsupported, caller falls back.
    if (graph->validate().is_good()
        && graph->build_operation_graph(handle).is_good()
        && graph->create_execution_plans({fe::HeurMode_t::A}).is_good()
        && graph->check_support(handle).is_good()
        && graph->build_plans(handle).is_good()) {
        int64_t ws = 0;
        if (graph->get_workspace_size(ws).is_good() && ws <= CONV3D_WS_CAP) {
            plan.graph     = graph;
            plan.workspace = ws;
            plan.supported = true;
        }
    }

    auto [ins, ok] = g_conv3d_cache.emplace(k, std::move(plan));
    GGML_UNUSED(ok);
    return ins->second;
}

static half * get_or_build_weight3d(const ggml_tensor * kernel, int OC, int IC,
                                    int KD, int KH, int KW, cudaStream_t stream) {
    auto it = g_weight3d_cache.find(kernel->data);
    if (it != g_weight3d_cache.end()) return it->second.d;

    const long n = (long)OC * IC * KD * KH * KW;
    half * d = nullptr;
    if (cudaMalloc(&d, n * sizeof(half)) != cudaSuccess) GGML_ABORT("cudnn conv3d weight cudaMalloc failed");

    const int bs = 256;
    const int gs = (int)((n + bs - 1) / bs);
    if (kernel->type == GGML_TYPE_F16) {
        kcrs3d_to_krsc3d_f16<half><<<gs, bs, 0, stream>>>((const half *)kernel->data, d, OC, IC, KD, KH, KW);
    } else {
        kcrs3d_to_krsc3d_f16<float><<<gs, bs, 0, stream>>>((const float *)kernel->data, d, OC, IC, KD, KH, KW);
    }
    g_weight3d_cache[kernel->data] = {d, (size_t)n};
    return d;
}

bool ggml_cuda_op_conv3d_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!getenv("GGML_CUDNN_CONV3D") && !getenv("GGML_CUDNN_CONV")) return false;

    const ggml_tensor * kernel = dst->src[0];   // ne=[KW,KH,KD,c*oc]
    const ggml_tensor * input  = dst->src[1];   // ne=[W,H,D,c*n]

    const int32_t * p = (const int32_t *) dst->op_params;
    const int SW=p[0], SH=p[1], SD=p[2], PW=p[3], PH=p[4], PD=p[5], DW=p[6], DH=p[7], DD=p[8];
    const int c=p[9], n=p[10], oc=p[11];

    if (!ggml_is_contiguous(kernel) || !ggml_is_contiguous(input)) return false;
    if (kernel->type != GGML_TYPE_F16 && kernel->type != GGML_TYPE_F32) return false;
    if (input->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;

    const int KW = kernel->ne[0], KH = kernel->ne[1], KD = kernel->ne[2];
    const int W  = input->ne[0],  H  = input->ne[1],  D  = input->ne[2];
    const int OW = dst->ne[0],     OH = dst->ne[1],     OD = dst->ne[2];

    if ((int64_t)kernel->ne[3] != (int64_t)c * oc) return false;
    if ((int64_t)input->ne[3]  != (int64_t)c * n)  return false;
    if ((int64_t)dst->ne[3]    != (int64_t)oc * n) return false;

    cudaStream_t  stream = ctx.stream();
    cudnnHandle_t handle = get_conv3d_handle();
    if (cudnnSetStream(handle, stream) != CUDNN_STATUS_SUCCESS) GGML_ABORT("cudnnSetStream (conv3d) failed");

    static int traced = 0;
    if (getenv("GGML_CUDNN_TRACE") && traced++ < 8) {
        fprintf(stderr, "[cudnn-conv3d] N=%d C=%d %dx%dx%d -> OC=%d %dx%dx%d  k=%dx%dx%d s=%d,%d,%d p=%d,%d,%d d=%d,%d,%d wt=%d\n",
                n, c, D, H, W, oc, OD, OH, OW, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, kernel->type);
    }

    std::lock_guard<std::mutex> lk(g_conv3d_mtx);

    conv3d_key key{n, c, D, H, W, oc, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW};
    conv3d_plan & plan = get_or_build_conv3d_plan(handle, key);
    if (!plan.supported) return false;   // cap exceeded / unsupported -> caller falls back

    half * w_krsc = get_or_build_weight3d(kernel, oc, c, KD, KH, KW, stream);

    // X: NCDHW f32 -> NDHWC f16 (pool temp), tiled transpose with S = D*H*W
    const int S_in = D * H * W;
    ggml_cuda_pool_alloc<half> x_ndhwc(ctx.pool(), (size_t)n * c * S_in);
    {
        dim3 blk(CV_TILE, CV_BR);
        dim3 grd((S_in + CV_TILE - 1) / CV_TILE, (c + CV_TILE - 1) / CV_TILE, n);
        ncdhw_f32_to_ndhwc_f16_tiled<<<grd, blk, 0, stream>>>((const float *)input->data, x_ndhwc.get(), n, c, S_in);
    }

    const int S_out = OD * OH * OW;
    ggml_cuda_pool_alloc<half> y_ndhwc(ctx.pool(), (size_t)n * oc * S_out);

    ggml_cuda_pool_alloc<uint8_t> ws(ctx.pool());
    void * ws_ptr = nullptr;
    if (plan.workspace > 0) { ws.alloc((size_t)plan.workspace); ws_ptr = ws.get(); }

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> vpack = {
        {X_UID, x_ndhwc.get()},
        {W_UID, w_krsc},
        {Y_UID, y_ndhwc.get()},
    };
    if (!plan.graph->execute(handle, vpack, ws_ptr).is_good()) GGML_ABORT("cudnn conv3d execute failed");

    // Y: NDHWC f16 -> NCDHW f32 directly into ggml dst, tiled transpose with S = OD*OH*OW
    {
        dim3 blk(CV_TILE, CV_BR);
        dim3 grd((oc + CV_TILE - 1) / CV_TILE, (S_out + CV_TILE - 1) / CV_TILE, n);
        ndhwc_f16_to_ncdhw_f32_tiled<<<grd, blk, 0, stream>>>(y_ndhwc.get(), (float *)dst->data, n, oc, S_out);
    }

    return true;
}

#else  // !GGML_CUDNN

bool ggml_cuda_conv3d_cudnn_available() { return false; }

bool ggml_cuda_op_conv3d_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
}

#endif // GGML_CUDNN
