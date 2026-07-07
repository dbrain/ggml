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
#include <string>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;

#define X_UID 1
#define W_UID 2
#define Y_UID 3

// Bound the per-call cuDNN workspace so a pathological shape cannot blow the VRAM budget.
// The LTX decoder ladder measured <=103 MB at high temporal-context (conv3d_golden.cu); a
// 1 GB cap leaves ~10x headroom while still rejecting anything absurd (-> caller falls back).
// Wan2.2's VAE convs are far larger (1280x704 full-frame, more channels): the heuristic's
// FAST plan needs >1 GB workspace, so a 1 GB cap REJECTS it -> slow fallback (measured 14 vs
// 5 s/tile). Env-tunable via GGML_CUDNN_CONV3D_WS_MB so the cap can admit the fast plan when
// VRAM headroom allows (the cuDNN-conv buffer is already ~7x lighter than im2col).
static int64_t conv3d_ws_cap() {
    static int64_t c = -1;
    if (c < 0) {
        const char * e = getenv("GGML_CUDNN_CONV3D_WS_MB");
        c = (e && atoll(e) > 0) ? ((int64_t)atoll(e) << 20) : ((int64_t)1 << 30);
    }
    return c;
}

// Optional temporal shape bucket for continuation decode. cuDNN appears to keep
// internal execution memory per unique conv3d shape, and the continuation path
// alternates nearby temporal depths (e.g. 13f/16f). Padding the input depth at
// the tail with zeros and discarding extra output frames is numerically neutral
// for the real output prefix, while making cuDNN reuse one cached shape.
static int conv3d_bucket_d() {
    static int d = -1;
    if (d < 0) {
        const char * e = getenv("GGML_CUDNN_CONV3D_BUCKET_D");
        d = (e && atoi(e) > 0) ? atoi(e) : 0;
    }
    return d;
}

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

// dtype helpers so the layout transposes are agnostic to the ggml-side activation dtype
// (F32 for the legacy path, F16 for the WAN_VAE_F16 F16-activation decode). cuDNN's IO is
// always HALF; only the ggml src/dst tensors differ.
static __device__ __forceinline__ float cv3d_to_f32(float v) { return v; }
static __device__ __forceinline__ float cv3d_to_f32(half  v) { return __half2float(v); }
static __device__ __forceinline__ void  cv3d_store(float * p, float v) { *p = v; }
static __device__ __forceinline__ void  cv3d_store(half  * p, float v) { *p = __float2half(v); }

// NCDHW (ggml src, Tin) -> NDHWC f16 (cuDNN X): per batch n, [C, S] -> [S, C], S = D*H*W.
template <typename Tin>
static __global__ void ncdhw_to_ndhwc_f16_tiled(const Tin * __restrict__ in, half * __restrict__ out,
                                                int N, int C, int S) {
    __shared__ float tile[CV_TILE][CV_TILE + 1];
    int n = blockIdx.z;
    const Tin * inb  = in  + (long)n * C * S;   // [C, S]
    half      * outb = out + (long)n * S * C;   // [S, C]

    int c0 = blockIdx.y * CV_TILE;
    int s0 = blockIdx.x * CV_TILE;
    for (int j = 0; j < CV_TILE; j += CV_BR) {
        int c = c0 + threadIdx.y + j;
        int s = s0 + threadIdx.x;
        if (c < C && s < S) tile[threadIdx.y + j][threadIdx.x] = cv3d_to_f32(inb[(long)c * S + s]);
    }
    __syncthreads();
    int sT = s0 + threadIdx.y;   // becomes row
    int cT = c0 + threadIdx.x;   // becomes col (contiguous in out)
    for (int j = 0; j < CV_TILE; j += CV_BR) {
        int s = sT + j;
        if (cT < C && s < S) outb[(long)s * C + cT] = __float2half(tile[threadIdx.x][threadIdx.y + j]);
    }
}

// NCDHW real input -> NDHWC padded input. S_pad = D_pad*H*W; any tail-depth
// positions d >= D are materialized as zero, matching conv padding for the real
// output prefix.
template <typename Tin>
static __global__ void ncdhw_to_ndhwc_f16_tiled_pad_d(const Tin * __restrict__ in, half * __restrict__ out,
                                                      int N, int C, int D, int H, int W, int D_pad) {
    __shared__ float tile[CV_TILE][CV_TILE + 1];
    int n = blockIdx.z;
    const int S     = D * H * W;
    const int S_pad = D_pad * H * W;
    const Tin * inb  = in  + (long)n * C * S;
    half      * outb = out + (long)n * S_pad * C;

    int c0 = blockIdx.y * CV_TILE;
    int s0 = blockIdx.x * CV_TILE;
    for (int j = 0; j < CV_TILE; j += CV_BR) {
        int c = c0 + threadIdx.y + j;
        int s = s0 + threadIdx.x;
        float v = 0.0f;
        if (c < C && s < S_pad) {
            const int wh = H * W;
            const int d  = s / wh;
            const int rem = s - d * wh;
            if (d < D) {
                const int src_s = d * wh + rem;
                v = cv3d_to_f32(inb[(long)c * S + src_s]);
            }
        }
        tile[threadIdx.y + j][threadIdx.x] = v;
    }
    __syncthreads();
    int sT = s0 + threadIdx.y;
    int cT = c0 + threadIdx.x;
    for (int j = 0; j < CV_TILE; j += CV_BR) {
        int s = sT + j;
        if (cT < C && s < S_pad) outb[(long)s * C + cT] = __float2half(tile[threadIdx.x][threadIdx.y + j]);
    }
}

// NDHWC f16 (cuDNN Y) -> NCDHW (ggml dst, Tout): per batch n, [S, C] -> [C, S], S = OD*OH*OW.
template <typename Tout>
static __global__ void ndhwc_f16_to_ncdhw_tiled(const half * __restrict__ in, Tout * __restrict__ out,
                                                int N, int C, int S) {
    __shared__ float tile[CV_TILE][CV_TILE + 1];
    int n = blockIdx.z;
    const half * inb  = in  + (long)n * S * C;    // [S, C]
    Tout       * outb = out + (long)n * C * S;    // [C, S]

    int c0 = blockIdx.x * CV_TILE;
    // grid-stride over S tiles: grid.y is capped at 65535 (the CUDA grid.y limit) at launch, so a
    // large output spatial volume (S = OD*OH*OW, e.g. 2.3M on the full-res LTX 0.9.x decoder conv)
    // is walked in gridDim.y-strided s-blocks rather than overflowing grid.y -> "invalid argument".
    for (int s0 = blockIdx.y * CV_TILE; s0 < S; s0 += gridDim.y * CV_TILE) {
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
            if (c < C && sT < S) cv3d_store(&outb[(long)c * S + sT], tile[threadIdx.x][threadIdx.y + j]);
        }
        __syncthreads();
    }
}

// hi-precision X transpose: NCDHW (ggml src, Tin) -> NDHWC f32 (cuDNN X). Same as
// ncdhw_to_ndhwc_f16_tiled but keeps the transposed input in fp32 for the full-F32-IO head.2
// plan (X/W/Y all fp32). Tin may be f16 (WAN_VAE_F16 activations) or f32.
template <typename Tin>
static __global__ void ncdhw_to_ndhwc_f32_tiled(const Tin * __restrict__ in, float * __restrict__ out,
                                                int N, int C, int S) {
    __shared__ float tile[CV_TILE][CV_TILE + 1];
    int n = blockIdx.z;
    const Tin * inb  = in  + (long)n * C * S;   // [C, S]
    float     * outb = out + (long)n * S * C;   // [S, C]

    int c0 = blockIdx.y * CV_TILE;
    int s0 = blockIdx.x * CV_TILE;
    for (int j = 0; j < CV_TILE; j += CV_BR) {
        int c = c0 + threadIdx.y + j;
        int s = s0 + threadIdx.x;
        if (c < C && s < S) tile[threadIdx.y + j][threadIdx.x] = cv3d_to_f32(inb[(long)c * S + s]);
    }
    __syncthreads();
    int sT = s0 + threadIdx.y;   // becomes row
    int cT = c0 + threadIdx.x;   // becomes col (contiguous in out)
    for (int j = 0; j < CV_TILE; j += CV_BR) {
        int s = sT + j;
        if (cT < C && s < S) outb[(long)s * C + cT] = tile[threadIdx.x][threadIdx.y + j];
    }
}

template <typename Tin>
static __global__ void ncdhw_to_ndhwc_f32_tiled_pad_d(const Tin * __restrict__ in, float * __restrict__ out,
                                                      int N, int C, int D, int H, int W, int D_pad) {
    __shared__ float tile[CV_TILE][CV_TILE + 1];
    int n = blockIdx.z;
    const int S     = D * H * W;
    const int S_pad = D_pad * H * W;
    const Tin * inb  = in  + (long)n * C * S;
    float     * outb = out + (long)n * S_pad * C;

    int c0 = blockIdx.y * CV_TILE;
    int s0 = blockIdx.x * CV_TILE;
    for (int j = 0; j < CV_TILE; j += CV_BR) {
        int c = c0 + threadIdx.y + j;
        int s = s0 + threadIdx.x;
        float v = 0.0f;
        if (c < C && s < S_pad) {
            const int wh = H * W;
            const int d  = s / wh;
            const int rem = s - d * wh;
            if (d < D) {
                const int src_s = d * wh + rem;
                v = cv3d_to_f32(inb[(long)c * S + src_s]);
            }
        }
        tile[threadIdx.y + j][threadIdx.x] = v;
    }
    __syncthreads();
    int sT = s0 + threadIdx.y;
    int cT = c0 + threadIdx.x;
    for (int j = 0; j < CV_TILE; j += CV_BR) {
        int s = sT + j;
        if (cT < C && s < S_pad) outb[(long)s * C + cT] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// hi-precision weight reorder: ggml KCRS-3d -> KRSC-3d f32 (no fp16 down-convert), for the
// full-F32-IO head.2 plan. Source may be f16 or f32.
template <typename T>
static __global__ void kcrs3d_to_krsc3d_f32(const T * __restrict__ in, float * __restrict__ out,
                                            int OC, int IC, int KD, int KH, int KW) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long tot = (long)OC * IC * KD * KH * KW;
    if (idx >= tot) return;
    int kw = idx % KW;
    long t = idx / KW;
    int kh = t % KH; t /= KH;
    int kd = t % KD; t /= KD;
    int ic = t % IC; t /= IC;
    int oc = t;
    long o = ((((long)oc * KD + kd) * KH + kh) * KW + kw) * IC + ic;   // KRSC-3d
    out[o] = (float)in[idx];
}

// hi-precision Y transpose: NDHWC f32 (cuDNN F32-IO Y) -> NCDHW f32 (ggml dst). Mirrors
// ndhwc_f16_to_ncdhw_tiled but with no fp16 round-trip, so the fp32 conv output reaches the
// ggml dst intact (WAN_VAE_HEAD_F32 head.2: kills the per-channel fp16 quantization grid).
static __global__ void ndhwc_f32_to_ncdhw_f32_tiled(const float * __restrict__ in, float * __restrict__ out,
                                                    int N, int C, int S) {
    __shared__ float tile[CV_TILE][CV_TILE + 1];
    int n = blockIdx.z;
    const float * inb  = in  + (long)n * S * C;    // [S, C]
    float       * outb = out + (long)n * C * S;    // [C, S]

    int c0 = blockIdx.x * CV_TILE;
    // grid-stride over S tiles (see ndhwc_f16_to_ncdhw_tiled): grid.y capped at 65535 at launch,
    // large S walked in gridDim.y-strided s-blocks instead of overflowing grid.y.
    for (int s0 = blockIdx.y * CV_TILE; s0 < S; s0 += gridDim.y * CV_TILE) {
        for (int j = 0; j < CV_TILE; j += CV_BR) {
            int s = s0 + threadIdx.y + j;
            int c = c0 + threadIdx.x;
            if (s < S && c < C) tile[threadIdx.y + j][threadIdx.x] = inb[(long)s * C + c];
        }
        __syncthreads();
        int cT = c0 + threadIdx.y;   // becomes row
        int sT = s0 + threadIdx.x;   // becomes col (contiguous in out)
        for (int j = 0; j < CV_TILE; j += CV_BR) {
            int c = cT + j;
            if (c < C && sT < S) outb[(long)c * S + sT] = tile[threadIdx.x][threadIdx.y + j];
        }
        __syncthreads();
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
    int64_t     workspace = 0;
    bool        supported = false;   // false => caller must fall back
    std::string fail;                // per-stage failure summary when !supported (for loud logging)
};

struct conv3d_key {
    int64_t N, C, D, H, W, Cout, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW;
    bool    HP;   // hi-precision: F32-IO plan (fp32 Y) instead of the default fp16 Y
    bool operator==(const conv3d_key & o) const {
        return N==o.N && C==o.C && D==o.D && H==o.H && W==o.W && Cout==o.Cout &&
               KD==o.KD && KH==o.KH && KW==o.KW && SD==o.SD && SH==o.SH && SW==o.SW &&
               PD==o.PD && PH==o.PH && PW==o.PW && DD==o.DD && DH==o.DH && DW==o.DW &&
               HP==o.HP;
    }
};
struct conv3d_key_hash {
    size_t operator()(const conv3d_key & k) const {
        size_t h = 1469598103934665603ull;
        for (int64_t v : {k.N,k.C,k.D,k.H,k.W,k.Cout,k.KD,k.KH,k.KW,
                          k.SD,k.SH,k.SW,k.PD,k.PH,k.PW,k.DD,k.DH,k.DW,(int64_t)k.HP}) {
            h ^= (size_t)v; h *= 1099511628211ull;
        }
        return h;
    }
};

static std::mutex g_conv3d_mtx;
static std::unordered_map<conv3d_key, conv3d_plan, conv3d_key_hash> g_conv3d_cache;

static bool cudnn_conv3d_vram_trace() {
    return getenv("LONGCAT_VRAM_BREAKDOWN") || getenv("GGML_CUDNN_TRACE") || getenv("GGML_CONV3D_DBG");
}

// Drop every cached 3D-conv plan. Only the plan cache is cleared here; the
// g_weight3d_cache / g_weight3d_f32_cache raw-cudaMalloc reorder buffers (keyed
// by the persistent weight ptr) are deliberately kept. Some cuDNN-internal
// device reservations appear to be handle-owned rather than graph-owned, so
// ggml_cuda_cudnn_conv3d_release_handle is the stronger boundary reset used by
// the public API.
void ggml_cuda_cudnn_conv3d_release_plans() {
    std::lock_guard<std::mutex> lk(g_conv3d_mtx);
    g_conv3d_cache.clear();
}

void ggml_cuda_cudnn_conv3d_release_handle() {
    ggml_cuda_cudnn_conv3d_release_plans();
    if (g_cudnn_conv3d_handle) {
        if (cudnnDestroy(g_cudnn_conv3d_handle) != CUDNN_STATUS_SUCCESS) {
            GGML_ABORT("cudnnDestroy (conv3d) failed");
        }
        g_cudnn_conv3d_handle = nullptr;
    }
}

struct weight3d_buf { half * d = nullptr; size_t n = 0; };
static std::unordered_map<const void *, weight3d_buf> g_weight3d_cache;

// Separate fp32 weight cache for the full-F32-IO head.2 plan (KRSC-3d fp32).
struct weight3d_buf_f32 { float * d = nullptr; size_t n = 0; };
static std::unordered_map<const void *, weight3d_buf_f32> g_weight3d_f32_cache;

// WAN_VAE_CONV3D_NO_WINOGRAD (default ON; =0 restores Winograd): exclude Winograd and
// reduced-precision-reduction engine configs from EVERY cuDNN conv3d plan (encode + decode),
// forcing implicit-GEMM with an fp32 reduction. Winograd's input/output transforms + fp16 tiling
// imprint a fixed ~2px spatial pattern (screen-door grid) on the heavy interior decoder convs that
// the im2col F32 path avoids. Implicit-GEMM is the same speed class, so the conv3d decode speedup
// is kept. If excluding these leaves a shape with no engine, that conv silently re-allows Winograd
// (logged, once) so the decode never aborts. =0 restores Winograd to A/B the grid.
static bool conv3d_no_winograd() {
    static const int on = [] {
        const char * e = getenv("WAN_VAE_CONV3D_NO_WINOGRAD");
        return (e == nullptr || atoi(e) != 0) ? 1 : 0;
    }();
    return on != 0;
}

// Build ONE cuDNN plan for shape k. apply_deselect => drop Winograd / reduced-precision engine
// configs. Returns the plan (supported=false, with a fail summary, if cuDNN can't do the shape).
static conv3d_plan build_conv3d_plan_once(cudnnHandle_t handle, const conv3d_key & k, bool apply_deselect) {
    const int64_t N=k.N, C=k.C, D=k.D, H=k.H, W=k.W, Cout=k.Cout;
    const int64_t KD=k.KD, KH=k.KH, KW=k.KW;
    const int64_t OD = (D + 2*k.PD - k.DD*(KD-1) - 1)/k.SD + 1;
    const int64_t OH = (H + 2*k.PH - k.DH*(KH-1) - 1)/k.SH + 1;
    const int64_t OW = (W + 2*k.PW - k.DW*(KW-1) - 1)/k.SW + 1;

    conv3d_plan plan;
    size_t free_before = 0;
    const bool trace_vram = cudnn_conv3d_vram_trace();
    if (trace_vram) {
        size_t total_before = 0;
        cudaMemGetInfo(&free_before, &total_before);
    }

    auto graph = std::make_shared<fe::graph::Graph>();
    // hi-precision (k.HP, WAN_VAE_HEAD_F32 head.2): full F32 IO (X/W/Y all fp32) so cuDNN never
    // stores an fp16 output. Standard fp32 conv -> broadly supported. (HALF-in/FLOAT-out was
    // rejected by the HeurMode-A heuristic for head.2's 3D shape, silently reverting to the fp16
    // plan and leaving the grid; full fp32 IO avoids the mixed-dtype engine gap.)
    graph->set_io_data_type(k.HP ? fe::DataType_t::FLOAT : fe::DataType_t::HALF)
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
    // (Y dtype follows set_io_data_type above: FLOAT for k.HP, else HALF.)

    // hi-precision also tries the FALLBACK heuristic so a shape the A-heuristic misses still finds
    // an fp32 engine, instead of silently reverting to the fp16 plan (= the grid).
    std::vector<fe::HeurMode_t> heur = {fe::HeurMode_t::A};
    if (k.HP) { heur.push_back(fe::HeurMode_t::FALLBACK); }

    // Anything cuDNN can't do for this shape/arch -> mark unsupported, caller falls back.
    // Per-step results captured so GGML_CUDNN_TRACE/GGML_CONV3D_DBG can report exactly
    // why a shape ran cuDNN vs fell back (don't infer the path from s/it timing).
    const bool v_ok  = graph->validate().is_good();
    const bool b_ok  = v_ok && graph->build_operation_graph(handle).is_good();
    const bool p_ok  = b_ok && graph->create_execution_plans(heur).is_good();
    if (p_ok) {
        graph->deselect_workspace_greater_than(conv3d_ws_cap());
    }
    // Filter the candidate engine configs before check_support selects one: drop Winograd (and its
    // tile variants) + reduced-precision reduction. deselect_numeric_notes filters graph.plans in
    // place; if it empties the list, check_support fails -> the wrapper retries without deselect.
    if (p_ok && apply_deselect) {
        graph->deselect_numeric_notes({fe::NumericalNote_t::WINOGRAD,
                                       fe::NumericalNote_t::WINOGRAD_TILE_4x4,
                                       fe::NumericalNote_t::WINOGRAD_TILE_6x6,
                                       fe::NumericalNote_t::WINOGRAD_TILE_13x13,
                                       fe::NumericalNote_t::REDUCED_PRECISION_REDUCTION});
    }
    const bool s_ok  = p_ok && graph->check_support(handle).is_good();
    const bool bp_ok = s_ok && graph->build_plans(handle).is_good();
    int64_t ws = -1;
    if (bp_ok && graph->get_workspace_size(ws).is_good() && ws <= conv3d_ws_cap()) {
        plan.graph     = graph;
        plan.workspace = ws;
        plan.supported = true;
    }
    if (!plan.supported) {
        char buf[224];
        snprintf(buf, sizeof(buf), "validate=%d opgraph=%d plans=%d support=%d build=%d ws=%lldMB(cap=%lldMB) nowino=%d",
                 v_ok, b_ok, p_ok, s_ok, bp_ok, (long long)(ws >> 20), (long long)(conv3d_ws_cap() >> 20), (int)apply_deselect);
        plan.fail = buf;
    }
    if (getenv("GGML_CUDNN_TRACE") || getenv("GGML_CONV3D_DBG")) {
        fprintf(stderr, "[cudnn-conv3d] plan N=%lld C=%lld %lldx%lldx%lld->OC=%lld k=%lldx%lldx%lld s=%lld,%lld,%lld hp=%d nowino=%d : "
                "validate=%d opgraph=%d plans=%d support=%d build=%d ws=%lldMB(cap=%lldMB) -> %s\n",
                (long long)k.N,(long long)k.C,(long long)k.D,(long long)k.H,(long long)k.W,(long long)k.Cout,
                (long long)k.KD,(long long)k.KH,(long long)k.KW,(long long)k.SD,(long long)k.SH,(long long)k.SW,(int)k.HP,(int)apply_deselect,
                v_ok,b_ok,p_ok,s_ok,bp_ok,(long long)(ws>>20),(long long)(conv3d_ws_cap()>>20),
                plan.supported ? "CUDNN" : "FALLBACK(im2col)");
    }
    if (trace_vram) {
        size_t free_after = 0, total_after = 0;
        cudaMemGetInfo(&free_after, &total_after);
        fprintf(stderr,
                "[cudnn-conv3d-plan] build N=%lld C=%lld %lldx%lldx%lld->OC=%lld k=%lldx%lldx%lld "
                "hp=%d nowino=%d supported=%d ws=%lld MB free %.1f -> %.1f MB (delta %+.1f MB, used %.1f MB)\n",
                (long long) k.N, (long long) k.C, (long long) k.D, (long long) k.H, (long long) k.W,
                (long long) k.Cout, (long long) k.KD, (long long) k.KH, (long long) k.KW,
                (int) k.HP, (int) apply_deselect, (int) plan.supported, (long long) (ws >> 20),
                free_before / 1048576.0, free_after / 1048576.0,
                ((double) free_after - (double) free_before) / 1048576.0,
                (total_after - free_after) / 1048576.0);
    }
    return plan;
}

static conv3d_plan & get_or_build_conv3d_plan(cudnnHandle_t handle, const conv3d_key & k) {
    auto it = g_conv3d_cache.find(k);
    if (it != g_conv3d_cache.end()) return it->second;

    const bool nw = conv3d_no_winograd();
    conv3d_plan plan = build_conv3d_plan_once(handle, k, nw);
    if (!plan.supported && nw) {
        // Excluding Winograd/reduced-precision left no engine for this shape -> re-allow them so
        // the decode doesn't abort (this conv may keep the grid). Logged once per shape.
        fprintf(stderr, "[cudnn-conv3d] WAN_VAE_CONV3D_NO_WINOGRAD: no non-Winograd engine for "
                "C=%lld->OC=%lld %lldx%lldx%lld (%s) -> re-allowing Winograd for this conv (grid may persist here)\n",
                (long long)k.C, (long long)k.Cout, (long long)k.D, (long long)k.H, (long long)k.W, plan.fail.c_str());
        plan = build_conv3d_plan_once(handle, k, false);
    }
    if (k.HP && !plan.supported) {
        // LOUD: the head.2 F32-IO plan could not be built -> the op falls back to the fp16 HALF plan.
        fprintf(stderr, "[cudnn-conv3d] *** WAN_VAE_HEAD_F32 head.2 F32-IO plan UNSUPPORTED (%s) "
                "C=%lld->OC=%lld %lldx%lldx%lld -> WILL FALL BACK TO fp16 HALF ***\n",
                plan.fail.c_str(), (long long)k.C, (long long)k.Cout, (long long)k.D, (long long)k.H, (long long)k.W);
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

// fp32 variant for the full-F32-IO head.2 plan (KRSC-3d fp32). Cached separately from the fp16
// weights; head.2's weight is tiny (256*12*27 = ~83k elems) so the extra device buffer is trivial.
static float * get_or_build_weight3d_f32(const ggml_tensor * kernel, int OC, int IC,
                                         int KD, int KH, int KW, cudaStream_t stream) {
    auto it = g_weight3d_f32_cache.find(kernel->data);
    if (it != g_weight3d_f32_cache.end()) return it->second.d;

    const long n = (long)OC * IC * KD * KH * KW;
    float * d = nullptr;
    if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) GGML_ABORT("cudnn conv3d f32 weight cudaMalloc failed");

    const int bs = 256;
    const int gs = (int)((n + bs - 1) / bs);
    if (kernel->type == GGML_TYPE_F16) {
        kcrs3d_to_krsc3d_f32<half><<<gs, bs, 0, stream>>>((const half *)kernel->data, d, OC, IC, KD, KH, KW);
    } else {
        kcrs3d_to_krsc3d_f32<float><<<gs, bs, 0, stream>>>((const float *)kernel->data, d, OC, IC, KD, KH, KW);
    }
    g_weight3d_f32_cache[kernel->data] = {d, (size_t)n};
    return d;
}

bool ggml_cuda_op_conv3d_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!getenv("GGML_CUDNN_CONV3D") && !getenv("GGML_CUDNN_CONV")) return false;

    const ggml_tensor * kernel = dst->src[0];   // ne=[KW,KH,KD,c*oc]
    const ggml_tensor * input  = dst->src[1];   // ne=[W,H,D,c*n]

    const int32_t * p = (const int32_t *) dst->op_params;
    const int SW=p[0], SH=p[1], SD=p[2], PW=p[3], PH=p[4], PD=p[5], DW=p[6], DH=p[7], DD=p[8];
    const int c=p[9], n=p[10], oc=p[11];
    // hi-precision (WAN_VAE_HEAD_F32 head.2): request an F32-IO cuDNN plan (fp32 output). Only
    // meaningful when the ggml dst is F32 (we write fp32 Y straight into it); head.2 forces its
    // result to F32, so this holds — guard anyway so a stray F16 dst can't take the fp32 path.
    bool hi_prec = (p[12] != 0) && (dst->type == GGML_TYPE_F32);

    const bool conv3d_dbg = getenv("GGML_CUDNN_TRACE") || getenv("GGML_CONV3D_DBG");
    #define CONV3D_REJECT(why) do { \
        if (conv3d_dbg) fprintf(stderr, "[cudnn-conv3d] REJECT(%s) C=%d->OC=%d %dx%dx%d k=%dx%dx%d " \
            "in.type=%d dst.type=%d w.type=%d contig(k=%d,in=%d) -> FALLBACK(slow conv2d_kernel)\n", \
            why, c, oc, (int)input->ne[2],(int)input->ne[1],(int)input->ne[0], \
            (int)kernel->ne[2],(int)kernel->ne[1],(int)kernel->ne[0], \
            input->type, dst->type, kernel->type, \
            ggml_is_contiguous(kernel), ggml_is_contiguous(input)); \
        return false; } while (0)
    if (!ggml_is_contiguous(kernel) || !ggml_is_contiguous(input)) CONV3D_REJECT("noncontig");
    if (kernel->type != GGML_TYPE_F16 && kernel->type != GGML_TYPE_F32) CONV3D_REJECT("wtype");
    // Activations may be F32 (legacy) or F16 (WAN_VAE_F16 F16-activation decode). cuDNN runs
    // HALF internally either way; the layout transposes below cast in/out per the ggml dtype.
    if (input->type != GGML_TYPE_F32 && input->type != GGML_TYPE_F16) CONV3D_REJECT("in-dtype");
    if (dst->type   != GGML_TYPE_F32 && dst->type   != GGML_TYPE_F16) CONV3D_REJECT("dst-dtype");

    const int KW = kernel->ne[0], KH = kernel->ne[1], KD = kernel->ne[2];
    const int W  = input->ne[0],  H  = input->ne[1],  D  = input->ne[2];
    const int OW = dst->ne[0],     OH = dst->ne[1],     OD = dst->ne[2];

    if ((int64_t)kernel->ne[3] != (int64_t)c * oc) return false;
    if ((int64_t)input->ne[3]  != (int64_t)c * n)  return false;
    if ((int64_t)dst->ne[3]    != (int64_t)oc * n) return false;

    cudaStream_t  stream = ctx.stream();
    cudnnHandle_t handle = get_conv3d_handle();
    if (cudnnSetStream(handle, stream) != CUDNN_STATUS_SUCCESS) GGML_ABORT("cudnnSetStream (conv3d) failed");

    const int D_plan = conv3d_bucket_d() > D ? conv3d_bucket_d() : D;
    const int OD_plan = (D_plan + 2*PD - DD*(KD-1) - 1)/SD + 1;
    const bool bucket_d = D_plan != D;

    static int traced = 0;
    if (getenv("GGML_CUDNN_TRACE") && traced++ < 8) {
        fprintf(stderr, "[cudnn-conv3d] N=%d C=%d %dx%dx%d(planD=%d) -> OC=%d %dx%dx%d(planOD=%d)  k=%dx%dx%d s=%d,%d,%d p=%d,%d,%d d=%d,%d,%d wt=%d\n",
                n, c, D, H, W, D_plan, oc, OD, OH, OW, OD_plan, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, kernel->type);
    }

    std::lock_guard<std::mutex> lk(g_conv3d_mtx);

    conv3d_key key{n, c, D_plan, H, W, oc, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, hi_prec};
    conv3d_plan * plan = &get_or_build_conv3d_plan(handle, key);
    if (!plan->supported && hi_prec) {
        // No F32-IO engine for this shape -> fall back to the standard fp16 plan (the same one
        // head.2 runs today, so it is known-supported) rather than aborting the whole decode.
        // get_or_build_conv3d_plan already printed the loud UNSUPPORTED reason once for the shape.
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[cudnn-conv3d] WAN_VAE_HEAD_F32 FIX INACTIVE: head.2 F32-IO plan unsupported "
                    "(%s) -> fp16 output, unpatchify GRID PRESENT. Raise GGML_CUDNN_CONV3D_WS_MB, or use "
                    "WAN_VAE_HEAD_F32=0 to A/B intentionally.\n", plan->fail.c_str());
        }
        hi_prec = false;
        conv3d_key key_h{n, c, D_plan, H, W, oc, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, false};
        plan = &get_or_build_conv3d_plan(handle, key_h);
    } else if (hi_prec) {
        // F32-IO plan engaged: confirm once so the fix can be verified from the log.
        static bool announced = false;
        if (!announced) {
            announced = true;
            fprintf(stderr, "[cudnn-conv3d] WAN_VAE_HEAD_F32 FIX ACTIVE: head.2 F32-IO cuDNN plan engaged "
                    "(fp32 output, no unpatchify grid) C=%d->OC=%d %dx%dx%d.\n", c, oc, D, H, W);
        }
    }
    if (!plan->supported) return false;   // cap exceeded / unsupported -> caller falls back

    // Weight (KRSC-3d), reordered + cached per ptr: fp32 for the hi_prec full-F32-IO plan, else fp16.
    void * w_ptr = hi_prec
        ? (void *) get_or_build_weight3d_f32(kernel, oc, c, KD, KH, KW, stream)
        : (void *) get_or_build_weight3d    (kernel, oc, c, KD, KH, KW, stream);

    // X: NCDHW (ggml) -> NDHWC (cuDNN), tiled transpose with S = D*H*W. fp32 for hi_prec, else fp16.
    const int S_in = D * H * W;
    const int S_in_plan = D_plan * H * W;
    ggml_cuda_pool_alloc<half>  x_half(ctx.pool());
    ggml_cuda_pool_alloc<float> x_f32 (ctx.pool());
    void * x_ptr = nullptr;
    {
        dim3 blk(CV_TILE, CV_BR);
        dim3 grd((S_in_plan + CV_TILE - 1) / CV_TILE, (c + CV_TILE - 1) / CV_TILE, n);
        if (hi_prec) {
            x_f32.alloc((size_t)n * c * S_in_plan);  x_ptr = x_f32.get();
            if (input->type == GGML_TYPE_F16) {
                if (bucket_d) ncdhw_to_ndhwc_f32_tiled_pad_d<half><<<grd, blk, 0, stream>>>((const half *)input->data, x_f32.get(), n, c, D, H, W, D_plan);
                else          ncdhw_to_ndhwc_f32_tiled<half><<<grd, blk, 0, stream>>>((const half *)input->data, x_f32.get(), n, c, S_in);
            } else {
                if (bucket_d) ncdhw_to_ndhwc_f32_tiled_pad_d<float><<<grd, blk, 0, stream>>>((const float *)input->data, x_f32.get(), n, c, D, H, W, D_plan);
                else          ncdhw_to_ndhwc_f32_tiled<float><<<grd, blk, 0, stream>>>((const float *)input->data, x_f32.get(), n, c, S_in);
            }
        } else {
            x_half.alloc((size_t)n * c * S_in_plan); x_ptr = x_half.get();
            if (input->type == GGML_TYPE_F16) {
                if (bucket_d) ncdhw_to_ndhwc_f16_tiled_pad_d<half><<<grd, blk, 0, stream>>>((const half *)input->data, x_half.get(), n, c, D, H, W, D_plan);
                else          ncdhw_to_ndhwc_f16_tiled<half><<<grd, blk, 0, stream>>>((const half *)input->data, x_half.get(), n, c, S_in);
            } else {
                if (bucket_d) ncdhw_to_ndhwc_f16_tiled_pad_d<float><<<grd, blk, 0, stream>>>((const float *)input->data, x_half.get(), n, c, D, H, W, D_plan);
                else          ncdhw_to_ndhwc_f16_tiled<float><<<grd, blk, 0, stream>>>((const float *)input->data, x_half.get(), n, c, S_in);
            }
        }
    }

    const int S_out = OD * OH * OW;
    const int S_out_plan = OD_plan * OH * OW;
    // hi_prec -> the cuDNN plan's Y is FLOAT (fp32 store); else HALF (fp16). Only the requested
    // buffer is allocated; the other stays empty (default-constructed pool_alloc, no allocation).
    ggml_cuda_pool_alloc<half>  y_half (ctx.pool());
    ggml_cuda_pool_alloc<float> y_f32  (ctx.pool());
    void * y_ptr = nullptr;
    if (hi_prec) { y_f32.alloc((size_t)n * oc * S_out_plan);  y_ptr = y_f32.get();  }
    else         { y_half.alloc((size_t)n * oc * S_out_plan); y_ptr = y_half.get(); }

    ggml_cuda_pool_alloc<uint8_t> ws(ctx.pool());
    void * ws_ptr = nullptr;
    if (plan->workspace > 0) { ws.alloc((size_t)plan->workspace); ws_ptr = ws.get(); }

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> vpack = {
        {X_UID, x_ptr},
        {W_UID, w_ptr},
        {Y_UID, y_ptr},
    };
    size_t free_before_exec = 0;
    const bool trace_exec = cudnn_conv3d_vram_trace();
    if (trace_exec) {
        size_t total_before_exec = 0;
        cudaStreamSynchronize(stream);
        cudaMemGetInfo(&free_before_exec, &total_before_exec);
    }
    if (!plan->graph->execute(handle, vpack, ws_ptr).is_good()) GGML_ABORT("cudnn conv3d execute failed");
    if (trace_exec) {
        cudaStreamSynchronize(stream);
        size_t free_after_exec = 0, total_after_exec = 0;
        cudaMemGetInfo(&free_after_exec, &total_after_exec);
        fprintf(stderr,
                "[cudnn-conv3d-exec] N=%d C=%d %dx%dx%d(planD=%d)->OC=%d k=%dx%dx%d ws=%lld MB "
                "free %.1f -> %.1f MB (delta %+.1f MB, used %.1f MB)\n",
                n, c, D, H, W, D_plan, oc, KD, KH, KW, (long long)(plan->workspace >> 20),
                free_before_exec / 1048576.0, free_after_exec / 1048576.0,
                ((double)free_after_exec - (double)free_before_exec) / 1048576.0,
                (total_after_exec - free_after_exec) / 1048576.0);
    }

    // Y: NDHWC (cuDNN) -> NCDHW (ggml dst), tiled transpose with S = OD*OH*OW.
    {
        dim3 blk(CV_TILE, CV_BR);
        int gy = (int)((S_out + CV_TILE - 1) / CV_TILE);
        if (gy > 65535) { gy = 65535; }   // CUDA grid.y limit; kernels grid-stride the rest
        dim3 grd((oc + CV_TILE - 1) / CV_TILE, gy, n);
        if (hi_prec) {
            // fp32 Y -> fp32 dst, no fp16 round-trip (the grid fix).
            ndhwc_f32_to_ncdhw_f32_tiled<<<grd, blk, 0, stream>>>((const float *)y_ptr, (float *)dst->data, n, oc, S_out);
        } else if (dst->type == GGML_TYPE_F16) {
            ndhwc_f16_to_ncdhw_tiled<half><<<grd, blk, 0, stream>>>((const half *)y_ptr, (half *)dst->data, n, oc, S_out);
        } else {
            ndhwc_f16_to_ncdhw_tiled<float><<<grd, blk, 0, stream>>>((const half *)y_ptr, (float *)dst->data, n, oc, S_out);
        }
    }

    return true;
}

#else  // !GGML_CUDNN

bool ggml_cuda_conv3d_cudnn_available() { return false; }

void ggml_cuda_cudnn_conv3d_release_plans() {}
void ggml_cuda_cudnn_conv3d_release_handle() {}

bool ggml_cuda_op_conv3d_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
}

#endif // GGML_CUDNN
