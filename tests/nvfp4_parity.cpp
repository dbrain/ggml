// Standalone NVFP4 mul_mat parity + perf harness for realistic flux2 DiT shapes.
// Compares: CUDA NVFP4 mul_mat (FP4-MMA on Blackwell, or DP4A fallback via
//   GGML_NVFP4_NO_MMA=1) against an fp32 oracle (CPU F32 x F32 with the SAME
//   dequantized weight). Reports cosine similarity, max relative error, s/it.
//
// Oracle = dequant(NVFP4 weight)->f32 then f32 matmul, so we isolate the
// activation-quantization + MMA accumulation error of the kernel (NOT the
// weight-quantization error, which is identical to both and to the render).
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

static std::vector<float> rand_vec(size_t n, unsigned seed, float scale) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = d(rng);
    return v;
}

// Run an [N,K] x [K,M] mul_mat on a backend. wtype = weight type (F32 oracle or NVFP4).
// w_f32: the raw f32 weight [N*K] (row-major N rows of K). a_f32: activation [M*K].
// Returns output [M*N] (M rows of N), plus measured ms/iter over `iters`.
static std::vector<float> run_mulmat(ggml_backend_t backend, ggml_type wtype,
                                     int64_t N, int64_t K, int64_t M,
                                     const std::vector<float>& w_f32,
                                     const std::vector<float>& a_f32,
                                     int iters, double* ms_per_iter) {
    size_t buf_sz = 16*1024*1024;
    ggml_init_params ip = { buf_sz, nullptr, true };
    ggml_context* ctx = ggml_init(ip);

    // ggml mul_mat(a, b): a=[K,N] (ne0=K, ne1=N) weight, b=[K,M] activation, out=[N,M]
    ggml_tensor* a = ggml_new_tensor_2d(ctx, wtype, K, N);
    ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    ggml_set_input(a);
    ggml_set_input(b);

    ggml_tensor* out = ggml_mul_mat(ctx, a, b);
    ggml_set_output(out);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // allocate the whole graph (inputs + intermediates + output) on the backend
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, gf);

    // upload weight: if NVFP4, quantize from f32; else copy f32.
    if (wtype == GGML_TYPE_F32) {
        ggml_backend_tensor_set(a, w_f32.data(), 0, ggml_nbytes(a));
    } else {
        std::vector<char> q(ggml_nbytes(a));
        ggml_quantize_chunk(wtype, w_f32.data(), q.data(), 0, N, K, nullptr);
        ggml_backend_tensor_set(a, q.data(), 0, ggml_nbytes(a));
    }
    ggml_backend_tensor_set(b, a_f32.data(), 0, ggml_nbytes(b));

    // warmup
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    auto t1 = std::chrono::high_resolution_clock::now();
    *ms_per_iter = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    std::vector<float> res(M*N);
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return res;
}

// dequantize NVFP4 weight to f32 by round-tripping through ggml (CPU): quantize then
// dequantize so the oracle uses the IDENTICAL weight bits the GPU kernel sees.
static std::vector<float> dequant_roundtrip(ggml_type wtype, int64_t N, int64_t K,
                                            const std::vector<float>& w_f32) {
    std::vector<char> q((size_t)ggml_row_size(wtype, K) * N);
    ggml_quantize_chunk(wtype, w_f32.data(), q.data(), 0, N, K, nullptr);
    std::vector<float> deq((size_t)N*K);
    const auto* tt = ggml_get_type_traits(wtype);
    tt->to_float(q.data(), deq.data(), (int64_t)N*K);
    return deq;
}

static void stats(const std::vector<float>& got, const std::vector<float>& ref,
                  double* cosine, double* max_rel) {
    double dot=0, ng=0, nr=0, mrel=0;
    for (size_t i=0;i<ref.size();++i) {
        double g=got[i], r=ref[i];
        dot += g*r; ng += g*g; nr += r*r;
        double denom = std::max(std::abs(r), 1e-3);
        double rel = std::abs(g-r)/denom;
        if (rel>mrel) mrel=rel;
    }
    *cosine = dot / (std::sqrt(ng)*std::sqrt(nr) + 1e-30);
    *max_rel = mrel;
}

int main() {
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (!cuda) { fprintf(stderr, "no cuda backend\n"); return 1; }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 8);

    bool no_mma = getenv("GGML_NVFP4_NO_MMA") != nullptr;
    printf("# NVFP4 mul_mat parity (path=%s) vs fp32 oracle (same dequant weight)\n",
           no_mma ? "DP4A_fallback" : "BLACKWELL_FP4_MMA");
    printf("%-26s %-12s %-14s %-10s\n", "shape(N,K,M)", "cosine", "max_rel_err", "ms/it");

    struct Shape { int64_t N, K, M; };
    std::vector<Shape> shapes = {
        {3072, 3072, 1024}, {3072, 3072, 4096},
        {9216, 3072, 1024}, {9216, 3072, 4096},
        {3072, 12288, 1024}, {3072, 12288, 4096},
    };

    for (auto s : shapes) {
        fprintf(stderr, "[dbg] shape N=%ld K=%ld M=%ld\n", (long)s.N,(long)s.K,(long)s.M);
        auto w = rand_vec((size_t)s.N*s.K, 1234, 0.05f);  // weight magnitude ~ DiT linear
        auto act = rand_vec((size_t)s.M*s.K, 5678, 1.0f);
        fprintf(stderr, "[dbg] dequant_roundtrip...\n");

        // fp32 oracle: dequantized NVFP4 weight (CPU) x f32 activation (CPU f32 matmul)
        auto wdeq = dequant_roundtrip(GGML_TYPE_NVFP4, s.N, s.K, w);
        fprintf(stderr, "[dbg] cpu oracle...\n");
        double ms_cpu;
        auto oracle = run_mulmat(cpu, GGML_TYPE_F32, s.N, s.K, s.M, wdeq, act, 1, &ms_cpu);

        // CUDA NVFP4 path
        double ms_gpu;
        int iters = (s.M*s.N > 20000000) ? 20 : 50;
        auto got = run_mulmat(cuda, GGML_TYPE_NVFP4, s.N, s.K, s.M, w, act, iters, &ms_gpu);

        double cos, mrel;
        stats(got, oracle, &cos, &mrel);
        char shp[40];
        snprintf(shp, sizeof shp, "(%ld,%ld,%ld)", (long)s.N,(long)s.K,(long)s.M);
        printf("%-26s %-12.6f %-14.4f %-10.3f\n", shp, cos, mrel, ms_gpu);
        fflush(stdout);
    }
    ggml_backend_free(cuda);
    ggml_backend_free(cpu);
    return 0;
}
