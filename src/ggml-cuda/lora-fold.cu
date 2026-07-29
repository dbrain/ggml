// GPU LoRA fold for NVFP4 weights.
//
// The CPU implementation (longcat-avatar.cpp src/model/adapter/lora_fold.hpp) is correct but
// costs ~80 s for a 1632-module adapter over a 22B DiT, which is worse than the ~44 s the
// per-step runtime adapter costs for a SINGLE render -- so a CPU fold only pays off from the
// second render onward. That break-even is the only reason pre-folded variant ggufs still
// have to exist, and each one is 16.26 GB of NVMe.
//
// The work is trivially parallel and the GPU is idle during model load, so it belongs here.
// The whole job is two steps per tensor:
//   1. delta[out, in] = sum_l scale_l * (up_l @ down_l)   -- one cuBLAS SGEMM per module,
//      accumulated with beta=1 so stacked LoRAs and packed row-segments just sum.
//   2. requantise the NVFP4 nibbles in place against that delta, keeping every UE4M3 block
//      scale EXACTLY as stored (frozen grid) and rounding STOCHASTICALLY.
//
// Why stochastic: measured on nvfp4-CLEAN.gguf + ltx2.3-audio-reactive-v2 @1.5, the base's
// own nvfp4 error is 0.094 relative while the LoRA delta is only 0.015-0.047. The delta is
// SMALLER than the quantisation noise it has to survive, so round-to-nearest lands almost
// none of it (projection 0.062) -- it pulls back to the code the weight already sat on.
// Stochastic rounding is unbiased in expectation and lands 0.944.
//
// The RNG is counter-based on (seed, row, element index) and mirrors the CPU version
// bit-for-bit, so a fold is reproducible and CPU/GPU agree.

#include "common.cuh"

#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#define QK_NVFP4_LF      64
#define QK_NVFP4_SUB_LF  16
#define NVFP4_BLOCK_SZ   36   // 4 UE4M3 scale bytes + 32 packed-nibble bytes

// kvalues_fp4 magnitudes, i.e. TWICE the E2M1 values -- paired with a UE4M3 decode that
// returns HALF the true scale, so the product is the true value. Same convention as
// dequantize_row_nvfp4, which is what makes these bytes byte-compatible.
__constant__ float c_e2m1_mag[8]      = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f};
__constant__ uint8_t c_lower_by_int[12] = {0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6};
__constant__ float c_inv_gap[8]       = {1.f, 1.f, 1.f, 1.f, 0.5f, 0.5f, 0.25f, 0.f};

__device__ __forceinline__ uint64_t lf_mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float lf_ue4m3(uint8_t x) {
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    const int e = (x >> 3) & 0xF;
    const int m = x & 0x7;
    const float raw = (e == 0) ? ldexpf((float) m, -9) : ldexpf(1.0f + (float) m / 8.0f, e - 7);
    return raw * 0.5f;
}

__device__ __forceinline__ uint8_t lf_code(float mag, float u) {
    if (!(mag > 0.0f)) {
        return 0;
    }
    if (mag >= 12.0f) {
        return 7;
    }
    const int lo  = c_lower_by_int[(int) mag];
    const float p = (mag - c_e2m1_mag[lo]) * c_inv_gap[lo];
    return (uint8_t) (lo + (u < p ? 1 : 0));
}

// One thread per 16-element sub-block: out * nb * 4 threads.
static __global__ void lf_requantise(uint8_t * __restrict__ blocks,
                                     const float * __restrict__ delta,
                                     int64_t in, int64_t nb, float inv_wglobal, uint64_t seed) {
    const int64_t tid   = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_sub = nb * 4;                 // 16-element sub-blocks per output row
    const int64_t o     = blockIdx.y;             // output row
    if (tid >= n_sub) {
        return;
    }
    const int64_t b = tid >> 2;                   // block within the row
    const int     s = (int) (tid & 3);            // sub-block within the block

    uint8_t * blk      = blocks + (o * nb + b) * NVFP4_BLOCK_SZ;
    const float d      = lf_ue4m3(blk[s]);
    if (!(d > 0.0f)) {
        return;                                   // zero-scale sub-block stays zero
    }
    const float inv_d      = 1.0f / d;
    const int64_t base     = b * QK_NVFP4_LF + s * QK_NVFP4_SUB_LF;
    const float * drow     = delta + o * in;
    const uint64_t row_seed = seed ^ lf_mix64((uint64_t) o);
    uint8_t * qs           = blk + 4;

    for (int j = 0; j < 8; ++j) {
        const uint8_t packed = qs[s * 8 + j];
        const uint64_t h     = lf_mix64(row_seed ^ (uint64_t) (base + j));
        uint8_t out_lo = 0, out_hi = 0;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const uint8_t code = (packed >> (half * 4)) & 0x0F;
            const int64_t idx  = base + j + half * 8;
            const float cur    = c_e2m1_mag[code & 0x7] * d * ((code & 0x8) ? -1.0f : 1.0f);
            const float tgt    = cur + drow[idx] * inv_wglobal;
            const float mag    = fabsf(tgt) * inv_d;
            const uint32_t bits = (uint32_t) (((half == 0) ? h : (h >> 32)) & 0xFFFFFFu);
            uint8_t nc = lf_code(mag, (float) bits * (1.0f / 16777216.0f));
            if (nc != 0 && tgt < 0.0f) {
                nc |= 0x8;
            }
            if (half == 0) { out_lo = nc; } else { out_hi = nc; }
        }
        qs[s * 8 + j] = (uint8_t) (out_lo | (out_hi << 4));
    }
}

// Device scratch, reused across tensors: a 22B DiT folds ~1344 of them and per-tensor
// cudaMalloc/cudaFree would serialise the whole pass on the allocator.
struct lf_scratch {
    void * ptr    = nullptr;
    size_t size   = 0;
    void * get(size_t n) {
        if (n > size) {
            if (ptr) cudaFree(ptr);
            if (cudaMalloc(&ptr, n) != cudaSuccess) { ptr = nullptr; size = 0; return nullptr; }
            size = n;
        }
        return ptr;
    }
};
static lf_scratch g_delta, g_blocks, g_down, g_up;

// PINNED host bounce buffer for the NVFP4 block array.
//
// MEASURED (SD_LORA_FOLD_PROFILE, 588 tensors): the two block transfers are 91% of the fold
// -- upload 1685 ms, download 3064 ms -- against 282 ms of GEMM+kernel. Per tensor that is
// 3.2 GB/s up and 1.8 GB/s down, far below what the link does, because the destination is
// PAGEABLE: it is the mmap'd model, so the driver pins and unpins ~2400 pages per transfer
// and the download additionally faults COW pages in the MAP_PRIVATE mapping. Staging through
// pinned memory turns both legs into full-rate DMA plus a plain host memcpy.
struct lf_pinned {
    void * ptr  = nullptr;
    size_t size = 0;
    void * get(size_t n) {
        if (n > size) {
            if (ptr) cudaFreeHost(ptr);
            if (cudaHostAlloc(&ptr, n, cudaHostAllocDefault) != cudaSuccess) {
                ptr = nullptr; size = 0; return nullptr;   // caller falls back to pageable
            }
            size = n;
        }
        return ptr;
    }
};
static lf_pinned g_pin;
static cublasHandle_t g_blas = nullptr;

// Phase timers, SD_LORA_FOLD_PROFILE=1. The fold is ~22 ms/tensor against ~2 ms of actual
// work, so the interesting number is WHICH phase eats the other 20 -- and the candidates
// (pageable-memory PCIe, per-tensor memset, sync stalls) are not distinguishable by staring.
struct lf_prof {
    double memset_ms = 0, up_lora_ms = 0, gemm_ms = 0, up_blk_ms = 0, kern_ms = 0, dn_blk_ms = 0;
    int n = 0;
    bool on = false;
};
static lf_prof g_prof;

static bool lf_profile_enabled() {
    const char * v = getenv("SD_LORA_FOLD_PROFILE");
    return v && v[0] && v[0] != '0';
}

// Wall-clock around a synchronised point. Only used when profiling is on, so the extra
// cudaDeviceSynchronize() cannot distort the numbers it is not being asked to produce.
static double lf_now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#define LF_TICK(v) do { if (g_prof.on) { cudaDeviceSynchronize(); (v) = lf_now_ms(); } } while (0)
#define LF_TOCK(acc, v) do { if (g_prof.on) { cudaDeviceSynchronize(); (acc) += lf_now_ms() - (v); } } while (0)

// ggml_cuda_lora_module is declared in ggml-cuda.h, which common.cuh already includes.
extern "C" {

bool ggml_cuda_lora_fold_nvfp4(void * blocks, int64_t in, int64_t out, float inv_wglobal,
                               const struct ggml_cuda_lora_module * mods, int n_mods,
                               uint64_t seed) {
    if (!blocks || !mods || n_mods <= 0 || in <= 0 || out <= 0 || in % QK_NVFP4_LF != 0) {
        return false;
    }
    if (!g_blas) {
        if (cublasCreate(&g_blas) != CUBLAS_STATUS_SUCCESS) {
            return false;
        }
        // cuBLAS may serve SGEMM from TF32 tensor cores under CUBLAS_DEFAULT_MATH on
        // Ampere and later, which would compute the delta with a 10-bit mantissa. The
        // delta is already SMALLER than the NVFP4 quantisation step it has to survive, so
        // it is the last thing that should be computed at reduced precision -- and this
        // GEMM is nowhere near the bottleneck, so full fp32 is free here.
        cublasSetMathMode(g_blas, CUBLAS_PEDANTIC_MATH);
    }

    const int64_t nb          = in / QK_NVFP4_LF;
    const size_t delta_bytes  = (size_t) out * in * sizeof(float);
    const size_t block_bytes  = (size_t) out * nb * NVFP4_BLOCK_SZ;

    float * d_delta = (float *) g_delta.get(delta_bytes);
    uint8_t * d_blk = (uint8_t *) g_blocks.get(block_bytes);
    if (!d_delta || !d_blk) {
        return false;
    }
    g_prof.on = lf_profile_enabled();
    double t = 0;
    // No cudaMemset: the FIRST module's GEMM runs with beta=0 and writes the whole buffer,
    // subsequent ones accumulate with beta=1. That removes a full [out,in] f32 clear per
    // tensor -- 67 MB at 4096x4096, ~1344 times over a fold.
    // Exception: a packed projection whose modules do not cover every output row would leave
    // the uncovered rows undefined, so those still need the clear.
    int64_t covered = 0;
    for (int i = 0; i < n_mods; ++i) {
        covered += mods[i].rows;
    }
    const bool need_clear = covered < out;
    if (need_clear) {
        LF_TICK(t);
        if (cudaMemset(d_delta, 0, delta_bytes) != cudaSuccess) {
            return false;
        }
        LF_TOCK(g_prof.memset_ms, t);
    }

    // delta[out, in] += scale * up[rows, rank] @ down[rank, in], accumulated per module.
    // Row-major [M, N] is column-major [N, M], so the row-major product U*D becomes the
    // column-major product D*U with the operands in the order below and no transposes.
    for (int i = 0; i < n_mods; ++i) {
        const struct ggml_cuda_lora_module & m = mods[i];
        if (m.rows <= 0 || m.rank <= 0 || m.row_begin < 0 || m.row_begin + m.rows > out) {
            return false;
        }
        float * dn = (float *) g_down.get((size_t) m.rank * in * sizeof(float));
        float * up = (float *) g_up.get((size_t) m.rows * m.rank * sizeof(float));
        if (!dn || !up) {
            return false;
        }
        LF_TICK(t);
        if (cudaMemcpy(dn, m.down, (size_t) m.rank * in * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(up, m.up,   (size_t) m.rows * m.rank * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            return false;
        }
        LF_TOCK(g_prof.up_lora_ms, t);
        LF_TICK(t);
        const float alpha = m.scale;
        const float beta  = (need_clear || i > 0) ? 1.0f : 0.0f;
        if (cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N,
                        (int) in, (int) m.rows, (int) m.rank,
                        &alpha,
                        dn, (int) in,
                        up, (int) m.rank,
                        &beta,
                        d_delta + m.row_begin * in, (int) in) != CUBLAS_STATUS_SUCCESS) {
            return false;
        }
        LF_TOCK(g_prof.gemm_ms, t);
    }

    LF_TICK(t);
    void * pin = g_pin.get(block_bytes);
    if (pin != nullptr) {
        memcpy(pin, blocks, block_bytes);
        if (cudaMemcpy(d_blk, pin, block_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            return false;
        }
    } else if (cudaMemcpy(d_blk, blocks, block_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    LF_TOCK(g_prof.up_blk_ms, t);
    LF_TICK(t);
    const int threads = 256;
    dim3 grid((unsigned) ((nb * 4 + threads - 1) / threads), (unsigned) out, 1);
    lf_requantise<<<grid, threads>>>(d_blk, d_delta, in, nb, inv_wglobal, seed);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }
    LF_TOCK(g_prof.kern_ms, t);
    LF_TICK(t);
    if (pin != nullptr) {
        if (cudaMemcpy(pin, d_blk, block_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
        memcpy(blocks, pin, block_bytes);
    } else if (cudaMemcpy(blocks, d_blk, block_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    LF_TOCK(g_prof.dn_blk_ms, t);
    g_prof.n++;
    return true;
}

void ggml_cuda_lora_fold_release(void) {
    if (g_prof.on && g_prof.n > 0) {
        fprintf(stderr,
                "[lora-fold] %d tensors: memset %.0f ms | lora-up %.0f | gemm %.0f | "
                "blk-up %.0f | kernel %.0f | blk-down %.0f  (total %.1f s)\n",
                g_prof.n, g_prof.memset_ms, g_prof.up_lora_ms, g_prof.gemm_ms,
                g_prof.up_blk_ms, g_prof.kern_ms, g_prof.dn_blk_ms,
                (g_prof.memset_ms + g_prof.up_lora_ms + g_prof.gemm_ms + g_prof.up_blk_ms +
                 g_prof.kern_ms + g_prof.dn_blk_ms) / 1000.0);
        g_prof = lf_prof();
    }
    for (lf_scratch * s : {&g_delta, &g_blocks, &g_down, &g_up}) {
        if (s->ptr) { cudaFree(s->ptr); s->ptr = nullptr; s->size = 0; }
    }
    if (g_pin.ptr) { cudaFreeHost(g_pin.ptr); g_pin.ptr = nullptr; g_pin.size = 0; }
    if (g_blas) { cublasDestroy(g_blas); g_blas = nullptr; }
}

}  // extern "C"
