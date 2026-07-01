// LongCat lap-31.2 / WAN SLA — CPU-precomputed BSA bitmap state for the FA MMA-f16
// kernel's K-tile skip.
//
// HISTORY OF THE BUG (fixed here): these used to be a single set of `extern __device__`
// symbols DEFINED in fattn.cu and DECLARED here for the kernel TUs. Without
// `-rdc` / CUDA_SEPARABLE_COMPILATION, cross-TU `__device__` symbols DO NOT share
// storage: the host set fattn.cu's copy via cudaMemcpyToSymbol, but the MMA flash
// kernel — compiled in a SEPARATE template-instance TU
// (fattn-mma-f16-instance-ncols1_*-ncols2_1.cu) — read its OWN zero copy. So the
// bitmap skip never engaged and Wan SLA sparse output was byte-identical to dense.
//
// FIX (approach B — kernel-args via per-TU symbols, NOT -rdc, to avoid flash-attn
// perf risk): make the device symbols STATIC (one private copy per TU) and have
// launch_fattn — which is template-instantiated in the SAME TU as the kernel it
// launches — cudaMemcpyToSymbol the values from a host-side struct right before the
// launch. Same TU ⇒ the symbol the launcher writes is the symbol the kernel reads.
// The host struct is the single source of truth (set by the extern-C setters in
// fattn.cu); a generation counter makes the per-TU sync a no-op unless the state
// actually changed, so the normal (non-sparse) flash-attn path pays nothing.

#pragma once

#include "common.cuh"   // CUDA_CHECK, cuda runtime (cudaMemcpyToSymbol)

#include <cstdint>

// --- Per-TU device-resident copies. STATIC ⇒ each TU gets its own; the launcher and
//     the kernel in a given instance TU share THIS TU's copy. [[maybe_unused]] because
//     only the MMA kernel TU reads them; other fattn TUs include this header too. ---
[[maybe_unused]] static __device__ const uint32_t * g_longcat_fa_bsa_bitmap_dev    = nullptr;
[[maybe_unused]] static __device__ int              g_longcat_fa_bsa_n_kwords_dev  = 0;
[[maybe_unused]] static __device__ int              g_longcat_fa_bsa_n_qtiles_dev  = 0;
[[maybe_unused]] static __device__ int              g_longcat_fa_bsa_mask_free_dev = 0;  // WAN SLA: maskless-path skip
[[maybe_unused]] static __device__ int              g_longcat_fa_bsa_n_ktiles_dev  = 0;  // WAN SLA: scope to matching K-tile count
[[maybe_unused]] static __device__ int              g_longcat_fa_bsa_n_heads_dev   = 0;  // WAN SLA Stage 1: per-head bitmap

// --- Host-side authoritative state. Set by the extern-C setters in fattn.cu (driven by
//     wan_sla.hpp / longcat_avatar.hpp). `generation` bumps on every change so each TU
//     syncs its device copies exactly once per change. Host symbols link normally
//     cross-TU (no -rdc issue) — that's the whole point. ---
struct longcat_fa_bsa_host_state_t {
    const void * bitmap_dev;   // device ptr to the bitmap (or nullptr = disabled)
    int          n_kwords;
    int          n_qtiles;
    int          mask_free;
    int          n_ktiles;
    int          n_heads;
    unsigned     generation;   // bumped by every setter; drives the per-TU resync
};

// Defined in fattn.cu (single TU). Read by launch_fattn in each instance TU.
const longcat_fa_bsa_host_state_t & ggml_cuda_longcat_fa_bsa_host_state();

// Sync THIS TU's device-symbol copies from the host state. `static inline` ⇒ internal
// linkage, so the function-local `synced_gen` is per-TU (independent cache per kernel
// instance TU). Generation-gated: a no-op (one host load, one int compare) unless the
// BSA state changed since this TU last synced. Call right before the kernel launch.
static inline void longcat_fa_bsa_sync_this_tu() {
    static unsigned synced_gen = (unsigned)-1;
    const longcat_fa_bsa_host_state_t & h = ggml_cuda_longcat_fa_bsa_host_state();
    if (h.generation == synced_gen) {
        return;
    }
    const void * ptr = h.bitmap_dev;
    CUDA_CHECK(cudaMemcpyToSymbol(g_longcat_fa_bsa_bitmap_dev,    &ptr,         sizeof(void *)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_longcat_fa_bsa_n_kwords_dev,  &h.n_kwords,  sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_longcat_fa_bsa_n_qtiles_dev,  &h.n_qtiles,  sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_longcat_fa_bsa_mask_free_dev, &h.mask_free, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_longcat_fa_bsa_n_ktiles_dev,  &h.n_ktiles,  sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_longcat_fa_bsa_n_heads_dev,   &h.n_heads,   sizeof(int)));
    synced_gen = h.generation;
}
