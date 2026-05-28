// LongCat lap-31.2 — cross-TU declarations for the CPU-precomputed BSA bitmap
// device symbols. Definitions live in fattn.cu (single TU). Consumers (the FA
// MMA-f16 kernel template) get this header via fattn-mma-f16.cuh. fattn.cu can
// safely include the same chain by defining LONGCAT_FA_BSA_BITMAP_DEFINING_TU
// before the include so the extern declarations are skipped in the defining TU.

#pragma once

#include <cstdint>

#ifndef LONGCAT_FA_BSA_BITMAP_DEFINING_TU
extern __device__ const uint32_t * g_longcat_fa_bsa_bitmap_dev;
extern __device__ int              g_longcat_fa_bsa_n_kwords_dev;
extern __device__ int              g_longcat_fa_bsa_n_qtiles_dev;
#endif
