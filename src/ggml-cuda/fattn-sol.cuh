#pragma once

#include "common.cuh"

// Experimental native Sol-Attn route for MiniMax-H3-style attention.  This is
// deliberately a narrowly guarded alternative to dense/SA3 attention: callers
// must opt in with GGML_H3_SOL_ATTN=1 and unsupported graphs return false.
//
// The persistent F32-fragment experiment has a stricter selector and never
// changes this default: it additionally requires FUSED=1, PERSISTENT=1, and
// ALL_EXACT=1 (or DENSE_CHECK=1).  Without every condition it falls back to
// the known bin65 fused/global-state kernel or normal dense dispatch.
bool ggml_cuda_flash_attn_ext_sol(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Debug-only paired reference check: `sol_out` is the result produced by the
// ALL_EXACT native path and `dense_out` is the subsequent normal dispatcher
// result for the exact same ggml attention node.
void ggml_cuda_flash_attn_ext_sol_compare(ggml_backend_cuda_context & ctx,
                                          const ggml_tensor * dense_out,
                                          const void * sol_out);
