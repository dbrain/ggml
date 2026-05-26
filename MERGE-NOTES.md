# dbrain/ggml consolidation — merge notes

Branch `sync-upstream-v0.13`: the fork's 25 custom commits rebased from
ggml-org/ggml **v0.11.0 → v0.13.0** (`e705c5fe`, 2026-05-25). 0 behind upstream,
22 files / +1383 −127. RTX 3060 (sm_86) is the only target.

## Conflicts hit + how resolved
- **snake** (add/add): upstream independently added `snake.cu` as an **auto-fusion**
  (`ggml_cuda_op_snake_fused`, detects `add(x, mul(sqr(sin(mul(a,x))), inv_b))` and
  fuses; wired at ggml-cuda.cu). The fork has an explicit `GGML_OP_SNAKE` op
  (`ggml_snake()`, applies `exp()` internally — qwen3-tts vocoder contract).
  **Both kept** — different math contracts, can't substitute. Fork op kernel renamed
  `snake_kernel`→`snake_op_kernel` to avoid colliding with upstream's fusion kernel.
- **im2col** (content): both fixed the `gridDim.y > 65535` overflow. Upstream's
  in-kernel grid-stride loop is cleaner than the fork's WIP host-side `iow_base`
  chunking → **took upstream**, dropped the fork WIP.
- **concat** (content): upstream added PDL launch wrapper; fork templated for F16/I32.
  **Took the fork's templated version** (F32/F16/I32). PDL skipped — see below.
- **mul_mat hook** (adjacency): upstream added `GGML_LOG_WARN_ONCE` next to where the
  fork adds the hook typedef → **kept both**.

## Worth adopting from upstream (the "should we take this" list)
- **im2col grid-stride** — ADOPTED. Strictly better than the fork's WIP chunking.
- **snake auto-fusion** — KEPT (coexists). **CANDIDATE**: if qwen3-tts's vocoder graph
  can express snake as primitives, it could drop the explicit `ggml_snake()` op and let
  upstream auto-fuse — then we delete the fork's snake op + CPU path entirely. Needs a
  numerical check (the fork op applies `exp(alpha)/exp(beta)`; the fusion expects
  `a`/`inv_b` pre-exp'd, so the graph would need the exp baked in). Evaluate post-baseline.
- **upstream's snake kernel** has BF16 + `ggml_cuda_cast` + fastdiv — more complete than
  the fork's; it's the one now used by the fusion path.
- **v0.13.0 at large**: 120 upstream commits of new ops (CONV_2D, CONV_2D_DW, …), kernel
  fixes and perf now sit under the fork for free.

## Immaterial on our hardware (safe to skip)
- **PDL (programmatic dependent launch)**: `ggml_cuda_pdl_sync()` / `ggml_cuda_pdl_lc()`
  compile to **nothing** unless `GGML_CUDA_USE_PDL` AND CC ≥ Hopper (9.0). On sm_86 it's a
  no-op. So any PDL-related upstream churn (concat launch wrapper, etc.) can be dropped
  without functional or perf impact here. Revisit only if this fork ever targets Hopper.

## ⚠️ Upcoming: longcat-avatar ggml merge (Tier 3) — overlap to watch
longcat's ggml is **leejet lineage** (sd.cpp), 26 commits / +1702 lines — the biggest,
most-drifted surface. It touches several areas THIS branch already changed, so the
longcat merge must reconcile (not blindly add):
- **concat**: longcat added F16; this branch already has F16+I32 templated concat → likely
  redundant/overlapping. Take this branch's version, verify longcat's needs are covered.
- **im2col**: this branch is at upstream's grid-stride 2D im2col; longcat adds a separate
  **im2col_3d** (smem-halo + fastdiv) — additive (different entry), but double-check the
  shared kernel/helpers don't collide.
- **rope fusion**: this branch has `af3d8f82` (I32 row_indices in rope+view+set_rows
  fusion); longcat has `GGML_OP_ROPE_PE` (fused interleaved RoPE) — different ops, reconcile
  naming/dispatch.
- **norm**: longcat has fused LayerNorm+MUL+ADD-across-views; upstream has auto-fusions
  (RMS_NORM+MUL etc.) — check for overlap/superseding.
- **cpy / scale_cast / mul_add_bcast**: longcat fused ops — check vs upstream's current cpy.
- **ggml-alloc**: longcat has a view-output-storage fix; this branch has a sched hash-set
  fix (`2028dcb8`) — different, but both touch alloc/sched, watch for interaction.
- Lineage caveat: leejet/ggml carries sd.cpp-specific base patches; longcat's commits sit on
  those, so the merge is a port (cherry-pick + adapt), not a clean rebase.

## Not yet validated
This branch is rebased but **not yet compiled**. Validation = build a consumer
(siglip2 first — has the cosine gate) against it; v0.13.0 has 120 commits of API drift, so
the consumer's own C++ may need adapting too, separate from ggml compiling.
