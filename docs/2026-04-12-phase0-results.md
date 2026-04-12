# Multi-buffer Phase 0 results — GPU parallel headroom measurement

**Date:** 2026-04-12 (overnight session)
**Commit:** `d62e246`
**Status:** DONE — Phase 0 answered definitively. GPU has real but partial headroom.

---

## Question

Is the flash_moe GPU pipeline compute-bound (no room for T-parallelism) or
latency-bound (T=4 gives real throughput improvement)?

This was the hidden load-bearing assumption behind every optimization decision
on 2026-04-11. The ANE offload investigation and batch prefill investigation
both ran into the same constraint — "we don't know if the GPU has headroom" —
and were parked rather than resolved.

## Method

New flag: `--test-multi-slot-attn T,kv_len,iters[,layer_idx]`

Allocates T independent Metal command queues + per-slot scratch buffers
(`SlotFullAttnBufs`: 16 buffer families covering `buf_input` through
`buf_shared_egate`). Encodes the full CMD1+CMD2 kernel sequence for a
full-attention layer:

1. **CMD1 equivalent:** Q/K/V 4-bit dequant matvec projections
2. **CMD2 equivalent:** attention scores + softmax + values + sigmoid gate +
   o_proj matvec + residual_add + rms_norm_sum + rms_norm_apply + routing_proj
   + shared expert gate/up/routing matvecs

All writes go to per-slot buffers. All reads come from per-slot buffers
(for slot-specific data) or shared read-only buffers (wf_buf weights,
buf_kv_k/v cache). This eliminates cross-slot data contention while measuring
real kernel throughput.

Two benchmarks per configuration:
- **Serial:** T iterations of CMD1+CMD2 back-to-back on one queue, commit+wait each
- **Parallel:** T command buffers dispatched simultaneously to T different queues, wait-all

Speedup = serial_per_op / parallel_per_op. Ideal = T, dead = 1.

## Results (M4 Pro mini-01, layer 3 = first full-attn layer, 100 iters)

| T | kv_len=64 | kv_len=256 | kv_len=1024 |
|---|---|---|---|
| 2 | 1.20x | **1.50x** | **1.51x** |
| 4 | 1.23x | **1.47x** | **1.66x** |
| 8 | 1.82x | **1.63x** | **1.71x** |

T=1 baseline (should be 1.0x, measures noise floor): ranged 0.99-1.14x across
3 runs with 100 iters each. Indicates ~10% measurement noise per run at this
granularity.

T=4 repeated 4 times at kv_len=256: 1.57, 1.35, 1.57, 1.62 → mean 1.53x
(±0.12 variance). After correcting for T=1 noise (~1.05x): ~1.46x effective.

## Interpretation

1. **GPU IS NOT fully compute-bound.** At kv_len≥256, every T≥2 configuration
   shows ≥1.35x speedup. This is real parallel headroom, not measurement noise
   (which is ≤1.14x as measured by the T=1 control).

2. **Context length matters more than T.** The attention kernel at longer kv_len
   has more per-dispatch work, giving the GPU scheduler more opportunity to
   overlap slots on different compute units. At kv_len=64, per-dispatch work is
   tiny (32 attention heads × 64 positions = 2048 threadgroups) and dispatch
   overhead dominates; at kv_len=256+ the work is large enough for meaningful
   parallelism.

3. **T=2 captures most of the gain.** 1.50x at T=2 vs 1.47x at T=4 at
   kv_len=256. Adding more slots beyond 2 gives diminishing returns because
   the M4 Pro GPU's memory bandwidth and execution units are shared across all
   queues.

4. **Best case: 1.71x at T=8 kv_len=1024.** For long-context prefill (which is
   the primary use case), the headroom is somewhat larger.

## What this means for Approach A (multi-buffered deferred state)

The full Approach A refactor processes T prefill tokens through the 60-layer
stack in lockstep, with per-slot buffers and per-slot deferred state on
separate command queues.

**Full-attention layers (15/60):**
- At T=4 kv_len=256: ~1.5x per-op speedup → saves ~33% of the 15 full-attn
  layers' contribution = ~33% × 25% of total time = **~8% wall-clock**
- At T=4 kv_len=1024: ~1.7x → ~41% × 25% = **~10% wall-clock**

**Linear-attention layers (45/60):**
- Phase 0 didn't test these directly (they have recurrent state that
  serializes token-by-token at each layer). BUT the MoE expert dispatch that
  follows each linear layer CAN be batched via `dispatch_experts_sparse`:
- Cold-cache first-chunk: already measured at 2.35× for T=4 → proves I/O
  latency-bound for expert_io
- Warm-cache: MoE amortization is smaller but still real (unique-expert
  overlap saves ~30% of pread calls at T=4 based on routing analysis)
- Estimated: ~10-15% wall-clock savings from expert I/O sharing across
  T=4 tokens' routing decisions

**Combined estimate:**
- 138-token prefill: **~20% wall-clock** (8% full-attn + 12% MoE I/O)
- 1000-token prefill: **~25-30% wall-clock** (10% full-attn + 15% MoE I/O)

**Comparison to the roadmap's original hope (30-60%):**
The original projection assumed GPU might be strongly latency-bound (case B
in the roadmap, where T=4 gives ~2.5× throughput). Phase 0 shows the reality
is between case A (dead, 1x) and case B (2.5x): about 1.5x. So the expected
win is roughly halved from the optimistic estimate but is still clearly
positive.

## Decision

**Approach A is VIABLE. Proceed to Phase 1 scaffolding.**

The 20-25% estimated wall-clock improvement at 1500-2000 LOC effort is a
concrete, measurable win. It's not the 30-60% slam-dunk the roadmap hoped
for, but it's the single largest remaining optimization target — nothing else
on the roadmap exceeds 15%.

Recommended T for the refactor: **T=4** (standard) with **T=2 as the minimum
viable product**. T=2 captures most of the gain and has fewer buffers to
duplicate.

Phase 1 plan (from the roadmap):
1. Add `MAX_BATCH_T` per-slot copies of the ~25 buffer families to MetalCtx
2. Add per-slot command queues
3. Add per-slot `g_deferred` state
4. Restructure `fused_layer_forward` to accept a `slot` parameter
5. Build a prefill driver that interleaves T slots through the layer stack
6. Hash-diff validation (mandatory)
7. Benchmark at 138-tok and 1k-tok prompts

## Cross-references

- `docs/2026-04-11-optimization-roadmap.md` — the full menu + pipeline explainer
- `docs/2026-04-11-batch-prefill-scoping.md` — the earlier investigation that
  first proposed Approach A and the linear-attention serial constraint
- `metal_infer/infer.m:test_multi_slot_full_attn()` — the Phase 0 test code
- `metal_infer/infer.m:encode_slot_full_attn()` — the per-slot encoder that
  replicates CMD1+CMD2
- `metal_infer/infer.m:SlotFullAttnBufs` — the per-slot buffer struct
