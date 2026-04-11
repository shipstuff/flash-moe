# Batch prefill — scoping

**Date:** 2026-04-11
**Status:** Scoping only, no code yet
**Goal:** Reduce long-context prefill wall time. Currently 22 min for 5k token prefill = the dominant UX cost for any long-context workflow.

## Current state

Prefill processes one prompt token at a time through `fused_layer_forward`. Per-token cost ≈ generation cost ≈ 226–257 ms at 3–5k context. So:

| Prompt | Prefill wall |
|---|---|
| 1k tokens | 3 min |
| 2.4k | 8 min |
| 3k | 12 min |
| 5k | 22 min |
| 8k (extrapolated) | ~38 min |

Per-token breakdown at 3k context (TQ enabled):
```
expert_io       2.140 ms  (~50%)
cmd1_wait       1.052 ms  (~25%)  q/k/v projections + linear attn pipeline
cmd2_wait       0.408 ms  (~10%)  attention + o_proj + residual + norm + routing
cpu_attn        0.063 ms
cmd2_encode     0.108 ms
tq_kernel       0.089 ms
total_layer     3.821 ms
```

## What batching would buy us

If we process T prompt tokens through each layer simultaneously instead of one at a time, the savings depend on what part of the work is amortized:

| Component | Per-token cost | Behaviour under batching | Notes |
|---|---|---|---|
| **q/k/v/o_proj** | ~0.5 ms | One dispatch with T inputs amortizes the dispatch overhead AND the per-output-channel dequant work (which is currently done T times for the same weight matrix). Realistic 2–3× speedup on this fraction. | Needs a `dequant_matmul_4bit` kernel or sgemm-after-CPU-dequant. |
| **attention** | ~0.4 ms (TQ) | One dispatch with T queries × (kv_len + T) keys. T× more compute but 1 dispatch. Causal mask required. | Needs a new attention kernel that takes T queries. TQ-compatible version even more involved. |
| **MoE expert dispatch (`expert_io` + cmd3)** | ~2.5 ms | If T tokens route to **disjoint** experts: no savings (still T·K reads). If routing **overlaps**: load each unique expert once, run it on the subset of tokens that selected it. Needs sparse-MoE batching. | The biggest bucket and the trickiest. |
| **routing softmax + topK** | ~0.005 ms | T separate routings, trivial to vectorize. | Tiny win. |
| **residual + RMS norm** | ~0.03 ms | T separate ops, trivial to vectorize. | Tiny win. |

**Realistic upside for prefill speedup:**
- T=4, no MoE batching: ~30% faster (cmd1+cmd2 amortized only) → 22 min → 16 min for 5k
- T=4, with MoE batching: ~50% faster → 22 min → 11 min for 5k
- T=16, with MoE batching: ~70% faster → 22 min → 7 min for 5k

To reach the "5–10× faster" textbook expectation we'd need T ≥ 32 with full MoE batching, plus the `expert_io` per-layer cost dropping below the per-batch attention cost. Whether that holds on M4 Pro with the SSD-bound expert reads is unclear without measurement.

## Code surface (estimate)

The current code is heavily T=1 optimized. New code needed:

| Piece | LOC est. | Notes |
|---|---|---|
| `dequant_matmul_4bit` Metal kernel | 150 | Variant of existing matvec kernel that takes T input rows. Per-channel dequant reused across T. |
| `attn_scores_batched_T` Metal kernel | 120 | T queries vs (kv_len + T) keys, causal mask. |
| `attn_softmax_batched_T` Metal kernel | 60 | Same shape, per-row softmax over kv_len. |
| `attn_values_batched_T` Metal kernel | 100 | Output is (T, head_dim). |
| `tq_fused_attention_T` Metal kernel | 200 | TQ-compatible version of the above (only if we want batched prefill to work in TQ mode). |
| `fused_layer_forward_batch` C function | 500 | Mirrors `fused_layer_forward` but accepts (T, HIDDEN_DIM). |
| Sparse MoE dispatch logic (C) | 200 | Bucket tokens by routed-expert, run each expert on its subset, scatter results. |
| Prefill loop refactor (C) | 60 | Chunk `pt->count` tokens into batches of T and call the new function. |
| Validation (CPU reference, comparison harness) | 200 | Crucial — every change is a correctness landmine. |

**Total: ~1600 lines.** Plus iterative debugging (the kind we just lived through with TQ — multiple passes finding bugs via per-layer hash diff).

**Realistic effort: 2–4 days of focused work.**

## Risk

The `fused_layer_forward` path is *very* hot. Reproducing its T=1 correctness in a T=N variant is the kind of work where every off-by-one ruins generation quality. The TQ debugging session today found 8 separate bugs over ~3 hours. A T=N batched path will have a similar bug-density risk profile.

We **must** build a CPU reference that runs the same prompt through the new batch path and compares against the baseline T=1 path layer-by-layer with the FNV hash diagnostic approach we used for TQ. Without that, debugging will be agony.

## Smaller stepping stones we could do first

If the full batch refactor is too big, we can do strict subsets:

1. **Batch only q/k/v/o_proj projections** — keep attention and MoE per-token. Estimated 15–20% prefill speedup. ~400 LOC new code.
2. **Batch only the GPU attention dispatches** (one TQ kernel call per T tokens) — keeps MoE per-token. Another 15–20%. ~500 LOC.
3. **Sparse MoE batching only** — keep projections and attention per-token. The dominant 50% of layer time. ~400 LOC.

These can be combined incrementally and each is independently verifiable.

## Recommendation

The **highest expected ROI for the smallest scope** is option 1 (batched projections). It touches the smallest amount of code and the projection path is the cleanest to validate (deterministic matmul). It buys 15–20% with ~1 day of work.

If that succeeds and we want more, option 3 (sparse MoE) is the biggest single-component win and unblocks larger T values for future stages.

**Decision needed:** commit to option 1 as a focused day-scale task, OR defer batch prefill to a separate planning cycle and pick a different "higher value" target now.
