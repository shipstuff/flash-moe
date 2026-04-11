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

---

## 2026-04-11 update

The three batched primitives **all exist on main** with passing synthetic tests:
- `dequant_matmul_4bit` Metal kernel (Option 1, commit `6ae50c6`)
- `attn_*_batched_T` Metal kernels with causal mask (Option 2, commit `9e01863`)
- `dispatch_experts_sparse` C function (Option 3, commit `a0bc022`)

A `fused_layer_forward_batch(T, ...)` scaffolding function and a
`--batch-prefill T` CLI flag are also in place (commit `19fc94a`). The
current stub implementation just loops the per-token `fused_layer_forward`
T times with `complete_deferred_experts()` between each call to avoid
clobbering the GPU pipeline state. This is **bit-for-bit equivalent** to
the per-token path (verified with the same prompt across `--batch-prefill`
1, 4, and 8 — same generated token IDs).

**Stub overhead measurement (45-token prompt):**
| Path | prefill ms/token | gen tok/s |
|---|---|---|
| `--batch-prefill 1` (per-token) | 179 ms | 5.73 |
| `--batch-prefill 4` (stub T=4)  | 230 ms (+28%) | 6.12 |

The stub is ~28% **slower** because every per-token call inside the
batched outer loop pays the cost of `complete_deferred_experts` and the
cmd3↔cmd1 pipelining (which the per-token loop relies on for back-to-back
layer overlap) is broken across token boundaries.

**To get a real speedup the stub must be replaced with a from-scratch
`fused_layer_forward_batch` that uses the batched primitives natively.**
That is the multi-day refactor estimated above. The component primitives
are ready; only the integration work remains. Suggested next stages, in
order:

1. **Buffer + helper plumbing** — `buf_input_batch[T,HIDDEN_DIM]`,
   `buf_qkv_batch_*`, and a `gpu_encode_batch_matmul_T()` helper that
   wraps `dequant_matmul_4bit` for T-input projections.
2. **Stage 1 — full-attention path** — q/k/v projections via
   `dequant_matmul_4bit`, batched RoPE + Q/K RMS norm (CPU loop, T is
   small), KV cache T-position append, batched attention via the
   Option 2 kernels, batched o_proj, per-token routing, sparse MoE
   via Option 3, per-token combine. Linear-attention layers stay on
   the per-token fallback (delta-net is stateful).
3. **Validation** — `--test-batch-prefill` harness that runs T=1 and
   T=N on the same prompt and diffs the per-layer hidden hashes (the
   approach used for the TQ debugging). MUST be in place before each
   stage to catch regressions early.
4. **Real-model benchmark** — same long-context sweep as the TQ work
   (1k, 2k, 3k contexts). Compare wall-time prefill against
   `--batch-prefill 1`.

---

## 2026-04-11 afternoon update — native batched path (status: cold-cache only)

The first real integration (`c08cf07` + `ddf25d1`, sub-agent branch
`integration-batch-prefill`) replaced the stub with a native
full-attention batched path. It found and fixed a real bug:
`dispatch_experts_sparse` was calling `gpu_expert_forward(..., already_in_buffer=1)`
after preading into `buf_multi_expert_data[0]`, but `gpu_expert_forward`
always reads from `buf_expert_data`. The fix passes
`already_in_buffer=(s > 0)` so the first token of each unique expert
does the copy and subsequent tokens reuse.

**Correctness:** T=1/4/8 bit-identical token sequences at both 18-token
(clockmaker) and 138-token (forest/cartographer) prompts. Natural EOS
generation. ✅

**Wall-time sweep (warm page cache, M4 Pro):**
| Prompt | T=1 rest avg | T=4 rest avg | T=8 rest avg | T=4 total vs T=1 |
|---|---|---|---|---|
| 18 tok  | 125 ms | 152 ms | 164 ms | ~flat |
| 50 tok  | 150 ms | 144 ms | 143 ms | ~flat |
| 138 tok | 150 ms | **236 ms** | **190 ms** | **+57% slower** |

**Cold-cache first-chunk only:**
| Prompt | T=1 first | T=4 first | Speedup |
|---|---|---|---|
| 15 tok  | 622 ms | 265 ms | 2.35x |
| 138 tok | 684 ms | 859 ms | 0.80x (slower) |

### Why the per-layer batching loses at long prompts

Traced the architecture carefully. The per-token prefill loop is fast
because of **intra-token layer pipelining**: each layer's
`fused_layer_forward` fast path *finalizes the previous layer's GPU MoE
dispatch while starting the next layer's CMD1*. The CMD3↔CMD1 overlap
eliminates the GPU wait between layers. This is structurally
**layers-inside-tokens**.

The native batched path inverts the loop to **layers-outside-tokens**
(runs T tokens through layer L, then T tokens through layer L+1). That
inversion breaks the pipeline. Worse, 45/60 layers are linear-attention
and fall back to the sequential path where each call to
`fused_layer_forward` inside the inner T-token loop must call
`complete_deferred_experts()` to ensure `hidden_batch[t-1]` has been
finalized before `hidden_batch[t]` can be used as input to the same
layer. That's (T-1)×45×60 GPU stalls per prefill chunk — the regression
source.

Net: the only benefit that survives is sparse MoE expert I/O
amortization on the *first* chunk when the page cache is cold. Past the
first chunk the stall cost dominates.

### Status: cold-cache-only opt-in

`--batch-prefill T` is correct but currently a **net loss for warm-cache
real prompts**. Gated as opt-in (default T=1). Useful only for:
- Very short (<20 tok) prompts on genuinely cold page cache
- Scenarios where prefill is a one-shot first-chunk cost

### Next steps — needs a deeper refactor

To get a real speedup, one of:

**Approach A (Multi-buffered deferred state):** keep the loop in
layers-inside-tokens form but run T token streams in lockstep — T
separate `buf_input_batch[t]` + T separate deferred states, all
pipelined. Preserves the existing CMD3↔CMD1 overlap and naturally
batches expert loading because T tokens want the same layer's experts
at the same time.

**Approach B (MoE cross-token decoupling):** keep layers-inside-tokens
per-token but queue the MoE routing decisions across adjacent tokens at
the same layer, then dispatch via sparse-MoE once T routing decisions
have accumulated. Smaller surgical change than A. Requires carefully
decoupling MoE completion from the deferred-expert pipeline.

Both are scoped as multi-session refactors with hash-diff validation at
every stage. Sub-agents spawned to explore each in parallel.
