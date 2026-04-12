# flash_moe optimization roadmap — pipeline explainer + full menu

**Date:** 2026-04-11 evening, after Phase 0 + Phase 1a ANE investigation
**Purpose:** Explain the current warm-cache pipeline architecture clearly enough
that any future reader understands why certain optimizations are viable and
others are blocked, then catalog every optimization avenue we've considered with
honest value/effort/risk estimates.

**Decision from this pass:** Multi-buffered deferred state (Approach A from batch
prefill scoping) is the next target, starting with a measurement-driven Phase 0
that costs 1-2 days to answer "is the GPU compute-bound or latency-bound for
multi-stream execution?" This question has been the hidden load-bearing
assumption behind every optimization decision today and nobody has actually
measured it.

Mixed-bit per-expert quantization is deprioritized — external projects have
explored it with mixed results and the quality risk isn't clearly worth the
10-20% upside.

---

# Part 1 — What `fused_layer_forward` actually does

For a single token, each layer runs three Metal command buffers. The structure
is the source of most optimization constraints, so understanding it is essential
before evaluating any refactor.

## CMD1 — attention prep (submit, wait)

```
input_norm(hidden)               → buf_input      (GPU)
Q_proj × buf_input               → q              (GPU, 4-bit matvec)
K_proj × buf_input               → k              (GPU)
V_proj × buf_input               → v              (GPU)
```

Single command buffer, several compute encoders, committed and waited on from
CPU. While CPU waits, GPU runs the kernels.

## CMD2 — attention finish + post-attention (submit, wait)

```
attention (scores, softmax, values)      (GPU, OR CPU for sequence < 32)
o_proj(attn_out)                 → buf_output     (GPU)
residual_add(buf_output, buf_residual) → buf_h_mid (GPU)
post_attn_norm(buf_h_mid)        → normalized_mid (GPU)
routing_proj(normalized_mid)     → gate_scores    (GPU)
shared_expert.gate/up/act/down projections         (GPU)
```

Again single command buffer, multiple encoders, committed and waited on. While
CPU waits, GPU runs these encoders and CPU does parallel pread of routed experts
from SSD into the malloc cache (or OS page cache).

## CMD3 — MoE experts + finalize (submit, DOES NOT wait)

```
for each routed expert k:
    expert_k_forward(h_mid)      → expert_out[k]  (GPU, parallel)
shared_expert SwiGLU + down      → shared_out     (GPU)
gpu_moe_combine: buf_moe_hidden = h_mid
                 + Σ(w[k] × expert_out[k])
                 + w_shared × shared_out
rms_norm(buf_moe_hidden)         → buf_input      (GPU)
                                  ^^^^^^^^^
                       note: writes to the SAME buf_input
                       that the NEXT layer's CMD1 will read
```

**The clever part:** CMD3 is dispatched *async*. The CPU doesn't wait on it.
Instead the code records the command buffer handle and some bookkeeping in a
global struct `g_deferred` and immediately returns to the caller.

## The fast path — how layers overlap

On the next layer's call, `fused_layer_forward` sees `g_deferred.active == true`
and takes the fast path:

1. **CMD1 of layer N+1 can be submitted immediately.** Metal's serial command
   queue guarantees CMD1(N+1) won't actually execute on GPU until CMD3(N)
   finishes, because both touch `buf_input`. The GPU handles the ordering. The
   CPU does not have to wait.
2. While GPU is still running CMD3(N), the CPU prepares CMD2(N+1)'s encoder
   setup and does its own scratch work.
3. By the time CPU needs to wait on CMD1(N+1) to complete — which implicitly
   requires CMD3(N) to finish first — both have had a chance to run concurrently.

## Timing breakdown (warm cache, 8.4 tok/s)

```
cmd1_wait      0.858 ms  (CPU blocked on GPU attention kernels)
cmd2_wait      0.426 ms  (CPU blocked on GPU post-attention + shared-expert prep)
expert_io      1.337 ms  (CPU blocked on pread() of routed experts +
                          GPU concurrently running CMD3 of the previous layer)
total_layer    2.703 ms  (sum of the waits, NOT wall clock)
```

The measured 8.4 tok/s gives ~119 ms/token for generation ÷ 60 layers =
**1.98 ms per layer wall clock**. That's 0.72 ms less than the sum of waits,
because `expert_io` overlaps with CMD3 of the prior layer, not just pread.

**GPU utilization estimate:**
- GPU busy during cmd1_wait: 0.858 ms
- GPU busy during cmd2_wait: 0.426 ms
- GPU busy during expert_io running CMD3: ~1.0 of the 1.337 ms
- Total GPU-busy per layer: ~2.28 ms out of 2.7 ms = **~85% GPU utilization**
- CPU busy: ~0.082 ms per layer = **~3% of layer time**

So the pipeline is GPU + SSD I/O co-saturated with CPU barely participating.

---

# Part 2 — The architectural constraint that blocks most optimizations

Everything the GPU does is a function of the hidden state chain:

```
hidden_0 → attn_0 → MoE_0 → hidden_1 → attn_1 → MoE_1 → hidden_2 → ... → hidden_60
```

Each arrow is a strict data dependency. You cannot start `attn_N+1` until
`MoE_N` is done. The whole 60-layer chain is serialized within a single token.

The current pipeline cheats by overlapping *one layer's* CMD3 with the *next
layer's* CMD1 via the serial command queue's implicit ordering. That's the
only cross-layer overlap available to a single-token stream.

**Within a single token stream, you cannot do better than this.** The pipeline
is architecturally optimal for its topology. Any further speedup must either:

1. Attack the constant factor of individual kernels (kernel fusion, mixed-bit,
   expert clustering — each potentially saves 5-15%)
2. Break the serial chain by running multiple independent token streams in
   parallel (Approach A / multi-buffered state — potentially saves 15-100%)

Everything else we've considered falls into one of those two categories, and
the outcomes depend on which bucket it attacks.

---

# Part 3 — What "multi-buffered deferred state" actually means

This is the key idea that's been on the table since the batch prefill
investigation. It's worth understanding precisely because it determines whether
the top-line speedup is available or not.

## The single-buffer problem

`g_deferred` is a single global struct. It tracks the one pending CMD3's
command buffer, output hidden pointer, expert indices, scratch state, etc.

The Metal context has a single `buf_input`, `buf_output`, `buf_moe_hidden`,
`buf_multi_expert_out[K]`, etc. There's exactly one set of scratch GPU buffers,
used by whichever token is currently being processed.

If we want to process 2 tokens through layer N in parallel, they would both try
to write to the same `buf_input` on the same command queue. Serialized or
corrupted — neither useful.

## The multi-buffer fix

Duplicate every shared buffer per slot, and duplicate `g_deferred` per slot:

```c
struct MetalCtx {
    id<MTLBuffer> buf_input[MAX_BATCH_T];
    id<MTLBuffer> buf_output[MAX_BATCH_T];
    id<MTLBuffer> buf_moe_hidden[MAX_BATCH_T];
    id<MTLBuffer> buf_multi_expert_out[MAX_BATCH_T][MAX_K];
    // ... ~25 buffer families, each × MAX_BATCH_T
};

static DeferredExperts g_deferred[MAX_BATCH_T];
```

And the call pattern becomes:

```c
for layer L in 0..60:
    for slot t in 0..T:
        fused_layer_forward_slot(L, t, hidden_batch[t],
                                 /* uses buf_*[t] and g_deferred[t] */);
```

Agent A's scoping pass in the batch prefill investigation counted ~25 buffer
families, ~200 references in `fused_layer_forward` and its Metal encoder
helpers. Each reference needs to become `buf_*_slot[slot]` or equivalent.
Estimated: **1500-2000 LOC** mechanical churn.

## How parallelism unlocks

With per-slot buffers and per-slot deferred state, at layer N you can:

- Dispatch slot 0's CMD1(N) — reads `buf_input[0]`, writes Q/K/V slot 0
- Dispatch slot 1's CMD1(N) — reads `buf_input[1]`, writes Q/K/V slot 1
- Dispatch slot 2's CMD1(N) — ...
- Dispatch slot 3's CMD1(N) — ...

These have no data dependencies on each other. If placed on different command
queues, Metal can execute them in parallel on GPU's compute units. On a single
serial queue they still serialize, so this likely needs T separate command
queues or a concurrent queue type.

Meanwhile the previous layer's CMD3 is still running async for each slot. So at
any instant you could have CMD3(layer N-1) × 4 slots + CMD1(layer N) × 4 slots
all in flight — 8 in-flight GPU operations instead of the current 2.

## Ideal-case math

Current: `60 layers × 1.98 ms/layer = 119 ms per token`
Ideal T=4: `60 layers × (1.98 ms for 4 tokens in parallel) = 119 ms for 4 tokens`
         = **30 ms per token, ~4× speedup**

Actual number depends entirely on GPU compute headroom and memory bandwidth. If
the GPU is already at 100% utilization for one slot's work, adding more slots
just queues up work and gives no wall-clock improvement. If the GPU has
significant idle time or is latency-bound on memory / dispatches, multi-slot
fills those gaps and gives meaningful speedup.

---

# Part 4 — The load-bearing uncertainty: compute-bound vs latency-bound

Everything in this analysis has been implicitly assuming one answer or the other
to this question:

**Is the flash_moe pipeline compute-bound (GPU ALU at ceiling) or latency-bound
(GPU waiting on memory, dispatch, or I/O) when processing a single token?**

## Case A — compute-bound

If the GPU's arithmetic units are pegged by the single-token workload, then
running T=4 tokens takes ~4× the single-token wall time. No gain. Approach A is
a 2000-LOC refactor for zero benefit. All the other "add more parallel compute"
ideas (ANE offload, multi-stream) hit the same wall.

In this world, the only wins come from reducing per-layer work: smaller
kernels, reduced precision, fused dispatches, mixed-bit quantization, expert
clustering.

## Case B — latency-bound

If the GPU is waiting on memory reads, dispatch overhead, or coordinating with
other subsystems (CPU/SSD), then running T=4 tokens takes ~1.5× the single-token
wall time. Per-token wall clock drops ~2.5×. Approach A becomes a major win.

In this world, ANE offload also becomes viable (we'd be filling GPU idle time
with useful work).

## Empirical evidence we do have

From the sub-agent's `dispatch_experts_sparse` integration tests:

**Cold cache, 15-token prompt:**
- `--batch-prefill 1`: first chunk 622 ms
- `--batch-prefill 4`: first chunk **265 ms** (2.35× faster)

That 2.35× came from sparse MoE bucketing: 4 tokens' routing decisions pooled
into 10-12 unique experts instead of 16, pread once each. Which proves that for
cold `expert_io`, T=4 batching is **strongly latency-bound** and gives huge
wins.

**Warm cache, 138-token prompt:**
- `--batch-prefill 1`: 150 ms/token rest avg
- `--batch-prefill 4`: 236 ms/token rest avg (+57% regression)

That regression came from the linear-attention fallback breaking the intra-token
pipeline (see `docs/2026-04-11-batch-prefill-scoping.md`). It does NOT prove
warm-cache is compute-bound — it proves the partial implementation was flawed.
A full Approach A could give a different answer.

## What's needed to resolve this

A Phase-0-style measurement that implements multi-buffered deferred state for a
single restricted case (full-attention layers only, where state is just KV cache
append and trivially parallelizable) and measures whether the GPU gives back
speedup at T=4. This is 1-2 days of focused work — similar in scope to the ANE
Phase 0 dispatch bench. The result:

- If T=4 gives ≥ 2× full-attention throughput → GPU is latency-bound, commit to
  the full Approach A refactor, expected 30-60% total wall-clock win
- If T=4 gives ~1× full-attention throughput → GPU is compute-bound, Approach A
  is dead, focus on constant-factor optimizations (mixed-bit, fusion)

Either answer closes the uncertainty that's been blocking decision-making all
session.

---

# Part 5 — Full optimization menu

Every optimization avenue we've considered, with honest value / probability /
effort / risk estimates. Organized by which bucket they attack.

## Category 1 — Attack `expert_io` (49.5% of per-layer time)

This is the biggest single bucket and the most obvious target.

### 1a. Mixed-bit per-expert quantization — previously #1, now deprioritized

**Idea:** Per-expert sensitivity analysis. Keep quantization-sensitive experts
at 4-bit (preserves quality on tool calling, structured output, etc). Demote
quantization-tolerant experts to 2-bit. Reduces I/O proportional to the % of
experts demoted.

**Expected value:** 10-20% wall-clock at 50% demotion rate. Scales with
demotion fraction.

**Effort:** 3-5 days — offline sensitivity tool + conversion pipeline + runtime
loader switchable per-expert. Much of the 2-bit loader plumbing already exists
but is a global flag, not per-expert.

**Risk:** LOW. Sensitivity analysis is measurable at every step. Bad demotions
can be caught by comparing logits before/after.

**Deprioritization reason (2026-04-11):** User has seen mixed results in other
MoE projects — the quality risk isn't clearly worth the upside. Parking for
later reconsideration.

### 1b. Expert compression in-RAM — previously tried LZ4, regressed

**Idea:** Keep experts in RAM but compressed (zstd, LZFSE, or bitpacking). CPU
decompresses during the `expert_io` wait window when CPU is otherwise idle.
Previous LZ4 attempt failed because decompression cost > savings, but the
context was different (cache was smaller, LZ4 was CPU-heavy).

**Expected value:** 5-15% uncertain.

**Effort:** 2-3 days.

**Risk:** MEDIUM. Previously failed. Different compression algorithms might
work better on this workload.

### 1c. Expert prefetch prediction — previously tried, regressed -58%

**Idea:** Predict which experts a token will route to BEFORE the routing step,
prefetch them speculatively. Previous attempts used simple temporal locality
(26% hit rate) which was far too noisy.

**Better version:** Train a small neural predictor (MLP or tiny transformer)
on hidden state → predicted top-K, using the routing log data we already
collect via `--collect-routing`. Requires ML research.

**Expected value:** uncertain. Highly dependent on predictor accuracy.

**Effort:** 1 week + research.

**Risk:** HIGH. Research-scale, not engineering-scale.

## Category 2 — Break the serial ceiling (unlock T-parallelism)

These are the high-ceiling options. They either unlock the serial constraint or
confirm it's immovable.

### 2a. Multi-buffered deferred state (Approach A) — NEW #1

**Idea:** Explained in detail above. T parallel token streams through the layer
stack with per-slot buffers and per-slot deferred state, running on multiple
Metal command queues to get cross-slot GPU parallelism.

**Expected value:** 15-100% wall-clock. Range is wide because it depends on the
compute-bound vs latency-bound question.

**Effort decomposition:**
- **Phase 0 measurement:** 1-2 days. Implement multi-buffered state for
  full-attention layers ONLY (simple KV cache append, no recurrence). Benchmark
  at T=4 warm cache. Answers the compute-vs-latency question definitively.
- **Phase 1 (if Phase 0 positive):** 4-6 days. Extend to all layers EXCEPT
  linear attention. Keep linear attention on per-token serial path. Should
  still give meaningful win via sparse-MoE amortization + full-attn parallelism.
- **Phase 2 (if Phase 1 positive):** 3-5 days. Implement parallel scan for
  linear-attention recurrent state. Unlocks the full Approach A.

**Risk:** HIGH. 1500-2000 LOC mechanical refactor once committed. Per-layer
BPDBG hash diff validation approach (from TurboQuant debugging) is the
mandatory checkpoint technique.

**Decision:** Phase 0 is worth doing regardless of other priorities because it
closes the biggest unknown in the optimization landscape. If it comes back
negative, we know Approach A is dead and can focus on constant-factor wins with
confidence.

### 2b. ANE offload of linear-attention layers — blocked, see Phase 1a

**Status:** Phase 0 + Phase 1a complete 2026-04-11. Structural viability
confirmed (100% ANE placement, LUT4 at 1.41 ms/layer p50). BUT the layer
dependency chain serializes ANE and GPU work, making it a 43% wall-clock
regression vs the current pipelined GPU path. Blocked by the same constraint
as Approach A — only viable if multi-token batching (Approach A) lands first.

**Durable findings still valid:** LUT4 is always right for ANE deployment,
Swift CoreML dispatch overhead is ~1-2 ms stable, `MTLBuffer ↔ MLMultiArray`
transfer is <5 μs zero-copy. See `docs/2026-04-11-ane-phase1a-results.md`.

**Revisit:** after Approach A Phase 0, OR if power efficiency becomes a
constraint.

## Category 3 — Attack GPU compute (`cmd1_wait + cmd2_wait` = 47%)

These reduce per-layer constant factors without changing the serial topology.
Stackable with everything else.

### 3a. Kernel fusion — NEW #2

**Idea:** Merge Metal command encoders that don't need CPU work between them,
reducing command buffer overhead and giving the GPU scheduler more to chew on
between CPU syncs.

**Already done:** TurboQuant `tq_pack_update + tq_fused_attention + sigmoid_gate`
into `cmd_fused` (commit `8a9078f`), saved ~3 ms/token = ~2%.

**Remaining opportunities:**
- `o_proj + residual + norm + routing_proj` already fused in CMD2; could push
  further into shared-expert projections
- `attention_scores + softmax + values` are currently separate encoders in the
  non-TQ path — could fuse into one
- CMD3's `expert_combine + rms_norm` already fused in the fast path

**Expected value:** 5-10% more, stacking on top of what's already fused.

**Effort:** 1-2 days per fusion target. Incremental, revertable per commit.

**Risk:** LOW. Surgical and correctness-verifiable via hash diff.

### 3b. Attention kernel register tiling — not previously tried

**Idea:** Current `attn_scores / attn_softmax / attn_values` kernels may be
suboptimally tiled for M4 Pro's shared memory / register file. Better tiling
could cut `cmd1_wait` (currently ~0.86 ms × 60 = 51 ms/token).

**Expected value:** 5-15% reduction in `cmd1_wait` = 3-8% wall clock.

**Effort:** 2-3 days. Requires Metal kernel expertise, profiling with xctrace.

**Risk:** MEDIUM. Shader-level tuning is finicky and hardware-specific.

### 3c. Reduced-precision attention inner loops — explicitly ruled out

**Idea:** bf16 accumulator in matmul, fp32 output. Faster per layer.

**Status:** Ruled out by user on quality grounds. 4-bit is already at the
quality cliff and layered quantization risks pushing over.

**Keeping on the list for completeness only.**

## Category 4 — Attack the I/O subsystem

### 4a. Expert clustering by co-activation

**Idea:** Cluster experts that frequently co-fire into the same file or page
so a single pread loads multiple. Reduces syscall count and amplifies sequential
read locality.

**Data available:** The `--collect-routing` binary format is already
implemented and `scripts/analyze_routing.py` can compute co-firing statistics.

**Expected value:** 5-15% reduction in `expert_io`. Depends on how sharp the
clustering is in real workloads — our earlier analysis showed only ~42%
cross-prompt overlap in top-N, which is 3× better than random but not
overwhelming.

**Effort:** 2-4 days — offline analysis + repack script + runtime loader for
co-located experts.

**Risk:** MEDIUM. Gains are bounded by actual co-activation patterns.

### 4b. Persistent universal hot set

**Idea:** Pin the 4-16 experts per layer that are hot in the *intersection* of
many diverse prompts. Much smaller than per-prompt top-N but universally hot.
Combined with LRU for the rest, gives a small stable win.

**Status:** Earlier persistent-working-set analysis (#26) showed naive top-N is
~42% of optimal cross-prompt. But *universal hot set* (tiny, 4-16 per layer)
hasn't been measured specifically.

**Expected value:** 2-5%. Small because the universal hot set is small.

**Effort:** 1-2 days.

**Risk:** LOW. Incremental refinement of existing LRU.

## Category 5 — System-level

### 5a. Server-mode cross-request batching

**Idea:** In server mode, concurrent requests can share expert loads. Request A
routes to expert 42; request B also routes to expert 42 — load it once for
both. Same sparse-MoE trick as batch prefill, but across independent requests.

**Expected value:** depends on workload. For heavy concurrent user sessions,
20-50% shared expert loads likely. For single-user mode, 0%.

**Effort:** 3-5 days — coordination layer in the HTTP server, routing pool.

**Risk:** MEDIUM. Only helps the right workload.

### 5b. Generation speculative decoding with verification batching

**Idea:** Small fast draft model (e.g., a quantized 1-2B) drafts 4-8 tokens.
flash_moe verifies them in a batch using Approach A. Accepted drafts advance
the position. Rejected drafts fall back to normal generation.

**Requires:** Approach A (for the verification batch) + draft model + token
acceptance logic.

**Expected value:** 20-50% if draft accuracy is 60%+. 0% if lower.

**Effort:** multi-week.

**Risk:** HIGH. Research-scale and depends on Approach A landing first.

---

# Part 6 — Ranking

Force-ranked by `(expected_value × probability_of_success) / effort_days`. I've
included the deprioritized ones at the bottom so you can see everything in one
place.

| Rank | Target | Expected value | Probability | Effort | Notes |
|---|---|---|---|---|---|
| **1** | **Multi-buffer Phase 0 (2a)** | 0-80% | Unknown | **1-2 days** | Cheap measurement to unblock the biggest win. Either answer is valuable. |
| 2 | Kernel fusion (3a) | 5-10% per target | High | 1-2 days | Stackable with anything. Incremental wins. |
| 3 | Expert clustering (4a) | 5-15% | Medium | 2-4 days | Routing log infrastructure already exists. |
| 4 | Attention kernel tiling (3b) | 5-15% | Medium | 2-3 days | Requires Metal kernel expertise. |
| 5 | Persistent universal hot set (4b) | 2-5% | High | 1-2 days | Small but cheap. |
| 6 | Multi-buffer Phase 1 (2a cont.) | 30-60% | Depends on Phase 0 | 4-6 days | Only if Phase 0 is positive. |
| 7 | Server cross-request batching (5a) | 0-50% | Workload-dependent | 3-5 days | Only for concurrent-user mode. |
| 8 | Multi-buffer Phase 2 — parallel scan (2a cont.) | +10-20% | Depends on Phases 0/1 | 3-5 days | Unlocks the full ceiling. |
| — | Mixed-bit experts (1a) | 10-20% | High | 3-5 days | **Deprioritized** 2026-04-11 (user has seen mixed results in other projects). |
| — | ANE offload (2b) | Blocked | — | — | Blocked on Approach A landing first. Phase 0+1a findings still durable. |
| — | Expert compression revisit (1b) | 5-15% | Low | 2-3 days | Previously failed with LZ4. |
| — | Expert predictor (1c) | uncertain | Low | 1 week + research | Research scale. |
| — | Reduced-precision attention (3c) | — | — | — | User ruled out (quality cliff). |
| — | Speculative decoding (5b) | 20-50% | Depends on Approach A | multi-week | Research-scale and dependent on 2a. |

## Structural observation

Items 1-5 attack **independent** buckets:

- Multi-buffer (2a) attacks the serial ceiling
- Kernel fusion (3a) attacks GPU encoder overhead
- Expert clustering (4a) attacks SSD I/O
- Attention kernel tiling (3b) attacks GPU attention compute
- Persistent hot set (4b) attacks cache hit rate

They don't fight over the same time budget and can in principle be stacked.
Diminishing returns apply (as one bucket shrinks, the others become a larger
fraction of the total), but compounding is real.

If all five land successfully, the theoretical stacked upper bound is something
like `(1 - 0.5) × (1 - 0.1) × (1 - 0.1) × (1 - 0.1) × (1 - 0.03) = ~0.35x` of
current latency, i.e. roughly **3× speedup**. In practice the real number would
be more like 1.5-2× because each step hits diminishing returns and Approach A's
ceiling may be lower than the theoretical max.

---

# Part 7 — Decision for the next working session

**Start with Multi-buffer Phase 0** (item 1 in the ranking).

It's the cheapest item on the list (1-2 days), it resolves the biggest hidden
uncertainty (compute-bound vs latency-bound), and its outcome determines
whether 30-60% wall-clock wins are reachable or whether we should focus on
1-2% constant-factor optimizations.

Specifically Phase 0 should:

1. Implement multi-buffered deferred state for full-attention layers ONLY.
   Linear attention layers continue using the existing per-token path. This
   tests whether multi-streaming works where state is easy.
2. Add `MAX_BATCH_T` per-slot copies of: `buf_input`, `buf_output`,
   `buf_moe_hidden`, `g_deferred`, `buf_attn_q/out/gate/scores`,
   `buf_multi_expert_*`, `buf_shared_*`. Skip buffer families that only linear
   attention touches.
3. Restructure the full-attention path in `fused_layer_forward` to accept a
   `slot` parameter.
4. Build a minimal benchmark that runs T=4 prefill on a prompt with only
   full-attention layers represented (or measure at full-attention layer
   boundaries specifically).
5. Compare T=1 vs T=4 per-token wall clock. If T=4 at full-attn is ≥50%
   faster, Approach A is viable and we commit to Phase 1. If it's flat or
   slower, Approach A is dead and we focus on kernel fusion and expert
   clustering as the stackable small-wins path.

**Validation approach** (mandatory): per-layer BPDBG hash diff, same as the
TurboQuant and batch-prefill sub-agent work. T=1 and T=4 hidden states must
match bit-exactly where state permits.

**Revisit cadence:** After Phase 0 resolves, re-rank based on the answer.

---

# Cross-references

- `docs/2026-04-11-batch-prefill-scoping.md` — the original batch prefill
  investigation that established the loop-inversion and layer-dependency
  findings. Afternoon update has the load-bearing trace.
- `docs/2026-04-11-ane-offload-scoping.md` — ANE offload scoping doc.
- `docs/2026-04-11-ane-phase1a-results.md` — Phase 0 + Phase 1a ANE
  measurements that confirmed ANE was blocked by the same constraint.
- `metal_infer/infer.m:4107-4124` — `complete_deferred_experts` /
  `discard_deferred_experts` implementation.
- `metal_infer/infer.m:4873-6600` — `fused_layer_forward` main body (where
  all the multi-buffering work would happen).
- `metal_infer/infer.m:9065-9180` — the prefill driver that calls
  `fused_layer_forward` per-token. This is the outer loop that would need to
  interleave T slots.
