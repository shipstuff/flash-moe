# ANE Phase 1a results — single-layer and 4-bundle conversion feasibility

**Date:** 2026-04-11 evening
**Branch from:** `docs/2026-04-11-ane-offload-scoping.md` Phase 1 stepping stone
**Outcome:** **Structural viability confirmed BUT wall-clock extrapolation
reveals an architectural block specific to MoE models. Strategy is NOT viable
for flash_moe without multi-token batching.**

## Summary

Three phases of measurement were run:

1. **Test A — fp16 single layer** (negative): per-layer 8.94 ms → 402 ms/token extrapolated, regression
2. **Test B — LUT4 single layer** (apparent positive): per-layer 1.41 ms → 63 ms/token extrapolated, ~35% projected speedup
3. **Test C — LUT4 4-layer bundle** (apparent better): per-layer 1.21 ms effective → ~55 ms/token extrapolated
4. **Architectural review** (decisive negative): the 4-bundle test does not model flash_moe's actual runtime because it omits MoE dispatches between layers. The real per-layer cost is 1.41 ms (unbundled), serialized through the layer dependency chain, producing a 43% regression vs the current GPU-only pipelined path.

The structural measurements are all correct. The optimistic extrapolations in Phase 0 and early Phase 1a were wrong because they assumed ANE and GPU could run in parallel on different parts of the same token's forward pass. They can't — each layer's MLP (MoE on GPU) must complete before the next layer's attention (on ANE) can start.

## Tests run

### Test A — fp16 single layer (unquantized baseline)

Script: `ane_bench/scripts/convert_fm_linear_l0_test.py`

Converted one `Qwen35GatedDeltaNet` layer with flash_moe's dimensions (Hv=64, conv_dim=12288) and random weights. `ct.convert` at fp16, ComputeUnit.CPU_AND_NE, iOS18 target.

**anemll-profile:**
```
Model size: 225.2 MB   Total ops: 166   ANE: 74 (100%)   CPU: 0
Graph interruptions: 0   Timeline: main ANE 3-165
linear 5 ops  1.180 ms/op  5.900 ms total  85.7% of cost  Comp bound
TOTAL sequential: 6.882 ms   Measured: 8.928 ms/prediction  (112 iter/s)
```

**ane_dispatch_bench (Obj-C):**
```
min: 8.642  p50: 8.940  p90: 9.615  p99: 10.217  max: 10.243 (ms)
110.5 predictions/sec
```

**Verdict:** structurally fine but would be 45 × 8.94 = 402 ms/token linear compute, a catastrophic regression.

### Test B — LUT4 single layer (per_grouped_channel, group_size=8)

Script: `ane_bench/scripts/convert_fm_linear_l0_lut4_test.py`

Same model as Test A, with LUT4 palettization added after `ct.convert`:
```python
cto_coreml.OpPalettizerConfig(
    mode="kmeans", nbits=4,
    granularity="per_grouped_channel",
    group_size=8, num_kmeans_workers=1,
)
```

**Size reduction:** 225 MB → 57 MB (4×, as expected for fp16 → 4-bit weights)

**anemll-profile:**
```
Total ops: 166   ANE: 74 (100%)   CPU: 0   Interruptions: 0
TOTAL sequential: 12.703 ms   Measured: 1.746 ms/prediction  (572 iter/s)
Speedup: 7.3× vs sequential estimate
```

**ane_dispatch_bench (Obj-C):**
```
min: 1.246  p50: 1.410  p90: 1.609  p99: 2.020  max: 2.196 (ms)
699.9 predictions/sec
```

**Verdict:** 6.3× faster per-layer than fp16. The initial "linear ops are Comp-bound so LUT4 won't help" hypothesis was wrong — ANE has a fundamentally different and much faster code path for LUT4 weights. This is a durable finding for all future ANE work.

### Test C — LUT4 4-layer bundle

Script: `ane_bench/scripts/convert_fm_linear_bundle4_lut4_test.py`

Stacked four independently-weighted `Qwen35GatedDeltaNet` instances in a single wrapper, threading hidden state and 4 state/conv pairs through them sequentially. Otherwise identical config to Test B.

**Scaling check:** 545 PyTorch ops vs 137 for single layer (4×, as expected)

**anemll-profile:**
```
Total ops: 655   ANE: 296 (100%)   CPU: 0   Interruptions: 0
TOTAL sequential: 50.812 ms  (4 × 12.703, linear scaling)
Measured: 4.980 ms/prediction  (200.8 iter/s)
Speedup: 10.2× vs sequential estimate
Per-layer effective: 1.245 ms
```

**ane_dispatch_bench (Obj-C):**
```
min: 4.668  p50: 4.828  p90: 6.060  p99: 7.027  max: 7.146 (ms)
196.8 predictions/sec
Per-layer effective: 1.207 ms
```

**Apparent result:** bundling saved ~14% per-layer (1.410 → 1.207 ms) in Obj-C, confirming that dispatch overhead amortizes across bundled layers.

## The architectural block

Here's what the bundle test doesn't model: **flash_moe's real layer sequence is** `attn → MoE → attn → MoE → ...`, not `attn → attn → attn → attn`. A 4-layer bundle on ANE computes:

```
x1 = attn(x0)                   # ignores MLP
x2 = attn(x1)                   # wrong — should be attn(mlp(x1) + x1)
x3 = attn(x2)                   # wrong
x4 = attn(x3)                   # wrong
```

But the mathematically correct sequence is:
```
x1 = mlp(attn(x0)) + residual
x2 = mlp(attn(x1)) + residual
x3 = mlp(attn(x2)) + residual
x4 = mlp(attn(x3)) + residual
```

**anemll-qwen35's super-blocks include the MLP.** Their Qwen3.5-9B is dense, so MLP is a simple `gate * up * down` FFN that fits inside the ANE-resident super-block. Their 4-layer bundle at 9.28 ms total correctly computes 4 layers of attn+mlp.

**flash_moe's MLP is MoE.** 512 experts per layer, routed per-token, streamed from SSD as needed. ANE has no equivalent streaming model type — it requires pre-compiled `.mlpackage` bundles with fixed weights. We cannot include MoE in the ANE super-block.

So for flash_moe, every layer is a separate ANE call: the bundle would have to break after each attention, transfer hidden state to GPU, run MoE, transfer back, start the next attention.

## Revised wall-clock math (the honest version)

Per-layer serialized through ANE→GPU→ANE→GPU:

| Layer type | ANE work | GPU work | Serial per-layer total |
|---|---|---|---|
| linear (45 layers) | 1.41 ms (attn) | ~1.4 ms (MoE cmd3) | 2.81 ms |
| full attn (15 layers) | (none — full attn stays GPU) | ~1.5 ms attn + 1.4 ms MoE | 2.9 ms |

**Wall clock per token:**
- 45 × 2.81 + 15 × 2.9 = **170 ms/token**
- Current GPU-only pipelined: 119 ms/token
- **ANE offload: +43% REGRESSION**

The GPU-only path wins because Metal's serial command queue provides CMD3(layer N-1) ↔ CMD1(layer N) overlap inside a single engine. Moving CMD1 to ANE gives up that overlap — ANE and Metal don't share a command queue, so there's no cross-engine dependency primitive that lets ANE start layer N+1's attention while Metal is still running layer N's MoE.

## Why the Phase 0 extrapolation looked positive

Phase 0 measured the single-super-block dispatch time (6.64 ms for 4 layers on anemll-qwen35's LUT4 super-block). I extrapolated to flash_moe as "15 super-blocks × ~7 ms = 100 ms/token on ANE, with GPU running MoE in parallel at ~78 ms/token, wall clock max(100, 78) = ~100 ms/token = 33% faster than 150 ms baseline."

**That extrapolation had two errors:**

1. It assumed flash_moe super-blocks would match anemll-qwen35's timing (≈6.64 ms per 4 layers). But anemll-qwen35's super-block includes the MLP work, which is DENSE and fits. flash_moe's super-block cannot include the MoE MLP, so it would only contain attention — 4 attention layers' worth of work, not 4 full layers. And separately: flash_moe's attention compute is larger (Hv=64 vs 32), so even just the attention bundle is ~1.5-2× larger.

2. It assumed ANE and GPU work run **in parallel**. They don't — the dependency chain forces serial execution per layer unless multiple tokens are processed simultaneously (which was already ruled out in the batch prefill investigation).

## Why this converges with the batch prefill finding

The batch prefill investigation (docs/2026-04-11-batch-prefill-scoping.md) concluded that warm-cache speedups are blocked by a ~97% GPU saturation ceiling + the fact that 45/60 layers have per-token recurrent state that can't be multi-streamed. Two sub-agent investigations (Approach A multi-buffered deferred state, Approach B MoE cross-token decoupling) both confirmed the block.

**ANE offload hits the same block from a different direction.** It doesn't care about GPU saturation per se, but the layer dependency chain means ANE work must serialize through GPU work for any single token. And multi-token batching — which would let ANE and GPU run on different tokens simultaneously and finally provide useful overlap — is the same Approach A constraint that couldn't be resolved for batch prefill.

**The deep conclusion:** flash_moe's warm-cache single-token inference is architecturally serialized. No hardware accelerator can break the dependency chain alone. Only a multi-token path breaks it, and multi-token requires solving the multi-buffered state problem first.

## When ANE offload would become viable

The strategy is viable if any ONE of these becomes true:

1. **Multi-token batching lands.** If Approach A (multi-buffered deferred state) is implemented, then 8 token streams can run through the layer stack in lockstep. ANE processes 8 tokens' attention per super-block while GPU processes 8 tokens' MoE per layer, trading serial dependency for batched parallelism. Effort: 9-15 days for Approach A alone, plus 9-15 days for ANE integration.

2. **The model becomes dense (or partially dense).** If future flash_moe variants include some layers with a dense MLP instead of MoE, those layers can be bundled on ANE and amortize dispatch. Purely hypothetical today.

3. **Power efficiency becomes a hard constraint.** The ANE path has lower power draw than the GPU path. If we're running on battery or under sustained thermal pressure, a 43% wall-clock regression might be acceptable for 30-50% lower power. Would need measurement to confirm.

4. **Cold-cache scenarios dominate.** Warm-cache wall clock is 119 ms/token. Cold cache (first query of a session, after system reboot) is probably 200-300 ms/token because `expert_io` balloons. If ANE compute can hide behind cold `expert_io` waits, the offload could be neutral or positive there. Not measured today.

## Per-test evidence table

| Test | Variant | Measured p50 | Per-layer effective | 45-layer extrapolation | Wall clock w/ serial MoE |
|---|---|---|---|---|---|
| A | fp16 single | 8.940 ms | 8.940 ms | 402 ms | ~466 ms/token (huge regression) |
| B | LUT4 single | 1.410 ms | 1.410 ms | 63 ms | ~170 ms/token (43% regression) |
| C | LUT4 4-bundle | 4.828 ms | 1.207 ms | 54 ms * | ~170 ms/token (invalid model) |

\* The bundle's 4-layer number is incorrect as a flash_moe extrapolation because it skips MLP between layers. It represents what a dense model would achieve.

## What Phase 1a did prove

1. **flash_moe's dimensions compile fully ANE-resident.** 100% ANE placement, 0 CPU ops, 0 graph interruptions for both fp16 and LUT4 variants. The larger Hv=64 and conv_dim=12288 do not trigger any ANE incompatibility.
2. **LUT4 palettization works on flash_moe shapes and gives a ~6× per-layer speedup vs fp16.** Always use LUT4 for ANE deployment.
3. **Bundling 4 layers saves ~14% per-layer** in terms of ANE dispatch overhead — but only for models where consecutive layers can be bundled, which excludes MoE.
4. **Swift CoreML dispatch overhead at per-prediction granularity** is stable at ~1 ms and doesn't scale with model size. Not a bottleneck.

These findings are real and durable; they'll inform future ANE work on any model type. But none of them beat flash_moe's layer dependency chain.

## Recommendation

**Pivot to priority #2 (mixed-bit per-expert quantization).** That optimization attacks the dominant bucket (expert_io = 49.5% of per-layer time) without introducing any cross-engine dependency. The ANE offload can be revisited if/when multi-token batching becomes a separate engineering effort, but as a standalone project it's blocked by the same architectural constraint that blocked batch prefill.

## Commits affected

- `ane_bench/scripts/convert_fm_linear_l0_test.py` (Test A, fp16 reference)
- `ane_bench/scripts/convert_fm_linear_l0_lut4_test.py` (Test B, LUT4 single layer)
- `ane_bench/scripts/convert_fm_linear_bundle4_lut4_test.py` (Test C, LUT4 4-bundle)
- `ane_bench/ane_dispatch_bench.m` (added `--fm-linear` and `--fm-bundle4` signatures)
- This file

Compiled `.mlmodelc` artifacts (225 MB + 57 MB + 226 MB) are gitignored.
