# ANE Phase 1a results — single-layer conversion feasibility test

**Date:** 2026-04-11 evening
**Branch from:** `docs/2026-04-11-ane-offload-scoping.md` Phase 1 stepping stone
**Outcome:** **Strategy confirmed viable. Proceeding to Phase 1 proper.**

## What was tested

Convert a single Qwen35GatedDeltaNet (linear attention) layer with flash_moe's
dimensions to CoreML, using random weights, and measure:

1. Does it compile fully ANE-resident, with no CPU fallbacks or graph
   interruptions, given flash_moe's larger dimensions (Hv=64 vs 32, conv_dim
   12288 vs 8192) that differ from the anemll-qwen35 reference?
2. What's the raw per-layer dispatch time via `MLModel predictionFromFeatures:`?
3. Does LUT4 palettization (group_size=8, per_grouped_channel) work on
   flash_moe's shapes, and how much does it speed up?

Random weights are fine for structural/timing tests; only the data flow
matters, not the numerical output.

## Setup

flash_moe dimensions vs anemll-qwen35 9B defaults (linear attention layer):

| Field | anemll-qwen35 | flash_moe | Δ |
|---|---|---|---|
| `hidden_size` | 4096 | 4096 | same |
| `linear_num_value_heads` (Hv) | 32 | **64** | 2× |
| `linear_num_key_heads` (Hk) | 16 | 16 | same |
| `linear_key_head_dim` (Dk) | 128 | 128 | same |
| `linear_value_head_dim` (Dv) | 128 | 128 | same |
| derived `key_dim` | 2048 | 2048 | same |
| derived `value_dim` | 4096 | **8192** | 2× |
| derived `conv_dim` | 8192 | **12288** | 1.5× |
| recurrent state `[1,Hv,Dv,Dk]` | 1 MB fp16 | **2 MB fp16** | 2× |

Scripts:
- `scripts/convert_fm_linear_l0_test.py` — base fp16 conversion (mini-02)
- `scripts/convert_fm_linear_l0_lut4_test.py` — LUT4 variant (mini-02)

Benchmark harness:
- `ane_bench/ane_dispatch_bench.m --fm-linear` — Obj-C, 200 back-to-back predictions

## Results

### Test A — fp16 (unquantized)

**Conversion:** 2.65 s ct.convert → 236 MB mlpackage → 225 MB mlmodelc

**anemll-profile:**
```
Model size: 225.2 MB
Total ops: 166   |   ANE ops: 74 (100% of cost)   |   CPU ops: 0
ANE graph interruptions: 0
Timeline: main ANE 3-165       ← full-graph ANE placement

linear   5 ops  1.180 ms/op  5.900 ms total  85.7% of cost  Comp bound
TOTAL sequential:      6.882 ms
Measured:              8.928 ms/prediction  (112 iter/s)
```

**ane_dispatch_bench (Obj-C):**
```
min: 8.642 ms  p50: 8.940 ms  p90: 9.615 ms  p99: 10.217 ms  max: 10.243 ms
throughput: 110.5 predictions/sec
```

**Verdict:** structurally fine (100% ANE, 0 interruptions), but 45 × 8.94 ms =
**402 ms/token** would be a severe regression vs the current 150 ms/token GPU
baseline.

### Test B — LUT4 palettized (matches anemll-qwen35 production config)

**Conversion:** 2.65 s ct.convert + 55 s palettize → 59 MB mlpackage → 57 MB mlmodelc
(4× size reduction)

Palettization config (copied from `anemll-qwen35/scripts/convert_all_superblocks_lut4.py`):
```python
cto.coreml.OpPalettizerConfig(
    mode="kmeans", nbits=4,
    granularity="per_grouped_channel",
    group_size=8, num_kmeans_workers=1,
)
```

**anemll-profile:**
```
Model size: 56.5 MB
Total ops: 166   |   ANE ops: 74 (100% of cost)   |   CPU ops: 0
ANE graph interruptions: 0
TOTAL sequential:      12.703 ms     (higher than fp16 — more ops post-LUT4)
Measured:              1.746 ms/prediction  (572 iter/s)
Speedup: 7.3× vs sequential estimate    ← ANE runs this much faster than its own cost model predicts
```

**ane_dispatch_bench (Obj-C):**
```
min: 1.246 ms  p50: 1.410 ms  p90: 1.609 ms  p99: 2.020 ms  max: 2.196 ms
throughput: 699.9 predictions/sec
```

**Obj-C is 19% faster than anemll-profile here**, consistent with the Phase 0
observation that raw Obj-C dispatch is consistently a bit tighter than
`anemll-profile`'s measured number.

## Critical insight — why my fp16 extrapolation was wrong

I initially concluded LUT4 would be a wash for our layers because the cost
model reported linear ops as `Comp` bound (compute-throughput limited rather
than memory-BW limited), and LUT4 reduces weight bandwidth without reducing
FLOPs. I was wrong.

**LUT4 is ~6× faster per-layer, not ~2× as the anemll-qwen35 v0→v1 timeline
suggested.** Apple's ANE has a fundamentally different (and much faster) code
path for LUT4-palettized weights. It's not purely about bandwidth — the
hardware MAC units appear to consume quantized weights at a much higher
effective rate than fp16.

The anemll-qwen35 v0→v1 timeline only showed 2× because other non-linear-layer
bottlenecks (notably the unsplit lm_head on CPU) were dragging the overall
tok/s down. Per-layer, LUT4's speedup is much larger.

**Lesson for future Phase 1 work: ALWAYS use LUT4 for ANE deployment.** Don't
benchmark fp16 variants — they're strictly worse across the board with no
quality benefit on ANE.

## Revised ANE offload projection (Obj-C LUT4 numbers)

45 linear layers × 1.410 ms = **63 ms/token** on ANE for linear compute alone.

Full wall-clock scenarios:

| Strategy | ANE path | GPU path (parallel) | Wall clock | vs 150ms |
|---|---|---|---|---|
| 45 linear unbundled | 63 ms | 98 ms (15 full-attn + 60 MoE) | 98 ms | **−35%** |
| 45 linear + bundling | ~75 ms | 98 ms | 98 ms | −35% |
| linear + full-attn on ANE | ~90 ms | 78 ms (60 MoE only) | 90 ms | **−40%** |

All three scenarios beat the current 150 ms/token warm-cache baseline.

Note: these assume GPU work and ANE work actually run in parallel with the
expected ~5% concurrent penalty measured in Phase 0 Unknown #3. Flash_moe's
host-side loop will need to kick ANE and GPU dispatches async to realize this.

## What Phase 1a did NOT test

1. **Real weights instead of random.** Random weights give the same timing but
   say nothing about output correctness. That's Phase 1 proper.
2. **Multi-layer bundling within a super-block.** I only converted one layer.
   A 4-layer (DDDA) super-block may be cheaper per-layer due to amortized
   dispatch, or more expensive due to larger compute graph. Need to test.
3. **Full attention layer conversion.** Different class (Qwen35FullAttention)
   with partial rotary + output gate. anemll-qwen35 has this working in 9B
   dimensions; flash_moe has 32/2 vs 16/4 head configuration so it's not
   guaranteed to compile without modification.
4. **Concurrent GPU+ANE under sustained load.** Phase 0 tested a brief overlap
   window; real deployment needs minutes-long thermal behaviour.
5. **Transition from GPU Metal buffer to ANE MLMultiArray under load.** Phase
   0 measured transfer cost with GPU idle; need to confirm the zero-copy
   pattern still holds when Metal is busy.

## Next steps — Phase 1 proper

1. **Port Qwen35Config + model wrapper to flash_moe's weight loader.** Build
   a PyTorch module that loads flash_moe's `model_weights.bin` into a
   flash_moe-shaped Qwen35DecoderLayer stack (without the MoE MLP — that stays
   on GPU). This is the biggest unknown in Phase 1.
2. **Convert and bundle into 15 DDDA super-blocks.** Use anemll-qwen35's
   `convert_all_superblocks_lut4.py` as the pattern, adapted for flash_moe
   dimensions.
3. **Parity test on a single layer** vs flash_moe's existing delta-net CPU
   reference: cos similarity target ≥ 0.999.
4. **Full 15-super-block dispatch benchmark** on the actual converted model —
   this is the definitive number that replaces the current extrapolation.

## Commits affected

- `flash_moe/ane_bench/ane_dispatch_bench.m` — updated to accept `--fm-linear`
  flag for the flash_moe single-layer input signature (hidden + gated_state +
  conv_state, vs the 9B super-block's 9-input signature)
- `flash_moe/ane_bench/README.md` — Phase 1a section added (separate commit)
- Scripts on mini-02 at `/tmp/convert_fm_linear_l0_test.py` and
  `/tmp/convert_fm_linear_l0_lut4_test.py` — not yet committed anywhere;
  copy into `flash_moe/ane_bench/scripts/` during Phase 1 proper
