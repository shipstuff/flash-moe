# ANE bench — Phase 0 decision gate for the linear-attention offload

This directory holds the measurement harnesses for the **Phase 0** decision
gate of the ANE offload strategy scoped in
`docs/2026-04-11-ane-offload-scoping.md`.

## Goal

Before committing to the full ANE port (9-15 focused days across 5 phases),
measure the three critical unknowns that could kill the strategy:

1. **Swift / Obj-C CoreML per-prediction dispatch overhead** — if `MLModel
   predictionFromFeatures:` costs several ms of overhead above the raw
   kernel time, the 45-per-token call count makes the strategy infeasible.
2. **MTLBuffer ↔ MLMultiArray transfer cost** — hidden state moves between
   Metal and CoreML every layer transition; if this isn't zero-copy, we pay
   45× per token.
3. **GPU + ANE simultaneous workload contention** — flash_moe keeps the GPU
   busy with MoE expert dispatch; if ANE and GPU contend on shared DRAM
   bandwidth, the parallel gain evaporates.

## Harnesses

### `ane_dispatch_bench.m`

Loads one pre-compiled LUT4 super-block (from `anemll-qwen35`) and runs 200
back-to-back predictions with warmup, computing min / p50 / p90 / p99 / max
latency. Compares against the `anemll-qwen35` reference of 9.28 ms pure-ANE
time for super-block 0.

### `ane_transfer_bench.m`

Measures four MLMultiArray/MTLBuffer interaction patterns 10,000 times each:
- Zero-copy wrap via `initWithDataPointer:` (the intended production path)
- Alloc fresh + memcpy (naive fallback)
- Readback memcpy (MLMultiArray → MTLBuffer)
- Alloc only (isolates the alloc cost)

## Build

```bash
clang -O2 -fobjc-arc -framework Foundation -framework CoreML \
      ane_dispatch_bench.m -o ane_dispatch_bench

clang -O2 -fobjc-arc -framework Foundation -framework CoreML -framework Metal \
      ane_transfer_bench.m -o ane_transfer_bench
```

## Model bundle

`superblock0.mlmodelc/` is copied from
`carl@192.168.0.62:~/models/anemll-qwen3.5-9b/qwen3_5_superblock0_lut4.mlmodelc/`.
It is gitignored (413 MB). To reproduce:

```bash
rsync -a carl@192.168.0.62:/Users/carl/models/anemll-qwen3.5-9b/qwen3_5_superblock0_lut4.mlmodelc/ \
      ./superblock0.mlmodelc/
```

## Results — 2026-04-11, M4 Pro mini-01

### Dispatch overhead (ane_dispatch_bench, solo)

```
Model load: 5273.1 ms (one-shot)
Warmup converges at call 3 to ~6.7 ms.

200 back-to-back predictions:
  min:  6.567 ms
  p50:  6.642 ms
  p90:  6.861 ms
  p99:  7.444 ms
  max:  7.586 ms
  mean: 6.706 ms
  throughput: 149.1 predictions/sec
```

**Verdict: VIABLE.** p50 is *faster* than the `anemll-qwen35` reference of
9.28 ms (their measurement included Python `coremltools` overhead; raw Obj-C
is tighter). No thermal throttling visible across 200 tight calls.

### Transfer cost (ane_transfer_bench)

```
Strategy A — zero-copy wrap (initWithDataPointer):
  min=0.000 us  p50=0.000 us  p99=1.073 us  mean=0.491 us

Strategy B — alloc + memcpy:
  min=0.000 us  p50=0.954 us  p99=4.053 us  mean=1.050 us

Strategy C — readback memcpy:
  min=0.000 us  p50=0.000 us  p99=1.073 us  mean=0.131 us

Strategy D — alloc only:
  min=0.000 us  p50=0.000 us  p99=1.073 us  mean=0.283 us
```

**Verdict: free.** All paths are sub-5-microseconds. Per-token overhead
across 45 layer transitions = ~22 us (~0.02 ms), negligible versus the
~33 ms/token expected win.

### GPU + ANE concurrent load

Ran `ane_dispatch_bench` concurrently with `TQ_KV=1 ./infer --tokens 128`
(flash_moe normal warm-cache inference).

```
ANE under concurrent GPU load:
  p50: 6.902 ms  (+4% vs solo)
  p90: 7.207 ms  (+5%)
  p99: 8.516 ms  (+14%)
  max: 10.002 ms (+32% tail)

GPU inference alongside ANE bench:
  Generation: 6.05 tok/s (TQ_KV=1, 128 tok)
  Historical baseline at same config: 5.65-5.91 tok/s
  → GPU is unchanged / slightly faster
```

**Verdict: no meaningful contention** on Apple M4 Pro unified memory. ANE and
GPU can run in parallel at ~95% of their solo rates for the hot path. Tail
latency is worse but still well within budget (10 ms worst case = 1.8 ms
above reference).

## Combined Phase 0 conclusion

**All three decision gates passed. ANE offload proceeds to Phase 1 (weight
conversion).**

Revised per-token budget estimate for the full port:
```
ANE path (15 super-blocks × 6.64 ms p50 warm + ~5% concurrent penalty):
  = 15 × ~6.97 ms  = ~104 ms/token
GPU path (60 MoE dispatches in parallel):
  = ~78 ms/token
Transitions (45 × ~0.5 us):
  = ~0.02 ms/token negligible
Wall clock:
  = max(104, 78) + 0.02  = ~104 ms/token
Vs current warm-cache baseline (~150 ms/token):
  = ~31% wall-clock speedup
```

Conservative estimate because:
- Super-block 0 is the first super-block and may be faster/slower than
  average; need to measure the full 15-super-block sum once we have them
- flash_moe's 2× value-head count may increase per-super-block time (needs
  actual shape measurement during Phase 1)
- Sustained thermal throttling under minutes-long inference not measured
  (Phase 2 validation)

Non-conservative (optimistic) side:
- Mini-01 is slightly faster than mini-02 where the reference was measured
- We use Obj-C directly (no Python overhead)
- super-block 0 measurements assume full 4-layer bundling; any simpler
  per-layer .mlpackage split adds overhead only if we need to split

See `../docs/2026-04-11-ane-offload-scoping.md` for the rest of the port plan.
