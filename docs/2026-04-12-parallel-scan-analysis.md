# GatedDeltaNet parallel scan analysis — infeasible, pivot to parallel projections

**Date:** 2026-04-12
**Finding:** Parallel scan for the delta-rule recurrence is theoretically valid
(the recurrence IS affine) but computationally infeasible on M4 Pro because
the scan composition step requires [128×128] matrix multiplications — 65× more
FLOPs than the sequential recurrence. The pragmatic alternative is to
parallelize the PROJECTIONS (90%+ of per-layer compute) and keep the
recurrence sequential.

---

## The recurrence

GatedDeltaNet's per-token state update (from `fused_layer_forward` lines
2898-2928):

```
S_decay = g * S                                    # [Dv, Dk] = [128, 128]
kv_mem = S_decay @ k                               # [Dv] — predict v from state
delta = (v - kv_mem) * beta                         # [Dv] — error signal
S_new = S_decay + outer(delta, k)                   # [Dv, Dk] — update
y = S_new @ q                                       # [Dv] — output
```

Expanding: `S_new = S @ (g·(I - β·k·k^T)) + β·v⊗k = S @ A + B`.

This is an affine map `f(S) = S @ A + B` where:
- A = `g · (I - β · k k^T)` — a [Dk, Dk] rank-1-updated identity, depends on (g, β, k)
- B = `β · v ⊗ k` — a [Dv, Dk] matrix, depends on (v, k, β)

Neither A nor B depends on S. The recurrence is AFFINE and parallel scan
applies in theory.

## Why parallel scan is infeasible

The parallel scan composition step for two affine maps (A₁, B₁) ∘ (A₂, B₂)
= (A₁ @ A₂, B₁ @ A₂ + B₂) requires a [128 × 128] matrix multiplication.

For T=4 with log₂(4)=2 scan rounds:
- 6 matrix multiplications of [128 × 128] per v-head
- 64 v-heads → 384 matmuls
- Each matmul: 128³ = 2.1M FLOPs → total 768M FLOPs

Compare to sequential recurrence:
- 4 tokens × 64 v-heads × ~32K FLOPs per token per head = 8.2M FLOPs total

**Parallel scan uses 94× more FLOPs than sequential.** On M4 Pro's ~3.6 TFLOPS
fp32 peak, the parallel scan would take ~0.3 ms vs sequential's ~0.02 ms
(at realistic GPU utilization). Net regression.

Even with rank-1 structural optimizations (exploiting the fact that A is a
rank-1-updated identity), the composition can be done in O(T × Dk²) per
v-head but this matches the sequential recurrence's O(T × Dv × Dk) complexity.
The parallel scan reduces STEPS from T to log(T) but each step has similar
FLOPs to the sequential approach.

**Contrast with Mamba/S4:** those models have DIAGONAL A matrices, so scan
composition is element-wise O(D) instead of O(D³). GatedDeltaNet's rank-1
structure from the outer product makes composition quadratic in Dk.

## The pragmatic alternative: parallel projections

Per-layer breakdown of the GPU CMD1 for linear-attention:
```
qkv projection:     4096 → 12288    ~40M FLOPs + 6.3MB weight stream
z projection:        4096 → 8192     ~33M FLOPs + 4.2MB weight stream
b projection:        4096 → 64       ~0.5M FLOPs
a projection:        4096 → 64       ~0.5M FLOPs
conv1d_step:         12288 × 4       ~0.05M FLOPs
rms_norm_qk:         2 × 16 heads    ~0.01M FLOPs
compute_decay_beta:  64 heads        ~0.001M FLOPs
delta_net_step:      64 × 128 × 128  ~2M FLOPs + 1MB state read/write
gated_rms_norm:      64 × 128        ~0.5M FLOPs
```

**Projections are ~90% of CMD1 FLOPs.** The recurrence is ~2.6%. By
parallelizing the projections across T tokens on separate Metal command
queues (same technique validated in Phase 0 at ~1.5x for T=4), we get:

```
Serial per-token:        0.85 ms (all of CMD1)
Parallel projections T=4: 0.85 × 0.9 / 1.5 + 0.85 × 0.1 = 0.60 ms effective
Speedup:                 1.43× per linear layer
```

Across 45 linear layers: saves ~11.5 ms/token = ~10% wall-clock. Combined
with full-attention multi-slot (~1.3 ms) + expert I/O amortization (~6 ms):
**total ~15-16% at T=4, ~20% at T=8 on 1k+ context.**

This requires NO new Metal kernels — just per-slot dispatch of existing
`gpu_encode_batch_matvec` + sequential `delta_net_step`.

## Cross-references

- The affine formulation S_new = S @ A + B is from rewriting lines 2909-2928
  of infer.m (the CPU delta-net reference path)
- The GPU delta-net kernel is `delta_net_step` in shaders.metal
- Phase 0 multi-slot benchmark: commit `d62e246`
- Optimization roadmap: `docs/2026-04-11-optimization-roadmap.md`
