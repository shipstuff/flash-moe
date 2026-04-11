# ANE offload scoping — Qwen3.5-397B-A17B MoE

**Date:** 2026-04-11
**Status:** Scoping only — no code yet. Tagged for a later focused multi-session effort.
**Related project:** `carl@192.168.0.62:~/projects/anemll-qwen35/` (working Qwen3.5-9B dense port to ANE at ~11.4 tok/s)

---

## Why this document exists

Today's session established that **the per-token prefill fast path is ~97% GPU-saturated** on warm cache (derived from CLAUDE.md timing breakdown: `cmd1_wait + cmd2_wait + expert_io = 2.62/2.70 ms per layer`). Two batch-prefill refactor approaches (multi-buffered deferred state, MoE cross-token decoupling) were investigated by parallel sub-agents and both concluded that warm-cache speedups are blocked by this saturation ceiling — there's no GPU idle time to fill with additional parallel work.

**ANE is the only unused compute accelerator on the machine.** On an M4 Pro running flash_moe, the Apple Neural Engine (16 compute cores) is 0% utilized. If we can move some of flash_moe's per-layer work onto ANE, we add real capacity rather than fighting for GPU gaps. This doc scopes that port.

---

## What already exists — `anemll-qwen35` on mini-02

A parallel effort has ported **Qwen3.5-9B (dense)** to Apple Neural Engine. As of 2026-04-11 afternoon (their session 2):

- **End-to-end generation working** at ~11.4 tok/s wall-clock with a full KV cache
- 32 decoder layers in the pattern `DDDA ×8` (24 GatedDeltaNet linear attention + 8 Qwen3Next-style gated full attention)
- LUT4 palettized super-blocks, LUT6 split lm_head
- 84.4 ms/token pure ANE time (Python `coremltools` overhead adds ~10-20 ms)
- 4/5 top-1 logit agreement vs mlx-lm reference on real prompts (the miss is an fp16 tied-logit tiebreak)
- 0 CPU fallbacks, 0 graph interruptions in the compiled ANE super-blocks

**Crucially:** the same PyTorch reference and converter pipeline handles both of flash_moe's layer types cleanly with pure ANE ops:
- `Qwen35GatedDeltaNet` — depthwise Conv1d kernel=4, recurrent state `[B, Hv, Dv, Dk]`, decay/beta/alpha projections, RMS-norm + scale q/k, per-timestep recurrent update. Pure-ops (`mul`/`sum`/`where`/`silu`/`softplus`/`exp`), no custom kernels.
- `Qwen35FullAttention` (Qwen3Next-style) — q_proj output split into `(queries, gate)`, partial rotary (first 64 of 256 dims), `output = o_proj(attn_output * sigmoid(gate))`. Partial rotary is new to ANEMLL but is ANE-legal via static slice + concat.

**Speed reference (same M4 Pro mini):**

| Variant | Backend | tok/s | Note |
|---|---|---|---|
| Qwen3.5-9B MLX 4bit | GPU (Metal) | 44.72 | baseline |
| Qwen3.5-9B MLX 8bit | GPU (Metal) | 26.25 | |
| Qwen3.5-9B bf16 | GPU (Metal) | 13.92 | |
| Qwen3.5-9B LUT4 | ANE | 11.39 | full KV |

ANE is ~3.5× slower than GPU on the same dense model. The project frames ANE's value as **power efficiency and offloading GPU for other workloads**, not raw throughput.

### Reference files on mini-02

- `~/projects/anemll-qwen35/STATUS.md` — current state + benchmarks
- `~/projects/anemll-qwen35/PORT_PLAN.md` — architecture deep-dive, sanity checks, verified weight shapes
- `~/projects/anemll-qwen35/CLAUDE.md` — gotchas (norm-weight +1.0 shift, `cache=None` forces CPU, `view_as` unsupported, etc.)
- `~/projects/anemll-qwen35/model/qwen3_5_model.py` — PyTorch scaffold (~8.954B params)
- `~/projects/anemll-qwen35/scripts/convert_all_superblocks_lut4.py` — super-block converter
- `~/projects/anemll-qwen35/scripts/chain_runner_kv.py` — Python chain runner with KV cache
- `~/benchmarks/qwen35-9b-ane-chain.json` — benchmark results

---

## Architecture alignment — flash_moe vs anemll-qwen35

flash_moe runs Qwen3.5-397B-A17B (**MoE**). anemll-qwen35 runs Qwen3.5-9B (**dense**). The transformer backbone is strikingly similar:

| | anemll-qwen35 (9B dense) | flash_moe (397B MoE) | aligned? |
|---|---|---|---|
| `hidden_size` | 4096 | 4096 | ✅ exact |
| `head_dim` | 256 | 256 | ✅ exact |
| Layer pattern | `DDDA ×8` (32 layers) | `DDDA ×15` (60 layers) | ✅ structural |
| `FULL_ATTN_INTERVAL` | 4 | 4 | ✅ exact |
| `partial_rotary_factor` | 0.25 (→ 64 rotary dims) | 0.25 (→ 64 rotary dims) | ✅ exact |
| `rope_theta` | 10,000,000 | (to verify) | likely ✅ |
| Full-attn heads | 16 Q / 4 KV | 32 Q / 2 KV | ⚠️ different |
| `linear_num_value_heads` | 32 | 64 | ⚠️ 2× in flash_moe |
| `linear_num_key_heads` | 16 | 16 | ✅ exact |
| `linear_key_head_dim` | 128 | 128 | ✅ exact |
| `linear_value_head_dim` | 128 | 128 | ✅ exact |
| `conv_dim` (conv1d) | 8192 (`2*Hk*Dk + Hv*Dv`) | 10240 (`2*2048 + 8192`) | ⚠️ larger |
| `CONV_KERNEL_SIZE` | 4 | 4 | ✅ exact |
| MLP | dense FFN (`gate/up/down`) | 512 MoE experts, K=4, plus shared expert | ❌ fundamental |

**Takeaway:** the two layer types flash_moe and anemll-qwen35 share (GatedDeltaNet + Qwen3NextAttention) are nearly identical. The weight converter and the ANE-compiled super-blocks **could** be reused with dimension substitutions — we wouldn't be starting from zero. Full attention has a different num-head ratio (32/2 vs 16/4) but the kernel structure is the same. GatedDeltaNet's `Hv` doubled (32 → 64) and recurrent state is correspondingly bigger.

**The MoE MLP is the immovable mismatch.** flash_moe stores 512 experts × 60 layers = 30,720 expert tiles on SSD in a custom 4-bit packed format, streamed into GPU buffers via `pread`. Converting and streaming these through ANE is out of scope — ANE needs pre-compiled `.mlpackage` bundles with fixed weights. Any offload must leave MoE on the GPU path.

---

## Why ANE offload matters *specifically* for flash_moe

The key insight from today's saturation analysis:

```
Warm-cache per-layer breakdown (CLAUDE.md 128-tok gen, TQ disabled):
    expert_io      1.337 ms  (49.5%)  [CPU pread + GPU cmd3 running in parallel]
    cmd1_wait      0.858 ms  (31.8%)  [CPU blocked on GPU attention kernels]
    cmd2_wait      0.426 ms  (15.8%)  [CPU blocked on GPU residual/norm kernels]
    CPU active     0.082 ms  ( 3.0%)
    total_layer    2.703 ms

GPU busy   ≈ 2.62 ms / 2.70 ms  =  ~97% utilization
```

Per-token prefill runs at ~150 ms/token which matches this per-layer rate × 60 layers. The GPU is essentially saturated.

**Implications:**
- Adding more GPU-side parallelism (batch prefill) cannot beat 150 ms/token. Both batch prefill agents confirmed this.
- CPU optimization is capped at 0.082 ms × 60 = ~5 ms/token = 3% win.
- **The only place to add compute parallelism is ANE.** If we offload the linear-attention layers' matmul + delta-net recurrence to ANE, the GPU spends its cycles only on full-attention (15 layers) + MoE (60 layers of `expert_io` and expert dispatch).

---

## Realistic offload strategies

### Strategy 1: Linear attention (45/60 layers) on ANE — **recommended**

Move all 45 GatedDeltaNet layers to ANE. GPU keeps full attention (15 layers) and all 60 layers of MoE expert dispatch.

**Per-token budget estimate:**
```
ANE path:
    45 linear layers × ~2.5 ms/layer  =  112 ms  (extrapolated from anemll-qwen35's
                                                  ~10 ms per DDDA super-block = ~2.5 ms
                                                  per linear layer; flash_moe's 2×
                                                  value heads may bump this to ~3-3.5 ms
                                                  so call it 135-160 ms conservatively)
GPU path (runs in parallel with ANE):
    15 full attention layers  ≈ 15 × 1.3 ms (cmd1+cmd2)  ≈ 20 ms
    60 MoE dispatches         ≈ 60 × 1.3 ms (expert_io + cmd3)  ≈ 80 ms
    ...and all of this work is dependent on the previous linear
    layer's hidden state, so GPU stalls waiting for ANE between
    each D→A and A→D transition.

Realistic wall clock = max(ANE, GPU-only-work) + transition overhead
```

**The big unknown: transition overhead.** Per-layer, hidden state (4096 floats = 16 KB) must move:
- GPU Metal shared-storage buffer → CPU memory (free on Apple Silicon unified memory, but requires command buffer commit+wait for correctness)
- CPU memory → CoreML `MLTensor` input (copy)
- CoreML prediction (the 2.5-3.5 ms ANE compute)
- CoreML output `MLTensor` → CPU (copy)
- CPU memory → next layer's Metal buffer (free, unified memory)

**If transition overhead is <1 ms, wall clock ≈ max(135-160, 100) = 135-160 ms/token**. That's roughly **break-even or ~10% faster** than current 150 ms/token.

**If transition overhead is ~2-3 ms per transition** (which is realistic for Python `coremltools`; Swift `MLModel` is faster but unknown), wall clock = 135-160 + 45 × 2-3 ms = 225-295 ms/token. **Significantly worse.**

The whole strategy hinges on the CoreML per-prediction overhead from Swift/Obj-C. The anemll-qwen35 project's Python chain runner reports ~10-20 ms of Python+coremltools overhead per full 32-layer pass — that's ~300 μs per super-block, which is promising IF the same scales to Swift. Unknown without measurement.

**This strategy's decision gate: measure CoreML dispatch overhead first.**

### Strategy 2: Full attention (15/60 layers) on ANE — **not recommended**

Move the 15 full-attention layers to ANE. GPU keeps linear attention + MoE.

Unlikely to win because:
- ANE full attention compute is roughly equivalent to GPU full attention (both ~1-1.5 ms/layer)
- Transition overhead would likely exceed the compute saving
- Full attention is only 22% of layer time; the potential win is small anyway

Skip unless Strategy 1 is infeasible.

### Strategy 3: Per-token lm_head on ANE — **tiny win, low risk**

anemll-qwen35 runs their split-16 LUT6 lm_head on ANE at 7.38 ms per token. flash_moe currently runs lm_head on GPU (inside the final commit). Moving it to ANE would free up GPU time for the next token's prefill but saves only ~1-2 ms/token since lm_head is a tiny fraction of total layer time. Not worth the integration work unless bundled with Strategy 1.

### Strategy 4: KV cache attention compute on ANE — **not viable**

The ANE project has KV-aware super-blocks where the K/V cache is passed as explicit input/output tensors. This could in principle let ANE run the attention scores computation.

Problem: flash_moe's KV cache is either 250+ MB float (full) or 33 MB compressed (TurboQuant) and is actively updated every token. Shoveling this through CoreML every layer is prohibitive — tensor copy overhead would dwarf the compute saving.

Skip.

### Strategy 5: MoE experts on ANE — **not viable**

Discussed earlier. flash_moe's 512 experts × 60 layers are streamed on-demand from SSD at 7 MB each. ANE has no equivalent streaming model; it requires fixed `.mlpackage` bundles compiled ahead of time. We'd have to:
- Pre-compile all 30,720 expert tiles into CoreML models (infeasible — hundreds of GB of compiled artifacts)
- Or compile *all active experts for a request* up-front (infeasible — routing decisions are per-token)

Skip.

---

## Integration cost — Strategy 1 breakdown

Assume we commit to linear-attention-on-ANE. The work is:

### Phase 0: Measurement (1-2 days) — **decision gate**

Before anything else, measure:
1. **Swift/Obj-C CoreML prediction overhead** at per-layer granularity. Build a minimal Obj-C harness that loads one of anemll-qwen35's `_lut4.mlmodelc` super-blocks and calls `[model predictionFromFeatures:]` in a tight loop. Measure p50, p90 of dispatch latency. If p90 > 1 ms, Strategy 1 is dead on arrival.
2. **Data transfer cost** from a `MTLBuffer` (shared storage) to an `MLMultiArray` and back. Metal+CoreML on Apple Silicon *should* be zero-copy via `MLMultiArray dataFromMTLBuffer` or similar, but I'm not sure this is supported. If not, we're doing 45×2 memcpy per token.
3. **ANE thermal throttling** under sustained load. Run a 30-second anemll-qwen35 generation loop and watch powermetrics. If ANE drops to 50% after 10 seconds, the 2.5 ms/layer number is misleading.

**Go/no-go on Phase 0**. If overhead is too high, pivot to reduced-precision attention or kernel fusion instead.

### Phase 1: Weight conversion (2-3 days)

1. Port `model/qwen3_5_model.py` from anemll-qwen35 to flash_moe's dimensions (64 Hv instead of 32, 32/2 attn heads, 60 layers, MoE placeholder). Load from `model_weights.bin` instead of safetensors.
2. Write a "linear-layers-only" exporter that dumps just the 45 GatedDeltaNet layers as a stack of `.mlpackage` super-blocks (or one per layer, TBD).
3. Run anemll-qwen35's converter chain on the exported PyTorch model. Adapt `convert_all_superblocks_lut4.py` for flash_moe's layer count and dim.
4. Parity test: run the 45 ANE linear layers against flash_moe's existing delta-net CPU reference on the same hidden inputs. Target: cos similarity ≥ 0.999.

### Phase 2: Swift/Obj-C CoreML bridge (3-5 days)

1. Write a small Objective-C class `FMANELinearStack` that:
   - Loads 45 compiled `.mlmodelc` bundles at startup (or one large one if we bundle them)
   - Exposes a C function `fmane_linear_run(int layer_idx, float *hidden_in, float *hidden_out, void *delta_state_in_out, void *conv_state_in_out)`
   - Uses `MLPredictionOptions` with `usesCPUOnly = NO` and prefers `.cpuAndNeuralEngine` compute unit
2. Wire into `infer.m` at the linear-attn dispatch site (currently `cpu_conv1d_step` + `gpu_delta_net_step` paths) behind an env var `FMANE=1`
3. Manage the delta-net state tensor lifecycle (currently `linear_states[NUM_LINEAR_LAYERS]`) as CoreML input/output pairs

### Phase 3: Integration + pipelining (2-3 days)

1. Make CoreML dispatch asynchronous so GPU full-attention + MoE work can overlap with ANE linear compute. This is the whole point.
2. Wire the existing per-token prefill loop to pass `hidden` through the alternating pattern: ANE → GPU full-attn → ANE → ... with dependency management
3. Handle T=1 fast path correctly
4. Ensure TQ KV-cache mode still works (TQ only touches full attention layers, which stay on GPU — no interaction with ANE path)

### Phase 4: Validation (1-2 days)

1. Per-layer hash diff of baseline vs FMANE=1 on a 138-tok prompt (same method as TQ debugging)
2. Token-sequence identity across multiple prompts
3. Power measurement: powermetrics sampling during 60-second generation on FMANE=1 vs baseline. Expected: ANE rail spikes, GPU rail drops.
4. Wall-clock benchmark: 1k, 2k, 3k context prefill + generation. Target: ≥10% wall-clock improvement OR equivalent throughput at ≥15% lower power.

**Total estimate: 9-15 focused days** for a working Phase-0-through-Phase-4 delivery. Similar scale to the TQ KV-cache effort. Expect ~8-12 bugs found the hard way (anemll-qwen35's CLAUDE.md documents several ANE-specific traps).

---

## Specific unknowns that block commitment

These are the questions that must be answered before investing in the port:

1. **Swift `MLModel predictionFromFeatures:` p50 latency** for a single lut4 super-block on the M4 Pro ANE. Assumed "a few hundred μs" but not measured. **This is the single most important number.**
2. **`MTLBuffer` ↔ `MLMultiArray` zero-copy** availability on macOS 15+. If unsupported, we pay 2 × memcpy × 45 transitions × 60 layers.
3. **ANE sustained throughput under thermal pressure**. The 2.5 ms/layer number is from warm idle measurements, not sustained load.
4. **Whether flash_moe's `linear_num_value_heads=64` doubles ANE runtime** proportionally or stays at ~2.5 ms. GatedDeltaNet's recurrent state is `[B, 64, 128, 128]` here vs `[B, 32, 128, 128]` in anemll-qwen35 — 2× memory bandwidth, but compute may be less linear.
5. **GPU + ANE simultaneous workloads on M4 Pro**. Are there shared-memory-bandwidth constraints that make them interfere? Our 97% GPU saturation already pushes memory bandwidth; adding ANE on top may cause contention.

---

## Parallel concerns

Because anemll-qwen35 has an active project with its own sub-agents, **any work here should coordinate with that repo**, either:
- As a dependency: flash_moe reuses their PyTorch reference + converter pipeline; contribute dim-parameterization back upstream
- As a fork: clone `~/projects/anemll-qwen35/model/qwen3_5_model.py` + converter scripts into `flash_moe/ane/` and diverge

Fork is simpler short-term; dependency is better long-term for maintenance. TBD when the port starts.

---

## When to revisit

Pick this back up when one of the following is true:
- A smaller-scope GPU-side optimization (reduced-precision attention, kernel fusion) has landed and we want to chase the next 10%
- Power consumption becomes a constraint (e.g. running flash_moe on battery, or multi-tenant with UI workloads)
- anemll-qwen35 publishes a reusable Swift CoreML bridge for their project that we can adapt
- The Phase 0 measurements are done opportunistically and look favorable

## Cross-references

- Sibling scoping doc: `docs/2026-04-11-batch-prefill-scoping.md` (the batch prefill investigation that established the GPU saturation constraint)
- Top-level plan: `../CLAUDE.md` research questions section (expert_io, cmd1_wait, mixed-bit experts)
- Project on mini-02: `carl@192.168.0.62:~/projects/anemll-qwen35/`
