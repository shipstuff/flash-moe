# Post-repack end-to-end benchmarks

**Date:** 2026-04-10 (overnight session)
**Hardware:** Apple M4 Pro (skynet-m4-mini-01)
**Model:** Qwen3.5-397B-A17B-4bit (60 layers, 512 experts, K=4 active)
**Commit:** 8f76d02 (+ repack_experts_v2.py + rebuilt packed_experts)
**Success criterion (user):** coherent output, no token-loop. Speed not required.

---

## The bug that was fixed

The prior `packed_experts/layer_XX.bin` files were generated using
`model/expert_index.json` (dated 2026-04-10 10:34) whose `abs_offset` fields
stored `data_offsets[0]` directly instead of `(8 + hdr_len + data_offsets[0])`.
The off-by-47245 error (exactly the safetensors file header size) caused every
expert block in every layer to be populated with bytes from neighboring tensors.
The Apr 10 10:35 rebuild faithfully copied these wrong bytes. Result: every
`./infer` run NaNed out in the forward pass and emitted token 0 ("!") forever.

**Fix:** `repack_experts_v2.py` parses each safetensors file's own header and
computes absolute offsets per spec (`data_start + data_offsets[0]`). All 60
layer files were rebuilt in 307s (202.5 GB written, ~0.68 GB/s avg). The
internal spot-check compares bytes for experts `{0, 1, 255, 511}` across all
9 components against `os.pread()` from the source tensor — every layer passed.

---

## Sanity test

```
Prompt: "The capital of France is"
Output: "Paris. <|im_end|>"
Top-5 logits: ĠParis=16.41 (winner), Ġ(=14.12, Ġa=14.29, Ġ______=14.26, ĊĊ=14.10
hidden rms after final_norm: 1.7411
logits rms:                  2.8243
```

Real finite values. No NaN. Correct answer. EOS emitted after "Paris.".

## Baseline (no cache, OS page cache only)

```
Prompt:  "Once upon a time in a small village, there lived an old clockmaker who"
Tokens:  128 requested, 104 generated (model self-terminated with <|im_end|>)
TTFT:    4311 ms (5 prompt tokens)
Gen:     6.01 tok/s
Total:   21.5 s
```

Output is coherent: the model produced a fully formed short story ending with
"...completing a full circle in exactly 60 seconds, but it did so in a series
of jerky movements, pausing for a brief moment at the 12 o'clock position
before continuing its journey around the clock face."

### Per-layer timing

```
deferred_wait:   0.000
deferred_cpu:    0.001
input_norm:      0.000
cmd1_submit:     0.019
cmd1_wait:       0.858  ← attention
cpu_attn:        0.008
cmd2_encode:     0.019
cmd2_wait:       0.426  ← residual/norm
routing_cpu:     0.002
expert_io:       1.337  ← expert SSD load (dominant)
cmd3_encode:     0.032
tq_kernel_time:  0.000  (disabled)
total_layer:     2.703
```

Profile is consistent with the "real baseline" documented in top-level CLAUDE.md
(5.86 tok/s, expert_io dominant at 41-50% of layer time).

---

## malloc cache path — `lz4_comp_buf` uninitialized stack memory bug

Second bug discovered and fixed. Every site that stack-allocated
`InferPreadTask tasks[MAX_K]` (7 call sites across parallel_pread_experts,
parallel_pread_experts_into, infer_prefetch worker, malloc-cache miss,
Metal-LRU-cache miss, and the pred sync-pread path) never zeroed the
`lz4_comp_buf`/`lz4_comp_size` fields added later for optional LZ4 reads.
`io_pool_worker` checks `if (t->lz4_comp_buf && t->lz4_comp_size > 0)` and
with garbage stack values would take the LZ4 decompression path — pread into
a wild pointer, return -1, and the cache never populated. This explains the
historical `WARNING: expert X pread: -1/7077888` messages **and** the bogus
"14.5–15.5 tok/s with malloc cache" benchmarks in `results.tsv` that were
actually the GPU computing on uninitialized memory (0% hit rate while the
cache "worked" on garbage bytes).

Fix: `memset(tasks, 0, sizeof(tasks))` at each stack allocation. After fix:

```
malloc-cache-64,  128tok, K=4: 6.13 tok/s, 0% hit rate (thrashing — 64 slots
                                                         vs 240 active/token,
                                                         coherent output)
malloc-cache-512, 96tok,  K=4: 6.07 tok/s, 32.2% hit rate (8569/26640),
                                coherent output
no malloc cache,  128tok, K=4: 6.01 tok/s (baseline, see above)
```

The malloc cache does not accelerate this workload: OS page cache already
covers the hot set. This confirms the historical "Trust the OS" finding from
the top-level CLAUDE.md. The eviction fix in `8f76d02` is also confirmed
working on hardware (no silent mismatches at the observed hit rate).

---

## TurboQuant (TQ_KV=1) — FIXED 2026-04-11

**Status:** TurboQuant now produces fully coherent output end-to-end at all
tested context lengths. Force-fallback gate replaced by `TQ_KV_SAFE=1` opt-in
for A/B testing.

### The bug that mattered most

`tq_pack_update` writes the per-position L2 norms via the `k_norms_out` /
`v_norms_out` kernel parameters at buffer indices 4 / 6. The C dispatch was
binding the wrong buffers there:

```objc
[tq_enc setBuffer:g_metal->buf_tq_encode_k_norms offset:0 atIndex:4];  // wrong
[tq_enc setBuffer:g_metal->buf_tq_encode_v_norms offset:0 atIndex:6];  // wrong
```

`buf_tq_encode_*_norms` are tiny per-step scratch buffers used by an earlier,
abandoned design that precomputed norms on the CPU. The persistent
per-position cache (`buf_tq_k_norms[fa_idx]` / `buf_tq_v_norms[fa_idx]`),
which `tq_fused_attention` reads at runtime, was **never written to**. It
stayed all zeros forever. Every TQ-decoded K/V value was therefore
`centroid * 0 = 0`, the attention contribution from the full-attention layers
vanished, and the model produced bad output once `gpu_attn_fuse` engaged.

The fix is one line per buffer:

```objc
[tq_enc setBuffer:g_metal->buf_tq_k_norms[fa_idx] offset:0 atIndex:4];
[tq_enc setBuffer:g_metal->buf_tq_v_norms[fa_idx] offset:0 atIndex:6];
```

### How we found it

Per-layer hidden-state hashes (gated on `TQ_KV_DBG=1`) for `baseline` vs
`TQ_KV_FORCE=1` showed first divergence at **gen=16, layer 3** — exactly
when `kv_len=32` made `gpu_attn_fuse` activate. The CPU KV cache hashes
matched bit-for-bit at that step, ruling out memory corruption. So the bug
had to be in the GPU attention path itself.

A diagnostic Phase-4 write that encoded `global_max`, `global_sum`, `inv_sum`,
`tg_max[0]`, `tg_sum[0]`, `tg_acc[0]`, `q_rot[0]`, `T_kv` into successive
`attn_out` slots revealed:

```
attn_out = 0.0  (global_max)
           32.0 (global_sum = T_kv exactly — uniform softmax!)
           0.0312 (inv_sum = 1/32)
           0.0  (tg_max[0])
           4.0  (tg_sum[0] = T_kv/num_sg, exp(0)=1 for every key)
           0.0  (tg_acc[0])
           ...  (q_rot[0])
           32.0 (T_kv)
```

`global_max=0` and `tg_max=0` mean every QK score was 0 → softmax was
uniform → attention output collapsed to a constant. That implied `k_val =
centroid * k_norm * tq_inv_scale = centroid * 0 * 1/16 = 0`. A second peek at
`k_norms[0..3]` and `v_norms[0..3]` confirmed the persistent cache was all
zeros, leading directly to the buffer-binding bug above.

### The full fix list (all needed for TQ to actually work)

1. **`tq_pack_update` buffer binding** — bind persistent norms cache, not
   the per-step scratch (the critical bug).
2. **Gram-Schmidt rotation matrix** — rewrite as standard row Gram-Schmidt
   so `Pi` is actually orthonormal.
3. **`tq_fused_attention` Phase 3** — broadcast `global_max` /
   `global_sum` via threadgroup memory; previously they were computed only
   on `lid==0` but used per-thread (other threads saw `-1e10` / `0`).
4. **`tq_fused_attention` Phase 4** — replace
   `sum += acc_rot[i] * inv_rot[j*N+d]` with a real
   `sum_j(tg_acc[j] * inv_rot[d*N+j])` matmul. The original was a scalar
   times a column sum.
5. **Quantization scale mismatch** — multiply by `sqrt(HEAD_DIM)` at
   encode and divide by `sqrt(HEAD_DIM)` at decode in all 4 TQ kernels so
   the centroids `{-1.5,-0.5,0.5,1.5}` match the per-dim distribution of
   `(Pi·k)/||k||`.
6. **`tq_pack_update` `kv_head=1` norm write** — was guarded by
   `if (tgid == 0)` which only fired for `kv_head=0`; changed to
   `if (lane == 0 && word_idx == 0)` so both heads write their norms.
7. **Double sigmoid gate** — `tq_fused_attention` Phase 5 was applying
   the sigmoid gate, then `cmd_fused`'s `sigmoid_gate` kernel applied it
   again. Removed Phase 5 (Enc A4 always runs).
8. **Pre-rotate Q on CPU** — to make `dot(Q, K) = dot(Pi@Q, Pi@K)` collapse
   correctly under orthonormal `Pi`, the kernel needs Q in the rotated
   basis. Added a one-time per-layer per-head matvec on CPU before the
   `memcpy` into `buf_attn_q` (gated on `g_metal->use_tq_kv`).

### Validated benchmarks (short context, scalar Q rotation)

```
Baseline      128tok, K=4: 5.91 tok/s, 104 coherent tokens (EOS at 104)
TQ_KV=1       128tok, K=4: 5.31 tok/s, 128 coherent tokens, KV cache 33.4 MB
TQ_KV=1       256tok, K=4: 5.15 tok/s, 256 coherent tokens, KV cache 33.4 MB
```

### Q rotation vectorized via Accelerate sgemm (2026-04-11)

The hand-rolled `Pi @ q` matvec inside `fused_layer_forward` was 32 × 256 ×
256 scalar muls × 60 layers = 31 M ops/token (~12 ms/token). Replaced with
a single `cblas_sgemm` per layer that does all 32 heads at once as
`q (NA, HD) × Pi^T (HD, HD) → q_rot (NA, HD)`. `cpu_attn` per layer
0.205 → 0.065 ms (3.2× faster). 128-tok TQ generation 5.31 → 5.65 tok/s.

### Long-context sweep (2026-04-11) — TQ wins where it matters

Same prompt construction repeated to grow the context, 32–64 generated
tokens per run, M4 Pro:

```
Context  Baseline cmd2_wait  TQ cmd2_wait  Baseline tok/s  TQ tok/s  TQ delta
~30 tok       0.434              0.423            5.91         5.65    -4.4%
~1k tok       0.710              0.423            4.98         5.40    +8.4%
~2.4k tok     1.056              0.406            3.98         4.69   +17.8%
```

`cmd2_wait` is the stall time for the command buffer that contains the
fused-layer GPU attention dispatches. Baseline grows roughly linearly with
`kv_len` because the float KV cache is scanned in full every step. TQ
stays flat (0.406–0.423 ms) because the compressed cache is constant size
per token regardless of how many tokens have been seen.

Per-layer breakdown at ~2.4k context, 48-token generation:

```
                  baseline    TQ_KV=1
cmd1_wait         1.032 ms    0.974 ms
cmd2_wait         1.056 ms    0.406 ms   ← attention dispatch flat in TQ
cpu_attn          0.005 ms    0.062 ms   ← Q rotation via sgemm
cmd2_encode       0.018 ms    0.098 ms
tq_kernel_time    0.000 ms    0.079 ms
expert_io         1.943 ms    1.881 ms
total_layer       4.109 ms    3.475 ms
generation        3.98 tok/s  4.69 tok/s
```

Crossover (TQ overhead = TQ savings) sits around **600–800 token
context**. Beyond that TQ wins both on memory footprint (7.5×) **and**
generation speed.

Theoretical scaling: TQ keeps `cmd2_wait ≈ 0.41 ms` indefinitely; baseline
grows ~0.20 ms per +1k tokens. Extrapolating:
- 4k context: baseline ~1.6 ms cmd2_wait → TQ ~+30%
- 8k context: baseline ~2.6 ms → TQ ~+50%

These will be measured directly once the long-prompt prefill (linear in
prompt length, ~8 minutes for 2k tokens) finishes for 4k.



### Per-layer timing comparison (TQ vs baseline at 128 tok)

```
                baseline    TQ_KV=1
expert_io       1.346 ms    1.390 ms
cmd1_wait       0.890 ms    0.913 ms
cpu_attn        0.008 ms    0.205 ms   ← CPU Q rotation overhead
cmd2_encode     0.018 ms    0.089 ms
cmd2_wait       0.434 ms    0.419 ms
tq_kernel_time  0.000 ms    0.070 ms   ← per-token tq_pack_update
total_layer     2.750 ms    3.069 ms
```

TQ adds ~0.32 ms/layer (12% slowdown at 128 tokens) but compresses the KV
cache by **7.5x**:

```
                Float KV     TQ KV
per layer/tok   2,048 B      272 B   (16 u32 x 2 heads + 2 norms)
8k tokens, 15 layers   252 MB   33.4 MB
32k tokens, 15 layers  1.0 GB   134 MB
128k tokens, 15 layers 4.0 GB   536 MB
```

The CPU Q rotation is the easy next optimization (the inner matvec is a
simple HEAD_DIM × HEAD_DIM dense matmul that vectorizes trivially). At long
context (>1k tokens) the smaller KV cache also shrinks `cmd1_wait` since the
attention dispatch reads less memory.

### Quality

For the same prompt the first 32 generated tokens are bit-for-bit identical
to baseline. Around token 32-33 the small 2-bit quantization noise (per-step
~2% L2 error in the attention output) causes argmax to flip to a different
token, and the two stories diverge into different (equally coherent) endings:

- **Baseline:** "...minute hand that moved at a constant speed, but it was
  the second hand that was truly special. It moved in a peculiar way,
  completing a full circle in exactly 60 seconds, but it did so in a series
  of jerky movements, pausing for a brief moment at the 12 o'clock
  position before continuing its journey around the clock face."
- **TQ:** "...minute hand that was 12 cm long and an hour hand that was 9
  cm long. The old clockmaker had a peculiar habit of checking the time on
  his clock at specific intervals, and he noticed something interesting
  about the angle between the hour and minute hands... 'Can anyone tell me
  the exact time when the angle between the hour and minute hands of this
  clock is exactly 90 degrees?'"

Both are grammatical, on-topic, and coherent for the full 256-token run.
This is the expected behavior of a 2-bit lossy KV cache compression.

---

## (historical) TurboQuant force-fallback (now removed)

TQ generation path has at least four bugs:

1. **Gram-Schmidt** in `metal_setup()` was incorrect (mixed row/column indices
   so `buf_tq_rot` wasn't orthonormal). Fixed.
2. **`tq_fused_attention` Phase 3** — `global_max`/`global_sum` computed only
   on `lid==0` but used per-thread; other threads saw initial `-1e10` / `0`.
   Fixed by broadcasting via `threadgroup float` scalars.
3. **`tq_fused_attention` Phase 4** — the "inverse rotation" was
   `sum_j(acc_rot[i] * inv_rot[j,d])` with `acc_rot[i]` outside the j-loop
   — i.e., a scalar times a column sum, not a matrix-vector product. Fixed
   by writing the softmax-normalized output vector into `tg_acc[0..HEAD_DIM)`
   via sid==0 and reading it as `sum_j(tg_acc[j] * inv_rot[j,d])` in Phase 4.
4. **Quantization scale mismatch** — after orthonormal rotation each per-dim
   component of `(Pi·k)/||k||` has std `~1/sqrt(HEAD_DIM) = 1/16`, but the
   encoder compares against thresholds `±1/0/-1` so almost every value lands
   in the middle centroid, and the decoder reconstructs as `centroid·||k||`
   (off by a factor of `sqrt(HEAD_DIM)`). The 2-bit centroids `{-1.5,-0.5,
   +0.5,+1.5}` assume unit-variance inputs. Needs a `sqrt(HEAD_DIM)` scale
   on both sides. **Not yet fixed.**

Because bug (4) is algorithmic and the overnight budget is constrained,
`TQ_KV=1` is now **disabled by default**: the init code prints a one-line
warning and forces `use_tq_kv=0` so the standard float KV cache is used.
Set `TQ_KV_FORCE=1` to run the broken path anyway. Post-fallback benchmark:

```
TQ_KV=1 (fallback) 128tok, K=4: 6.61 tok/s, TTFT 3883 ms, 104 coherent
                                              tokens, tq_kernel_time=0
  expert_io=1.102ms, cmd1_wait=0.832ms, cmd2_wait=0.435ms, total=2.451ms
```

Output coherent: "...it had a unique feature that made it quite famous in
the village. The clock had a minute hand that moved at a constant speed, but
it was the second hand that was truly..." — the model continues the
clockmaker story without drift.

---

## `--predict` temporal expert prediction — functional but net regression

```
--predict, 128tok, K=4, no cache: 2.51 tok/s, coherent story output
  hits=6250 misses=17822 rate=26.0% layers=6018
  cmd1_wait=2.351ms (baseline 0.858)
  expert_io=1.871ms (baseline 1.337)
  total_layer=6.383 (baseline 2.703)
```

Prediction stores the previous token's expert indices and uses them to
issue speculative async_pread on CMD1_wait. With the rebuilt packed files
the prediction mechanism and its hit-rate accounting work correctly, but
the 26% hit rate is insufficient to amortize the extra I/O — 74% of
predicted preads waste SSD bandwidth that would otherwise service cold
demand reads. Net result is a ~58% slowdown. Matches the historical
"-18% / 25% hit rate" finding from top-level CLAUDE.md. Functional,
coherent output, but this optimization should stay off by default.

---
