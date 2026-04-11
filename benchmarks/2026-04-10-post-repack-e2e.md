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

## TurboQuant (TQ_KV=1) — currently force-fallback, not fixed

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
