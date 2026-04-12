# Multi-slot prefill milestone: ~2× speedup with production config

**Date:** 2026-04-12
**Commit:** `d8cb44c` (Phase 2d code), benchmark results below

## Result

454-token prompt, T=4, `--malloc-cache 2581` (production config):

| | Serial | Multi-slot | Speedup |
|---|---|---|---|
| Run 1 rest avg | 205 ms/tok | **109 ms/tok** | **1.88×** |
| Run 2 rest avg | 200 ms/tok | **101 ms/tok** | **1.98×** |
| Average | 202 ms/tok | **105 ms/tok** | **1.93×** |
| First chunk | 2849-3068 ms | **932-939 ms** | **3.1×** |

**Total prefill wall time:** serial ~95 sec → multi-slot ~48 sec for 454 tokens.

## Per-chunk breakdown (warm, T=4)

| Chunk | kv_len range | Multi-slot ms/tok | Serial baseline | Speedup |
|---|---|---|---|---|
| 20 | ~76-79 | 83.1 | ~200 | 2.4× |
| 50 | ~196-199 | 81.7 | ~200 | 2.4× |
| 80 | ~316-319 | 91.3 | ~200 | 2.2× |
| 100 | ~396-399 | 78.9 | ~200 | 2.5× |

## Why it works

The ~2× speedup comes from the convergence of all Phase 1-2d optimizations:

1. **Parallel projection dispatch** across T=4 tokens on per-slot queues
   (CMD1 with 4 parallel command buffers) — ~1.5× on the dominant compute

2. **Deferred expert CMD3** on per-slot queues — restores CMD3→CMD1 overlap
   via Metal serial queue ordering. Each slot's queue runs: CMD3(layer N) →
   CMD1(layer N+1) back-to-back with zero CPU intervention.

3. **Per-slot GPU combine + norm** in CMD3 — eliminates CPU round-trip for
   the hidden→buf_input chain between layers.

4. **Skip-norm when GPU-combined** — avoids redundant norm on
   already-prepared `buf_input_slot[t]` from CMD3.

5. **Malloc cache integration** — per-token expert dispatch with 62% hit rate
   achieves near-zero I/O for the majority of expert loads.

## The malloc cache effect

Without malloc cache (OS page cache only):
- Serial: 169-183 ms/tok
- Multi-slot: 188 ms/tok (near-parity, +2.7%)

With malloc cache (18 GB, 2581 entries):
- Serial: 200-217 ms/tok (SLOWER due to 18 GB allocation competing with page cache)
- Multi-slot: 96-109 ms/tok (1.93× FASTER)

The serial path is hurt by the malloc cache because the 18 GB allocation
pushes OS page cache entries out → the 38% of experts that miss the
malloc cache now also miss the page cache → slower pread.

The multi-slot path overcomes this because T=4 tokens per chunk amortize
expert loads: each unique expert is loaded once per chunk and used by all
tokens that routed to it. The parallel projection dispatch + deferred
pipeline are also more effective when expert I/O time is reduced (cache
hits shorten the time between CMD3 dispatch and CMD1 start).

## How to activate

```bash
MULTI_SLOT_PREFILL=1 ./infer --prompt "<long prompt>" --tokens N \
  --k 4 --malloc-cache 2581 --batch-prefill 4
```

Default: `--batch-prefill 1` (serial per-token, unchanged behavior).

## Correctness

- Matching token sequences on "Hello world" (271, 9419, 0) ✓
- Coherent output on all prompt lengths tested (2, 18, 138, 454 tokens) ✓
- Precision: matches serial fast-path bf16 norm precision (GPU norm
  throughout, no CPU→GPU precision cascade)

## What's left to optimize

1. **Early-chunk overhead** (first ~8 chunks with kv_len < 32): these run
   at ~110 ms/tok which is still fast but slightly above the warm-cache
   baseline of ~80-90 ms/tok. Could be improved with CPU attention fallback
   at very low kv_len.

2. **High-kv_len slight regression**: chunks at kv_len > 300 show 91 ms/tok
   vs the 79-83 ms sweet spot at kv_len ~80-200. The attention kernel scales
   with kv_len and the Phase 0 GPU headroom diminishes at higher kv_len.

3. **Batched expert dispatch with cache integration**: `dispatch_experts_sparse`
   was tried but is 3.3× slower than per-token dispatch because it doesn't
   use the malloc cache and does synchronous gpu_expert_forward. A cache-aware
   variant would batch unique expert loads across T tokens AND use the cache.

4. **Reduce command buffer overhead**: multi-slot creates 12 cmd bufs/layer
   (4 × CMD1 + 4 × CMD2 + 4 × CMD3) vs serial's 3. Could potentially merge
   some command buffers.
