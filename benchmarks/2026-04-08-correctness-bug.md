# FlashMoE Correctness Bug: malloc-cache-64 3-token attractor loop

**Date:** 2026-04-08
**Found by:** seslly
**Commit:** `5060a65` (benchmarks)

## Symptom
- `malloc-cache-64`, 512 tokens: model produces "the name of the name of the name of..." × 512 — 3-token looping attractor
- Same binary, model, prompt, token count — baseline (no cache) works fine
- Expert cache hit rate: 0.0% (123,600 misses for 512 tokens) — cache not being reused
- malloc cache path loads expert weights fresh every step, bypassing OS page cache layer

## Hypothesis
malloc-cache-64 removes the OS page cache layer entirely. That layer may have been providing implicit memory barriers or coherency. Without it, memory ordering / DMA races cause expert weight corruption on some loads, leading to collapsed routing that locks the model into a 3-token attractor.

## Validation Plan
1. **Run baseline 512tok** → confirm output is coherent (no loop)
2. **Reproduce malloc-cache-64 loop** → confirm it reproduces consistently
3. **Instrument with memory fence** → add explicit GPU sync before expert loads in malloc path
4. **Check temperature/sampling config** → rule out greedy collapse
5. **Run 256tok baseline vs malloc-cache** → confirm 256tok baseline is also coherent

## Status
- [ ] Step 1: Run baseline 512tok (no cache) — confirm coherent output
- [ ] Step 2: Run malloc-cache-64 512tok 3x — check if loop is consistent
- [ ] Step 3: Add memory fence / sync to malloc cache path
- [ ] Step 4: Check sampling config
- [ ] Step 5: Fix and re-run
