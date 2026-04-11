# Flash-MoE Developer Guide

> **User-facing docs:** `README.md` (paper, results, architecture overview)
> **This file:** Current bugs, debugging patterns, project conventions, dev workflow

---

## What's Working

### CLI Inference (metal_infer/infer.m)
- Tokenization + vocab loading
- Model weights (`model_weights.bin`) + manifest (`model_weights.json`)
- 4-bit packed experts (`/models/Qwen3.5-397B-A17B-4bit/packed_experts/layer_XX.bin`)
  — rebuilt 2026-04-10 via `repack_experts_v2.py` with correct safetensors offsets
- Metal GPU pipeline (fused MoE layers via `fused_layer_forward`)
- Malloc-based expert LRU cache (`--malloc-cache N`) — lz4_comp_buf stack-init
  bug fixed 2026-04-10 (see below)
- OS page cache for expert I/O — 32 GB/s sequential reads
- **TurboQuant KV-cache (`TQ_KV=1`) — WORKS end-to-end as of 2026-04-11.**
  Coherent generation at 32 / 128 / 256+ tokens, 7.5x KV cache compression
  (33.4 MB vs 252 MB at 8k context), ~12% generation slowdown at 128 tokens
  from CPU Q rotation overhead (vectorizable). See below and the
  `benchmarks/2026-04-10-post-repack-e2e.md` write-up.
- Temporal expert prediction (`--predict`) — functional, but a net regression
  (~26% hit rate, -58% speed)
- Per-layer timing breakdown (`--timing`)
- **Batch prefill (`--batch-prefill T`) — COLD-CACHE-ONLY as of 2026-04-11.**
  Processes T prompt tokens per layer call, amortizing expert I/O across T tokens.
  Correctness validated: T=1/4/8 bit-identical tokens on 18 and 138-token prompts. ✅
  Cold-cache first chunk: 2.35x speedup (T=4 265ms vs T=1 622ms for 15-tok prompt).
  **Warm-cache wall-time is a REGRESSION at long prompts**: at 138 tok prompt,
  T=4 is +57% slower than T=1 (236 vs 150 ms rest-avg). Root cause: the loop
  inversion (layers outside, tokens inside) breaks the CMD3↔CMD1 pipelining
  that per-token prefill relies on, and 45/60 linear-attention layers fall
  through to a sequential path with `complete_deferred_experts()` between
  tokens. See `docs/2026-04-11-batch-prefill-scoping.md` afternoon update for
  the full trace and the two refactor approaches (multi-buffered deferred
  state, MoE cross-token decoupling). Flag is opt-in — default stays T=1.
  See also `dispatch_experts_sparse` aliasing bug fix note in "Common Errors".

### HTTP Server Mode (`--serve PORT`)
- OpenAI-compatible `/v1/chat/completions` API
- System prompt prefilling + KV cache snapshots
- Per-request timing

### Benchmark Results (2026-04-11 morning session, M4 Pro)
All runs coherent output unless noted. Two key updates today: TQ now works
end-to-end (8 bug fixes earlier in the day) and Q rotation is now done via
Accelerate sgemm (cpu_attn dropped 0.205 → 0.065 ms/layer).

Short context (16-token prompt):
```
no cache,           128tok, K=4: 5.91 tok/s, 104 coherent (EOS at 104)
TQ_KV=1 (sgemm),    128tok, K=4: 5.65 tok/s, 128 coherent
TQ_KV=1 (sgemm),    256tok, K=4: 5.15 tok/s, 256 coherent
malloc-cache-64,    128tok, K=4: 6.13 tok/s, 0% hit (thrashing), coherent
malloc-cache-512,    96tok, K=4: 6.07 tok/s, 32.2% hit, coherent
--predict,          128tok, K=4: 2.51 tok/s, 26% hit, coherent (-58% net regression)
```

Long context (TQ wins because it keeps cmd2_wait flat):
```
Context  Baseline cmd2_wait  TQ cmd2_wait  Baseline tok/s  TQ tok/s  TQ delta
~30 tok       0.434              0.423            5.91         5.65    -4.4%
~1k tok       0.710              0.423            4.98         5.40    +8.4%
~2.4k tok     1.056              0.406            3.98         4.69   +17.8%
~3k tok       1.221              0.408            3.72         4.27   +14.8%
```

Crossover ≈ 600–800 token context. Beyond that TQ wins on both memory
footprint (7.5x compression) and generation speed. The 3k delta is
slightly under the 2.4k delta because at 3k context the float KV cache
(251 MB) starts adding page-cache pressure that bumps `expert_io` by
~0.1 ms/layer in baseline; TQ keeps the cache at 33.4 MB so it stays out
of the way.
Real baseline per-layer: expert_io=1.337ms, cmd1_wait=0.858ms,
cmd2_wait=0.426ms, total_layer=2.703ms. This matches the "5.86 tok/s" reference
numbers in top-level `CLAUDE.md` (difference is measurement noise).

**The historical "14-15 tok/s with malloc cache" results in `results.tsv` were
bogus** — the malloc cache's pread path was silently failing due to an
uninitialized-stack-memory bug (lz4_comp_buf garbage → io_pool_worker took the
LZ4 decompress path → pread into wild pointer → returned -1 → GPU computed on
zero-filled cache buffers). The "fast" numbers were the speed of running a
not-actually-doing-I/O path with garbage expert data. See the benchmark file
`benchmarks/2026-04-10-post-repack-e2e.md` for full details.

---

## What's Broken

### Temporal prediction (`--predict`) — functional net regression
- 26% hit rate, -58% generation speed vs baseline. Matches historical
  "-18% / 25% hit rate" note in top-level CLAUDE.md. Feature works e2e, output
  coherent, but should stay off by default until a better prediction scheme
  is found.

---

## Project Structure

```
flash_moe/
├── README.md              ← Paper + architecture overview
├── AGENTS.md              ← This file (developer context)
├── CLAUDE.md              ← Symlink to AGENTS.md (for Claude Code)
│
├── metal_infer/
│   ├── infer.m            ← Main inference engine (Objective-C/Metal, ~7400 lines)
│   ├── shaders.metal      ← GPU kernels (MoE, attention, TQ KV-cache)
│   ├── infer              ← Compiled binary (make -j4)
│   ├── model_weights.bin  ← Shared embeddings + MLPs (5.5 GB)
│   ├── model_weights.json ← Manifest (tensor shapes, offsets)
│   ├── vocab.bin          ← BPE vocabulary
│   ├── packed_experts/    ← Per-layer MoE expert files (4-bit)
│   ├── packed_experts_2bit/
│   └── results.tsv        ← Benchmark results log
│
├── expert_index.json      ← Expert metadata (paths, sizes, shapes)
├── repack_experts.py     ← Re-pack raw experts into metal_infer format
└── paper/flash_moe.pdf   ← Published paper
```

### Key Files

| File | Purpose |
|------|---------|
| `metal_infer/infer.m` | Everything: main(), model loading, generation loop, Metal API |
| `metal_infer/shaders.metal` | GPU kernels (MoE, linear attention, TQ KV-cache) |
| `expert_index.json` | Model paths, expert shapes — read by `repack_experts.py` |
| `repack_experts.py` | Converts HuggingFace safetensors to packed binary format |

---

## How to Run

### CLI mode
```bash
cd ~/projects/turbomoe/flash_moe/metal_infer

# Basic (uses default model path from code)
./infer --prompt "Hello world" --tokens 32 --k 4

# With malloc cache (17GB for 2581 entries)
./infer --model /Users/carl/models/Qwen3.5-397B-A17B-4bit \
  --prompt "Write a story" --tokens 64 --k 4 --malloc-cache 2581 --timing

# With TQ KV-cache (4 TQ kernels in shaders.metal)
TQ_KV=1 ./infer --model /Users/carl/models/Qwen3.5-397B-A17B-4bit \
  --prompt "Hello" --tokens 8 --k 4 --malloc-cache 2581 --timing

# Server mode
./infer --model /Users/carl/models/Qwen3.5-397B-A17B-4bit \
  --serve 8080 --k 4 --malloc-cache 2581
```

### Rebuild
```bash
cd ~/projects/turbomoe/flash_moe/metal_infer
rm -f infer && make -j4
```

### Benchmark sweep
```bash
# From metal_infer/
for tok in 64 128 256; do
  ./infer --model /Users/carl/models/Qwen3.5-397B-A17B-4bit \
    --tokens $tok --k 4 --malloc-cache 2581 --timing \
    --prompt "Write a story about a robot who loves music." 2>&1 \
    | grep -E 'tok/s|TTFT|ttft|expert_io|cmd1_wait';
done
```

---

## Key Architecture Decisions

### Expert I/O: OS page cache, not Metal LRU
Testing showed OS page cache at 32 GB/s outperformed Metal LRU (38% faster). The LRU cache code is present but disabled by default (`cache_entries=0`). Malloc-based expert cache (`--malloc-cache N`) is the recommended path for multi-shot workloads.

### Linear attention: CPU fallback
`gpu_linear_attn_enabled=0` uses CPU/hybrid path. Default is GPU fused delta-net (`--gpu-linear`). The linear attention state is per-layer (`linear_states[]`) with 15 layers using linear attention at `FULL_ATTN_INTERVAL=4`.

### TurboQuant KV-Cache (TQ_KV)
Compresses K/V from fp16 to 2-bit Lloyd-Max quantized + random rotation. Storage: 68 bytes vs 512 bytes fp16 = 7.5x compression. Kernels in `shaders.metal`:
- `tq_encode_packed` - compress new K/V on write
- `tq_fused_attention` - attention on compressed data (no full decompression)
- `tq_pack_update` - update compressed KV cache
- `tq_dequant_all` - full dequant for final computation

TQ_KV is gated by env var `TQ_KV=1` and auto-falls back if pipelines unavailable.

### Batch prefill
All prompt tokens are embedded upfront into `embed_batch[]`, then processed in a batch prefill loop. Intermediate tokens use `discard_deferred_experts()` (no CPU readback). Only the last prefill token uses `complete_deferred_experts()` (full GPU readback + combine).

---

## Debugging

### Finding where it hangs
Add `fprintf(stderr, "CHECKPOINT\n")` through the init sequence, rebuild, run:
```bash
./infer --prompt "Hi" --tokens 4 2>&1 | grep CHECKPOINT
```

Key checkpoints in `main()`:
1. After `metal_setup()` - `[metal]`
2. After `io_pool_init()` - `[io]`
3. After `open_weights()` - `[weights]`
4. After `load_vocab()` - `[vocab]`
5. After expert file mmap loop - `[experts]`
6. After `printf("[init]"...)` - `[init]`
7. After prefill - `[ttft]`

### Checking expert files
```bash
# Verify all 60 layer files exist and have correct size
stat ~/models/Qwen3.5-397B-A17B-4bit/packed_experts/layer_00.bin
# Expected: 7077888 bytes per file (4-bit), 3932160 (2-bit)

# Count files
ls ~/models/Qwen3.5-397B-A17B-4bit/packed_experts/ | wc -l
# Expected: 60
```

### Metal shader debugging
```bash
# Force re-JIT compile (delete library cache)
rm -rf ~/Library/Developer/Xcode/DerivedData/*

# Check if TQ pipelines are found
TQ_KV=1 ./infer --model /Users/carl/models/Qwen3.5-397B-A17B-4bit \
  --prompt "Hi" --tokens 1 2>&1 | grep -i 'tq\|pipeline\|error\|warning'
```

### Checking model path
```bash
cat ~/projects/turbomoe/flash_moe/expert_index.json | grep model_path
# Must be: /Users/carl/models/Qwen3.5-397B-A17B-4bit
# NOT: /Users/danielwoods/... (old path from hf hub snapshot)
```

---

## Git Workflow

### Commits on mini-01 (no GitHub auth)
Changes are committed locally on `skynet-m4-mini-01.local`. To sync:
```bash
# On mini: commit with useful messages
git add -u && git commit -m "descriptive message"

# On another machine with GitHub auth:
git fetch mini-01
git log --oneline mini-01/main -5
git cherry-pick mini-01/main  # or merge/rebase
```

### Current HEAD
```
8f76d02 turboquant: TQ KV-cache kernels + eviction fix + model path fix
```

Working tree is clean (all changes committed). Untracked: `infer.dSYM/`, `output/`, backup files.

### What to commit
- `metal_infer/infer.m` - engine changes
- `metal_infer/shaders.metal` - GPU kernel changes
- `metal_infer/results.tsv` - new benchmark results
- `expert_index.json` - model path fixes, expert format changes
- `AGENTS.md` - this file

---

## Constants (infer.m)

```
NUM_LAYERS = 60
HIDDEN_DIM = 7168
NUM_EXPERTS = 256
K = 4  (default active experts per layer)
EXPERT_SIZE = 7077888  (4-bit expert byte size)
EXPERT_SIZE_2BIT = 3932160  (2-bit expert byte size)
FULL_ATTN_INTERVAL = 4  (every 4th layer uses full attention, rest use linear)
NUM_KV_HEADS = 2
HEAD_DIM = 256
VQ_WORDS_PER_HEAD = 16  (for TQ KV-cache)
```

---

## Environment Variables

| Var | Effect |
|-----|--------|
| `TQ_KV=1` | Enable TurboQuant KV-cache (auto-falls back if unavailable) |
| `TQ_HIST=1` | Record TQ latency histogram per layer |
| `HERMES_HOME` | Overrides `~/.hermes` for Hermes config profiles |

---

## Common Errors

### "ERROR: Failed to load weights"
Model path wrong or `model_weights.bin` missing. Check `expert_index.json` `model_path` field.

### "ERROR: Failed to encode prompt"
`encode_prompt_text_to_tokens()` fails. Usually `bpe_encode()` returns < 0. Check vocab is loaded.

### Metal pipeline not found (TQ_KV)
TQ kernels in `shaders.metal` compiled but `makePipe` cannot find them. Usually stale binary - `rm -f infer && make -j4` fixes.

### Expert size mismatch
If you see `expert_index.json` size is not equal to `EXPERT_SIZE` constant in `infer.m`, the repacking pipeline changed. Check `repack_experts.py` output against hardcoded constants.

### `malloc_cache_insert`: wrong entry evicted
The LRU eviction bug - `entry_idx` cleared before telemetry call, so wrong entry tracked as evicted. Fixed in `8f76d02`.

### `dispatch_experts_sparse`: wrong Metal buffer aliasing (FIXED 2026-04-11)
`dispatch_experts_sparse` pread'd expert data into `buf_multi_expert_data[0]` then called
`gpu_expert_forward(..., already_in_buffer=1)`. But `gpu_expert_forward` ALWAYS reads its
weight matrices from `buf_expert_data` — not `buf_multi_expert_data[0]`. With
`already_in_buffer=1`, the copy step was skipped, so the GPU used stale data from the
previous call. This caused all T>1 batched tokens to produce garbage hidden states.
Fix: pass `already_in_buffer=(s > 0)` — copy on first token (s==0), reuse on subsequent
tokens of the same unique expert (s>0). Fixed in `c08cf07` (integration-batch-prefill branch).

---

*Last verified: 2026-04-11. CLI hang is active - check "What's Broken" before running.*
