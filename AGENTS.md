# Flash-MoE Developer Guide

> **User-facing docs:** `README.md` (paper, results, architecture overview)
> **This file:** Current bugs, debugging patterns, project conventions, dev workflow

---

## What's Working

### CLI Inference (metal_infer/infer.m)
- Tokenization + vocab loading
- Model weights (`model_weights.bin`) + manifest (`model_weights.json`)
- 4-bit packed experts (`/models/Qwen3.5-397B-A17B-4bit/packed_experts/layer_XX.bin`)
- Metal GPU pipeline (fused MoE layers via `fused_layer_forward`)
- Malloc-based expert LRU cache (`--malloc-cache N`)
- OS page cache for expert I/O — 32 GB/s sequential reads
- TurboQuant KV-cache shaders (TQ kernels compile, Metal library JIT succeeds)
- Per-layer timing breakdown (`--timing`)

### HTTP Server Mode (`--serve PORT`)
- OpenAI-compatible `/v1/chat/completions` API
- System prompt prefilling + KV cache snapshots
- Per-request timing

### Benchmark Results (2026-04-09, M4 Pro)
```
128tok, M256 malloc cache, K=4, no TQ:  14.72 tok/s
128tok, M256 malloc cache, K=4, +TQ_KV: 14.80 tok/s
256tok, M256 malloc cache, K=4, no TQ:  14.44 tok/s
256tok story, M256, K=4, +TQ_KV:         14.98 tok/s
128tok, M512 malloc cache:               14.41 tok/s
64-entry cold baseline (no malloc):        8.64 tok/s
```

---

## What's Broken

### CLI hangs after tokenization (as of 2026-04-10)
- **Symptom:** `Tokens (1): [9419]` prints, then hangs — no `[init]` header, no `[ttft]`
- **Location:** Hang is after tokenization but before Metal init/weights loading completes
- **Likely cause:** `metal_setup()` or `io_pool_init()` — not yet isolated
- **Date:** Started after `packed_experts` rebuild at 2026-04-10 10:35 AM
- **Packed expert format changed:** `expert_index.json` shows `expert_size: 2097152` vs hardcoded `EXPERT_SIZE: 7077888` in `infer.m` — mismatch not yet confirmed as root cause

### TQ shader hang (pre-existing, not yet fixed)
- **Symptom:** With TQ KV pipelines enabled (`TQ_KV=1`), generation hangs at token [5834] post-TTFT
- **Location:** Inside the generation loop — hangs before summary output
- **Note:** TQ kernels compile and pipeline creation succeeds, but execution hangs
- **Status:** Unfixed; hang is in TQ execution path, not shader compilation

### malloc_cache_insert eviction bug (partially fixed)
- **Bug:** In `expert_cache_evict()` LRU path, `cache_telemetry_evict()` was called with already-cleared `layer_idx[target]`/`expert_idx[target]`
- **Fix committed:** Capture old indices before clearing, call telemetry first, then clear entry_idx
- **Status:** Fix committed in `8f76d02` but not yet verified on hardware

### Temporal prediction (`--predict`) not yet tested
- Flag `g_pred_enabled` wired up but no end-to-end verification

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

---

*Last verified: 2026-04-10. CLI hang is active - check "What's Broken" before running.*
