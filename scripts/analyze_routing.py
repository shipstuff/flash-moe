#!/usr/bin/env python3
"""
Analyze a --collect-routing binary log from flash_moe/infer.

Binary record format (per routing decision, one per layer per token):
    int32  layer_idx
    int32  K
    float32[HIDDEN_DIM=4096]  hidden    (skipped — not needed for working-set analysis)
    int32[K] expert_indices

Constants must match infer.m:
    NUM_LAYERS = 60
    HIDDEN_DIM = 4096  (the actual compiled constant — AGENTS.md docstring says 7168 but it's stale)
    NUM_EXPERTS = 512
    EXPERT_SIZE_4BIT = 7077888 bytes  (~6.75 MB)

Outputs:
  1. Hit-rate-vs-pinned-N curves per layer (top-N most frequent experts).
  2. Cross-layer view: aggregate hit rate at uniform N across all layers
     (so you can size a "persistent expert working set" budget).
  3. Token-temporal locality: how often does token T+1 reuse an expert from
     token T at the same layer? (informs whether a recency cache helps.)
  4. Per-layer skewness (Gini) — flat distributions are bad candidates for
     a working set; heavily skewed layers are good candidates.

Usage:
    ./infer --collect-routing /tmp/routing.bin --prompt "..." --tokens 256
    python3 scripts/analyze_routing.py /tmp/routing.bin
"""
import sys
import struct
import argparse
from collections import defaultdict, Counter

NUM_LAYERS = 60
HIDDEN_DIM = 4096
NUM_EXPERTS = 512
EXPERT_BYTES_4BIT = 7077888
HEADER_BYTES = 8                       # int32 layer_idx + int32 K
HIDDEN_BYTES = HIDDEN_DIM * 4          # 16384


def parse_log(path, skip_hidden=True):
    """Yield (layer_idx, [expert_idx, ...]) tuples."""
    with open(path, "rb") as f:
        while True:
            hdr = f.read(HEADER_BYTES)
            if len(hdr) < HEADER_BYTES:
                return
            layer_idx, K = struct.unpack("<ii", hdr)
            if skip_hidden:
                f.seek(HIDDEN_BYTES, 1)
            else:
                f.read(HIDDEN_BYTES)
            exp_bytes = f.read(K * 4)
            if len(exp_bytes) < K * 4:
                return
            experts = list(struct.unpack(f"<{K}i", exp_bytes))
            yield layer_idx, experts


def collect(path):
    """Single pass: per-layer counter + per-layer ordered token route list."""
    per_layer_counts = [Counter() for _ in range(NUM_LAYERS)]
    per_layer_routes = [[] for _ in range(NUM_LAYERS)]   # list of frozensets
    total_records = 0
    K_seen = None
    for layer, exps in parse_log(path):
        if 0 <= layer < NUM_LAYERS:
            per_layer_counts[layer].update(exps)
            per_layer_routes[layer].append(frozenset(exps))
        if K_seen is None:
            K_seen = len(exps)
        total_records += 1
    return per_layer_counts, per_layer_routes, total_records, K_seen


def hit_rate_at_N(counter, N, K):
    """If we pin the top-N experts of this layer, what fraction of activations hit?"""
    if not counter:
        return 0.0
    top = counter.most_common(N)
    hits = sum(c for _, c in top)
    total = sum(counter.values())
    return hits / total if total else 0.0


def gini(values):
    """Gini coefficient of inequality on a list of counts. 0=flat, 1=skewed."""
    if not values:
        return 0.0
    vs = sorted(values)
    n = len(vs)
    s = sum(vs)
    if s == 0:
        return 0.0
    cum = 0
    for i, v in enumerate(vs, 1):
        cum += i * v
    return (2 * cum) / (n * s) - (n + 1) / n


def temporal_reuse(routes):
    """Per-layer fraction of token-T experts that also appeared in token T-1."""
    overlaps = []
    for layer_routes in routes:
        if len(layer_routes) < 2:
            overlaps.append(0.0)
            continue
        total = 0
        reused = 0
        for prev, curr in zip(layer_routes, layer_routes[1:]):
            total += len(curr)
            reused += len(curr & prev)
        overlaps.append(reused / total if total else 0.0)
    return overlaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="binary routing log from --collect-routing")
    ap.add_argument("--Ns", default="1,2,4,8,16,32,64,128,256",
                    help="working-set sizes to evaluate per layer (comma list)")
    ap.add_argument("--per-layer", action="store_true",
                    help="print full per-layer hit-rate table")
    args = ap.parse_args()

    Ns = [int(x) for x in args.Ns.split(",")]

    print(f"Reading {args.path} ...", file=sys.stderr)
    counts, routes, total_records, K = collect(args.path)
    print(f"  {total_records} routing records, K={K}", file=sys.stderr)
    if total_records == 0:
        print("Empty log.", file=sys.stderr)
        sys.exit(1)

    tokens_per_layer = total_records // NUM_LAYERS
    print(f"  ~{tokens_per_layer} tokens per layer", file=sys.stderr)
    print()

    # ---- Aggregate hit rate vs uniform N ----
    print("=== Hit rate vs uniform working-set size N (same N every layer) ===")
    print(f"{'N/layer':>8} {'hit_rate':>10} {'RAM_GB':>9} {'total_experts':>14}")
    layer_total_acts = sum(sum(c.values()) for c in counts)
    for N in Ns:
        hit_acts = 0
        for c in counts:
            top = c.most_common(N)
            hit_acts += sum(v for _, v in top)
        hr = hit_acts / layer_total_acts if layer_total_acts else 0.0
        ram_gb = N * NUM_LAYERS * EXPERT_BYTES_4BIT / (1024**3)
        print(f"{N:8d} {hr*100:9.2f}% {ram_gb:8.2f} {N*NUM_LAYERS:14d}")
    print()

    # ---- Per-layer skewness (Gini) ----
    ginis = [gini(list(c.values())) for c in counts]
    avg_gini = sum(ginis) / len(ginis)
    print(f"=== Per-layer routing skewness (Gini, 1.0=concentrated) ===")
    print(f"avg Gini = {avg_gini:.3f}")
    flat = sorted(range(NUM_LAYERS), key=lambda i: ginis[i])[:5]
    skewed = sorted(range(NUM_LAYERS), key=lambda i: ginis[i], reverse=True)[:5]
    print(f"  flattest layers (worst for pinning): {[(l, round(ginis[l],3)) for l in flat]}")
    print(f"  most skewed layers (best for pinning): {[(l, round(ginis[l],3)) for l in skewed]}")
    print()

    # ---- Token-temporal locality ----
    overlaps = temporal_reuse(routes)
    avg_ovr = sum(overlaps) / len(overlaps)
    print(f"=== Token-temporal reuse (T+1 vs T, same layer) ===")
    print(f"avg overlap = {avg_ovr*100:.1f}% (fraction of K experts at token T+1 that")
    print(f"  also routed at token T). Random baseline = K/NUM_EXPERTS ~= {K/NUM_EXPERTS*100:.2f}%.")
    print()

    # ---- Greedy budget allocation ----
    # Given a total expert-slot budget, pick the per-layer Ns that maximize hits.
    # For each layer build "marginal gain" curve: gain[i] = count of i-th expert.
    print("=== Greedy budget allocation (variable N per layer) ===")
    print("Goal: given a total slot budget B, distribute slots across layers to maximize hits.")
    marginals = []  # (gain, layer_idx) tuples, all experts
    for li, c in enumerate(counts):
        sorted_counts = sorted(c.values(), reverse=True)
        for v in sorted_counts:
            marginals.append((v, li))
    marginals.sort(key=lambda x: -x[0])

    # Walk down the sorted marginal-gain list, accumulate.
    cum_hits = 0
    print(f"{'budget_B':>10} {'RAM_GB':>9} {'hit_rate':>10}")
    target_budgets = [60, 120, 240, 480, 960, 1920, 3840, 7680, 15360, 30720]
    next_idx = 0
    for B in target_budgets:
        while next_idx < len(marginals) and next_idx < B:
            cum_hits += marginals[next_idx][0]
            next_idx += 1
        hr = cum_hits / layer_total_acts if layer_total_acts else 0.0
        ram_gb = B * EXPERT_BYTES_4BIT / (1024**3)
        print(f"{B:10d} {ram_gb:8.2f} {hr*100:9.2f}%")
    print()

    if args.per_layer:
        print("=== Per-layer detailed table ===")
        cols = " ".join(f"N={N:>4}" for N in Ns)
        print(f"layer  unique  gini  reuse  {cols}")
        for l in range(NUM_LAYERS):
            uniq = sum(1 for v in counts[l].values() if v > 0)
            row = " ".join(f"{hit_rate_at_N(counts[l], N, K)*100:6.1f}" for N in Ns)
            print(f"{l:5d}  {uniq:6d}  {ginis[l]:.2f}  {overlaps[l]*100:5.1f}  {row}")


if __name__ == "__main__":
    main()
