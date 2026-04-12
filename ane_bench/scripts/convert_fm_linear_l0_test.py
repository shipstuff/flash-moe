"""Phase 1a feasibility test: convert ONE GatedDeltaNet linear-attention layer
with flash_moe's dimensions (random weights) and verify it compiles fully
ANE-resident.

flash_moe dims vs anemll-qwen35 9B defaults:
  hidden_size             4096     (same)
  num_attention_heads     32  (vs  16  in 9B)    — full attn, not traced here
  num_key_value_heads      2  (vs  4   in 9B)    — full attn, not traced here
  linear_num_value_heads  64  (vs  32  in 9B)    ← the big one, doubles Hv
  linear_num_key_heads    16  (same)
  linear_key_head_dim    128  (same)
  linear_value_head_dim  128  (same)
  linear_conv_kernel_dim   4  (same)

Derived:
  key_dim   = 16 * 128 = 2048
  value_dim = 64 * 128 = 8192
  conv_dim  = 2*2048 + 8192 = 12288    (vs 8192 in 9B — +50%)
  state shape: [1, 64, 128, 128] — 1.0 MB fp16 per layer (vs 0.5 MB in 9B)
  conv_state  : [1, 3, 12288]    — 72 KB fp16 per layer (vs 48 KB in 9B)

Success = (a) ct.convert completes, (b) anemll-profile reports 0 CPU ops and
0 graph interruptions on the compiled .mlmodelc, (c) a dispatch benchmark
shows per-prediction latency within a reasonable factor of the 9B super-block
baseline (9.28 ms pure ANE / 6.64 ms Obj-C in our ane_dispatch_bench).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path.home() / "projects" / "anemll-qwen35" / "model"))
from qwen3_5_model import (  # noqa: E402
    Qwen35Config,
    Qwen35GatedDeltaNet,
    set_rmsnorm_mode,
)

OUT_DIR = Path("/tmp/fm_ane_phase1a")
OUT_DIR.mkdir(exist_ok=True)


class FMLinearWrapper(nn.Module):
    """Bare GatedDeltaNet forward with explicit state tensors.

    Matches the reference convert_part2_layer0.py wrapper pattern so the
    resulting .mlpackage has the expected input/output layout:
        in:  hidden_states [1,1,H], gated_state [1,Hv,Dv,Dk], conv_state [1,K-1,conv_dim]
        out: output_hidden_states [1,1,H], new_gated_state, new_conv_state
    """
    def __init__(self, layer: Qwen35GatedDeltaNet):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, gated_state, conv_state):
        out, new_state, new_conv = self.layer(hidden_states, gated_state, conv_state)
        return out, new_state, new_conv


def main():
    import coremltools as ct

    # flash_moe-shaped config
    cfg = Qwen35Config(
        hidden_size=4096,
        num_attention_heads=32,       # flash_moe 32 Q heads
        num_key_value_heads=2,        # flash_moe 2 KV heads
        head_dim=256,
        linear_num_value_heads=64,    # ← THE flash_moe-specific change
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    print(f"[fm-p1a] config: Hv={cfg.linear_num_value_heads}, "
          f"key_dim={cfg.linear_key_dim}, value_dim={cfg.linear_value_dim}, "
          f"conv_dim={cfg.linear_conv_dim}")

    # Build a single GatedDeltaNet layer with RANDOM weights (we're testing shapes,
    # not correctness of output values)
    torch.manual_seed(42)
    layer = Qwen35GatedDeltaNet(cfg)
    # Random-init the weights explicitly (they're zero-ish by default)
    for p in layer.parameters():
        if p.dim() >= 2:
            nn.init.xavier_normal_(p, gain=0.02)
        elif p.dim() == 1:
            nn.init.normal_(p, mean=0.0, std=0.02)
    # Init A_log to something non-degenerate (anemll-qwen35 CLAUDE.md warned
    # about A_log values becoming subnormals in fp16 — keep them in a safe range)
    with torch.no_grad():
        layer.A_log.fill_(-2.0)
        layer.dt_bias.fill_(0.0)
    layer.eval()
    set_rmsnorm_mode(layer, "ane")

    wrapper = FMLinearWrapper(layer).eval().float()

    Hv = cfg.linear_num_value_heads
    Dv = cfg.linear_value_head_dim
    Dk = cfg.linear_key_head_dim
    conv_dim = cfg.linear_conv_dim
    K = cfg.linear_conv_kernel_dim
    B, T = 1, 1

    h_ex = torch.zeros((B, T, cfg.hidden_size), dtype=torch.float32)
    gs_ex = torch.zeros((B, Hv, Dv, Dk), dtype=torch.float32)
    cs_ex = torch.zeros((B, K - 1, conv_dim), dtype=torch.float32)

    print(f"[fm-p1a] input shapes: h={tuple(h_ex.shape)}, gs={tuple(gs_ex.shape)}, cs={tuple(cs_ex.shape)}")
    print(f"[fm-p1a] tracing...")
    t0 = time.perf_counter()
    traced = torch.jit.trace(wrapper, (h_ex, gs_ex, cs_ex))
    print(f"[fm-p1a] traced in {time.perf_counter() - t0:.2f}s")

    # Sanity: eager vs trace
    with torch.no_grad():
        h_in = torch.randn_like(h_ex) * 0.02
        gs_in = torch.zeros_like(gs_ex)
        cs_in = torch.zeros_like(cs_ex)
        eager = wrapper(h_in, gs_in, cs_in)
        traced_out = traced(h_in, gs_in, cs_in)
    for i, (e, t_) in enumerate(zip(eager, traced_out)):
        diff = (e - t_).abs().max().item()
        print(f"[fm-p1a] trace check output {i}: max_abs {diff:.2e}")

    print("[fm-p1a] running ct.convert (FP16, CPU_AND_NE, iOS18)...")
    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(B, T, cfg.hidden_size), dtype=np.float16),
            ct.TensorType(name="gated_state", shape=(B, Hv, Dv, Dk), dtype=np.float16),
            ct.TensorType(name="conv_state", shape=(B, K - 1, conv_dim), dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="output_hidden_states", dtype=np.float16),
            ct.TensorType(name="new_gated_state", dtype=np.float16),
            ct.TensorType(name="new_conv_state", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    print(f"[fm-p1a] ct.convert done in {time.perf_counter() - t0:.2f}s")

    out_path = OUT_DIR / "fm_linear_l0.mlpackage"
    mlmodel.save(str(out_path))
    size_mb = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file()) / 1e6
    print(f"[fm-p1a] saved {out_path} ({size_mb:.1f} MB)")
    print(f"[fm-p1a] to compile: xcrun coremlcompiler compile {out_path} {OUT_DIR}")


if __name__ == "__main__":
    main()
