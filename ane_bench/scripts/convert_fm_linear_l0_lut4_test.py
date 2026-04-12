"""Phase 1a LUT4 variant: same flash_moe single linear layer, but with
per-channel LUT4 palettization on the weight matrices. Tests whether
quantization brings per-layer ANE time down.

Hypothesis: the cost model showed linear ops as 'Comp' bound (compute, not
memory), so LUT4 may give zero speedup. This test confirms or refutes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path.home() / "projects" / "anemll-qwen35" / "model"))
from qwen3_5_model import Qwen35Config, Qwen35GatedDeltaNet, set_rmsnorm_mode  # noqa: E402

OUT_DIR = Path("/tmp/fm_ane_phase1a")


class FMLinearWrapper(nn.Module):
    def __init__(self, layer: Qwen35GatedDeltaNet):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, gated_state, conv_state):
        out, new_state, new_conv = self.layer(hidden_states, gated_state, conv_state)
        return out, new_state, new_conv


def main():
    import coremltools as ct
    import coremltools.optimize.coreml as cto_coreml

    cfg = Qwen35Config(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_value_heads=64,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )

    torch.manual_seed(42)
    layer = Qwen35GatedDeltaNet(cfg)
    for p in layer.parameters():
        if p.dim() >= 2:
            nn.init.xavier_normal_(p, gain=0.02)
        elif p.dim() == 1:
            nn.init.normal_(p, mean=0.0, std=0.02)
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

    print("[fm-p1a-lut4] tracing...")
    traced = torch.jit.trace(wrapper, (h_ex, gs_ex, cs_ex))

    print("[fm-p1a-lut4] ct.convert (fp16)...")
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
    print(f"[fm-p1a-lut4] base convert done in {time.perf_counter() - t0:.2f}s")

    # Apply LUT4 palettization — matches anemll-qwen35/scripts/convert_all_superblocks_lut4.py
    print("[fm-p1a-lut4] palettizing to LUT4 (per_grouped_channel group_size=8)...")
    t0 = time.perf_counter()
    config = cto_coreml.OptimizationConfig(
        global_config=cto_coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=4,
            granularity="per_grouped_channel",
            group_size=8,
            num_kmeans_workers=1,
        )
    )
    mlmodel_lut4 = cto_coreml.palettize_weights(mlmodel, config=config)
    print(f"[fm-p1a-lut4] palettize done in {time.perf_counter() - t0:.2f}s")

    out_path = OUT_DIR / "fm_linear_l0_lut4.mlpackage"
    mlmodel_lut4.save(str(out_path))
    size_mb = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file()) / 1e6
    print(f"[fm-p1a-lut4] saved {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
