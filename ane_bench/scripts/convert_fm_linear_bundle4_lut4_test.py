"""Phase 1a addendum: bundle 4 flash_moe GatedDeltaNet layers into ONE
compiled .mlmodelc to measure bundling economics.

Single-layer LUT4 measurement (2026-04-11) was 1.410 ms p50 Obj-C. This
script stacks 4 independently-weighted linear layers with their own state
tensors and asks: what's the per-layer cost when bundled?

Rationale: anemll-qwen35's reference super-block 0 runs 4 layers (DDDA) in
~6.64 ms Obj-C (Phase 0). That's 1.66 ms/layer average bundled, vs ~2.3 ms
the single-layer equivalent would be if unbundled. Bundling appears to
amortize ~0.6-1 ms of dispatch overhead per layer.

The production flash_moe port will bundle DDDA (3 linear + 1 full attention),
but for Phase 1a we only have the GatedDeltaNet class working. DDDD (4
linear) is a reasonable surrogate — slightly conservative because all 4
layers are the "heavier" linear type, while real DDDA has 1 full-attention
layer which is lighter. So if DDDD-bundled hits N ms, DDDA-bundled will be
at most N ms and probably a bit less.
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
OUT_DIR.mkdir(exist_ok=True)


class FMLinearBundle4(nn.Module):
    """Stack of 4 independently-weighted Qwen35GatedDeltaNet layers.

    Threads hidden state through sequentially; each layer has its own
    recurrent state + conv state pair. Input/output signature:
        in:   hidden_states [1,1,H]
              gated_state_{0..3} [1,Hv,Dv,Dk]
              conv_state_{0..3}  [1,K-1,conv_dim]
        out:  output_hidden_states [1,1,H]
              new_gated_state_{0..3}, new_conv_state_{0..3}
    """
    def __init__(self, cfg: Qwen35Config):
        super().__init__()
        self.layers = nn.ModuleList([
            Qwen35GatedDeltaNet(cfg) for _ in range(4)
        ])

    def forward(self,
                hidden_states,
                gated_state_0, conv_state_0,
                gated_state_1, conv_state_1,
                gated_state_2, conv_state_2,
                gated_state_3, conv_state_3):
        x = hidden_states
        x, ng0, nc0 = self.layers[0](x, gated_state_0, conv_state_0)
        x, ng1, nc1 = self.layers[1](x, gated_state_1, conv_state_1)
        x, ng2, nc2 = self.layers[2](x, gated_state_2, conv_state_2)
        x, ng3, nc3 = self.layers[3](x, gated_state_3, conv_state_3)
        return x, ng0, nc0, ng1, nc1, ng2, nc2, ng3, nc3


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
    print(f"[fm-p1a-b4] config: Hv={cfg.linear_num_value_heads}, "
          f"conv_dim={cfg.linear_conv_dim}, 4 layers bundled")

    torch.manual_seed(42)
    model = FMLinearBundle4(cfg)
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.xavier_normal_(p, gain=0.02)
        elif p.dim() == 1:
            nn.init.normal_(p, mean=0.0, std=0.02)
    # Safe A_log/dt_bias per-layer
    with torch.no_grad():
        for l in model.layers:
            l.A_log.fill_(-2.0)
            l.dt_bias.fill_(0.0)
    model.eval()
    set_rmsnorm_mode(model, "ane")
    model = model.float()

    Hv = cfg.linear_num_value_heads
    Dv = cfg.linear_value_head_dim
    Dk = cfg.linear_key_head_dim
    conv_dim = cfg.linear_conv_dim
    K = cfg.linear_conv_kernel_dim
    B, T = 1, 1

    def z(shape): return torch.zeros(shape, dtype=torch.float32)

    h_ex = z((B, T, cfg.hidden_size))
    state_shape = (B, Hv, Dv, Dk)
    conv_shape = (B, K - 1, conv_dim)
    ex_args = (
        h_ex,
        z(state_shape), z(conv_shape),
        z(state_shape), z(conv_shape),
        z(state_shape), z(conv_shape),
        z(state_shape), z(conv_shape),
    )

    print("[fm-p1a-b4] tracing...")
    t0 = time.perf_counter()
    traced = torch.jit.trace(model, ex_args)
    print(f"[fm-p1a-b4] traced in {time.perf_counter() - t0:.2f}s")

    print("[fm-p1a-b4] ct.convert (fp16)...")
    t0 = time.perf_counter()
    inputs = [
        ct.TensorType(name="hidden_states", shape=(B, T, cfg.hidden_size), dtype=np.float16),
    ]
    for i in range(4):
        inputs.append(ct.TensorType(name=f"gated_state_{i}", shape=state_shape, dtype=np.float16))
        inputs.append(ct.TensorType(name=f"conv_state_{i}",  shape=conv_shape,  dtype=np.float16))
    outputs = [
        ct.TensorType(name="output_hidden_states", dtype=np.float16),
    ]
    for i in range(4):
        outputs.append(ct.TensorType(name=f"new_gated_state_{i}", dtype=np.float16))
        outputs.append(ct.TensorType(name=f"new_conv_state_{i}",  dtype=np.float16))

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    print(f"[fm-p1a-b4] base convert done in {time.perf_counter() - t0:.2f}s")

    # LUT4 palettization
    print("[fm-p1a-b4] palettizing LUT4 (per_grouped_channel, group_size=8)...")
    t0 = time.perf_counter()
    cfg_p = cto_coreml.OpPalettizerConfig(
        mode="kmeans", nbits=4, granularity="per_grouped_channel",
        group_size=8, num_kmeans_workers=1,
    )
    mlmodel_lut4 = cto_coreml.palettize_weights(
        mlmodel, cto_coreml.OptimizationConfig(global_config=cfg_p)
    )
    print(f"[fm-p1a-b4] palettize done in {time.perf_counter() - t0:.2f}s")

    out_path = OUT_DIR / "fm_linear_bundle4_lut4.mlpackage"
    mlmodel_lut4.save(str(out_path))
    size_mb = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file()) / 1e6
    print(f"[fm-p1a-b4] saved {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
