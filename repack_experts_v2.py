#!/usr/bin/env python3
"""Rebuild packed_experts/layer_XX.bin using correct safetensors offsets.

Background: The prior repack used expert_index.json whose abs_offset values were
written as raw data_offsets[0] instead of (data_start + data_offsets[0]). This
off-by-47245 (= file_header_size) error caused every expert block to contain
bytes from neighboring tensors, producing NaN forward passes.

This script derives offsets directly from each safetensors file's own header
(the authoritative source) and writes layer files in the layout infer.m expects:

Per-expert block (7,077,888 bytes, in order):
  gate_proj.weight   offset        0, size 2,097,152 (U32)
  gate_proj.scales   offset 2,097,152, size   131,072 (BF16)
  gate_proj.biases   offset 2,228,224, size   131,072 (BF16)
  up_proj.weight     offset 2,359,296, size 2,097,152 (U32)
  up_proj.scales     offset 4,456,448, size   131,072 (BF16)
  up_proj.biases     offset 4,587,520, size   131,072 (BF16)
  down_proj.weight   offset 4,718,592, size 2,097,152 (U32)
  down_proj.scales   offset 6,815,744, size   131,072 (BF16)
  down_proj.biases   offset 6,946,816, size   131,072 (BF16)

Layer file = 512 experts × 7,077,888 bytes = 3,623,878,656 bytes.

Usage:
  python repack_experts_v2.py --layer 0             # rebuild one layer (quick test)
  python repack_experts_v2.py                        # rebuild all 60 layers
  python repack_experts_v2.py --verify 0             # verify layer 0 against source
"""
import argparse
import json
import os
import struct
import sys
import time
from glob import glob

MODEL_DIR = '/Users/carl/models/Qwen3.5-397B-A17B-4bit'
PACKED_DIR = os.path.join(MODEL_DIR, 'packed_experts')
NUM_LAYERS = 60
NUM_EXPERTS = 512
EXPERT_SIZE = 7_077_888
LAYER_SIZE = NUM_EXPERTS * EXPERT_SIZE  # 3,623,878,656

# Component layout within one expert block (must match infer.m EXPERT_SIZE offsets)
COMPONENTS = [
    ('gate_proj.weight',        0, 2_097_152, 'U32'),
    ('gate_proj.scales',  2_097_152,   131_072, 'BF16'),
    ('gate_proj.biases',  2_228_224,   131_072, 'BF16'),
    ('up_proj.weight',    2_359_296, 2_097_152, 'U32'),
    ('up_proj.scales',    4_456_448,   131_072, 'BF16'),
    ('up_proj.biases',    4_587_520,   131_072, 'BF16'),
    ('down_proj.weight',  4_718_592, 2_097_152, 'U32'),
    ('down_proj.scales',  6_815_744,   131_072, 'BF16'),
    ('down_proj.biases',  6_946_816,   131_072, 'BF16'),
]


def build_tensor_map():
    """Scan all safetensors files, return {tensor_name: (path, abs_off, size, dtype, shape)}."""
    tensor_map = {}
    files = sorted(glob(os.path.join(MODEL_DIR, 'model-*.safetensors')))
    if not files:
        print(f'FATAL: no safetensors files found in {MODEL_DIR}', file=sys.stderr)
        sys.exit(1)
    for fpath in files:
        with open(fpath, 'rb') as fp:
            hdr_len = struct.unpack('<Q', fp.read(8))[0]
            hdr = json.loads(fp.read(hdr_len).decode())
        data_start = 8 + hdr_len
        for k, info in hdr.items():
            if k == '__metadata__':
                continue
            if 'switch_mlp' not in k:
                continue
            do0, do1 = info['data_offsets']
            tensor_map[k] = (fpath, data_start + do0, do1 - do0,
                             info['dtype'], tuple(info['shape']))
    return tensor_map


def verify_layout(tensor_map):
    """Confirm per-expert sizes and dtypes match our expected layout."""
    dtype_expected = {
        'gate_proj.weight': 'U32',  'gate_proj.scales': 'BF16',  'gate_proj.biases': 'BF16',
        'up_proj.weight':   'U32',  'up_proj.scales':   'BF16',  'up_proj.biases':   'BF16',
        'down_proj.weight': 'U32',  'down_proj.scales': 'BF16',  'down_proj.biases': 'BF16',
    }
    # Also verify per-expert byte size
    expected_per_expert = {c[0]: c[2] for c in COMPONENTS}
    for layer in range(NUM_LAYERS):
        for comp, _, sz, dt in COMPONENTS:
            key = f'language_model.model.layers.{layer}.mlp.switch_mlp.{comp}'
            if key not in tensor_map:
                raise RuntimeError(f'missing tensor: {key}')
            _, _, total_sz, dtype, shape = tensor_map[key]
            if dtype != dt:
                raise RuntimeError(f'{key}: dtype {dtype} != expected {dt}')
            # First shape dim must be NUM_EXPERTS
            if shape[0] != NUM_EXPERTS:
                raise RuntimeError(f'{key}: shape[0]={shape[0]} != {NUM_EXPERTS}')
            per_expert = total_sz // NUM_EXPERTS
            if per_expert != sz:
                raise RuntimeError(f'{key}: per-expert {per_expert} != expected {sz}')
    print(f'Layout verified: {NUM_LAYERS} layers × {len(COMPONENTS)} components, all sizes/dtypes match')


def repack_layer(layer_idx, tensor_map, out_path):
    """Repack one layer from the source safetensors into layer_XX.bin."""
    t0 = time.monotonic()

    # Open output file (create or truncate-in-place)
    fd_out = os.open(out_path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        # Ensure correct size (first time / if changed)
        cur_size = os.fstat(fd_out).st_size
        if cur_size != LAYER_SIZE:
            os.ftruncate(fd_out, LAYER_SIZE)

        bytes_written = 0
        # cache fds for source files
        src_fds = {}
        try:
            for comp_name, dst_off_within_expert, comp_size, _dtype in COMPONENTS:
                key = f'language_model.model.layers.{layer_idx}.mlp.switch_mlp.{comp_name}'
                src_path, abs_off, total_sz, _dt, shape = tensor_map[key]
                if src_path not in src_fds:
                    src_fds[src_path] = os.open(src_path, os.O_RDONLY)
                src_fd = src_fds[src_path]

                # Per-expert stride in the source tensor
                per_expert = total_sz // NUM_EXPERTS
                assert per_expert == comp_size, \
                    f'{key}: per-expert {per_expert} != component {comp_size}'

                # Read the entire tensor for this layer in one pread (1 GB max for weights)
                data = os.pread(src_fd, total_sz, abs_off)
                if len(data) != total_sz:
                    raise IOError(f'{key}: short read {len(data)}/{total_sz}')

                # Scatter each expert's slice to the output file
                for expert_idx in range(NUM_EXPERTS):
                    src_slice = data[expert_idx * per_expert : (expert_idx + 1) * per_expert]
                    dst_off = expert_idx * EXPERT_SIZE + dst_off_within_expert
                    os.pwrite(fd_out, src_slice, dst_off)
                    bytes_written += len(src_slice)

                # Release the large buffer before loading the next component
                del data
        finally:
            for fd in src_fds.values():
                os.close(fd)
    finally:
        os.fsync(fd_out)
        os.close(fd_out)

    elapsed = time.monotonic() - t0
    return bytes_written, elapsed


def verify_layer(layer_idx, tensor_map, out_path):
    """Read back a few experts from the packed file and compare to source bytes."""
    fd_out = os.open(out_path, os.O_RDONLY)
    src_fds = {}
    mismatches = 0
    try:
        for comp_name, dst_off_within_expert, comp_size, _dtype in COMPONENTS:
            key = f'language_model.model.layers.{layer_idx}.mlp.switch_mlp.{comp_name}'
            src_path, abs_off, total_sz, _dt, _shape = tensor_map[key]
            if src_path not in src_fds:
                src_fds[src_path] = os.open(src_path, os.O_RDONLY)
            src_fd = src_fds[src_path]
            per_expert = total_sz // NUM_EXPERTS
            for expert_idx in [0, 1, 255, 511]:
                src_data = os.pread(src_fd, per_expert,
                                    abs_off + expert_idx * per_expert)
                dst_data = os.pread(fd_out, per_expert,
                                    expert_idx * EXPERT_SIZE + dst_off_within_expert)
                if src_data != dst_data:
                    mismatches += 1
                    print(f'  MISMATCH layer={layer_idx} expert={expert_idx} comp={comp_name}: '
                          f'src_first8={src_data[:8].hex()} dst_first8={dst_data[:8].hex()}')
    finally:
        for fd in src_fds.values():
            os.close(fd)
        os.close(fd_out)
    return mismatches == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, default=None,
                    help='Rebuild a single layer (0..59)')
    ap.add_argument('--verify', type=int, default=None,
                    help='Verify an existing layer without rewriting')
    ap.add_argument('--layers', default=None,
                    help='Layer spec: "0-4", "0,5,10", or "all" (default: all)')
    args = ap.parse_args()

    print('Building tensor map from safetensors headers...')
    t0 = time.monotonic()
    tensor_map = build_tensor_map()
    print(f'  {len(tensor_map)} switch_mlp tensors in {time.monotonic()-t0:.1f}s')

    verify_layout(tensor_map)

    if args.verify is not None:
        path = os.path.join(PACKED_DIR, f'layer_{args.verify:02d}.bin')
        ok = verify_layer(args.verify, tensor_map, path)
        print(f'Layer {args.verify} verification: {"PASS" if ok else "FAIL"}')
        sys.exit(0 if ok else 1)

    # Determine layers to process
    if args.layer is not None:
        layers = [args.layer]
    elif args.layers:
        layers = []
        for part in args.layers.split(','):
            part = part.strip()
            if '-' in part:
                a, b = part.split('-', 1)
                layers.extend(range(int(a), int(b) + 1))
            else:
                layers.append(int(part))
    else:
        layers = list(range(NUM_LAYERS))

    os.makedirs(PACKED_DIR, exist_ok=True)
    total_written = 0
    t_start = time.monotonic()
    for i, layer_idx in enumerate(layers):
        out_path = os.path.join(PACKED_DIR, f'layer_{layer_idx:02d}.bin')
        bytes_written, elapsed = repack_layer(layer_idx, tensor_map, out_path)
        total_written += bytes_written
        overall = time.monotonic() - t_start
        overall_gb = total_written / 1024**3
        tput = total_written / overall / 1024**3 if overall > 0 else 0
        eta = (len(layers) - i - 1) * (overall / (i + 1))
        print(f'  Layer {layer_idx:2d}: {bytes_written/1024**3:.2f} GB in {elapsed:.1f}s '
              f'({bytes_written/elapsed/1024**3:.2f} GB/s) | '
              f'Total: {overall_gb:.1f}/{len(layers)*LAYER_SIZE/1024**3:.1f} GB '
              f'({tput:.2f} GB/s avg) | ETA: {eta:.0f}s',
              flush=True)

        # Spot-verify this layer
        if not verify_layer(layer_idx, tensor_map, out_path):
            print(f'ABORTING: verification failed for layer {layer_idx}', flush=True)
            sys.exit(1)

    print(f'\nDONE: {total_written/1024**3:.1f} GB written in {time.monotonic()-t_start:.1f}s')


if __name__ == '__main__':
    main()
