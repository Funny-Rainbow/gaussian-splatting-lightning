#!/usr/bin/env python3
"""Export a Feature3DGS checkpoint to an extended PLY with feat_* attributes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from internal.utils.gaussian_utils import GaussianPlyUtils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Feature3DGS high-dimensional Gaussian features to PLY.")
    parser.add_argument("--checkpoint", required=True, help="Feature3DGS .ckpt file.")
    parser.add_argument("--output", required=True, help="Output extended PLY path.")
    parser.add_argument("--feature-key", default="renderer.features", help="State-dict key for per-Gaussian features.")
    parser.add_argument("--with-colors", action="store_true", help="Also write standard RGB fields from SH DC.")
    return parser.parse_args()


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def add_attributes(dtype_full: list, attribute_list: list, name_prefix: str, value: np.ndarray) -> None:
    for idx in range(value.shape[-1]):
        name = f"{name_prefix}_{idx}"
        dtype_full.append((name, "f4"))
        attribute_list.append((name, value[..., idx].astype(np.float32)))


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if args.feature_key not in state_dict:
        keys = "\n".join(k for k in sorted(state_dict) if "feature" in k or "renderer" in k)
        raise KeyError(f"Feature key not found: {args.feature_key}\nCandidate keys:\n{keys}")

    features = state_dict[args.feature_key].float()
    gaussian = GaussianPlyUtils.load_from_state_dict(state_dict).to_ply_format()
    if gaussian.xyz.shape[0] != features.shape[0]:
        raise RuntimeError(f"Gaussian/feature count mismatch: {gaussian.xyz.shape[0]} != {features.shape[0]}")

    xyz = gaussian.xyz.astype(np.float32)
    f_dc = gaussian.features_dc.reshape((gaussian.features_dc.shape[0], -1)).astype(np.float32)
    f_rest = (
        gaussian.features_rest.reshape((gaussian.features_rest.shape[0], -1)).astype(np.float32)
        if gaussian.sh_degrees > 0
        else np.zeros((f_dc.shape[0], 0), dtype=np.float32)
    )
    opacities = gaussian.opacities.astype(np.float32)
    scale = gaussian.scales.astype(np.float32)
    rotation = gaussian.rotations.astype(np.float32)
    feature_np = to_numpy(features).astype(np.float32)

    dtype_full = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    attribute_list = [("x", xyz[:, 0]), ("y", xyz[:, 1]), ("z", xyz[:, 2])]
    add_attributes(dtype_full, attribute_list, "f_dc", f_dc)
    add_attributes(dtype_full, attribute_list, "f_rest", f_rest)
    dtype_full.append(("opacity", "f4"))
    attribute_list.append(("opacity", opacities.squeeze(-1)))
    add_attributes(dtype_full, attribute_list, "scale", scale)
    add_attributes(dtype_full, attribute_list, "rot", rotation)
    add_attributes(dtype_full, attribute_list, "feat", feature_np)

    if args.with_colors:
        from internal.utils.sh_utils import eval_sh

        rgbs = np.clip((eval_sh(0, gaussian.features_dc, None) + 0.5), 0.0, 1.0)
        rgbs = (rgbs * 255).astype(np.uint8)
        dtype_full += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        attribute_list += [("red", rgbs[:, 0]), ("green", rgbs[:, 1]), ("blue", rgbs[:, 2])]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    for key, value in attribute_list:
        elements[key] = value

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(tmp))
    os.replace(tmp, output)
    print(f"Exported {xyz.shape[0]} Gaussians with {feature_np.shape[1]} feature dims to {output}")


if __name__ == "__main__":
    main()
