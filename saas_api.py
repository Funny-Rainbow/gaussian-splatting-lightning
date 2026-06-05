from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parent
UTILS_ROOT = ROOT / "utils"


def _ensure_import_paths() -> None:
    for candidate in (ROOT, UTILS_ROOT):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def run_keyframes(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    _ensure_import_paths()
    from utils.redundant_image_detection_sharpness_SaaS import video_mode

    def is_keyframe_factory(dist: float, ratio: float):
        def is_keyframe(xy0, xy1):
            normalized_dist = np.linalg.norm(xy0 - xy1, axis=-1)
            over_threshold_mask = normalized_dist > dist
            over_threshold_ratio = over_threshold_mask.sum() / (over_threshold_mask.shape[0] + 1)
            return xy0.shape[0] < 32 or over_threshold_ratio > ratio

        return is_keyframe

    args = SimpleNamespace(
        input=payload["input"],
        output_dir=payload.get("output_dir"),
        fps=int(payload.get("fps", 15)),
        dist=float(payload.get("dist", 0.07)),
        ratio=float(payload.get("ratio", 0.2)),
        max_size=int(payload.get("max_size", 1024)),
        save_max_long_edge=int(payload.get("save_max_long_edge", 0)),
        start_number=int(payload.get("start_number", 1)),
        filename_format=str(payload.get("filename_format", "%05d.jpg")),
        min_sharpness=float(payload.get("min_sharpness", 10.0)),
        sharpness_max_size=int(payload.get("sharpness_max_size", 640)),
        sharpness_grid=tuple(payload.get("sharpness_grid", (4, 4))),
        sharpness_tile_percentile=float(payload.get("sharpness_tile_percentile", 50.0)),
        sharpness_pre_blur_sigma=float(payload.get("sharpness_pre_blur_sigma", 0.0)),
        sharpness_downscale=float(payload.get("sharpness_downscale", 0.5)),
    )
    result = video_mode(args, is_keyframe_factory(args.dist, args.ratio), quiet=True)
    if isinstance(result, dict):
        return {"ok": True, **result}
    return {"ok": True, "result": result}


def _extract_arg_value(args: list[str], *flags: str) -> str | None:
    for index, arg in enumerate(args):
        if arg in flags:
            if index + 1 < len(args):
                return args[index + 1]
            return None
        for flag in flags:
            if arg.startswith(f"{flag}="):
                return arg.split("=", 1)[1]
    return None


def _resolve_output_root(args: list[str]) -> str | None:
    name = _extract_arg_value(args, "-n", "--name")
    if not name:
        return None

    output_base = _extract_arg_value(args, "--output") or str(ROOT / "outputs")
    version = _extract_arg_value(args, "-v", "--version")
    output_root = Path(output_base) / name
    if version:
        output_root = output_root / version
    return str(output_root)


def run_train(payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_import_paths()
    from internal.entrypoints.gspl import cli as gspl_cli

    args = list(payload["args"])
    gspl_cli(args=args)
    return {"ok": True, "output_root": _resolve_output_root(args)}


def run_transform(payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_import_paths()
    import numpy as np
    import torch
    from viser import transforms as vt

    from internal.utils.gaussian_model_editor import MultipleGaussianModelEditor
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.gaussian_utils import GaussianPlyUtils

    def rx(theta):
        return np.matrix(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )

    def ry(theta):
        return np.matrix(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )

    def rz(theta):
        return np.matrix(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

    with torch.no_grad():
        input_path = Path(payload["input"]).expanduser()
        output_path = Path(payload["output"]).expanduser()
        tx = float(payload.get("tx", 0))
        ty = float(payload.get("ty", 0))
        tz = float(payload.get("tz", 0))
        rx_value = float(payload.get("rx", 0))
        ry_value = float(payload.get("ry", 0))
        rz_value = float(payload.get("rz", 0))
        scale = float(payload.get("scale", 1))
        sh_factor = float(payload.get("sh_factor", 1.0))
        device = torch.device(payload.get("device", "cpu"))

        gaussian_model, _ = GaussianModelLoader.search_and_load(
            str(input_path),
            device=device,
            eval_mode=True,
            pre_activate=False,
        )
        gaussian_model_editor = MultipleGaussianModelEditor([gaussian_model], device=device)
        rot_mat = rx(rx_value) @ ry(ry_value) @ rz(rz_value)
        gaussian_model_editor.transform_with_vectors(
            idx=0,
            scale=scale,
            r_wxyz=vt.SO3.from_matrix(rot_mat).wxyz,
            t_xyz=np.asarray([tx, ty, tz]),
        )

        if sh_factor != 1.0:
            gaussian_model.shs_dc *= sh_factor
            gaussian_model.shs_rest *= sh_factor

        output_path.parent.mkdir(parents=True, exist_ok=True)
        GaussianPlyUtils.load_from_model(gaussian_model).to_ply_format().save_to_ply(str(output_path))

    return {"ok": True, "output": str(output_path)}


def run_prune_position_sigma(payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_import_paths()
    import torch

    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.gaussian_utils import GaussianPlyUtils

    def _prune_gaussian(gaussian: GaussianPlyUtils, keep_mask: torch.Tensor) -> GaussianPlyUtils:
        return GaussianPlyUtils(
            sh_degrees=gaussian.sh_degrees,
            xyz=gaussian.xyz[keep_mask],
            opacities=gaussian.opacities[keep_mask],
            features_dc=gaussian.features_dc[keep_mask],
            features_rest=gaussian.features_rest[keep_mask],
            scales=gaussian.scales[keep_mask],
            rotations=gaussian.rotations[keep_mask],
        )

    with torch.no_grad():
        input_path = Path(payload["input"]).expanduser()
        output_path = Path(payload["output"]).expanduser()
        sigma = float(payload.get("sigma", 3.0))
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        device = torch.device(payload.get("device", "cpu"))

        gaussian_model, _ = GaussianModelLoader.search_and_load(
            str(input_path),
            device=device,
            eval_mode=True,
            pre_activate=False,
        )
        gaussian = GaussianPlyUtils.load_from_model(gaussian_model)
        xyz = gaussian.xyz
        if xyz.shape[0] == 0:
            raise RuntimeError("input Gaussian model has no points")

        axis_mean = xyz.mean(dim=0)
        axis_std = xyz.std(dim=0, unbiased=False)
        lower = axis_mean - sigma * axis_std
        upper = axis_mean + sigma * axis_std
        keep_mask = torch.isfinite(xyz).all(dim=1)
        keep_mask &= ((xyz >= lower) & (xyz <= upper)).all(dim=1)

        original_count = int(xyz.shape[0])
        kept_count = int(keep_mask.sum().item())
        if kept_count == 0:
            raise RuntimeError(
                "position sigma pruning would remove all Gaussians "
                f"(input={input_path}, sigma={sigma})"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _prune_gaussian(gaussian, keep_mask).to_ply_format().save_to_ply(str(output_path))

    return {
        "ok": True,
        "input": str(input_path),
        "output": str(output_path),
        "sigma": sigma,
        "original_count": original_count,
        "kept_count": kept_count,
        "removed_count": original_count - kept_count,
        "thresholds": {
            "mean": axis_mean.detach().cpu().tolist(),
            "std": axis_std.detach().cpu().tolist(),
            "lower": lower.detach().cpu().tolist(),
            "upper": upper.detach().cpu().tolist(),
        },
    }
