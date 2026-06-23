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

    def _payload_bool(name: str, default: bool) -> bool:
        value = payload.get(name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

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
        auto_detect = _payload_bool("auto_detect", True)
        auto_sigma = float(payload.get("auto_sigma", 8.0))
        auto_max_to_p99_ratio = float(payload.get("auto_max_to_p99_ratio", 1.5))
        auto_max_fraction = float(payload.get("auto_max_fraction", 0.02))
        auto_min_count = int(payload.get("auto_min_count", 1))
        if auto_sigma <= 0:
            raise ValueError(f"auto_sigma must be positive, got {auto_sigma}")
        if auto_max_to_p99_ratio < 1.0:
            raise ValueError(f"auto_max_to_p99_ratio must be >= 1.0, got {auto_max_to_p99_ratio}")
        if not 0 < auto_max_fraction <= 1:
            raise ValueError(f"auto_max_fraction must be in (0, 1], got {auto_max_fraction}")
        if auto_min_count < 1:
            raise ValueError(f"auto_min_count must be >= 1, got {auto_min_count}")
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

        original_count = int(xyz.shape[0])
        auto_detection = None
        if auto_detect:
            finite_mask = torch.isfinite(xyz).all(dim=1)
            finite_xyz = xyz[finite_mask]
            finite_count = int(finite_xyz.shape[0])
            nonfinite_count = original_count - finite_count
            if finite_count == 0:
                raise RuntimeError("input Gaussian model has no finite positions")

            center = finite_xyz.median(dim=0).values
            radius = torch.linalg.vector_norm(finite_xyz - center, dim=1)
            median_radius = radius.median()
            mad = torch.abs(radius - median_radius).median()
            robust_sigma = 1.4826 * mad
            p99_radius = torch.quantile(radius, 0.99) if finite_count > 1 else radius.max()
            max_radius = radius.max()
            far_threshold = median_radius + auto_sigma * robust_sigma
            far_mask = radius > far_threshold
            far_count = int(far_mask.sum().item())
            candidate_count = far_count + nonfinite_count
            far_fraction = far_count / finite_count
            candidate_fraction = candidate_count / original_count
            distance_triggered = bool(
                max_radius.item() > far_threshold.item()
                and max_radius.item() >= auto_max_to_p99_ratio * p99_radius.item()
            )
            triggered = bool(
                candidate_count >= auto_min_count
                and candidate_fraction <= auto_max_fraction
                and (nonfinite_count > 0 or distance_triggered)
            )
            auto_detection = {
                "enabled": True,
                "triggered": triggered,
                "center": center.detach().cpu().tolist(),
                "median_radius": float(median_radius.item()),
                "mad_radius": float(mad.item()),
                "robust_sigma_radius": float(robust_sigma.item()),
                "p99_radius": float(p99_radius.item()),
                "max_radius": float(max_radius.item()),
                "far_threshold": float(far_threshold.item()),
                "far_count": far_count,
                "nonfinite_count": nonfinite_count,
                "candidate_count": candidate_count,
                "far_fraction": far_fraction,
                "candidate_fraction": candidate_fraction,
                "auto_sigma": auto_sigma,
                "auto_max_to_p99_ratio": auto_max_to_p99_ratio,
                "auto_max_fraction": auto_max_fraction,
                "auto_min_count": auto_min_count,
            }
            if not triggered:
                return {
                    "ok": True,
                    "skipped": True,
                    "reason": "auto_detection_not_triggered",
                    "input": str(input_path),
                    "output": str(output_path),
                    "sigma": sigma,
                    "original_count": original_count,
                    "auto_detection": auto_detection,
                }
        else:
            auto_detection = {"enabled": False}

        axis_mean = xyz.mean(dim=0)
        axis_std = xyz.std(dim=0, unbiased=False)
        lower = axis_mean - sigma * axis_std
        upper = axis_mean + sigma * axis_std
        keep_mask = torch.isfinite(xyz).all(dim=1)
        keep_mask &= ((xyz >= lower) & (xyz <= upper)).all(dim=1)

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
        "auto_detection": auto_detection,
        "thresholds": {
            "mean": axis_mean.detach().cpu().tolist(),
            "std": axis_std.detach().cpu().tolist(),
            "lower": lower.detach().cpu().tolist(),
            "upper": upper.detach().cpu().tolist(),
        },
    }


def run_render_tsdf_frames(payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_import_paths()
    import json

    import numpy as np
    import torch
    from PIL import Image

    from internal.cameras.cameras import Cameras
    from internal.utils.gaussian_model_loader import GaussianModelLoader

    model_path = Path(payload["model_path"]).expanduser()
    cameras_path = Path(payload["cameras_path"]).expanduser()
    output_dir = Path(payload["output_dir"]).expanduser()
    max_views = int(payload.get("max_views") or 0)
    device = torch.device(payload.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    cameras_data = json.loads(cameras_path.read_text(encoding="utf-8"))
    if not isinstance(cameras_data, list) or not cameras_data:
        raise ValueError(f"cameras.json must contain a non-empty list: {cameras_path}")
    if max_views > 0:
        cameras_data = cameras_data[:max_views]

    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    alpha_dir = output_dir / "alpha"
    for directory in (rgb_dir, depth_dir, alpha_dir):
        directory.mkdir(parents=True, exist_ok=True)

    gaussian_model, renderer = GaussianModelLoader.search_and_load(str(model_path), device)
    gaussian_model.freeze()
    gaussian_model.eval()
    renderer.eval()
    background = torch.zeros((3,), dtype=torch.float32, device=device)
    available_outputs = renderer.get_available_outputs()
    depth_type = "exp_depth" if "exp_depth" in available_outputs else "depth"
    render_types = ["rgb", depth_type]
    has_alpha = "alpha" in available_outputs
    if has_alpha:
        render_types.append("alpha")

    frames = []
    with torch.no_grad():
        for index, camera_info in enumerate(cameras_data):
            width = int(camera_info["width"])
            height = int(camera_info["height"])
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = np.asarray(camera_info["rotation"], dtype=np.float64)
            c2w[:3, 3] = np.asarray(camera_info["position"], dtype=np.float64)
            w2c = np.linalg.inv(c2w)
            camera = Cameras(
                R=torch.tensor(w2c[:3, :3], dtype=torch.float32).unsqueeze(0),
                T=torch.tensor(w2c[:3, 3], dtype=torch.float32).unsqueeze(0),
                fx=torch.tensor([float(camera_info["fx"])], dtype=torch.float32),
                fy=torch.tensor([float(camera_info["fy"])], dtype=torch.float32),
                cx=torch.tensor([float(camera_info["cx"])], dtype=torch.float32),
                cy=torch.tensor([float(camera_info["cy"])], dtype=torch.float32),
                width=torch.tensor([width], dtype=torch.int16),
                height=torch.tensor([height], dtype=torch.int16),
                appearance_id=torch.tensor([int(camera_info.get("appearance_id") or 0)], dtype=torch.long),
                normalized_appearance_id=torch.tensor([float(camera_info.get("normalized_appearance_id") or 0.0)], dtype=torch.float32),
                time=torch.tensor([float(camera_info.get("time") or 0.0)], dtype=torch.float32),
                distortion_params=None,
                camera_type=torch.zeros((1,), dtype=torch.int8),
            )[0].to_device(device)

            outputs = renderer(
                camera,
                gaussian_model,
                background,
                scaling_modifier=1.0,
                render_types=render_types,
            )
            rgb = outputs["render"].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            rgb_u8 = (rgb * 255.0).round().astype(np.uint8)
            depth = outputs[depth_type].detach().squeeze().cpu().numpy().astype(np.float32)
            depth[~np.isfinite(depth)] = 0.0
            depth[depth < 0.0] = 0.0
            if has_alpha and outputs.get("alpha") is not None:
                alpha = outputs["alpha"].detach().squeeze().clamp(0, 1).cpu().numpy()
            else:
                alpha = (depth > 0.0).astype(np.float32)

            stem = f"{index:05d}"
            rgb_path = rgb_dir / f"{stem}.png"
            depth_path = depth_dir / f"{stem}.npy"
            alpha_path = alpha_dir / f"{stem}.png"
            Image.fromarray(rgb_u8).save(rgb_path)
            np.save(depth_path, depth)
            Image.fromarray((alpha * 255.0).round().astype(np.uint8)).save(alpha_path)

            frames.append(
                {
                    "rgb": str(rgb_path.relative_to(output_dir)),
                    "depth": str(depth_path.relative_to(output_dir)),
                    "alpha": str(alpha_path.relative_to(output_dir)),
                    "intrinsics": {
                        "width": width,
                        "height": height,
                        "fx": float(camera_info["fx"]),
                        "fy": float(camera_info["fy"]),
                        "cx": float(camera_info["cx"]),
                        "cy": float(camera_info["cy"]),
                    },
                    "world_to_camera": w2c.tolist(),
                }
            )

    manifest_path = output_dir / "manifest.json"
    manifest = {"depth_scale": 1.0, "frames": frames}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "manifest": str(manifest_path),
        "frame_count": len(frames),
        "depth_type": depth_type,
        "alpha": has_alpha,
    }


def run_extract_tsdf_mesh(payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_import_paths()
    import json

    import numpy as np
    import open3d as o3d
    import torch

    from internal.cameras.cameras import Cameras
    from internal.utils.gaussian_model_loader import GaussianModelLoader

    model_path = Path(payload["model_path"]).expanduser()
    cameras_path = Path(payload["cameras_path"]).expanduser()
    output_path = Path(payload["output_path"]).expanduser()
    output_manifest_path = Path(payload.get("output_manifest_path") or output_path.with_suffix(".json")).expanduser()
    max_views = int(payload.get("max_views") or 0)
    mesh_resolution = int(payload.get("mesh_resolution") or 512)
    depth_trunc = float(payload.get("depth_trunc") or 6.0)
    alpha_threshold = float(payload.get("alpha_threshold") or 0.5)
    voxel_size = float(payload.get("voxel_size") or -1.0)
    sdf_trunc = float(payload.get("sdf_trunc") or -1.0)
    clean_mesh = bool(payload.get("clean_mesh", False))
    min_component_triangles = max(0, int(payload.get("min_component_triangles") or 0))
    keep_components = max(0, int(payload.get("keep_components") or 0))
    target_triangles = max(0, int(payload.get("target_triangles") or 0))
    decimate_enabled = bool(payload.get("decimate_enabled", False))
    device = torch.device(payload.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    if mesh_resolution <= 0:
        raise ValueError("mesh_resolution must be positive")
    if depth_trunc <= 0:
        raise ValueError("depth_trunc must be positive")

    def cleanup_mesh(mesh):
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        return mesh

    cameras_data = json.loads(cameras_path.read_text(encoding="utf-8"))
    if not isinstance(cameras_data, list) or not cameras_data:
        raise ValueError(f"cameras.json must contain a non-empty list: {cameras_path}")
    if max_views > 0:
        cameras_data = cameras_data[:max_views]

    voxel_size = voxel_size if voxel_size > 0 else depth_trunc / mesh_resolution
    sdf_trunc = sdf_trunc if sdf_trunc > 0 else 5.0 * voxel_size
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    gaussian_model, renderer = GaussianModelLoader.search_and_load(str(model_path), device)
    gaussian_model.freeze()
    gaussian_model.eval()
    renderer.eval()
    background = torch.zeros((3,), dtype=torch.float32, device=device)
    available_outputs = renderer.get_available_outputs()
    depth_type = "exp_depth" if "exp_depth" in available_outputs else "depth"
    render_types = ["rgb", depth_type]
    has_alpha = "alpha" in available_outputs
    if has_alpha:
        render_types.append("alpha")

    frame_logs = []
    used_views = 0
    with torch.no_grad():
        for index, camera_info in enumerate(cameras_data):
            width = int(camera_info["width"])
            height = int(camera_info["height"])
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = np.asarray(camera_info["rotation"], dtype=np.float64)
            c2w[:3, 3] = np.asarray(camera_info["position"], dtype=np.float64)
            w2c = np.linalg.inv(c2w)
            camera = Cameras(
                R=torch.tensor(w2c[:3, :3], dtype=torch.float32).unsqueeze(0),
                T=torch.tensor(w2c[:3, 3], dtype=torch.float32).unsqueeze(0),
                fx=torch.tensor([float(camera_info["fx"])], dtype=torch.float32),
                fy=torch.tensor([float(camera_info["fy"])], dtype=torch.float32),
                cx=torch.tensor([float(camera_info["cx"])], dtype=torch.float32),
                cy=torch.tensor([float(camera_info["cy"])], dtype=torch.float32),
                width=torch.tensor([width], dtype=torch.int16),
                height=torch.tensor([height], dtype=torch.int16),
                appearance_id=torch.tensor([int(camera_info.get("appearance_id") or 0)], dtype=torch.long),
                normalized_appearance_id=torch.tensor([float(camera_info.get("normalized_appearance_id") or 0.0)], dtype=torch.float32),
                time=torch.tensor([float(camera_info.get("time") or 0.0)], dtype=torch.float32),
                distortion_params=None,
                camera_type=torch.zeros((1,), dtype=torch.int8),
            )[0].to_device(device)

            outputs = renderer(
                camera,
                gaussian_model,
                background,
                scaling_modifier=1.0,
                render_types=render_types,
            )
            rgb = outputs["render"].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            rgb_u8 = np.asarray((rgb * 255.0).round(), dtype=np.uint8, order="C")
            depth = np.asarray(outputs[depth_type].detach().squeeze().cpu().numpy(), dtype=np.float32, order="C")
            valid = np.isfinite(depth) & (depth > 0.0) & (depth <= depth_trunc)
            if has_alpha and outputs.get("alpha") is not None:
                alpha = outputs["alpha"].detach().squeeze().cpu().numpy()
                valid &= alpha >= alpha_threshold
            if not np.any(valid):
                frame_logs.append({"index": index, "integrated": False})
                continue

            filtered_depth = np.where(valid, depth, 0.0).astype(np.float32, copy=False)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_u8),
                o3d.geometry.Image(np.asarray(filtered_depth, dtype=np.float32, order="C")),
                depth_scale=1.0,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=width,
                height=height,
                fx=float(camera_info["fx"]),
                fy=float(camera_info["fy"]),
                cx=float(camera_info["cx"]),
                cy=float(camera_info["cy"]),
            )
            volume.integrate(rgbd, intrinsic, w2c)
            used_views += 1
            frame_logs.append({"index": index, "integrated": True})

    if used_views == 0:
        raise RuntimeError("no valid depth views were integrated")

    mesh = volume.extract_triangle_mesh()
    original_vertices = int(np.asarray(mesh.vertices).shape[0])
    original_triangles = int(np.asarray(mesh.triangles).shape[0])
    if clean_mesh:
        mesh = cleanup_mesh(mesh)

    component_filter = {
        "enabled": bool(min_component_triangles > 0 or keep_components > 0),
        "min_component_triangles": min_component_triangles,
        "keep_components": keep_components,
        "component_count": 0,
        "kept_components": [],
        "removed_triangles": 0,
    }
    if component_filter["enabled"] and int(np.asarray(mesh.triangles).shape[0]) > 0:
        triangle_clusters, cluster_n_triangles, _cluster_area = mesh.cluster_connected_triangles()
        cluster_counts = np.asarray(cluster_n_triangles, dtype=np.int64)
        component_filter["component_count"] = int(cluster_counts.shape[0])
        if cluster_counts.shape[0] > 0:
            sorted_clusters = np.argsort(-cluster_counts)
            kept_clusters = []
            for cluster_id in sorted_clusters:
                cluster_id = int(cluster_id)
                if min_component_triangles > 0 and int(cluster_counts[cluster_id]) < min_component_triangles:
                    continue
                kept_clusters.append(cluster_id)
                if keep_components > 0 and len(kept_clusters) >= keep_components:
                    break
            if not kept_clusters:
                kept_clusters = [int(sorted_clusters[0])]

            triangle_cluster_ids = np.asarray(triangle_clusters, dtype=np.int64)
            remove_mask = ~np.isin(triangle_cluster_ids, np.asarray(kept_clusters, dtype=np.int64))
            removed_triangles = int(remove_mask.sum())
            if removed_triangles > 0:
                mesh.remove_triangles_by_mask(remove_mask.tolist())
                mesh.remove_unreferenced_vertices()
            component_filter["kept_components"] = [
                {"id": int(cluster_id), "triangles": int(cluster_counts[cluster_id])}
                for cluster_id in kept_clusters
            ]
            component_filter["removed_triangles"] = removed_triangles

    triangles_after_component_filter = int(np.asarray(mesh.triangles).shape[0])
    decimation = {
        "enabled": decimate_enabled,
        "target_triangles": target_triangles,
        "applied": False,
        "input_triangles": triangles_after_component_filter,
        "output_triangles": triangles_after_component_filter,
    }
    if decimate_enabled and target_triangles > 0 and triangles_after_component_filter > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        decimation["applied"] = True
        decimation["output_triangles"] = int(np.asarray(mesh.triangles).shape[0])

    if clean_mesh:
        mesh = cleanup_mesh(mesh)
    mesh.compute_vertex_normals()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output_path), mesh):
        raise RuntimeError(f"failed to write mesh: {output_path}")

    vertices = int(np.asarray(mesh.vertices).shape[0])
    triangles = int(np.asarray(mesh.triangles).shape[0])
    if decimation["applied"]:
        decimation["output_triangles"] = triangles
    manifest = {
        "mesh_path": str(output_path),
        "vertices": vertices,
        "triangles": triangles,
        "used_views": used_views,
        "skipped_views": len(cameras_data) - used_views,
        "voxel_size": voxel_size,
        "sdf_trunc": sdf_trunc,
        "depth_trunc": depth_trunc,
        "alpha_threshold": alpha_threshold,
        "mesh_resolution": mesh_resolution,
        "max_views": max_views,
        "frame_count": len(cameras_data),
        "depth_type": depth_type,
        "alpha": has_alpha,
        "clean_mesh": clean_mesh,
        "original_vertices": original_vertices,
        "original_triangles": original_triangles,
        "component_filter": component_filter,
        "decimation": decimation,
        "frames": frame_logs,
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, **manifest, "manifest": str(output_manifest_path)}
