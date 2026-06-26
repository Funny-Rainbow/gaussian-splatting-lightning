from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
import torch
from tqdm.auto import tqdm

from internal.utils.gaussian_model_loader import GaussianModelLoader


def _load_model_and_renderer(model_path: str, device: torch.device):
    loadable = GaussianModelLoader.search_load_file(model_path)
    dataparser_config = None
    dataset_path = None

    if loadable.endswith(".ckpt"):
        model, renderer, checkpoint = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
            loadable,
            device=device,
            eval_mode=True,
            pre_activate=True,
        )
        datamodule_hparams = checkpoint.get("datamodule_hyper_parameters", {})
        dataparser_config = datamodule_hparams.get("parser")
        dataset_path = datamodule_hparams.get("path")
    elif loadable.endswith(".ply"):
        model, renderer = GaussianModelLoader.initialize_model_and_renderer_from_ply_file(
            loadable,
            device=device,
            eval_mode=True,
            pre_activate=True,
        )
    else:
        raise ValueError(f"unsupported model file: {loadable}")

    model.freeze()
    renderer.eval()
    return loadable, model, renderer, dataparser_config, dataset_path


def _load_cameras(dataset_path: str, output_path: Path, dataparser_config, mask_dir: str | None, device: torch.device):
    if dataparser_config is None:
        from internal.dataparsers.colmap_dataparser import Colmap
        dataparser_config = Colmap()
    if mask_dir and hasattr(dataparser_config, "mask_dir"):
        dataparser_config.mask_dir = mask_dir

    outputs = dataparser_config.instantiate(
        path=dataset_path,
        output_path=str(output_path),
        global_rank=0,
    ).get_outputs()
    image_set = outputs.train_set
    cameras = [camera.to_device(device) for camera in image_set.cameras]
    return image_set, cameras


def _render_view(renderer, model, camera, bg_color: torch.Tensor):
    try:
        outputs = renderer(
            camera,
            model,
            bg_color=bg_color,
            render_types=["rgb", "alpha", "exp_depth"],
        )
        rgb = outputs["render"]
        depth = outputs.get("exp_depth")
        alpha = outputs.get("alpha")
        if depth is None:
            depth = outputs.get("depth")
        if depth is not None:
            return rgb, depth[:1], alpha
    except Exception:
        pass

    rgb_outputs = renderer(camera, model, bg_color=bg_color, render_types=["rgb"])
    depth_outputs = renderer(camera, model, bg_color=torch.zeros_like(bg_color), render_types=["depth"])
    rgb = rgb_outputs["render"]
    depth = depth_outputs.get("depth")
    if depth is None:
        raise RuntimeError("renderer did not return a usable depth map")
    return rgb, depth[:1], None


def _load_mask(mask_path: str | None, size: tuple[int, int]) -> np.ndarray | None:
    if not mask_path:
        return None
    path = Path(mask_path)
    if not path.is_file():
        return None
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != size:
            image = image.resize(size, getattr(Image, "Resampling", Image).NEAREST)
        return np.asarray(image) >= 128


def _camera_intrinsic(camera) -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(
        width=int(camera.width.item()),
        height=int(camera.height.item()),
        fx=float(camera.fx.item()),
        fy=float(camera.fy.item()),
        cx=float(camera.cx.item()),
        cy=float(camera.cy.item()),
    )


def _integrate_tsdf(args, model, renderer, image_set, cameras, device: torch.device):
    depth_trunc = args.depth_trunc
    voxel_size = args.voxel_size if args.voxel_size > 0 else depth_trunc / args.mesh_resolution
    sdf_trunc = args.sdf_trunc if args.sdf_trunc > 0 else 5.0 * voxel_size

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    bg_color = torch.zeros((3,), dtype=torch.float32, device=device)
    view_count = len(cameras) if args.max_views <= 0 else min(len(cameras), args.max_views)

    used_views = 0
    for index, camera in tqdm(list(enumerate(cameras[:view_count])), desc="CoMe TSDF views"):
        with torch.no_grad():
            rgb, depth, alpha = _render_view(renderer, model, camera, bg_color)

        rgb_np = np.asarray(
            np.clip(rgb.detach().permute(1, 2, 0).cpu().numpy(), 0.0, 1.0) * 255.0,
            dtype=np.uint8,
            order="C",
        )
        depth_np = np.asarray(depth.detach().squeeze(0).cpu().numpy(), dtype=np.float32, order="C")

        if alpha is not None:
            alpha_np = alpha.detach().squeeze(0).cpu().numpy()
            depth_np[alpha_np < args.alpha_threshold] = 0.0

        mask = _load_mask(image_set.mask_paths[index], (rgb_np.shape[1], rgb_np.shape[0]))
        if mask is not None:
            depth_np[~mask] = 0.0

        depth_np[~np.isfinite(depth_np)] = 0.0
        depth_np[depth_np <= 0.0] = 0.0
        if not np.any(depth_np > 0.0):
            continue

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_np),
            o3d.geometry.Image(depth_np),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(
            rgbd,
            _camera_intrinsic(camera),
            np.asarray(camera.world_to_camera.T.detach().cpu().numpy()),
        )
        used_views += 1

    if used_views == 0:
        raise RuntimeError("no valid depth views were integrated")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh, {
        "used_views": used_views,
        "voxel_size": voxel_size,
        "sdf_trunc": sdf_trunc,
        "depth_trunc": depth_trunc,
        "alpha_threshold": args.alpha_threshold,
    }


def _default_output_path(model_path: str) -> Path:
    path = Path(model_path)
    if path.is_file():
        root = path.parent
    else:
        root = path
    return root / "come_mesh" / "mesh.ply"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract a CoMe-style TSDF mesh from a GSPL model.")
    parser.add_argument("--model-path", required=True, help="GSPL output directory, checkpoint, or PLY path.")
    parser.add_argument("--dataset-path", help="COLMAP dataset path. Required for PLY input; checkpoint input can infer it.")
    parser.add_argument("--output", help="Output mesh PLY path. Defaults to <model>/come_mesh/mesh.ply.")
    parser.add_argument("--mask-dir", help="Optional COLMAP mask directory; 0-valued pixels are ignored.")
    parser.add_argument("--mesh-resolution", type=int, default=512)
    parser.add_argument("--voxel-size", type=float, default=-1.0)
    parser.add_argument("--sdf-trunc", type=float, default=-1.0)
    parser.add_argument("--depth-trunc", type=float, default=6.0)
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    output_path = Path(args.output) if args.output else _default_output_path(args.model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loadable, model, renderer, dataparser_config, inferred_dataset_path = _load_model_and_renderer(args.model_path, device)
    dataset_path = args.dataset_path or inferred_dataset_path
    if not dataset_path:
        raise RuntimeError("--dataset-path is required when it cannot be inferred from checkpoint")

    image_set, cameras = _load_cameras(
        dataset_path=dataset_path,
        output_path=output_path.parent,
        dataparser_config=dataparser_config,
        mask_dir=args.mask_dir,
        device=device,
    )
    mesh, stats = _integrate_tsdf(args, model, renderer, image_set, cameras, device)
    o3d.io.write_triangle_mesh(str(output_path), mesh)

    manifest = {
        "model_path": args.model_path,
        "loadable_model": loadable,
        "dataset_path": dataset_path,
        "mesh_path": str(output_path),
        "vertices": int(np.asarray(mesh.vertices).shape[0]),
        "triangles": int(np.asarray(mesh.triangles).shape[0]),
        **stats,
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
