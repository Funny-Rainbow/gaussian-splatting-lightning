from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import trimesh
from scipy.spatial import Delaunay, cKDTree
from tqdm.auto import tqdm

from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.general_utils import build_rotation


_CORNERS = torch.tensor(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=torch.float32,
)


def _load_marching_tetrahedra() -> Callable:
    come_root = Path(os.environ.get("COME_ROOT", "/home/ubuntu/3dgs_deps/CoMe"))
    tetmesh_path = come_root / "utils" / "tetmesh.py"
    if not tetmesh_path.is_file():
        raise RuntimeError(
            f"CoMe tetmesh.py not found at {tetmesh_path}; set COME_ROOT to a CoMe checkout"
        )
    spec = importlib.util.spec_from_file_location("come_tetmesh", tetmesh_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load CoMe tetmesh module from {tetmesh_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.marching_tetrahedra


def _try_come_triangulate(points: torch.Tensor) -> torch.Tensor | None:
    come_root = Path(os.environ.get("COME_ROOT", "/home/ubuntu/3dgs_deps/CoMe"))
    tetra_path = come_root / "submodules" / "tetra-triangulation"
    if tetra_path.is_dir():
        import sys

        sys.path.insert(0, str(tetra_path))
    try:
        from tetranerf.utils.extension import cpp  # type: ignore

        return cpp.triangulate(points)
    except Exception as exc:
        print(f"CoMe tetra-triangulation extension unavailable, using scipy fallback: {exc}", flush=True)
        return None


def _get_gaussian_tensors(model):
    xyz = model.get_xyz if hasattr(model, "get_xyz") else model.get_means()
    scales = model.get_scaling if hasattr(model, "get_scaling") else model.get_scales()
    rotations = model.get_rotation if hasattr(model, "get_rotation") else model.get_rotations()
    opacities = model.get_opacity if hasattr(model, "get_opacity") else model.get_opacities()
    if opacities.ndim == 2:
        opacities = opacities[:, 0]
    return xyz.detach(), scales.detach(), rotations.detach(), opacities.detach()


def _select_gaussians(xyz, scales, rotations, opacities, args):
    mask = opacities > args.opacity_cutoff_tetra
    xyz = xyz[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    opacities = opacities[mask]
    if xyz.shape[0] == 0:
        raise RuntimeError("no Gaussians survived opacity cutoff")

    if args.max_gaussians > 0 and xyz.shape[0] > args.max_gaussians:
        score = opacities * scales.max(dim=-1).values
        indices = torch.topk(score, k=args.max_gaussians, largest=True).indices
        xyz = xyz[indices]
        scales = scales[indices]
        rotations = rotations[indices]
        opacities = opacities[indices]
    return xyz, scales, rotations, opacities


def _bounded_scales(scales, opacities, mode: str):
    if mode == "sigma3":
        return scales * 3.0
    if mode == "sigma333":
        return scales * 3.33
    if mode != "stp":
        raise ValueError(f"unsupported bounding mode: {mode}")
    safe = torch.clamp(255.0 * opacities[:, None], min=1.000001)
    return scales * torch.sqrt(2.0 * torch.log(safe))


def _generate_tetra_points(xyz, scales, rotations, opacities, args):
    bounded_scales = _bounded_scales(scales, opacities, args.bounding_mode)
    rotation_mats = build_rotation(rotations)
    corners = _CORNERS.to(device=xyz.device, dtype=xyz.dtype)
    vertices = corners[None, :, :] * bounded_scales[:, None, :]
    vertices = torch.bmm(rotation_mats, vertices.transpose(1, 2)).transpose(1, 2)
    vertices = vertices + xyz[:, None, :]
    vertices = vertices.reshape(-1, 3).contiguous()
    points = torch.cat([vertices, xyz], dim=0)

    scale = bounded_scales.max(dim=-1, keepdim=True).values
    points_scale = torch.cat([scale.repeat(1, 8).reshape(-1, 1), scale], dim=0)
    return points, points_scale


def _triangulate(points: torch.Tensor, args) -> tuple[torch.Tensor, str]:
    cells = None
    actual_backend = args.triangulation
    if args.triangulation == "come":
        cells = _try_come_triangulate(points)
        if cells is None:
            actual_backend = "scipy"
    elif args.triangulation != "scipy":
        raise ValueError(f"unsupported triangulation backend: {args.triangulation}")

    if cells is None:
        points_np = points.detach().cpu().numpy()
        cells_np = Delaunay(points_np, qhull_options="QJ").simplices
        cells = torch.from_numpy(np.asarray(cells_np, dtype=np.int64))
    return cells.to(device=points.device, dtype=torch.long), actual_backend


def _evaluate_opacity_field(
    query_points: torch.Tensor,
    xyz: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    args,
) -> torch.Tensor:
    device = query_points.device
    xyz_np = xyz.detach().cpu().numpy()
    tree = cKDTree(xyz_np)
    query_np = query_points.detach().cpu().numpy()
    k = min(args.k_neighbors, xyz.shape[0])

    out = torch.empty((query_points.shape[0],), dtype=torch.float32, device=device)
    rotation_mats = build_rotation(rotations)
    inv_scales = torch.clamp(scales, min=args.min_scale).reciprocal()

    for start in tqdm(range(0, query_points.shape[0], args.query_chunk), desc="CoMe tets opacity"):
        end = min(start + args.query_chunk, query_points.shape[0])
        _, nn_idx = tree.query(query_np[start:end], k=k, workers=-1)
        if k == 1:
            nn_idx = nn_idx[:, None]
        nn_idx_t = torch.as_tensor(nn_idx, device=device, dtype=torch.long)
        q = query_points[start:end]
        centers = xyz[nn_idx_t]
        local = q[:, None, :] - centers
        rots = rotation_mats[nn_idx_t].transpose(-1, -2)
        local = torch.einsum("bkjl,bkl->bkj", rots, local)
        normed = local * inv_scales[nn_idx_t]
        power = -0.5 * (normed * normed).sum(dim=-1)
        contrib = opacities[nn_idx_t] * torch.exp(torch.clamp(power, min=-60.0, max=0.0))
        alpha = 1.0 - torch.exp(-contrib.sum(dim=-1))
        out[start:end] = alpha.clamp_(0.0, 1.0)
    return out


def _extract_tets_mesh(args, model):
    marching_tetrahedra = _load_marching_tetrahedra()
    xyz, scales, rotations, opacities = _get_gaussian_tensors(model)
    xyz, scales, rotations, opacities = _select_gaussians(xyz, scales, rotations, opacities, args)
    points, points_scale = _generate_tetra_points(xyz, scales, rotations, opacities, args)
    cells, triangulation_backend = _triangulate(points, args)

    alpha = _evaluate_opacity_field(points, xyz, scales, rotations, opacities, args)
    sdf = (alpha - args.opacity_level)[None]
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(
        points[None],
        cells,
        sdf,
        points_scale[None],
    )
    end_points, end_sdf = verts_list[0]
    faces = faces_list[0].detach().cpu().numpy()

    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    mid_points = (left_points + right_points) * 0.5

    for _ in range(args.binary_steps):
        alpha = _evaluate_opacity_field(mid_points, xyz, scales, rotations, opacities, args)
        mid_sdf = (alpha - args.opacity_level).reshape(-1, 1)
        same_side = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))
        left_sdf[same_side] = mid_sdf[same_side]
        right_sdf[~same_side] = mid_sdf[~same_side]
        flat = same_side.flatten()
        left_points[flat] = mid_points[flat]
        right_points[~flat] = mid_points[~flat]
        mid_points = (left_points + right_points) * 0.5

    mesh = trimesh.Trimesh(vertices=mid_points.detach().cpu().numpy(), faces=faces, process=False)
    stats = {
        "selected_gaussians": int(xyz.shape[0]),
        "tetra_points": int(points.shape[0]),
        "tetra_cells": int(cells.shape[0]),
        "opacity_level": float(args.opacity_level),
        "opacity_cutoff_tetra": float(args.opacity_cutoff_tetra),
        "k_neighbors": int(args.k_neighbors),
        "binary_steps": int(args.binary_steps),
        "bounding_mode": args.bounding_mode,
        "triangulation": triangulation_backend,
        "requested_triangulation": args.triangulation,
    }
    return mesh, stats


def _default_output_path(model_path: str) -> Path:
    path = Path(model_path)
    root = path.parent if path.is_file() else path
    return root / "come_mesh" / "mesh_tets.ply"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract a CoMe-style tets mesh from a GSPL model.")
    parser.add_argument("--model-path", required=True, help="GSPL output directory, checkpoint, or PLY path.")
    parser.add_argument("--dataset-path", help="Accepted for parity with TSDF backend; not used by tets backend.")
    parser.add_argument("--output", help="Output mesh PLY path. Defaults to <model>/come_mesh/mesh_tets.ply.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-gaussians", type=int, default=12000)
    parser.add_argument("--k-neighbors", type=int, default=16)
    parser.add_argument("--query-chunk", type=int, default=8192)
    parser.add_argument("--binary-steps", type=int, default=6)
    parser.add_argument("--opacity-level", type=float, default=0.5)
    parser.add_argument("--opacity-cutoff-tetra", type=float, default=0.0039)
    parser.add_argument("--bounding-mode", choices=("stp", "sigma3", "sigma333"), default="stp")
    parser.add_argument("--triangulation", choices=("come", "scipy"), default="come")
    parser.add_argument("--min-scale", type=float, default=1e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    output_path = Path(args.output) if args.output else _default_output_path(args.model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loadable = GaussianModelLoader.search_load_file(args.model_path)
    if loadable.endswith(".ckpt"):
        model, _, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
            loadable,
            device=device,
            eval_mode=True,
            pre_activate=True,
        )
    elif loadable.endswith(".ply"):
        model, _ = GaussianModelLoader.initialize_model_and_renderer_from_ply_file(
            loadable,
            device=device,
            eval_mode=True,
            pre_activate=True,
        )
    else:
        raise ValueError(f"unsupported model file: {loadable}")

    model.freeze()
    model.eval()
    mesh, stats = _extract_tets_mesh(args, model)
    mesh.export(output_path)

    manifest = {
        "model_path": args.model_path,
        "loadable_model": loadable,
        "dataset_path": args.dataset_path,
        "mesh_backend": "tets",
        "mesh_path": str(output_path),
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        **stats,
    }
    output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
