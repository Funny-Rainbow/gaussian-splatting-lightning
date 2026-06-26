# CoMe Branch Deployment

This branch carries the GSPL-side CoMe validation files used by the SaaS experiments:

- `configs/come_geometry.yaml`
- `internal/metrics/come_geometry.py`
- `internal/entrypoints/come_tsdf_mesh_extraction.py`
- `internal/entrypoints/come_tets_mesh_extraction.py`

It is intended for local validation and controlled SaaS worker experiments. It is not a complete control-plane or frontend feature by itself.

## Local Checkout

Use the existing local GSPL environment first:

```bash
cd /home/ubuntu/3dgs_deps/gaussian-splatting-lightning
git fetch origin
git checkout come
git pull --ff-only origin come
```

Expected local runtime:

```text
/home/ubuntu/3dgs_deps/gaussian-splatting-lightning
/home/ubuntu/miniconda3/envs/gspl/bin/python
/home/ubuntu/miniconda3/envs/colmap-build/bin/colmap
```

Verify the CoMe metric module imports and compiles:

```bash
/home/ubuntu/miniconda3/envs/gspl/bin/python -m py_compile \
  internal/metrics/come_geometry.py
```

## CoMe Geometry Training

Train on a COLMAP dataset with the CoMe geometry losses enabled:

```bash
cd /home/ubuntu/3dgs_deps/gaussian-splatting-lightning
/home/ubuntu/miniconda3/envs/gspl/bin/python main.py fit \
  --trainer.check_val_every_n_epoch 99999 \
  --trainer.max_steps 30000 \
  --config configs/bilagrid_fused.yaml \
  --config configs/gsplat_v1_SaaS.yaml \
  --config configs/gns.yaml \
  --config configs/come_geometry.yaml \
  --model.density.budget 200000 \
  --model.save_ply true \
  --data.parser Colmap \
  --data.parser.reorient True \
  --data.path /path/to/colmap_dataset \
  --data.image_uint8 true \
  -n come-validation
```

For background-removal datasets, add the background-removal config and mask options:

```bash
/home/ubuntu/miniconda3/envs/gspl/bin/python main.py fit \
  --trainer.check_val_every_n_epoch 99999 \
  --trainer.max_steps 30000 \
  --config configs/background_removal_w_loop.yaml \
  --config configs/bilagrid_fused.yaml \
  --config configs/gsplat_v1_SaaS.yaml \
  --config configs/gns.yaml \
  --config configs/come_geometry.yaml \
  --model.density.budget 100000 \
  --model.save_ply true \
  --data.parser Colmap \
  --data.parser.reorient True \
  --data.parser.mask_dir /path/to/undistorted_masks \
  --data.allow_mask_interpolation true \
  --data.mask_interpolation_mode nearest \
  --data.path /path/to/undistorted_dataset \
  --data.image_uint8 true \
  -n come-bg-validation \
  -v bkg_removal-bilagrid
```

## Mesh Extraction

Extract a TSDF mesh from a trained model:

```bash
cd /home/ubuntu/3dgs_deps/gaussian-splatting-lightning
/home/ubuntu/miniconda3/envs/gspl/bin/python -m internal.entrypoints.come_tsdf_mesh_extraction \
  --model-path outputs/come-validation \
  --dataset-path /path/to/colmap_dataset \
  --mesh-resolution 512 \
  --max-views 0
```

The default output is:

```text
<model-path>/come_mesh/mesh.ply
<model-path>/come_mesh/mesh.json
```

For masked datasets:

```bash
/home/ubuntu/miniconda3/envs/gspl/bin/python -m internal.entrypoints.come_tsdf_mesh_extraction \
  --model-path outputs/come-bg-validation/bkg_removal-bilagrid \
  --dataset-path /path/to/undistorted_dataset \
  --mask-dir /path/to/undistorted_masks \
  --mesh-resolution 512
```

The experimental tets backend is also available:

```bash
/home/ubuntu/miniconda3/envs/gspl/bin/python -m internal.entrypoints.come_tets_mesh_extraction \
  --model-path outputs/come-bg-validation/bkg_removal-bilagrid \
  --dataset-path /path/to/undistorted_dataset \
  --max-gaussians 12000 \
  --k-neighbors 16 \
  --binary-steps 6 \
  --opacity-level 0.5 \
  --triangulation come
```

## SaaS Wrapper Sync

If deploying from the SaaS wrapper repository instead of checking out this branch directly, sync the override package:

```bash
cd /home/ubuntu/3dgs_gspl_come
git checkout come
bash scripts/sync-external-repo-overrides.sh
```

The wrapper sync path overwrites only the CoMe validation files listed at the top of this document.

## Rollback

To leave the CoMe branch and return to the SaaS baseline:

```bash
cd /home/ubuntu/3dgs_deps/gaussian-splatting-lightning
git checkout saas
git status --short
```

If files were copied manually into another branch, restore them from git before switching workflows:

```bash
git checkout -- configs/come_geometry.yaml \
  internal/metrics/come_geometry.py \
  internal/entrypoints/come_tsdf_mesh_extraction.py \
  internal/entrypoints/come_tets_mesh_extraction.py
```
