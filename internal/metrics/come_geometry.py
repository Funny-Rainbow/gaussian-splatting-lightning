from dataclasses import dataclass

import torch
import torch.nn.functional as F
from gsplat.utils import depth_to_normal

from .vanilla_metrics import VanillaMetrics, VanillaMetricsImpl


@dataclass
class ComeGeometryMetrics(VanillaMetrics):
    depth_normal_from_iter: int = 0
    depth_normal_lambda: float = 0.05
    depth_normal_alpha_threshold: float = 0.05
    flatten_lambda: float = 0.01
    alpha_entropy_from_iter: int = 0
    alpha_entropy_lambda: float = 0.0
    background_depth_from_iter: int = 7000
    background_depth_lambda: float = 0.1
    background_depth_key: str = "hard_inverse_depth"
    background_depth_dilate_pixels: int = 0
    background_depth_clamp_max: float = 0.0

    def instantiate(self, *args, **kwargs):
        return ComeGeometryMetricsModule(self)


class ComeGeometryMetricsModule(VanillaMetricsImpl):
    def _depth_normal_loss(self, outputs):
        exp_depth = outputs.get("exp_depth")
        normal = outputs.get("normal")
        alpha = outputs.get("alpha")
        preprocessed_camera = outputs.get("preprocessed_camera")
        if exp_depth is None or normal is None or alpha is None or preprocessed_camera is None:
            return None

        w2c, K, _ = preprocessed_camera
        normals_from_depth = depth_to_normal(
            exp_depth.detach().permute(1, 2, 0),
            torch.linalg.inv(w2c[0]),
            K[0],
        ).permute(2, 0, 1)
        alpha_weight = alpha.squeeze(0).detach().clamp(0.0, 1.0)
        valid = alpha_weight > self.config.depth_normal_alpha_threshold
        if not torch.any(valid):
            return None

        normal = F.normalize(normal, dim=0)
        normals_from_depth = F.normalize(normals_from_depth, dim=0)
        cosine_error = 1.0 - (normal * normals_from_depth).sum(dim=0).clamp(-1.0, 1.0)
        return (cosine_error * alpha_weight)[valid].mean()

    def _flatten_loss(self, gaussian_model):
        scales = gaussian_model.get_scales()
        if scales.shape[-1] < 1:
            return None
        return scales[..., -1].mean()

    def _alpha_entropy_loss(self, outputs):
        alpha = outputs.get("alpha")
        if alpha is None:
            return None
        alpha = alpha.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(alpha * alpha.log() + (1.0 - alpha) * (1.0 - alpha).log())
        return entropy.mean()

    def _background_depth_loss(self, batch, outputs):
        depth = outputs.get(self.config.background_depth_key)
        if depth is None:
            return None
        mask = batch[1][2]
        if mask is None:
            return None
        foreground = mask[:1, ...].to(torch.bool)
        dilate_pixels = int(self.config.background_depth_dilate_pixels)
        if dilate_pixels > 0:
            kernel_size = dilate_pixels * 2 + 1
            foreground = F.max_pool2d(
                foreground.to(depth.dtype).unsqueeze(0),
                kernel_size=kernel_size,
                stride=1,
                padding=dilate_pixels,
            ).squeeze(0).to(torch.bool)

        background = torch.logical_not(foreground)
        valid = background & torch.isfinite(depth) & (depth > 0)
        if not torch.any(valid):
            return None
        valid_depth = depth[valid]
        if self.config.background_depth_clamp_max > 0:
            valid_depth = valid_depth.clamp_max(self.config.background_depth_clamp_max)
        return valid_depth.mean()

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs):
        metrics, pbar = super().get_train_metrics(
            pl_module=pl_module,
            gaussian_model=gaussian_model,
            step=step,
            batch=batch,
            outputs=outputs,
        )

        if step >= self.config.depth_normal_from_iter and self.config.depth_normal_lambda > 0:
            depth_normal = self._depth_normal_loss(outputs)
            if depth_normal is not None:
                loss = depth_normal * self.config.depth_normal_lambda
                metrics["loss"] = metrics["loss"] + loss
                metrics["come_depth_normal"] = loss
                pbar["come_depth_normal"] = False

        if self.config.flatten_lambda > 0:
            flatten = self._flatten_loss(gaussian_model)
            if flatten is not None:
                loss = flatten * self.config.flatten_lambda
                metrics["loss"] = metrics["loss"] + loss
                metrics["come_flatten"] = loss
                pbar["come_flatten"] = False

        if step >= self.config.alpha_entropy_from_iter and self.config.alpha_entropy_lambda > 0:
            alpha_entropy = self._alpha_entropy_loss(outputs)
            if alpha_entropy is not None:
                loss = alpha_entropy * self.config.alpha_entropy_lambda
                metrics["loss"] = metrics["loss"] + loss
                metrics["come_alpha_entropy"] = loss
                pbar["come_alpha_entropy"] = False

        if step >= self.config.background_depth_from_iter and self.config.background_depth_lambda > 0:
            background_depth = self._background_depth_loss(batch, outputs)
            if background_depth is not None:
                loss = background_depth * self.config.background_depth_lambda
                metrics["loss"] = metrics["loss"] + loss
                metrics["come_background_depth"] = loss
                pbar["come_background_depth"] = False

        return metrics, pbar
