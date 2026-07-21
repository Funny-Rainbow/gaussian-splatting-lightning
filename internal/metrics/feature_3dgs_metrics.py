from dataclasses import dataclass
from typing import Tuple, Dict, Any, Literal
import torch
import torch.nn.functional as F

from .metric import Metric, MetricImpl


@dataclass
class Feature3DGSMetrics(Metric):
    feature_prior_path: str = ""
    feature_prior_weight: float = 0.0
    feature_prior_loss_type: Literal["l1", "l2", "cosine"] = "l1"
    feature_prior_sample_size: int = 0
    feature_prior_seed: int = 0

    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return Feature3DGSMetricImpl(self)


class Feature3DGSMetricImpl(MetricImpl):
    def setup(self, stage: str, pl_module):
        self.feature_prior = None
        self.feature_prior_generator = None
        if not self.config.feature_prior_path or self.config.feature_prior_weight <= 0:
            return

        loaded_prior = torch.load(self.config.feature_prior_path, map_location=pl_module.device)
        if isinstance(loaded_prior, dict):
            loaded_prior = loaded_prior.get("features", loaded_prior.get("renderer.features"))
        if loaded_prior is None:
            raise ValueError(f"No feature tensor found in feature_prior_path={self.config.feature_prior_path}")
        loaded_prior = loaded_prior.to(device=pl_module.device, dtype=torch.float32)
        expected_shape = tuple(pl_module.renderer.features.shape)
        if tuple(loaded_prior.shape) != expected_shape:
            raise ValueError(
                f"feature_prior_path shape mismatch: got {tuple(loaded_prior.shape)}, expected {expected_shape}"
            )
        self.feature_prior = loaded_prior
        self.feature_prior_generator = torch.Generator(device=pl_module.device)
        self.feature_prior_generator.manual_seed(int(self.config.feature_prior_seed))

    @staticmethod
    def _move_channel_first(tensor: torch.Tensor, channels: int) -> torch.Tensor:
        if tensor.ndim != 3 or tensor.shape[0] == channels:
            return tensor
        for dim, size in enumerate(tensor.shape):
            if size == channels:
                order = [dim] + [i for i in range(tensor.ndim) if i != dim]
                return tensor.permute(*order).contiguous()
        return tensor

    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs) -> Tuple[Dict[str, float], Dict[str, bool]]:
        metrics = {}
        metrics_pbar = {}

        _, _, gt_feature_map = batch

        feature_map = outputs["features"]
        if gt_feature_map.ndim == 4 and gt_feature_map.shape[0] == 1:
            gt_feature_map = gt_feature_map.squeeze(0)
        if feature_map.ndim == 4 and feature_map.shape[0] == 1:
            feature_map = feature_map.squeeze(0)
        feature_map = self._move_channel_first(feature_map, pl_module.renderer.n_feature_dims)
        gt_feature_map = self._move_channel_first(gt_feature_map, feature_map.shape[0])
        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0)

        l1_loss = torch.abs((feature_map - gt_feature_map)).mean()

        metrics["loss"] = l1_loss
        metrics_pbar["loss"] = True

        return metrics, metrics_pbar

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, metrics_pbar = super().get_train_metrics(
            pl_module=pl_module,
            gaussian_model=gaussian_model,
            step=step,
            batch=batch,
            outputs=outputs,
        )
        if self.feature_prior is None or self.config.feature_prior_weight <= 0:
            return metrics, metrics_pbar

        features = pl_module.renderer.features
        prior = self.feature_prior
        sample_size = int(self.config.feature_prior_sample_size)
        if sample_size > 0 and sample_size < features.shape[0]:
            ids = torch.randint(
                low=0,
                high=features.shape[0],
                size=(sample_size,),
                device=features.device,
                generator=self.feature_prior_generator,
            )
            features = features[ids]
            prior = prior[ids]

        if self.config.feature_prior_loss_type == "l1":
            prior_loss = torch.abs(features - prior).mean()
        elif self.config.feature_prior_loss_type == "l2":
            prior_loss = ((features - prior) ** 2).mean()
        elif self.config.feature_prior_loss_type == "cosine":
            prior_loss = (1.0 - F.cosine_similarity(features, prior, dim=-1)).mean()
        else:
            raise ValueError(f"Unsupported feature_prior_loss_type={self.config.feature_prior_loss_type}")

        prior_loss = prior_loss * float(self.config.feature_prior_weight)
        metrics["loss"] = metrics["loss"] + prior_loss
        metrics["feature_prior"] = prior_loss
        metrics_pbar["feature_prior"] = True
        return metrics, metrics_pbar
