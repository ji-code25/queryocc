import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import lightning as L
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT

from queryocc.config.base_config import InstantiateConfig
from queryocc.utils.fetchers import fetch, fetch_label_mapper, fetch_model
from queryocc.viz.plotters import (
    get_bev_features_figure,
    get_probe_figure_3d_semantics_splat_bev,
    get_probe_figure_occupancy_splat_bev,
)

logger = logging.getLogger(__name__)


@dataclass
class VisualizerConfig(InstantiateConfig):
    _target: type = field(default_factory=lambda: Visualizer)
    name: str = "viz"
    epoch_interval: int = 4

    train_batch_interval: int = 10_000
    val_batch_interval: int = 200
    test_batch_interval: int = 1_000

    f_viz_bev_features: bool = False
    """ PCA-reduce the BEV feature map to RGB and log it as an image. """
    f_viz_occ_3d_bev: bool = False
    """ Occ3D predictions vs ground truth as BEV splats (semantic + binary), with the lidar overlaid. """


class WandbGeneralCallback(L.Callback):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.name = config.name
        self.epoch_interval = config.epoch_interval
        self.train_batch_interval = config.train_batch_interval
        self.val_batch_interval = config.val_batch_interval
        self.test_batch_interval = config.test_batch_interval

    def _get_figure(self, batch, outputs, trainer, batch_idx, prefix):
        raise NotImplementedError()

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        *args,
        **kwargs,
    ) -> None:
        if trainer.current_epoch % self.epoch_interval == 0:
            if batch_idx % self.train_batch_interval == 0:
                self._visualize_batch(batch, outputs, trainer, batch_idx, f"train/{self.name}")

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        *args,
        **kwargs,
    ) -> None:
        if trainer.current_epoch % self.epoch_interval == 0:
            if batch_idx % self.val_batch_interval == 0:
                self._visualize_batch(batch, outputs, trainer, batch_idx, f"val/{self.name}")

    @rank_zero_only
    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        *args,
        **kwargs,
    ) -> None:
        if batch_idx % self.test_batch_interval == 0:
            self._visualize_batch(batch, outputs, trainer, batch_idx, f"test/{self.name}")

    def _visualize_batch(self, batch, outputs, trainer: L.Trainer, batch_idx: int, prefix: str):
        figure = self._get_figure(batch, outputs, trainer, batch_idx, prefix)
        self._log_plotly_figures(figure, trainer)

    def _log_plotly_figures(self, figure, trainer):
        trainer.logger.experiment.log(figure)


class Visualizer(WandbGeneralCallback):
    config = VisualizerConfig

    def __init__(
        self,
        config: VisualizerConfig,
    ):
        super().__init__(config)
        self.config = config

    def _get_bev_features_figure(self, batch, outputs, trainer, batch_idx, prefix, **kwargs) -> dict:
        figures = {}
        if self.config.f_viz_bev_features and "info" in outputs:
            info = outputs["info"]
            bev_features = fetch(info, None, "bev_features")
            if bev_features is not None:
                try:
                    figure = get_bev_features_figure(bev_features, color_by="PCA")
                    figures[f"{prefix}/bev_features/pca"] = figure
                except Exception as e:
                    logger.error(f"Could not visualize bev features: {e}")

        return figures

    def _get_occ_3d_bev_figure(self, batch, outputs, trainer, batch_idx, prefix, **kwargs) -> dict:
        figures = {}
        if self.config.f_viz_occ_3d_bev and "occ3d_labels" in batch:
            try:
                occ3d_labels = fetch(batch, None, "occ3d_labels")
                occ3d_semantic_preds = fetch(outputs, None, "occ3d_semantic_preds")
                occ3d_binary_preds = fetch(outputs, None, "occ3d_binary_preds")

                model = fetch_model(trainer)
                occ3d_queries = model._create_occ3d_queries()[:, :3]

                label_mapper = fetch_label_mapper(trainer)

                occ3d_binary_target = occ3d_labels != label_mapper.free_idx
                occ3d_num_samples_per_dim = (200, 200, 16)
                occ3d_probe_range = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)
                lidars = fetch(batch, None, "lidars")
                if occ3d_semantic_preds is not None:
                    occ3d_labels_colors = label_mapper.learning_to_color(occ3d_labels)
                    occ3d_semantic_preds_colors = label_mapper.learning_to_color(occ3d_semantic_preds)

                    occ3d_labels_occupied = occ3d_labels != label_mapper.free_idx
                    occ3d_semantic_preds_occupied = occ3d_semantic_preds != label_mapper.free_idx

                    gt_figure = get_probe_figure_3d_semantics_splat_bev(
                        occ3d_queries,
                        occ3d_labels_occupied,
                        occ3d_labels_colors,
                        occ3d_probe_range,
                        occ3d_num_samples_per_dim,
                        pcd=lidars,
                    )
                    figures[f"{prefix}/occ_3d_bev/labels"] = gt_figure

                    pred_figure = get_probe_figure_3d_semantics_splat_bev(
                        occ3d_queries,
                        occ3d_semantic_preds_occupied,
                        occ3d_semantic_preds_colors,
                        occ3d_probe_range,
                        occ3d_num_samples_per_dim,
                        pcd=lidars,
                    )
                    figures[f"{prefix}/occ_3d_bev/preds"] = pred_figure

                if occ3d_binary_preds is not None:
                    pred_figure = get_probe_figure_occupancy_splat_bev(
                        occ3d_queries,
                        occ3d_binary_preds.bool(),
                        occ3d_probe_range,
                        occ3d_num_samples_per_dim,
                        pcd=lidars,
                    )
                    figures[f"{prefix}/occ_3d_bev/preds_binary"] = pred_figure

                    target_figure = get_probe_figure_occupancy_splat_bev(
                        occ3d_queries,
                        occ3d_binary_target.bool(),
                        occ3d_probe_range,
                        occ3d_num_samples_per_dim,
                        pcd=lidars,
                    )
                    figures[f"{prefix}/occ_3d_bev/labels_binary"] = target_figure

            except Exception as e:
                logger.error(f"Could not visualize 3D-occ bev: {e}")
        return figures

    def _get_figure(self, batch, outputs, trainer, batch_idx, prefix, **kwargs):
        bev_features_figure = self._get_bev_features_figure(batch, outputs, trainer, batch_idx, prefix, **kwargs)
        occ_3d_bev_figure = self._get_occ_3d_bev_figure(batch, outputs, trainer, batch_idx, prefix, **kwargs)

        return {
            **bev_features_figure,
            **occ_3d_bev_figure,
        }
