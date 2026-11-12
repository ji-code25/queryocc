from types import MethodType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
lightning = pytest.importorskip("lightning")

from queryocc.config.trainer import TrainingConfig  # noqa: E402
from queryocc.data.datasets.zod_base import ZODBaseDatasetConfig  # noqa: E402
from queryocc.data.joint_datamodule import JointDataModule, JointDataModuleConfig  # noqa: E402
from queryocc.model.pre_train import PreTrainConfig, PreTrainModel  # noqa: E402
from queryocc.train import build_checkpoint_callbacks, write_evaluation_results  # noqa: E402


def _configured_dataset():
    dataset = ZODBaseDatasetConfig()
    dataset.load_image = True
    dataset.load_vfm_stuff = True
    dataset.load_depth_maps = True
    dataset.load_offset_pointclouds = True
    dataset.load_offset_panoptic_labels = True
    dataset.load_current_lidar_panoptic_labels = True
    dataset.add_missing_points = True
    dataset.load_occ3d_val = True
    dataset.vfm_cfg.read_metric3d_depth = True
    dataset.vfm_cfg.read_grounding_sam = True
    dataset.vfm_cfg.read_grounding_sam_scores = True
    dataset.vfm_cfg.create_pseudo_pointclouds = True
    dataset._occ3d_only_eval = True
    return dataset


def test_occ3d_only_eval_keeps_train_inputs_and_strips_eval_supervision():
    module = JointDataModule(JointDataModuleConfig(dataset=_configured_dataset()))
    train_config, eval_config = module._dataset_configs()

    assert train_config.load_image
    assert train_config.load_vfm_stuff
    assert train_config.load_depth_maps
    assert train_config.load_offset_pointclouds
    assert train_config.load_offset_panoptic_labels
    assert train_config.vfm_cfg.read_metric3d_depth
    assert train_config.vfm_cfg.read_grounding_sam
    assert train_config.vfm_cfg.create_pseudo_pointclouds

    assert eval_config.load_image
    assert eval_config.load_occ3d_val
    assert not eval_config.load_vfm_stuff
    assert not eval_config.load_depth_maps
    assert not eval_config.load_offset_pointclouds
    assert not eval_config.load_offset_panoptic_labels
    assert not eval_config.load_current_lidar_panoptic_labels
    assert not eval_config.add_missing_points
    assert not eval_config.vfm_cfg.read_metric3d_depth
    assert not eval_config.vfm_cfg.read_grounding_sam
    assert not eval_config.vfm_cfg.read_grounding_sam_scores
    assert not eval_config.vfm_cfg.create_pseudo_pointclouds

    assert train_config is not eval_config
    assert train_config.vfm_cfg is not eval_config.vfm_cfg
    assert "_occ3d_only_eval" not in module._dataset_kwargs(eval_config)


def test_occ3d_only_requires_occ3d_metrics():
    with pytest.raises(ValueError, match="requires compute_occ3d_metrics"):
        PreTrainConfig(occ3d_only_eval=True, compute_occ3d_metrics=False)


def _minimal_model(call_log):
    model = object.__new__(PreTrainModel)
    lightning.LightningModule.__init__(model)
    model.config = SimpleNamespace(
        freeze_encoder=False,
        compute_occ3d_metrics=True,
        occ3d_only_eval=True,
        logging=SimpleNamespace(bev_features=False),
        net=SimpleNamespace(
            image_encoder_name="cerveza",
            cerveza_encoder=SimpleNamespace(lss=SimpleNamespace(supervise_depth=True)),
        ),
    )
    model.net = SimpleNamespace(config=SimpleNamespace(use_one_head=True))
    model.visualization_mode = False

    def encode(self, _batch):
        call_log.append("encode")
        return torch.zeros((1, 1), dtype=torch.float32), {"img_depths": torch.zeros((1, 1, 1, 1, 1))}

    def occ3d(self, _batch, _bev_feats, result):
        call_log.append("occ3d")
        result["occ3d_binary_preds"] = torch.zeros((1, 1), dtype=torch.int32)
        return result

    def forbidden_depth(self, *_args, **_kwargs):
        raise AssertionError("evaluation called the depth-supervision branch")

    def forbidden_pseudo(self, *_args, **_kwargs):
        raise AssertionError("evaluation called the pseudo-supervision branch")

    def metrics(self, _batch, _result, split):
        call_log.append(f"metrics:{split}")

    def forbidden_val_log(self, *_args, **_kwargs):
        raise AssertionError("Occ3D-only evaluation logged pseudo/depth validation loss")

    model._encode = MethodType(encode, model)
    model._step_occ3d = MethodType(occ3d, model)
    model._step_img_depth = MethodType(forbidden_depth, model)
    model._step_one_head = MethodType(forbidden_pseudo, model)
    model._compute_occ3d_metrics = MethodType(metrics, model)
    model._log_val = MethodType(forbidden_val_log, model)
    return model


def test_occ3d_only_validation_never_calls_pseudo_or_depth(monkeypatch):
    calls = []
    model = _minimal_model(calls)
    monkeypatch.setattr(PreTrainModel, "_do_occ3d_step", property(lambda _self: True))

    # Deliberately omit all pseudo-label and depth targets.  Images/calibration
    # are represented by the mocked encoder; only Occ3D targets reach metrics.
    result = model.validation_step({"id": ["sample"], "occ3d_labels": torch.ones(1)}, 0)

    assert "occ3d_binary_preds" in result
    assert calls == ["encode", "occ3d", "metrics:val"]


def test_training_step_path_still_enables_supervision(monkeypatch):
    calls = []
    model = _minimal_model(calls)
    monkeypatch.setattr(PreTrainModel, "_do_occ3d_step", property(lambda _self: False))

    def depth(self, _batch, _img_depths, result):
        calls.append("depth")
        return result

    def pseudo(self, _batch, _bev_feats, result):
        calls.append("pseudo")
        return result

    model._step_img_depth = MethodType(depth, model)
    model._step_one_head = MethodType(pseudo, model)
    model.step({"id": ["sample"], "depth_maps": {}}, 0)

    assert calls == ["encode", "depth", "pseudo"]


def test_best_and_epoch_recovery_checkpoints_are_independent():
    config = TrainingConfig(monitor_key="val/occ3d/score", monitor_mode="max")
    best, recovery, scheduled = build_checkpoint_callbacks(config)

    assert best is not None
    assert best.monitor == "val/occ3d/score"
    assert best.save_top_k == 1
    assert not best.save_last
    assert recovery.monitor is None
    assert recovery.save_top_k == 0
    assert recovery.save_last
    assert recovery._every_n_epochs == 1
    assert recovery._save_on_train_epoch_end is True
    assert best.state_key != recovery.state_key
    assert scheduled is None


def test_no_validation_uses_scheduled_and_recovery_checkpoints_only():
    config = TrainingConfig(checkpoint_epochs=(10, 11, 12), monitor_key="val/score")
    config.trainer.limit_val_batches = 0
    best, recovery, scheduled = build_checkpoint_callbacks(config)

    assert best is None
    assert recovery.save_last
    assert scheduled.epochs == frozenset((10, 11, 12))


def test_recovery_checkpoint_is_written_before_the_next_epoch(tmp_path):
    class ToyModel(lightning.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def training_step(self, batch, _batch_idx):
            return (self.weight * batch[0]).sum()

        def validation_step(self, _batch, _batch_idx):
            self.log("val/score", self.weight.detach(), on_epoch=True)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

        def train_dataloader(self):
            return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.ones(2, 1)), batch_size=1)

        def val_dataloader(self):
            return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.ones(1, 1)), batch_size=1)

    best, recovery, scheduled = build_checkpoint_callbacks(
        TrainingConfig(
            checkpoint_epochs=(1, 2),
            monitor_key="val/score",
            monitor_mode="max",
        )
    )

    class RecoveryAudit(lightning.Callback):
        recovered_global_step = None

        def on_train_epoch_start(self, trainer, _module):
            if trainer.current_epoch == 1:
                checkpoint = torch.load(recovery.last_model_path, map_location="cpu", weights_only=False)
                self.recovered_global_step = checkpoint["global_step"]

    audit = RecoveryAudit()
    trainer = lightning.Trainer(
        max_epochs=2,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        callbacks=[best, scheduled, recovery, audit],
    )
    trainer.fit(ToyModel())

    assert audit.recovered_global_step == 2
    assert torch.load(recovery.last_model_path, map_location="cpu", weights_only=False)["global_step"] == 4
    assert (tmp_path / "checkpoints/epoch-01.ckpt").is_file()
    assert (tmp_path / "checkpoints/epoch-02.ckpt").is_file()


def test_standalone_evaluation_results_are_machine_readable(tmp_path):
    checkpoint = tmp_path / "epoch-10.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "epoch-10.json"
    config = TrainingConfig(
        load_ckpt_from=checkpoint,
        evaluation_results_path=output,
        evaluation_checkpoint_epoch=10,
    )

    write_evaluation_results(config, [{"test/metric": torch.tensor(1.25)}])

    import json

    payload = json.loads(output.read_text())
    assert payload["checkpoint_epoch"] == 10
    assert payload["checkpoint_path"] == str(checkpoint.resolve())
    assert payload["metrics"]["test/metric"] == 1.25
