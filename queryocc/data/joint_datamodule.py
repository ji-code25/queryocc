import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Type

import cv2
import numpy as np
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

import queryocc.config.base_config as cfg
from queryocc.data.datasets.argoverse2 import AV2DatasetConfig
from queryocc.data.datasets.nuscenes_dataset import NuScenesDatasetConfig
from queryocc.data.datasets.zod_base import ZODBaseDatasetConfig, new_base_collate_fn
from queryocc.data.utils.distributed_weighted_sampler import DistributedWeightedSampler, is_ddp_active

logger = logging.getLogger(__name__)


def _worker_init_fn(_worker_id: int) -> None:
    import torch

    torch.set_num_threads(1)
    cv2.setNumThreads(0)


@dataclass
class JointDataModuleConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: JointDataModule)

    av2_data: Path = Path("/datasets/av2")
    """The path to av2 dataset."""
    nuscenes_data: Path = Path("/datasets/nuscenes")
    """The path to nuscenes dataset."""

    version: Literal["full", "mini"] = "mini"
    """What version of the dataset to use."""
    splits: Literal["original", "geo_near"] = "original"
    """ The splits to use. Only applicable for Argoverse2 and nuScenes datasets."""

    debug: bool = False
    """If to run overfitting on one sample"""
    debug_sample_idx: int = 0
    """Which train sample to overfit on in debug mode (useful when only some samples have local labels)."""

    # dataloader parameters
    batch_size: int = 1
    """The batch size."""
    val_batch_size: Optional[int] = None
    """Per-device batch size for the validation/test dataloaders; None follows the train batch size.
    Validation peak memory scales with this (e.g. the occ3d metrics decode 640k queries per sample),
    so large-train-batch runs can validate at batch size 1 without affecting training throughput.
    Unlike batch_size, this value is used as-is (no division by world size)."""
    num_workers: int = 16
    """The number of train-dataloader workers (per rank)."""
    val_num_workers: int = 2
    """Workers (per rank) for the val/test loaders, kept small ON PURPOSE. Non-persistent val
    workers are not always reaped promptly (Lightning doesn't reliably trigger the iterator GC),
    so a leftover val pool can coexist with the persistent train pool. With train num_workers x
    ranks already near the core count, a full second val pool (num_workers x ranks) tips the node
    into a system-CPU context-switch storm that starves training and only gets worse over the run.
    Bounding val workers caps that leaked pool so total processes stay under the core count even
    if reaping fails. Val is infrequent (limit_val_batches) so the throughput cost is negligible."""
    pin_memory: bool = True
    """Whether to pin memory."""
    persistent_workers: bool = True
    """Keep dataloader workers alive across epochs and validation rounds; avoids paying
    worker startup + dataset re-initialization at every val_check_interval."""
    prefetch_factor: int = 2
    """The prefetch factor."""
    shuffle_test: bool = False
    """Whether to shuffle the test dataloader."""
    shuffle_val: bool = False
    """Shuffle validation deterministically by training epoch. With a fractional or fixed
    ``limit_val_batches``, this draws a new reproducible subset at each validation event."""
    balanced_dataset_sampling: bool = False
    """Whether to balance the dataset sampling."""
    num_samples_per_rank: Optional[int] = None
    """The number of samples per rank per epoch when using multiple datasets."""

    dataset: ZODBaseDatasetConfig = field(default_factory=ZODBaseDatasetConfig)
    """The dataset configuration."""
    datasets_to_include: Tuple[str, ...] = ("nuscenes",)
    """The datasets to include."""

    train_dataset_usage_ratio: float = 1.0  # should > 0 and <= 1
    """The ratio of the training dataset to use."""

    train_dataset_usage_num: Optional[int] = None
    """The number of samples to use from the training dataset. -1 means all samples."""

    run_test_on_enitre_dataset: bool = False
    """If to run test on the entire dataset."""

    val_on_sensor_only: bool = False
    """If to validate only on sensor."""

    val_on_nuscenes_only: bool = False
    """If to validate only on nuscenes."""

    min_dataset_length: int = 1000

    # Extra dataset configs per dataset
    av2_cfg: AV2DatasetConfig = field(default_factory=AV2DatasetConfig)
    """Dataset configs for Argoverse2"""

    def __post_init__(self) -> None:
        self.dataset.version = self.version
        assert self.train_dataset_usage_ratio > 0 and self.train_dataset_usage_ratio <= 1, (
            "train_dataset_usage_ratio must be > 0 and <= 1"
        )
        if self.train_dataset_usage_ratio < 1.0:
            assert self.train_dataset_usage_num is None, (
                "train_dataset_usage_num must be None when train_dataset_usage_ratio is set"
            )
        if self.train_dataset_usage_num is not None:
            # should not be used at the same time
            assert self.train_dataset_usage_ratio == 1.0, (
                "train_dataset_usage_ratio must be 1.0 when train_dataset_usage_num is set"
            )


class JointDataModule(LightningDataModule):
    """`LightningDataModule` for the ZOD dataset."""

    config: JointDataModuleConfig

    def __init__(self, config: JointDataModuleConfig) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.config = config
        self.dataset_name = "autonomous_driving"

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.batch_size_per_device = self.config.batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def _dataset_configs(self) -> Tuple[ZODBaseDatasetConfig, ZODBaseDatasetConfig]:
        """Return isolated train/eval configs with eval supervision removed.

        ``PreTrainConfig.configure_dataset`` attaches the private policy flag;
        keeping it private prevents a second, conflicting CLI switch on the
        datamodule.  Deep copies are essential because VFM reader settings are
        nested mutable dataclasses.
        """
        train_config = deepcopy(self.config.dataset)
        eval_config = deepcopy(self.config.dataset)

        if getattr(eval_config, "_occ3d_only_eval", False):
            eval_config.load_vfm_stuff = False
            eval_config.load_depth_maps = False
            eval_config.load_offset_pointclouds = False
            eval_config.load_offset_panoptic_labels = False
            eval_config.load_current_lidar_panoptic_labels = False
            eval_config.add_missing_points = False

            eval_config.vfm_cfg.read_metric3d_depth = False
            eval_config.vfm_cfg.read_grounding_sam = False
            eval_config.vfm_cfg.read_grounding_sam_scores = False
            eval_config.vfm_cfg.create_pseudo_pointclouds = False

        return train_config, eval_config

    @staticmethod
    def _dataset_kwargs(config: ZODBaseDatasetConfig) -> dict:
        # Exclude runtime-only policy attributes such as
        # ``_occ3d_only_eval`` before constructing the concrete dataset config.
        return {key: value for key, value in config.__dict__.items() if not key.startswith("_")}

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.config.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.config.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.config.batch_size // self.trainer.world_size

        # The VFM/pseudo pointcloud arrays are variable-length per sample and can only be stacked
        # by the default collate when padded to fixed sizes, so padding is required when any
        # dataloader uses a per-device batch size > 1 (see VFMReaderConfig.pad_pseudo_data_for_batching).
        if max(self.batch_size_per_device, self.config.val_batch_size or 1) > 1:
            ds_cfg = self.config.dataset
            ds_cfg.vfm_cfg.pad_pseudo_data_for_batching = True
            ds_cfg.vfm_cfg.pad_to_num_timesteps = (
                ds_cfg.num_future_frames_to_sample
                + ds_cfg.num_past_frames_to_sample
                + (1 if ds_cfg.always_add_current_frame_to_offsets else 0)
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            train_datasets = []
            val_datasets = []
            train_config, eval_config = self._dataset_configs()

            for dataset_name in self.config.datasets_to_include:
                if dataset_name == "sensor":
                    train_dataset_cfg = self._dataset_kwargs(train_config)
                    eval_dataset_cfg = self._dataset_kwargs(eval_config)
                    av2_cfg = self.config.av2_cfg.__dict__.copy()
                    train_dataset_cfg = {**av2_cfg, **train_dataset_cfg}
                    eval_dataset_cfg = {**av2_cfg, **eval_dataset_cfg}
                    train_dataset_cfg["data"] = self.config.av2_data
                    eval_dataset_cfg["data"] = self.config.av2_data
                    train_dataset_cfg["subset"] = "sensor"
                    eval_dataset_cfg["subset"] = "sensor"
                    dataset_train = AV2DatasetConfig(**train_dataset_cfg).setup(
                        train_or_val="train", split=f"sensor_{self.config.splits}"
                    )
                    dataset_val = AV2DatasetConfig(**eval_dataset_cfg).setup(
                        train_or_val="val", split=f"sensor_{self.config.splits}"
                    )
                    train_datasets.append(dataset_train)
                    val_datasets.append(dataset_val)

                elif "nuscenes" in dataset_name:
                    train_dataset_cfg = self._dataset_kwargs(train_config)
                    eval_dataset_cfg = self._dataset_kwargs(eval_config)
                    train_dataset_cfg["data"] = self.config.nuscenes_data
                    eval_dataset_cfg["data"] = self.config.nuscenes_data

                    if "only_tokens" in dataset_name:
                        train_dataset_cfg["only_tokens"] = True
                        eval_dataset_cfg["only_tokens"] = True

                    dataset_train = NuScenesDatasetConfig(**train_dataset_cfg).setup(
                        train_or_val="train", split=self.config.splits
                    )
                    dataset_val = NuScenesDatasetConfig(**eval_dataset_cfg).setup(
                        train_or_val="val", split=self.config.splits
                    )
                    train_datasets.append(dataset_train)
                    val_datasets.append(dataset_val)

                else:
                    raise ValueError(f"Dataset {dataset_name} not supported")

            if self.config.debug:
                idx = self.config.debug_sample_idx
                self.data_train = Subset(dataset_train, [idx] * 100)
                self.data_val = Subset(dataset_train, [idx, idx])
                self.data_test = Subset(dataset_train, [idx, idx])
                return

            if len(train_datasets) > 1:
                print(f"Concatenating {len(train_datasets)} train datasets")
                for ds, name in zip(train_datasets, self.config.datasets_to_include):
                    print(f"Dataset {name} has {len(ds)} samples")
                if self.config.balanced_dataset_sampling:
                    print("Balancing dataset sampling is active")

            self.data_train = ConcatDataset(train_datasets)

            if self.config.train_dataset_usage_ratio < 1.0:
                train_size = int(len(self.data_train) * self.config.train_dataset_usage_ratio)
                self.data_train = Subset(self.data_train, list(range(train_size)))

            if self.config.train_dataset_usage_num is not None:
                train_size = self.config.train_dataset_usage_num
                if train_size == -1 or train_size > len(self.data_train):
                    idx_ = list(range(len(self.data_train)))
                else:
                    idx_ = np.random.choice(len(self.data_train), train_size, replace=False).tolist()
                    # make replicas so we get roughly 10_000 samples
                    if train_size < self.config.min_dataset_length:
                        idx_ = idx_ * (self.config.min_dataset_length // train_size + 1)
                        idx_ = idx_[: self.config.min_dataset_length]
                self.data_train = Subset(self.data_train, idx_)

            if self.config.val_on_sensor_only and "sensor" in self.config.datasets_to_include:
                idx = self.config.datasets_to_include.index("sensor")
                self.data_val = val_datasets[idx]

            elif self.config.val_on_nuscenes_only and "nuscenes" in self.config.datasets_to_include:
                idx = self.config.datasets_to_include.index("nuscenes")
                self.data_val = val_datasets[idx]

            else:
                self.data_val = ConcatDataset(val_datasets)

            self.data_test = self.data_val

            if self.config.run_test_on_enitre_dataset:
                self.data_test = ConcatDataset(val_datasets + train_datasets)

    def set_epoch_for_sampler(self, epoch: int):
        if hasattr(self, "train_sampler") and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(epoch)

    def _dataloader_kwargs(
        self,
        batch_size: Optional[int] = None,
        persistent_workers: Optional[bool] = None,
        num_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Shared DataLoader arguments. Worker-only options are omitted for num_workers=0.

        persistent_workers defaults to the config; pass False for val/test so their worker pool is
        torn down after each cycle. num_workers defaults to the config; val/test pass the smaller
        val_num_workers. Both matter because non-persistent val workers are not always reaped, so a
        leftover val pool (num_workers x ranks) can coexist with the persistent train pool; keeping
        the val count small bounds that leak so the node's cores aren't oversubscribed into a
        system-CPU context-switch storm that starves the train pipeline post-val.
        """
        n_workers = self.config.num_workers if num_workers is None else num_workers
        kwargs: Dict[str, Any] = {
            "batch_size": batch_size if batch_size is not None else self.batch_size_per_device,
            "num_workers": n_workers,
            "pin_memory": self.config.pin_memory,
            "collate_fn": new_base_collate_fn,
        }
        if n_workers > 0:
            kwargs["persistent_workers"] = (
                self.config.persistent_workers if persistent_workers is None else persistent_workers
            )
            kwargs["prefetch_factor"] = self.config.prefetch_factor
            kwargs["worker_init_fn"] = _worker_init_fn
        return kwargs

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        assert self.data_train is not None, "Data not loaded. Run `setup()` first."

        if self.config.balanced_dataset_sampling and isinstance(self.data_train, ConcatDataset):
            if len(self.data_train.datasets) > 1:
                # balance the dataset sampling
                num_samples = [len(dataset) for dataset in self.data_train.datasets]
                num_samples = np.array(num_samples)
                weights = 1 / num_samples
                weights = weights / weights.sum()
                weight_per_sample = [[w] * ns for ns, w in zip(num_samples, weights)]
                weight_per_sample = list(np.concatenate(weight_per_sample))
                num_samples = self.config.num_samples_per_rank if self.config.num_samples_per_rank is not None else None
                sampler = DistributedWeightedSampler(
                    weights=weight_per_sample, num_samples=num_samples, replacement=True, seed=42
                )
                self.train_sampler = sampler
                shuffle = None
        elif is_ddp_active():
            print("DDP Distribution Sampler is used.")
            sampler = DistributedSampler(self.data_train, shuffle=True, drop_last=False)
            shuffle = False
        else:
            print("Default sampler is used.")
            sampler = None
            shuffle = True

        return DataLoader(
            dataset=self.data_train,
            shuffle=shuffle,
            sampler=sampler,
            # Drop the final partial batch: at batch_size > 1 the epoch-end odd batch changes the
            # activation shapes, which desyncs the compiled/static-graph DDP ranks into different
            # collectives (deterministic NCCL deadlock at the epoch boundary, seen on both clusters
            # at the same step). Costs at most world_size x batch_size samples per epoch.
            drop_last=True,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        assert self.data_val is not None, "Data not loaded. Run `setup()` first."
        if is_ddp_active():
            sampler_val = DistributedSampler(self.data_val, shuffle=self.config.shuffle_val, drop_last=False)
        else:
            sampler_val = None

        return DataLoader(
            dataset=self.data_val,
            shuffle=self.config.shuffle_val if sampler_val is None else False,
            sampler=sampler_val,
            # Non-persistent + few workers: a leftover val pool (reaping is unreliable) must stay
            # small so it can't oversubscribe the cores alongside the train pool (see _dataloader_kwargs).
            **self._dataloader_kwargs(
                batch_size=self.config.val_batch_size,
                persistent_workers=False,
                num_workers=self.config.val_num_workers,
            ),
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        assert self.data_test is not None, "Data not loaded. Run `setup()` first."
        if is_ddp_active():
            sampler_test = DistributedSampler(self.data_test, shuffle=False, drop_last=False)
        else:
            sampler_test = None

        return DataLoader(
            dataset=self.data_test,
            shuffle=self.config.shuffle_test,
            sampler=sampler_test,
            **self._dataloader_kwargs(
                batch_size=self.config.val_batch_size,
                persistent_workers=False,
                num_workers=self.config.val_num_workers,
            ),
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
