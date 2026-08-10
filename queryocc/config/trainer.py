from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union

import yaml

from queryocc.data.joint_datamodule import JointDataModuleConfig
from queryocc.model.base import ModelConfig
from queryocc.viz.visualizer import VisualizerConfig


@dataclass
class TrainerConfig:
    accumulate_grad_batches: int = 1
    """The number of batches to accumulate before performing a backward pass."""

    val_check_interval: Union[int, float] = 0.5
    """The validation check interval."""

    limit_val_batches: Union[int, float] = 160
    """The number of validation batches."""

    limit_test_batches: Union[int, float] = 1.0
    """The number of test batches."""

    check_val_every_n_epoch: int = 1
    """The number of training epochs between validation."""

    num_sanity_val_steps: int = 2
    """Number of sanity validation steps to run before training. Set to 0 to disable."""

    max_steps: int = 1_000_000
    """The maximum number of steps."""

    strategy: str = "ddp"  # ddp_find_unused_parameters_true"
    """The strategy for distributed training."""

    precision: Optional[str] = "bf16-mixed"
    """The precision to use (None, 16-mixed)."""

    devices: int = 1
    """The number of devices to use."""

    num_nodes: int = 1
    """The number of nodes."""

    sync_batchnorm: bool = True
    """Sync batchnorm."""

    overfit_batches: int = 0
    """ The number of batches to overfit on."""

    gradient_clip_val: float = 2.5
    gradient_clip_algorithm: str = "norm"
    """Gradient clipping."""

    profiler: Union[str, None] = None
    """The profiler to use (simple, advanced, pytorch, xla)."""

    detect_anomaly: bool = False
    """Detect anomalies."""

    deterministic: bool = False
    """Deterministic training."""

    use_distributed_sampler: bool = True
    """Replace the sampler with a distributed sampler."""

    ddp_static_graph: Optional[bool] = None
    """Override the DDP static_graph setting. None = auto (True only when accumulate_grad_batches == 1).
    Set False to disable it explicitly, e.g. to test whether static_graph is the cause of a divergence.
    Not a real Lightning Trainer arg — popped from trainer_kwargs in train.py."""

    enable_model_summary: bool = False
    """Enable model summary."""


@dataclass
class TrainingConfig:
    out_dir: Path = Path("/outputs")
    """The output directory."""
    exp_name: str = "unnamed"
    """The experiment name."""
    wandb_group: Optional[str] = None
    """W&B group = the research track this run belongs to. Defaulted
    per config module — the file a preset lives in decides its track (see config/methods/__init__.py
    WANDB_TRACK_BY_MODULE); override per preset or on the CLI. None on an unmapped module falls back
    to the legacy derived name (<model-class>-<roi>[-<task>])."""
    debug: bool = False
    """Debug mode."""
    seed: int = 1337
    """The one seed to rule them all."""
    train: bool = True
    """Train the model."""
    test: bool = True
    """Test the model."""
    test_after_fit_ckpt_path: Literal["best", "last"] = "last"
    """The checkpoint to test after training."""

    monitor_key: str = "val/total_loss"
    """The key to monitor for early stopping."""
    monitor_patience: int = -1
    """The patience for early stopping. Set to -1 for no early stopping."""
    monitor_mode: str = "min"
    """The mode for early stopping."""
    monitor_strict: bool = True
    """If we want to strictly monitor the metric or not."""

    scale_lr_with_batch_size: bool = False
    """Include the per-device batch size in the cluster scaling rule (train.py:scale_with_num_gpus):
    max_lr is multiplied and max_steps divided by gpus x nodes x accumulation x per-device batch size
    instead of only gpus x nodes x accumulation. This makes max_lr scale linearly with the total
    effective batch and makes max_steps mean "total training samples", so runs with different
    per-device batch sizes train on the same data budget with a consistently scaled LR. Off by
    default: legacy presets with batch_size > 1 tuned their max_lr/max_steps under the old rule."""

    scale_with_num_gpus: bool = True
    """Apply the multi-GPU cluster scaling rule (train.py:scale_with_num_gpus): batch_size and max_lr
    multiplied, max_steps divided, by the device/node/accumulation counts. On by default (the historical
    behavior). Presets that set batch_size/max_lr/max_steps as the LITERAL global values for a fixed
    multi-GPU shape (the ``*_mg`` recipes) set this False so they run exactly as written — otherwise the
    rule silently re-scales them (e.g. bs12 -> 12/device instead of the intended 3/device -> OOM)."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    """The trainer config"""
    datamodule: JointDataModuleConfig = field(default_factory=JointDataModuleConfig)
    """The dataset configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    """The model configuration."""
    preset_name: Optional[str] = None
    """The preset this config was registered under, stamped by config/methods/_registry.py.
    Not meant to be set by hand — it identifies the run's architecture when loading released
    weights, and names the file fetched by a `hf://<owner>/<repo>` checkpoint spec."""
    load_ckpt_from: Optional[Path] = None
    """Weights to load: a checkpoint file, a run directory (see load_ckpt_from_best), or a
    released checkpoint on the Hugging Face Hub as `hf://<owner>/<repo>[/<file>]` (the file
    defaults to `<preset>.pth`)."""
    load_ckpt_from_best: bool = False
    """If we want to load the best checkpoint if load_ckpt_from is a directory. Else load last."""
    resume_training: bool = False
    """If we want to resume training from the checkpoint. I.e., load optimizers etc. from the checkpoint. Has to be coupled with `load_ckpt_from`."""
    save_weights_only: bool = False
    """If we want to save only the weights of the model in a .pth file."""

    # logging
    use_logging: bool = True
    """Use logging."""
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    """The timestamp for the experiment."""
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)
    """The visualizer configuration."""
    allow_visualization: bool = True
    """Allow visualization."""

    validate_config: bool = True
    """Check the loaded checkpoint's sibling config.yml against this run's config. Only possible
    for a checkpoint still sitting in its training directory; automatically skipped for a released
    checkpoint, which carries its own preset stamp instead. Turning this off does NOT relax how
    strictly the weights are loaded — that is `allow_partial_load`."""
    allow_partial_load: bool = False
    """Let weights load with missing/unexpected keys instead of failing. Off by default: a silent
    partial load leaves part of the network at its random initialization, which looks like a
    trained model and scores like noise. Only turn it on when the mismatch is intended (e.g.
    initializing from a checkpoint with a different head)."""
    visualization_mode: bool = False

    disable_gradient_checkpointing_on_large_node: bool = True
    """Disable gradient checkpointing if on large node."""

    def get_base_dir(self) -> Path:
        """Get the base directory."""
        path = self.out_dir / "experiments" / self.exp_name / self.timestamp
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_config(self) -> None:
        """Save config to base directory"""
        base_dir = self.get_base_dir()
        assert base_dir is not None
        base_dir.mkdir(parents=True, exist_ok=True)
        config_yaml_path = base_dir / "config.yml"
        config_yaml_path.write_text(yaml.dump(self), "utf8")

    @classmethod
    def load_config_from_yaml(cls, config_yaml_path: Path) -> "TrainingConfig":
        """Load config from yaml file."""
        return yaml.load(config_yaml_path.read_text(), Loader=yaml.Loader)
