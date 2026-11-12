"""Train and evaluate QueryOcc.

Pick a method preset (e.g. [bright_yellow]queryocc-nusc[/bright_yellow]) and override any config
field from the CLI.

The most demonstrative and helpful example of the CLI structure is the difference in output between
the following commands:

    train.py -h
    train.py queryocc-nusc -h

In each of these examples, the -h applies to the previous subcommand (train.py and queryocc-nusc).

In the first example, we get the help menu for the train script.
In the second example, we get the help menu for the queryocc-nusc preset.

Your arguments apply to the preceding subcommand in your command, and thus where you put your
arguments matters:

    train.py (trainer args) queryocc-nusc (model args)
"""

import functools
import glob
import json
import logging
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import lightning as L
import lightning.pytorch.callbacks as cb
import torch
import tyro
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler, PyTorchProfiler
from lightning.pytorch.strategies import DDPStrategy

from queryocc.callbacks.checkpoint import (
    EpochRecoveryCheckpoint,
    FinalStepModelCheckpoint,
    ScheduledEpochCheckpoint,
    WeightsOnlyCheckpoint,
)
from queryocc.callbacks.sampler import ValidationEpochScheduler, ValidationSamplerEpochSetter
from queryocc.config.method_configs import AnnotatedBaseConfigUnion, convert_markup_to_ansi
from queryocc.config.trainer import TrainingConfig
from queryocc.data.joint_datamodule import JointDataModule
from queryocc.model.base import BaseModel
from queryocc.model.pre_train import PreTrainConfig
from queryocc.utils import cluster
from queryocc.utils.release import read_release_header, resolve_checkpoint_spec
from queryocc.viz.visualizer import Visualizer

logger = logging.getLogger(__name__)


def setup():
    cluster.set_ulimit()
    torch.set_float32_matmul_precision("medium")  # highest, high, medium
    # Image/BEV shapes are fixed (images are padded to a common size), so letting cudnn
    # autotune conv algorithms is a free win; only the first step per shape pays for it.
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and torch.cuda.get_device_capability() < (7, 0):
        # Triton (the torch.compile backend) requires compute capability >= 7.0. On older GPUs
        # (e.g. Pascal dev machines) we run all torch.compile() sites eagerly instead of crashing.
        logger.warning(
            "GPU compute capability %d.%d < 7.0 does not support Triton; "
            "running all torch.compile() sites in eager mode.",
            *torch.cuda.get_device_capability(),
        )
        torch.compiler.set_stance("force_eager")


def is_large_node():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory > 35e9

    return False


def validate_loaded_checkpoint_config(config: TrainingConfig):
    assert config.load_ckpt_from is not None and config.load_ckpt_from.exists(), "Checkpoint not found"
    config_path = find_checkpoint_config(config.load_ckpt_from)
    assert config_path.exists(), f"Config not found at {config_path}"
    loaded_config = TrainingConfig.load_config_from_yaml(config_path)

    assert loaded_config.model.net.image_encoder == config.model.net.image_encoder, "Image encoder mismatch"
    assert loaded_config.model.net.lidar_encoder == config.model.net.lidar_encoder, "Lidar encoder mismatch"

    assert loaded_config.model.roi == config.model.roi, "ROI mismatch"

    assert loaded_config.datamodule.splits == config.datamodule.splits, "Splits mismatch"

    ld = loaded_config.datamodule.dataset
    nd = config.datamodule.dataset

    if config.model.net.image_encoder:
        assert ld.n_prev_images == nd.n_prev_images, (
            f"n_prev_images mismatch. Expected {ld.n_prev_images} but got {nd.n_prev_images}"
        )
        assert ld.prev_images_max_time == nd.prev_images_max_time, (
            f"prev_images_max_time mismatch. Expected {ld.prev_images_max_time} but got {nd.prev_images_max_time}"
        )
        assert ld.image_size == nd.image_size, f"image_size mismatch. Expected {ld.image_size} but got {nd.image_size}"

    if config.model.net.lidar_encoder:
        assert ld.n_prev_lidar == nd.n_prev_lidar, (
            f"n_future_images mismatch. Expected {ld.n_prev_lidar} but got {nd.n_prev_lidar}"
        )
        assert ld.prev_lidar_max_time == nd.prev_lidar_max_time, (
            f"prev_lidar_max_time mismatch. Expected {ld.prev_lidar_max_time} but got {nd.prev_lidar_max_time}"
        )

    del loaded_config


def find_checkpoint_config(checkpoint_path: Path) -> Path:
    """Return the nearest ancestor config for an experiment checkpoint."""
    for parent in checkpoint_path.parents:
        candidate = parent / "config.yml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No config.yml found above checkpoint {checkpoint_path}")


def scale_with_num_gpus(config: TrainingConfig) -> None:
    keys_to_scale = (
        "datamodule.batch_size",
        "model.max_lr",
    )

    keys_to_divide = ("trainer.max_steps",)

    num_gpus = torch.cuda.device_count()
    num_nodes = config.trainer.num_nodes
    num_acc = config.trainer.accumulate_grad_batches
    # Read before the loop below multiplies datamodule.batch_size by the device count.
    per_device_batch_size = config.datamodule.batch_size if config.scale_lr_with_batch_size else 1

    if num_gpus <= 1 and num_nodes <= 1:
        return

    def rsetattr(obj, attr, val):
        pre, _, post = attr.rpartition(".")
        return setattr(rgetattr(obj, pre) if pre else obj, post, val)

    def rgetattr(obj, attr, *args):
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [obj] + attr.split("."))

    for key in keys_to_scale:
        rsetattr(config, key, rgetattr(config, key) * num_gpus * num_nodes)

    for key in keys_to_divide:
        value = rgetattr(config, key)
        # Lightning uses a negative max_steps value to disable the step limit in
        # epoch-limited runs. Preserve that sentinel instead of scaling it into a
        # negative float.
        if key == "trainer.max_steps" and value < 0:
            continue
        rsetattr(config, key, value / num_gpus / num_nodes / num_acc / per_device_batch_size)

    if "model.max_lr" in keys_to_scale:
        # With scale_lr_with_batch_size, the LR scales linearly with the total effective batch
        # (gpus x nodes x accumulation x per-device batch) and max_steps means total samples.
        rsetattr(config, "model.max_lr", rgetattr(config, "model.max_lr") * num_acc * per_device_batch_size)


def _clean_config_for_wandb(config: TrainingConfig):
    def __recursive_clean(atr):
        if isinstance(atr, dict):
            for k, v in atr.items():
                if hasattr(v, "__dict__"):
                    v = v.__dict__
                if k.startswith("_"):
                    continue
                atr[k] = __recursive_clean(v)
        else:
            if isinstance(atr, Path):
                atr = str(atr)

        return atr

    _config = deepcopy(config.__dict__)
    _config = __recursive_clean(_config)
    return _config


def get_wandb_logger(config: TrainingConfig, model_name: str) -> WandbLogger:
    # Group = research track: stamped per config module in
    # config/methods/__init__.py, overridable per preset or on the CLI (--wandb_group).
    wandb_group_name = config.wandb_group

    if wandb_group_name is None:
        # Legacy fallback for presets whose module has no track mapping: derived
        # <model-class>-<roi>.
        wandb_group_name = f"{model_name}-{config.model.roi}"

    if not config.train:
        wandb_group_name = "Eval-" + wandb_group_name

    if config.exp_name == "unnamed":
        config.exp_name = config.timestamp

    wandb_logger = WandbLogger(
        name=config.exp_name,
        project="queryocc",
        offline=config.debug,
        save_dir=config.get_base_dir(),
        config=_clean_config_for_wandb(config),
        group=wandb_group_name,
    )

    return wandb_logger


def get_checkpoint(checkpoint_folder: Path, best: bool = True) -> Path:
    # find the "last.ckpt" can be in any subfolder
    ckpts = glob.glob(str(checkpoint_folder / "**/*.ckpt"), recursive=True)

    last_ckpt = [c for c in ckpts if "last" in c][0]
    best_ckpt = [c for c in ckpts if "last" not in c][0]

    ckpt = best_ckpt if best else last_ckpt
    return Path(ckpt)


def build_checkpoint_callbacks(config: TrainingConfig):
    """Create independent metric-best, scheduled, and recovery checkpoints."""
    best_checkpoint_callback = None
    if config.trainer.limit_val_batches != 0:
        best_checkpoint_callback = FinalStepModelCheckpoint(
            monitor=config.monitor_key,
            save_top_k=1,
            mode=config.monitor_mode,
            save_last=False,
        )
    recovery_checkpoint_callback = EpochRecoveryCheckpoint(
        monitor=None,
        save_top_k=0,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    scheduled_checkpoint_callback = (
        ScheduledEpochCheckpoint(config.checkpoint_epochs) if config.checkpoint_epochs else None
    )
    return best_checkpoint_callback, recovery_checkpoint_callback, scheduled_checkpoint_callback


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_evaluation_results(config: TrainingConfig, results) -> None:
    """Atomically persist one standalone full-validation result."""
    if config.evaluation_results_path is None:
        return
    if config.load_ckpt_from is None:
        raise ValueError("evaluation_results_path requires load_ckpt_from")
    if len(results) != 1:
        raise RuntimeError(f"Expected one test dataloader result, got {len(results)}")

    output_path = config.evaluation_results_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "checkpoint_path": str(config.load_ckpt_from.resolve()),
        "checkpoint_epoch": config.evaluation_checkpoint_epoch,
        "metrics": _json_safe(results[0]),
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)


def check_release_checkpoint(header: dict, config: TrainingConfig) -> None:
    """Refuse a released checkpoint that was trained with a different preset.

    The presets differ in architecture, so the mismatch would surface as a wall of
    missing/unexpected keys — or, with allow_partial_load, as a half-random model.
    """
    trained_with = header.get("preset")
    if trained_with is not None and config.preset_name is not None and trained_with != config.preset_name:
        raise ValueError(
            f"These weights were trained with the preset '{trained_with}', but you selected "
            f"'{config.preset_name}'. Run: python queryocc/train.py {trained_with} --no-train --test "
            f"--load_ckpt_from={config.load_ckpt_from}"
        )
    logger.info(
        "Loading released QueryOcc weights (preset=%s, queryocc=%s)",
        trained_with,
        header.get("git_sha", "unknown"),
    )


def load_model_weights_from_checkpoint(config: TrainingConfig) -> BaseModel:
    config.load_ckpt_from = resolve_checkpoint_spec(config.load_ckpt_from, preset=config.preset_name)
    assert config.load_ckpt_from.exists(), f"Resume weights not found at {config.load_ckpt_from}"

    if config.load_ckpt_from.is_dir():
        logger.info(
            "Loading the %s model weights from the folder you provided",
            "best" if config.load_ckpt_from_best else "last",
        )
        config.load_ckpt_from = get_checkpoint(config.load_ckpt_from, config.load_ckpt_from_best)

    logger.info("Loading the model weights from the checkpoint, but NOT the optimizers etc.")
    logger.info("Loading from %s", config.load_ckpt_from)

    # How strictly the weights load is independent of whether we can validate the config:
    # a released checkpoint has no sibling config.yml to check, but must still load in full.
    strict = not config.allow_partial_load
    if config.allow_partial_load:
        logger.warning("allow_partial_load is set: key mismatches will be reported but not fail the run.")

    release_header = read_release_header(config.load_ckpt_from)
    if release_header is not None:
        check_release_checkpoint(release_header, config)
        is_from_pretrain = True  # released weights are always a trained QueryOcc network
    elif config.validate_config:
        validate_loaded_checkpoint_config(config)
        config_path = find_checkpoint_config(config.load_ckpt_from)
        assert config_path.exists(), f"Config not found at {config_path}"
        loaded_config = TrainingConfig.load_config_from_yaml(config_path)
        is_from_pretrain = isinstance(loaded_config.model, PreTrainConfig)
    else:
        logger.warning("Skipping config validation, I hope you know what you are doing!")
        is_from_pretrain = True

    # A strict load overwrites every backbone weight, so the published backbone initialization is
    # dead work: skipping it makes evaluating a released checkpoint independent of the gated DINOv3
    # download (and of any torchvision fetch). Under allow_partial_load we have no such guarantee --
    # a checkpoint that omits the backbone would leave it randomly initialized rather than
    # pretrained, so keep the published weights as the fallback the partial load lands on.
    if strict:
        config.model.net.cerveza_encoder.image_encoder_pretrained = False

    model = config.model._target.load_from_checkpoint(
        config.load_ckpt_from,
        config=config.model,
        strict=strict,
        is_from_pretrain=is_from_pretrain,
        visualization_mode=config.visualization_mode,
    )

    return model


def main(config: TrainingConfig):
    L.seed_everything(config.seed, workers=True)

    config.model.configure_dataset(config.datamodule.dataset)
    setup()
    if config.scale_with_num_gpus:
        scale_with_num_gpus(config)
    if config.debug:
        config.datamodule.debug = True

    if is_large_node() and config.disable_gradient_checkpointing_on_large_node:
        logger.info("Disabling gradient checkpointing due to large node")
        config.model.net.gradient_checkpointing = False

    if config.load_ckpt_from is not None and not config.resume_training:
        model: BaseModel = load_model_weights_from_checkpoint(config)
    else:
        model: BaseModel = config.model.setup(visualization_mode=config.visualization_mode)

    dm: JointDataModule = config.datamodule.setup()

    wandb_logger = get_wandb_logger(config, model.__class__.__name__)

    best_checkpoint_callback, recovery_checkpoint_callback, scheduled_checkpoint_callback = build_checkpoint_callbacks(
        config
    )
    callbacks = [
        callback
        for callback in (
            best_checkpoint_callback,
            scheduled_checkpoint_callback,
            recovery_checkpoint_callback,
        )
        if callback is not None
    ] + [
        cb.LearningRateMonitor(logging_interval="step"),
        ValidationSamplerEpochSetter(),
    ]

    if config.validation_epochs:
        callbacks.append(ValidationEpochScheduler(config.validation_epochs, config.trainer.limit_val_batches))

    if config.save_weights_only:
        callbacks.append(
            WeightsOnlyCheckpoint(
                monitor=config.monitor_key,
                mode=config.monitor_mode,
            )
        )

    if config.monitor_patience > 0:
        callbacks.append(
            cb.EarlyStopping(
                monitor=config.monitor_key,
                patience=config.monitor_patience,
                mode=config.monitor_mode,
                min_delta=0.00,
                strict=config.monitor_strict,
            )
        )

    if config.allow_visualization:
        visualizer: Visualizer = config.visualizer.setup()
        callbacks.append(visualizer)

    # Build Trainer kwargs on a copy: strategy/profiler may be replaced by live objects
    # below, and config.trainer must stay yaml-serializable for save_config().
    trainer_kwargs = dict(config.trainer.__dict__)
    # Not a Lightning Trainer arg; consumed in the DDP block below.
    ddp_static_graph_override = trainer_kwargs.pop("ddp_static_graph", None)

    if trainer_kwargs.get("strategy") == "ddp":
        # Tuned DDP: grads alias the reducer buckets (removes a per-step gradient copy
        # and saves memory). Named variants like "ddp_find_unused_parameters_true" are
        # passed through unchanged.
        # static_graph=True is only safe without gradient accumulation: its first-iteration
        # delayed all-reduce assumes a synchronized backward, but Lightning runs the first
        # accumulation micro-batch under no_sync(), tripping
        # "expect_autograd_hooks_ INTERNAL ASSERT FAILED" in the DDP reducer.
        static_graph = config.trainer.accumulate_grad_batches == 1
        if ddp_static_graph_override is not None:
            static_graph = ddp_static_graph_override
        # 2h collective timeout (default 30 min): per-rank eval-graph compiles and RayIoU metric
        # syncs around validation can stall a single rank past 30 min, and the NCCL watchdog then
        # kills the whole run.
        # broadcast_buffers=False: DDP otherwise re-broadcasts rank-0's module buffers on every
        # forward (an extra ordered per-step collective). With sync_batchnorm=True the BN running
        # stats are already all-reduced inside the BN op, so the broadcast is redundant - and it is
        # the exact collective a straggling rank was observed piling into. Dropping it removes one
        # per-step sync point, reducing how badly per-step jitter serializes the whole job.
        trainer_kwargs["strategy"] = DDPStrategy(
            gradient_as_bucket_view=True,
            static_graph=static_graph,
            broadcast_buffers=False,
            timeout=timedelta(hours=2),
        )

    # configure torch profiler
    do_cuda_profiling = False
    if trainer_kwargs.get("profiler") == "pytorch":
        trainer_kwargs["profiler"] = PyTorchProfiler(
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=5, repeat=1),
            profile_memory=True,
        )
    elif trainer_kwargs.get("profiler") == "advanced":
        logger.info("Using advanced profiler")
        trainer_kwargs["profiler"] = AdvancedProfiler(dirpath=".", filename="perf_logs")
    elif trainer_kwargs.get("profiler") == "cuda":
        do_cuda_profiling = True
        trainer_kwargs["profiler"] = None

    # Anything Lightning writes without an explicit path — checkpoints when ModelCheckpoint
    # has no dirpath, and the lightning_logs/ fallback logger when use_logging is off — goes
    # to default_root_dir, which defaults to the CWD. Under the SLURM launchers the CWD is the
    # bind-mounted repo on the home filesystem, where a handful of 1.6 GB checkpoints exhausts
    # a 30 GB quota. Point it at the experiment directory on the large storage instead.
    trainer_kwargs["default_root_dir"] = str(config.get_base_dir())

    trainer = L.Trainer(
        logger=wandb_logger if config.use_logging else None,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    # save the config
    config.save_config()

    if config.train:
        if config.resume_training:
            assert config.load_ckpt_from is not None and config.load_ckpt_from.exists(), "Checkpoint not found"
            if config.load_ckpt_from.is_dir():
                logger.info("Loading the last model weights from the folder you provided")
                config.load_ckpt_from = get_checkpoint(config.load_ckpt_from, best=False)
            logger.info("Resuming training from %s", config.load_ckpt_from)

        if do_cuda_profiling:
            # do profiling and export as chrome trace
            profiler_ = PyTorchProfiler(
                dirpath=wandb_logger.save_dir,
                filename="profiler_log",
                schedule=torch.profiler.schedule(warmup=80, wait=10, active=3, repeat=0),
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )

            trainer_kwargs.pop("profiler")

            trainer = L.Trainer(
                logger=wandb_logger if config.use_logging else None,
                callbacks=callbacks,
                profiler=profiler_,
                **trainer_kwargs,
            )

            trainer.fit(
                model,
                datamodule=dm,
                ckpt_path=config.load_ckpt_from if config.resume_training else None,
                # Our own checkpoints contain config dataclasses, which the torch>=2.6
                # weights_only unpickler rejects; they are trusted artifacts from our runs.
                weights_only=False,
            )

        else:
            # run normally
            trainer.fit(
                model,
                datamodule=dm,
                ckpt_path=config.load_ckpt_from if config.resume_training else None,
                # Our own checkpoints contain config dataclasses, which the torch>=2.6
                # weights_only unpickler rejects; they are trusted artifacts from our runs.
                weights_only=False,
            )

        if config.test:
            available_checkpoint_paths = {
                "best": best_checkpoint_callback.best_model_path if best_checkpoint_callback else "",
                "last": recovery_checkpoint_callback.last_model_path,
            }
            checkpoint_names = (
                ("best", "last") if config.test_best_and_last_after_fit else (config.test_after_fit_ckpt_path,)
            )
            for checkpoint_name in checkpoint_names:
                checkpoint_path = available_checkpoint_paths[checkpoint_name]
                if not checkpoint_path:
                    raise RuntimeError(f"No {checkpoint_name} checkpoint was saved before post-fit evaluation")
                logger.info(
                    "Running complete post-fit evaluation from the %s checkpoint: %s",
                    checkpoint_name,
                    checkpoint_path,
                )
                trainer.test(model, datamodule=dm, ckpt_path=checkpoint_path, weights_only=False)

    if config.test and not config.train:
        test_results = trainer.test(model=model, datamodule=dm)
        if trainer.is_global_zero:
            write_evaluation_results(config, test_results)

    return model


def configure_logging() -> None:
    """Show our own INFO messages without turning on every library's.

    Nothing else configures logging, so `logger.info` calls (which checkpoint the run resolved,
    which weights it loaded) were previously dropped by the root logger's default WARNING level.
    Scoped to the `queryocc` namespace deliberately: root-level INFO drowns cluster logs in
    third-party chatter.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    # `logger` too, not just the package: run as a script this module is "__main__", so it is
    # not under the "queryocc" namespace and would miss the handler.
    for target in (logging.getLogger("queryocc"), logger):
        if not target.handlers:
            target.addHandler(handler)
            target.setLevel(logging.INFO)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    configure_logging()
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
