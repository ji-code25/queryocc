import os
from pathlib import Path

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only
from torch import save as torch_save


class WeightsOnlyCheckpoint(ModelCheckpoint):
    """Write a `.pth` holding just the weights, alongside the normal `.ckpt`.

    Kept load-compatible with the ordinary checkpoint (`{"state_dict": ...}` with unprefixed
    keys), so `--load_ckpt_from` accepts either file. Not the published artifact — that is
    `tools/release/export_weights.py`, which also strips the recomputable buffers and stamps the
    provenance header.
    """

    def __init__(self, **kwargs):
        super().__init__(
            save_weights_only=True, save_top_k=1, filename="{epoch}-{step}_weights_only", save_last=True, **kwargs
        )

    @rank_zero_only
    def _save_checkpoint(self, trainer, filepath):
        save_path = filepath.replace(".ckpt", ".pth")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # trainer.lightning_module, not trainer.model: under DDP the latter is the wrapper, whose
        # state dict prefixes every key with "module." and does not load back into the model.
        torch_save({"state_dict": trainer.lightning_module.state_dict()}, save_path)


class FinalStepModelCheckpoint(ModelCheckpoint):
    """Keep ``last.ckpt`` synchronized with the actual final optimizer step.

    Lightning normally skips its train-end save whenever *any* earlier last checkpoint
    exists.  If ``max_steps`` stops training between validation boundaries, that leaves
    ``last.ckpt`` stale even though fitting completed successfully.  Refresh only when the
    saved global step differs, preserving Lightning's ordinary best-checkpoint behavior.
    """

    def on_train_end(self, trainer, pl_module) -> None:
        super().on_train_end(trainer, pl_module)
        if self.save_last and self._last_global_step_saved != trainer.global_step:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))


class EpochRecoveryCheckpoint(FinalStepModelCheckpoint):
    """A distinct callback state for the per-epoch resumable ``last.ckpt``."""

    @property
    def state_key(self) -> str:
        return f"{super().state_key}:epoch-recovery"

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Write ``last.ckpt`` even when ``save_top_k=0``.

        Lightning 2.6 only calls its ordinary last-checkpoint writer here when
        a top-k checkpoint was also saved in the same hook.  Recovery should
        not create a second series of epoch-named checkpoints, so explicitly
        refresh the single last checkpoint instead.
        """
        del pl_module
        if self._should_skip_saving_checkpoint(trainer) or not self._should_save_on_train_epoch_end(trainer):
            return
        if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))


class ScheduledEpochCheckpoint(Callback):
    """Retain immutable checkpoints at explicit one-based epoch numbers."""

    def __init__(self, epochs) -> None:
        super().__init__()
        normalized = tuple(int(epoch) for epoch in epochs)
        if not normalized or min(normalized) < 1 or len(set(normalized)) != len(normalized):
            raise ValueError("checkpoint epochs must be unique positive one-based epoch numbers")
        self.epochs = frozenset(normalized)
        self.saved_paths: dict[int, str] = {}

    @property
    def state_key(self) -> str:
        epochs = ",".join(str(epoch) for epoch in sorted(self.epochs))
        return f"{self.__class__.__qualname__}[epochs={epochs}]"

    def state_dict(self) -> dict:
        return {"saved_paths": self.saved_paths}

    def load_state_dict(self, state_dict: dict) -> None:
        self.saved_paths = {int(epoch): str(path) for epoch, path in state_dict.get("saved_paths", {}).items()}

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        del pl_module
        one_based_epoch = trainer.current_epoch + 1
        if one_based_epoch not in self.epochs or trainer.sanity_checking:
            return

        checkpoint_dir = Path(trainer.default_root_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"epoch-{one_based_epoch:02d}.ckpt"
        self.saved_paths[one_based_epoch] = str(checkpoint_path)
        trainer.save_checkpoint(checkpoint_path, weights_only=False)
