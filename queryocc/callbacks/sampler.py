"""Callbacks that keep evaluation sampling reproducible across distributed ranks."""

from collections.abc import Iterable

from lightning.pytorch.callbacks import Callback


class ValidationSamplerEpochSetter(Callback):
    """Seed shuffled validation samplers from the current training epoch.

    ``DistributedSampler(shuffle=True)`` otherwise keeps epoch zero unless somebody
    explicitly calls ``set_epoch``.  Setting the same training epoch on every rank gives
    each periodic validation a new deterministic global permutation while preserving
    disjoint rank partitions.
    """

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        dataloaders = trainer.val_dataloaders
        if not isinstance(dataloaders, (list, tuple)):
            dataloaders = [dataloaders]
        for dataloader in dataloaders:
            sampler = getattr(dataloader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(trainer.current_epoch)


class ValidationEpochScheduler(Callback):
    """Enable in-fit validation only at selected one-based epoch numbers.

    Lightning supports a constant ``check_val_every_n_epoch`` interval, but not an
    explicit epoch list. Its own ``enable_validation`` property is gated by
    ``limit_val_batches``; toggling that value at epoch start preserves the already
    initialized validation loader and its configured batch limit.
    """

    def __init__(self, epochs: Iterable[int], limit_val_batches: int | float) -> None:
        self.epochs = frozenset(int(epoch) for epoch in epochs)
        if not self.epochs or min(self.epochs) < 1:
            raise ValueError("validation epochs must contain positive one-based epoch numbers")
        if limit_val_batches <= 0:
            raise ValueError("scheduled validation requires limit_val_batches > 0")
        self.limit_val_batches = limit_val_batches

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        del pl_module
        one_based_epoch = trainer.current_epoch + 1
        trainer.limit_val_batches = self.limit_val_batches if one_based_epoch in self.epochs else 0

    def on_fit_end(self, trainer, pl_module) -> None:
        del pl_module
        trainer.limit_val_batches = self.limit_val_batches
