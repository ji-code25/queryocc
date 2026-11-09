"""Callbacks that keep evaluation sampling reproducible across distributed ranks."""

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
