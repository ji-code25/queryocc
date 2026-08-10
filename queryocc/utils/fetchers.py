from typing import Optional

import lightning as L
from torch.utils.data import ConcatDataset, Subset


def fetch(batch: Optional[dict], outputs: Optional[dict], key: str, sample_idx=0):
    """Fetch key from the batch or outputs dicts if it exists."""
    if batch is not None and key in batch:
        out = batch[key].detach()[sample_idx, ...]
    elif outputs is not None and key in outputs:
        out = outputs[key].detach()[sample_idx, ...]
    else:
        out = None

    return out


def fetch_label_mapper(trainer: L.Trainer):
    dm: L.LightningDataModule = trainer.datamodule  # type: ignore
    return _label_mapper_from_dataset(dm.data_train)


def _label_mapper_from_dataset(dataset):
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            if hasattr(ds, "label_mapper"):
                return ds.label_mapper
        return None
    elif isinstance(dataset, Subset):
        return _label_mapper_from_dataset(dataset.dataset)
    else:
        return getattr(dataset, "label_mapper", None)


def fetch_model(trainer: L.Trainer):
    """Depending on GPU or DDP training, return the model"""
    if isinstance(trainer.model, L.LightningModule):
        model = trainer.model
    else:
        model = trainer.model.module  # type: ignore
    return model
