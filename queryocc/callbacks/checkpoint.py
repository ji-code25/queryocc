import os

from lightning.pytorch.callbacks import ModelCheckpoint
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
