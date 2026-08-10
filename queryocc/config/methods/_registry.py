"""Shared registry for method configs.

Family modules (``queryocc``, ...) import the two dicts below
and add their presets via :func:`register`. Keeping the registry in a tiny
dependency-free module lets the family modules import it without cycling through
the package ``__init__`` (which is what assembles and exposes them).
"""

import sys
from typing import Dict

from queryocc.config.trainer import TrainingConfig

method_configs: Dict[str, TrainingConfig] = {}
descriptions: Dict[str, str] = {}
# Family module (last path segment, e.g. "queryocc") that registered each preset. The package
# __init__ uses this to stamp per-track defaults (currently the W&B group) by config module, so a
# preset's home file decides its research track — no per-preset bookkeeping.
method_modules: Dict[str, str] = {}


def register(name: str, config: TrainingConfig, description: str) -> TrainingConfig:
    """Register ``config`` under ``name`` with its CLI ``description``.

    Returns the config so callers can keep mutating it in place — the
    deepcopy-and-tweak pattern used throughout the family modules:

        cfg = register("queryocc-nusc-av2", copy.deepcopy(method_configs["queryocc-nusc"]), "...")
        cfg.model.max_lr = 5e-5
    """
    if name in method_configs:
        raise ValueError(f"Duplicate method config name: {name!r}")
    # Stamp the preset onto its config so the run knows which preset it is: released
    # checkpoints record the preset they were trained with, and train.py refuses to load
    # one into a different preset (queryocc/utils/release.py). Set after the deepcopy in
    # the copy-and-tweak pattern, so a derived preset never inherits its parent's name.
    config.preset_name = name
    method_configs[name] = config
    descriptions[name] = description
    method_modules[name] = sys._getframe(1).f_globals.get("__name__", "").rsplit(".", 1)[-1]
    return config
