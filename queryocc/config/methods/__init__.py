"""Method configs, assembled from per-family modules.

Each family module registers its presets into the shared ``_registry`` dicts.
This package then sorts them and builds the Tyro subcommand union consumed by
``train.py``.

Public API (unchanged from the old monolithic ``method_configs`` module):
``method_configs``, ``descriptions``, ``AnnotatedBaseConfigUnion``,
``convert_markup_to_ansi``, ``sort_methods``.
"""

import importlib
from typing import OrderedDict

import tyro

from queryocc.config.base_config import CONSOLE
from queryocc.config.methods._registry import descriptions, method_configs, method_modules

# Import the family modules for their registration side effects. Done via
# importlib (not `import` statements) so the import sorter can't reorder them.
_FAMILY_MODULES = ("queryocc",)
for _family in _FAMILY_MODULES:
    importlib.import_module(f"{__name__}.{_family}")

# W&B group, decided by the config module a preset lives in. Presets can still
# override wandb_group individually (or via the CLI).
WANDB_TRACK_BY_MODULE = {
    "queryocc": "QueryOcc",
}
for _name, _cfg in method_configs.items():
    if _cfg.wandb_group is None:
        _cfg.wandb_group = WANDB_TRACK_BY_MODULE.get(method_modules.get(_name, ""))


def convert_markup_to_ansi(markup_string: str) -> str:
    """Convert rich-style markup to ANSI sequences for command-line formatting.

    Args:
        markup_string: Text with rich-style markup.

    Returns:
        Text formatted via ANSI sequences.
    """
    with CONSOLE.capture() as out:
        CONSOLE.print(markup_string, soft_wrap=True)
    return out.get()


def sort_methods(methods, method_descriptions):
    """Sort methods and descriptions by method name."""
    methods = OrderedDict(sorted(methods.items(), key=lambda x: x[0]))
    method_descriptions = OrderedDict(sorted(method_descriptions.items(), key=lambda x: x[0]))
    return methods, method_descriptions


method_configs, descriptions = sort_methods(method_configs, descriptions)


AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
