"""Backwards-compatible shim.

The method configs were split from this single 1500-line module into the
``queryocc.config.methods`` package (one module per method family). This shim keeps
the historical import path working — ``from queryocc.config.method_configs import
method_configs, AnnotatedBaseConfigUnion, convert_markup_to_ansi, ...`` — so the
many scripts and entry points that use it need no changes.

Extremely inspired by nerfstudio, credit to them for this pattern.
"""

from queryocc.config.methods import (
    AnnotatedBaseConfigUnion,
    convert_markup_to_ansi,
    descriptions,
    method_configs,
    sort_methods,
)

__all__ = [
    "AnnotatedBaseConfigUnion",
    "convert_markup_to_ansi",
    "descriptions",
    "method_configs",
    "sort_methods",
]
