"""Local import alias for the ADAPOLE-L2 worktree.

This package re-exports the upstream ``adapol`` package from the local
``adapole-l2`` source tree, so scripts can use ``import adapol_l2`` and avoid
confusing this branch with other installed ``adapol`` copies.
"""

from __future__ import annotations

import importlib
import sys

import adapol as _adapol

__all__ = []
for _name in getattr(_adapol, "__all__", ()):
    if hasattr(_adapol, _name):
        globals()[_name] = getattr(_adapol, _name)
        __all__.append(_name)

for _submodule in ("hybfit", "anacont", "fit_utils", "fit_utils_dlr"):
    sys.modules[f"{__name__}.{_submodule}"] = importlib.import_module(f"adapol.{_submodule}")
