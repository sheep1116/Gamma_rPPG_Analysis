"""Compatibility entry point; implementation lives in gamma_rppg.common.statistics."""
from importlib import import_module as _import_module

_implementation = _import_module("gamma_rppg.common.statistics")
globals().update({name: value for name, value in vars(_implementation).items()
                  if not name.startswith("__")})
