"""Compatibility entry point; implementation lives in gamma_rppg.common.config."""
from importlib import import_module as _import_module

_implementation = _import_module("gamma_rppg.common.config")
globals().update({name: value for name, value in vars(_implementation).items()
                  if not name.startswith("__")})
