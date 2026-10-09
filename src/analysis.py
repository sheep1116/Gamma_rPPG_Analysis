"""Compatibility entry point; implementation lives in gamma_rppg.analysis.roi_mechanism.analysis."""
from importlib import import_module as _import_module

_implementation = _import_module("gamma_rppg.analysis.roi_mechanism.analysis")
globals().update({name: value for name, value in vars(_implementation).items()
                  if not name.startswith("__")})
