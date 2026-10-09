"""Compatibility entry point; implementation lives in gamma_rppg.analysis.sensitivity."""
from importlib import import_module as _import_module

_implementation = _import_module("gamma_rppg.analysis.sensitivity")
globals().update({name: value for name, value in vars(_implementation).items()
                  if not name.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
