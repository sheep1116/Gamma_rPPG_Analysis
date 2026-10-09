"""Compatibility entry point; implementation lives in gamma_rppg.diagnostics.pure_structure."""
from importlib import import_module as _import_module

_implementation = _import_module("gamma_rppg.diagnostics.pure_structure")
globals().update({name: value for name, value in vars(_implementation).items()
                  if not name.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
