from __future__ import annotations

import json
import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union




def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_project_root(config: Dict[str, Any], override: Optional[str] = None) -> Path:
    """Resolve the top-level v5 result directory."""
    if override:
        return Path(override).expanduser()
    output = config.get("output", {})
    root = Path(output.get("root", "~/Result")).expanduser()
    name = str(output.get("experiment_name", "GammaBrightness_rPPG_v5"))
    return root / name


def resolve_mode_root(
    config: Dict[str, Any],
    mode: Optional[str] = None,
    override: Optional[str] = None,
) -> Path:
    """Resolve one isolated module directory under the project result root.

    ``override`` replaces the complete module directory. Otherwise ``mode`` is
    mapped through ``output.folders``. If no mode is supplied, the experiment's
    ``output_subdir`` is used.
    """
    if override:
        return Path(override).expanduser()
    project_root = resolve_project_root(config)
    output = config.get("output", {})
    experiment = config.get("experiment", {})
    selected = mode or experiment.get("output_subdir")
    if not selected:
        return project_root
    folders = output.get("folders", {})
    folder = folders.get(str(selected), selected)
    return project_root / str(folder)


def resolve_output_root(config: Dict[str, Any], override: Optional[str] = None) -> Path:
    """Backward-compatible alias used by experiment and summary modules."""
    return resolve_mode_root(config, override=override)


def dump_resolved_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write the exact resolved config beside results for reproducibility."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore
        p.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    except Exception:
        p.with_suffix(".json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_environment(value):
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        def replace(match):
            name, default = match.groups()
            result = os.environ.get(name)
            if result:
                return result
            if default is not None:
                return default
            raise ValueError(f"Required configuration environment variable is unset: {name}")
        return _ENV.sub(replace, value)
    return value


def _merge_config(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Union[str, Path], *, allow_empty: bool = False) -> Dict[str, Any]:
    """Load YAML/JSON, relative ``extends`` and explicit environment placeholders.

    Mapping overrides merge recursively; lists are replaced. Inheritance is
    resolved before environment expansion so callers see the complete config.
    """
    def read(current: Path, parents: tuple) -> Dict[str, Any]:
        current = current.expanduser().resolve()
        if current in parents:
            raise ValueError(f"Circular config inheritance: {current}")
        text = current.read_text(encoding="utf-8")
        if current.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML configuration requires PyYAML.") from exc
            data = yaml.safe_load(text)
        if data is None and allow_empty:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a mapping: {current}")
        base = data.pop("extends", None)
        if base is not None:
            if not isinstance(base, str):
                raise ValueError(f"Config extends must be a file path: {current}")
            data = _merge_config(read(current.parent / base, parents + (current,)), data)
        return data
    return _expand_environment(read(Path(path), ()))
