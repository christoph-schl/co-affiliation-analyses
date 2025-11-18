# src/maa/config/loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Type, Union, cast

from pydantic import ValidationError

from .models import BaseConfig
from .registry import MODEL_REGISTRY

try:
    import tomllib as _tomllib

    def _load_toml(path: Path) -> Dict[str, Any]:
        """
        Load a TOML file using Python's built-in tomllib (Python ≥ 3.11).

        :param path: Path to the TOML file.
        :return: Parsed TOML content as a dictionary.
        """
        return _tomllib.loads(path.read_text(encoding="utf-8"))

except ImportError:
    import toml as _toml

    def _load_toml(path: Path) -> Dict[str, Any]:
        """
        Load a TOML file using the external 'toml' package.

        :param path: Path to the TOML file.
        :return: Parsed TOML content as a dictionary.
        """
        return _toml.load(path)


def _get_group(raw: Dict[str, Any], group: str) -> Dict[str, Any]:
    """
    Retrieve a nested configuration group from a dict structure.

    :param raw: The parsed TOML root dictionary.
    :param group: Dot-separated group name (e.g., "network" or "a.b.c").
    :return: The dictionary representing the requested group.
    :raises KeyError: If the group path does not exist.
    :raises ValueError: If the group is not a table.
    """
    sub: Any = raw
    for key in group.split("."):
        if not isinstance(sub, dict) or key not in sub:
            raise KeyError(
                f"Config group '{group}' not found. " f"Top-level keys: {list(raw.keys())}"
            )
        sub = sub[key]

    if not isinstance(sub, dict):
        raise ValueError(f"Config group '{group}' must be a table.")

    return sub


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def load_toml_group_to_model(
    toml_path: Union[str, Path],
    group: str,
    *,
    validate_paths_exist: bool = False,
) -> BaseConfig:
    """
    Load a configuration section from a TOML file and instantiate its model.

    This reads a TOML file, extracts the requested group, looks up its
    corresponding Pydantic model in ``MODEL_REGISTRY`` (which must inherit
    from ``BaseConfig``), validates the section, and optionally verifies that
    all filesystem paths exist.

    :param toml_path: Path to the TOML configuration file.
    :param group: Dot-separated name of the config section to load.
    :param validate_paths_exist: If True, call ``instance.check_paths_exist()``.
    :return: A validated configuration model instance.
    :raises FileNotFoundError: If the TOML file does not exist.
    :raises KeyError: If the configuration group or model registry entry is missing.
    :raises ValueError: On Pydantic validation errors.
    """
    toml_file = Path(toml_path).expanduser().resolve()
    if not toml_file.exists():
        raise FileNotFoundError(f"TOML file not found: {toml_file}")

    raw = _load_toml(toml_file)
    group_dict = _get_group(raw, group)

    model_key = group.split(".")[-1]
    model_cls = MODEL_REGISTRY.get(model_key)

    if model_cls is None:
        raise KeyError(
            f"No registered model for group '{model_key}'. " f"Add it to MODEL_REGISTRY."
        )

    # tell the type checker what the model class is
    model_type = cast(Type[BaseConfig], model_cls)

    try:
        instance = model_type(**group_dict)
    except ValidationError as exc:
        raise ValueError(f"Validation failed for config group '{group}':\n{exc}") from exc

    if validate_paths_exist:
        instance.check_paths_exist()

    return instance
