from pathlib import Path
from typing import Any

from maa.config import load_toml_group_to_model


def load_config_for_stage(
    config_file: Path, stage_group: str, *, validate_paths_exist: bool = False
) -> Any:
    """
    High-level factory: given path to config and a stage group string
    (e.g. "network" or "bibliometrics.network")
    return a typed Pydantic config instance for that stage.
    """
    return load_toml_group_to_model(
        toml_path=config_file, group=stage_group, validate_paths_exist=validate_paths_exist
    )
