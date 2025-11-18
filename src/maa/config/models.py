# src/maa/config/models.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def _expand_str(s: str) -> str:
    return os.path.expandvars(os.path.expanduser(s))


class BaseConfig(BaseModel):
    """
    Minimal base config:
    - expands ~ and env vars
    - coerces path-like fields into Path
    - resolves relative paths using data_root when provided
    """

    data_root: Optional[Path] = Field(
        None, description="Optional root for resolving relative paths"
    )

    @field_validator("data_root", mode="before")
    def _normalize_data_root(cls, v: Any) -> Optional[Path]:
        if v is None:
            return None
        if isinstance(v, Path):
            return v.expanduser().resolve()
        return Path(_expand_str(str(v))).resolve()

    @model_validator(mode="before")
    def _coerce_path_like_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs before standard validation:
        Converts string path fields into Path objects,
        applying data_root to resolve relative paths.
        """
        data_root = values.get("data_root")

        for name, val in list(values.items()):
            if val is None or not isinstance(val, str):
                continue

            # check if field is annotated as Path or Optional[Path]
            annotated_as_path = False
            if name in cls.model_fields:
                ann = cls.model_fields[name].annotation
                if ann is Path:
                    annotated_as_path = True
                else:
                    args = getattr(ann, "__args__", ())
                    if Path in args:  # Optional[Path]
                        annotated_as_path = True

            # name heuristic fallback
            looks_like_path = "path" in name.lower() or name.lower().endswith("_dir")

            if annotated_as_path or looks_like_path:
                p = Path(_expand_str(val))
                if data_root and not p.is_absolute():
                    p = (Path(data_root) / p).resolve()
                else:
                    p = p.resolve()
                values[name] = p

        return values

    def check_paths_exist(self) -> None:
        missing = [
            f"{k} -> {v}"
            for k, v in self.__dict__.items()
            if isinstance(v, Path) and not v.exists()
        ]
        if missing:
            raise FileNotFoundError("Missing config paths:\n" + "\n".join(missing))


class NetworkConfig(BaseConfig):
    article_file_path: Path = Field(..., description="Parquet with article records")
    affiliation_file_path: Path = Field(..., description="Affiliation file")
    output_path: Path = Field(..., description="Directory or file to write outputs")
    year_gap_stable_links: int = Field(..., description="Year gap for stable links")


class ZnibConfig(BaseConfig):
    model_output_dir: Path = Field(..., description="Where to store model artifacts")
    seed: int = Field(42, description="Random seed")
    max_iter: int = Field(1000, description="Maximum iterations")
