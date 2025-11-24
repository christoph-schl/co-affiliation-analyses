import logging
from pathlib import Path
from typing import Any, Mapping

import structlog
import toml

from maa.config.constants import DEFAULT_CONFIG_CONTENT

_logger = structlog.getLogger(__name__)


def configure_logging(debug: bool) -> None:
    """
    Configure structlog for CLI usage.

    This configures a console-friendly renderer, timestamper and exception helpers,
    and sets the minimum log level according to `debug`.
    """
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.set_exc_info,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            min_level=logging.DEBUG if debug else logging.INFO
        ),
        cache_logger_on_first_use=True,
    )


def _make_toml_serializable(obj: Any) -> Any:
    """
    Recursively convert datatypes that toml.dumps can't handle (e.g. Path)
    into plain Python builtins (str, dict, list, int, bool, ...).
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {k: _make_toml_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_toml_serializable(v) for v in obj]
    # primitives (str, int, float, bool, None) are returned as-is
    return obj


def write_default_toml(path: Path, force: bool = False) -> None:
    """
    Write a default TOML configuration to the given file path.

    If `force` is True, overwrite an existing file. Otherwise raise FileExistsError.
    """
    if path.exists():
        if not force:
            _logger.warning("config.exists", path=str(path))
            raise FileExistsError(f"Config file already exists: {path}")
        _logger.info("config.overwrite", path=str(path))

    # ensure parent dirs exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # make sure content is serializable by toml
    serializable = _make_toml_serializable(DEFAULT_CONFIG_CONTENT)

    toml_text = toml.dumps(serializable)
    path.write_text(toml_text, encoding="utf-8")

    _logger.info("config.created", path=str(path))
