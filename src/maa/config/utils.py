# Copyright © 2025 Christoph Schlager, TU Wien

import hashlib
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Mapping, Optional

import requests
import structlog
import toml
from tqdm import tqdm

from maa.config.constants import DEFAULT_CONFIG_CONTENT

_logger = structlog.getLogger(__name__)

ZENODO_API = "https://zenodo.org/api/records"
DEFAULT_TIMEOUT = 30
RETRY_WAIT = 2
MAX_RETRIES = 3
CHUNK_SIZE = 1024 * 1024  # 1 MB


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def configure_logging(level: LogLevel = LogLevel.INFO) -> None:
    """
    Configure structlog for CLI or Jupyter usage.
    Idempotent: calling multiple times always results in the same configuration.
    """
    # Reset structlog completely
    structlog.reset_defaults()

    # Clear all handlers on the root logger
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Reconfigure standard logging
    logging.basicConfig(
        level=level.value,
        force=True,  # ensures old handlers are dropped across all loggers
    )

    # Reconfigure structlog
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level.value),
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


@dataclass(frozen=True)
class ZenodoFile:
    name: str
    url: str
    size: int
    checksum: Optional[str] = None


def _request_with_retries(url: str) -> requests.Response:
    """Perform an HTTP GET with retries and exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=DEFAULT_TIMEOUT, stream=True)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_WAIT * attempt
            _logger.warning(f"Retry {attempt}/{MAX_RETRIES} in {wait}s: {exc}")
            time.sleep(wait)
    raise RuntimeError(f"Failed to get response for url {url}")


def fetch_record(record_id: str) -> Any:
    """Fetch Zenodo record metadata."""
    url = f"{ZENODO_API}/{record_id}"
    response = _request_with_retries(url=url)

    if response is None:
        raise RuntimeError(f"Failed to fetch Zenodo record {record_id}")

    return response.json()


def parse_files(metadata: Any) -> List[ZenodoFile]:
    """Extract downloadable files from Zenodo metadata."""
    files = []
    for entry in metadata.get("files", []):
        files.append(
            ZenodoFile(
                name=entry["key"],
                url=entry["links"]["self"],
                size=entry.get("size", 0),
                checksum=entry.get("checksum"),
            )
        )
    return files


def _compute_checksum(path: Path, algorithm: str) -> str:
    """Compute checksum using the given algorithm (md5, sha256, etc.)."""
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"{algorithm}:{h.hexdigest()}"


def download_file(file: ZenodoFile, target_file_path: Path) -> Path:
    """Download a single file with a progress bar and checksum validation."""
    target_file_path.parent.mkdir(parents=True, exist_ok=True)
    destination = target_file_path

    response = _request_with_retries(url=file.url)

    with (
        destination.open("wb") as f,
        tqdm(
            total=file.size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=file.name,
            colour="cyan",
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    if file.checksum:
        algo, _ = file.checksum.split(":", 1)
        actual = _compute_checksum(destination, algo)
        if actual != file.checksum:
            raise ValueError(
                f"Checksum mismatch for {file.name}:\n"
                f"  expected {file.checksum}\n"
                f"  got      {actual}"
            )

    return destination


def download_zenodo_record(
    record_id: str,
    output_file_path: str | Path = "data",
    *,
    dry_run: bool = False,
) -> list[Path]:
    """
    Download all files from a Zenodo record.
    :param record_id: Zenodo record identifier
    :param output_file_path: Destination file path
    :param dry_run: If True, only list files without downloading
    :return:
    """
    metadata = fetch_record(record_id)
    files = parse_files(metadata)

    if not files:
        _logger.warning("No files found in this record.")
        return []

    _logger.info(f"Download Zenodo record {record_id}")
    _logger.info(f"Title: {metadata['metadata'].get('title', '—')}")

    if dry_run:
        for f in files:
            _logger.info(f"  • {f.name} ({f.size / 1e6:.2f} MB)")
        return []

    output_file_path = Path(output_file_path)
    downloaded: list[Path] = []

    for f in files:
        downloaded.append(download_file(file=f, target_file_path=output_file_path))

    return downloaded
