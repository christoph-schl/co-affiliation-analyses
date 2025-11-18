# src/maa/cli_network.py
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterable, Iterator

import nx2vos
import structlog

from src.maa.config import load_config_for_stage
from src.maa.config.constants import CONFIGURATION_PATH, ProcessingStage
from src.maa.config.models import NetworkConfig
from src.maa.dataframe.models.affiliation import read_affiliations
from src.maa.dataframe.models.article import read_articles
from src.maa.network.network import AffiliationNetworkProcessor

_logger = structlog.getLogger(__name__)

# =====================================
# Constants
# =====================================
NETWORK_COUNTRY = "AUT"
AFFILIATION_DIR = Path("links")
VOS_DIR = Path("vos/map")
AFFILIATION_LINKS_PREFIX = "affiliation_links"
VOS_MAP_PREFIX = "map"
VOS_NETWORK_PREFIX = "network"


@dataclass(frozen=True)
class YearGapEntry:
    """Represents one year-gap variant for generation (gap value + suffix)."""

    gap: int
    suffix: str


@dataclass(frozen=True)
class YearGapResult:
    """Container for results produced for one YearGapEntry."""

    suffix: str
    graph: Any
    link_gdf: Any


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


def iter_year_gaps(stable_gap: int) -> Iterator[YearGapEntry]:
    """
    Yield the YearGapEntry variants expected by the pipeline.

    Always yields:
      * gap=0, suffix='all'      -> include all links
      * gap=stable_gap, suffix='stable' -> only stable links according to config
    """
    yield YearGapEntry(gap=0, suffix="all")
    yield YearGapEntry(gap=stable_gap, suffix="stable")


def _ensure_parent(path: Path) -> None:
    """
    Ensure that the parent directory for `path` exists.

    Intended as a small helper so callers can create files safely.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _output_file_paths(output_root: Path, suffix: str) -> dict:
    """
    Return output file paths for a given suffix.
    """
    return {
        "links": output_root / AFFILIATION_DIR / f"{AFFILIATION_LINKS_PREFIX}_{suffix}.gpkg",
        "map": output_root / VOS_DIR / f"{VOS_MAP_PREFIX}_{suffix}.txt",
        "network": output_root / VOS_DIR / f"{VOS_NETWORK_PREFIX}_{suffix}.txt",
    }


def _write_outputs(graph: Any, link_gdf: Any, output_path: Path, suffix: str) -> None:
    """
    Write the produced graph and link GeoDataFrame to disk.

    Writes:
      * GeoPackage of affiliation links
      * VOS map file
      * VOS network file

    Raises any exception encountered when writing so callers can handle/log it.
    """
    paths = _output_file_paths(output_path, suffix)
    for p in paths.values():
        _ensure_parent(p)

    # Persist outputs - underlying libraries handle their own formats.
    link_gdf.to_file(paths["links"])
    nx2vos.write_vos_map(G=graph.graph, fname=paths["map"])
    nx2vos.write_vos_network(G=graph.graph, fname=paths["network"])


def get_network_from_config(
    article_df: Any, affiliation_gdf: Any, net_cfg: NetworkConfig
) -> Generator[YearGapResult, None, None]:
    """
    Build affiliation networks according to the provided configuration.

    This function is a generator yielding YearGapResult instances, one per
    year-gap variant produced by `iter_year_gaps`.

        :param article_df:
            DataFrame containing article data.
        :param affiliation_gdf:
            GeoDataFrame containing affiliation data.
        :param net_cfg:
            NetworkConfig instance loaded from configuration.
        :return:
            Iterator yielding YearGapResult objects.
    """

    processor = AffiliationNetworkProcessor(
        article_df=article_df,
        affiliation_gdf=affiliation_gdf,
        country_filter=NETWORK_COUNTRY,
    )

    for yg in iter_year_gaps(net_cfg.year_gap_stable_links):
        _logger.info("processing.year_gap", gap=yg.gap, suffix=yg.suffix)
        link_gdf = processor.get_affiliation_links(min_year_gap=yg.gap)
        graph = processor.get_affiliation_graph()
        yield YearGapResult(suffix=yg.suffix, graph=graph, link_gdf=link_gdf)


def write_outputs(results: Iterable[YearGapResult], output_path: Path, dry_run: bool) -> None:
    """
    Write network results to the output directory.

    :param results: Iterator of YearGapResult objects.
    :param output_path: Target directory for all output files.
    :param dry_run: If True, do not write any files.
    :return: None
    """
    for result in results:
        if dry_run:
            _logger.info("dry_run.write", suffix=result.suffix, output_path=str(output_path))
            continue

        try:
            _write_outputs(result.graph, result.link_gdf, output_path, result.suffix)
        except Exception as exc:  # noqa: BLE001 - intentional: surface to caller after logging
            _logger.error("write.failed", suffix=result.suffix, error=str(exc))
            raise
        else:
            _logger.info("write.success", suffix=result.suffix, path=str(output_path))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments for the network-building pipeline.

    :param argv: Optional argument list (defaults to sys.argv).
    :return: Parsed argument namespace.
    """

    p = argparse.ArgumentParser(description="Build affiliation networks from configuration.")
    p.add_argument(
        "--config",
        "-c",
        type=Path,
        default=CONFIGURATION_PATH,
        help="Path to config.toml",
    )
    p.add_argument(
        "--stage",
        "-s",
        default=ProcessingStage.PREPROCESSING.value,
        help="Processing stage group name defined in configuration.",
    )
    p.add_argument(
        "--validate-paths",
        action="store_true",
        dest="validate_paths",
        default=True,
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    CLI entry point: load configuration, read inputs, build networks and write outputs.
    This function:
      * parses CLI arguments
      * configures logging
      * loads the NetworkConfig for the requested stage
      * reads article and affiliation inputs
      * builds affiliation networks (yielding variants for 'all' and 'stable')
      * writes outputs (unless dry-run)

    :param argv: Optional argument list for testing (defaults to sys.argv).
    :return: None
    """

    args = _parse_args(argv)
    configure_logging(debug=args.debug)

    _logger.info("config.load", path=str(args.config), stage=args.stage)
    net_cfg: NetworkConfig = load_config_for_stage(
        config_file=args.config,
        stage_group=args.stage,
        validate_paths_exist=args.validate_paths,
    )

    _logger.info("read.articles", file=str(net_cfg.article_file_path))
    article_df = read_articles(net_cfg.article_file_path)

    _logger.info("read.affiliations", file=str(net_cfg.affiliation_file_path))
    affiliation_gdf = read_affiliations(net_cfg.affiliation_file_path)

    _logger.info("network.build", output_path=str(net_cfg.output_path))
    results = get_network_from_config(article_df, affiliation_gdf, net_cfg)

    write_outputs(results, net_cfg.output_path, args.dry_run)

    _logger.info("done")


if __name__ == "__main__":
    main()
