from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Iterator

import click
import nx2vos
import structlog

from maa.config import load_config_for_stage
from maa.config.constants import CONFIGURATION_PATH, ProcessingStage
from maa.config.models import NetworkConfig
from maa.config.utils import configure_logging
from maa.dataframe.models.affiliation import read_affiliations
from maa.dataframe.models.article import read_articles
from maa.network.network import AffiliationNetworkProcessor

_logger = structlog.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

NETWORK_COUNTRY = "AUT"

AFFILIATION_DIR = Path("links")
VOS_DIR = Path("vos/map")

AFFILIATION_LINKS_PREFIX = "affiliation_links"
VOS_MAP_PREFIX = "map"
VOS_NETWORK_PREFIX = "network"


# ============================================================
# Dataclasses
# ============================================================


@dataclass(frozen=True)
class YearGapEntry:
    """Defines a single year-gap variant (gap + name suffix)."""

    gap: int
    suffix: str


@dataclass(frozen=True)
class YearGapResult:
    """Result object for a computed network variant."""

    suffix: str
    graph: Any
    link_gdf: Any


# ============================================================
# Helper Logic
# ============================================================


def iter_year_gaps(stable_gap: int) -> Iterator[YearGapEntry]:
    """Yield configured year-gap variants."""
    yield YearGapEntry(gap=0, suffix="all")
    yield YearGapEntry(gap=stable_gap, suffix="stable")


def _ensure_parent(path: Path) -> None:
    """Ensure parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _output_file_paths(output_root: Path, suffix: str) -> Dict[str, Path]:
    """Return all output file paths for a given suffix."""
    return {
        "links": output_root / AFFILIATION_DIR / f"{AFFILIATION_LINKS_PREFIX}_{suffix}.gpkg",
        "map": output_root / VOS_DIR / f"{VOS_MAP_PREFIX}_{suffix}.txt",
        "network": output_root / VOS_DIR / f"{VOS_NETWORK_PREFIX}_{suffix}.txt",
    }


def _write_outputs(graph: Any, link_gdf: Any, output_root: Path, suffix: str) -> None:
    """Write graph + link GeoDataFrame + VOS files."""
    paths = _output_file_paths(output_root, suffix)

    for p in paths.values():
        _ensure_parent(p)

    link_gdf.to_file(paths["links"])
    nx2vos.write_vos_map(G=graph.graph, fname=paths["map"])
    nx2vos.write_vos_network(G=graph.graph, fname=paths["network"])


# ============================================================
# Core Processing Logic
# ============================================================


def get_network_from_config(
    article_df: Any, affiliation_gdf: Any, net_cfg: NetworkConfig
) -> Generator[YearGapResult, None, None]:
    """
    Build affiliation networks for each configured year-gap variant.

    This includes:
      • the complete dataset ("all"), and
      • the stable co-affiliation variant ("stable"),
    as defined in the network configuration.

    :param article_df:
        DataFrame containing article metadata.
    :param affiliation_gdf:
        GeoDataFrame containing affiliation information.
    :param net_cfg:
        NetworkConfig object defining year-gap parameters and paths.
    :Yields:
        YearGapResult:
            An object containing:
                • suffix: the variant name ("all", "stable", ...)
                • graph: the constructed affiliation graph
                • link_gdf: the GeoDataFrame of computed affiliation links
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
        yield YearGapResult(yg.suffix, graph, link_gdf)


def write_outputs(results: Iterable[YearGapResult], output_path: Path, dry_run: bool) -> None:
    """Write each year-gap result to disk."""
    for result in results:
        if dry_run:
            _logger.info("dry_run.write", suffix=result.suffix, output=str(output_path))
            continue

        try:
            _write_outputs(result.graph, result.link_gdf, output_path, result.suffix)
        except Exception as exc:  # noqa intentionally broad
            _logger.error("write.failed", suffix=result.suffix, error=str(exc))
            raise
        else:
            _logger.info("write.success", suffix=result.suffix, output=str(output_path))


# ============================================================
# CLICK CLI ENTRYPOINT
# ============================================================


@click.command(name="create-network", help="Build affiliation networks from configuration.")
@click.option(
    "--config",
    "-c",
    type=click.Path(path_type=Path),
    default=CONFIGURATION_PATH,
    show_default=True,
    help="Path to config.toml",
)
@click.option(
    "--stage",
    "-s",
    default=ProcessingStage.PREPROCESSING.value,
    show_default=True,
    help="Stage group defined in the config file.",
)
@click.option("--validate-paths", is_flag=True, help="Validate paths exist before running.")
@click.option("--dry-run", is_flag=True, help="Do not write any files.")
@click.option("--debug", is_flag=True, help="Enable verbose logging.")
def main(config: Path, stage: str, validate_paths: bool, dry_run: bool, debug: bool) -> None:
    """CLI entry point for building affiliation networks."""

    configure_logging(debug=debug)

    _logger.info("config.load", config=str(config), stage=stage)
    net_cfg: NetworkConfig = load_config_for_stage(
        config_file=config,
        stage_group=stage,
        validate_paths_exist=validate_paths,
    )

    _logger.info("read.articles", file=str(net_cfg.article_file_path))
    article_df = read_articles(net_cfg.article_file_path)

    _logger.info("read.affiliations", file=str(net_cfg.affiliation_file_path))
    affiliation_gdf = read_affiliations(net_cfg.affiliation_file_path)

    _logger.info("network.build.start", output=str(net_cfg.output_path))
    results = get_network_from_config(article_df, affiliation_gdf, net_cfg)

    write_outputs(results, net_cfg.output_path, dry_run=dry_run)

    _logger.info("network.build.done")


if __name__ == "__main__":
    main()
