# Co-Affiliation Analysis (CAA)

**Co-Affiliation Analysis (CAA)** is a Python package for analysing **institutional pairing effects induced by authors holding multiple institutional affiliations simultaneously**.

Rather than modelling collaboration or co-authorship, CAA focuses explicitly on **concurrent multi-affiliation at the author level** and the resulting **co-affiliation structures between institutions**. These structures arise when a single author lists more than one affiliation on the same publication.

CAA provides a **reproducible, configuration-driven, dependency-aware analysis pipeline** to identify, quantify, and model these co-affiliation patterns, enabling systematic investigation of **institutional co-affiliation structures and their structural properties** within the research ecosystem.

---

## Features

* Construction of co-affiliation networks induced by simultaneous multi-affiliation
* Multiple network variants (full-range, stable)
* Estimation of Zero-Inflated Negative Binomial (ZNIB) gravity models
* Separate modelling of intra- and inter-institutional co-affiliation intensity
* Enrichment of institutional pairs with travel distance and travel-time metrics
* Identification of top-performing research organisations using mwPR
* Fully configurable plotting and export pipeline
* Dependency-aware, self-contained pipeline stages
* Command-line interface, notebooks, and Docker support

---

## Installation

CAA can be installed using either **Poetry** or **Conda**, depending on your workflow.

### Using Poetry (recommended)

```bash
pip install poetry
poetry install
poetry shell
```

### Using Conda

```bash
conda env create -f environment.yml
conda activate caa
```

Update an existing Conda environment:

```bash
conda env update -f environment.yml --prune
```

---

## Pipeline Design Philosophy

CAA is organised as a **hierarchical, dependency-aware pipeline**.

Each processing step:

* Automatically executes all required upstream steps
* Automatically creates required intermediate outputs
* Can be run independently without manual preparation

### Important Implication

You **do not need** to run pipeline steps sequentially by hand.

For example:

* Running `create_gravity_models.py` will:

  1. Create all required co-affiliation networks
  2. Generate gravity model input datasets
  3. Fit the specified ZNIB gravity models

The same principle applies to later stages such as plotting or routing enrichment:
**each script ensures that all required inputs exist before continuing**.



---

## Quick Start

CAA is controlled via a single configuration file.
Each CLI script corresponds to a fully worked example notebook in the `notebooks/` directory.

### Create a Default Configuration File

```bash
create-default-config
```

**Options**

* `--output, -o <path>` — output location
* `--force, -f` — overwrite existing configuration

---

## Configuration File Structure

CAA is controlled via a single **TOML configuration file**.
All pipeline stages are **configuration-driven** and **dependency-aware**.

### Inheritance and Dependency Model

The `[network]` section defines the **core data sources and temporal scope** of the analysis.

All other sections (`[gravity]`, `[impact]`, `[routing]`) **inherit these settings implicitly**.

This means:

* You never need to run the network stage manually
* Running a downstream stage automatically creates all required upstream outputs
* All downstream stages inherit the configuration and outputs of the network stage

For example:

* Running `create_gravity_models.py` will:

  1. Build the co-affiliation networks defined in `[network]`
  2. Generate gravity model input data
  3. Fit the specified ZNIB models

---

## Configuration Sections

### `[network]` — Core Data and Network Construction

Defines the **input data**, **output location**, and **structural parameters** for all co-affiliation networks.

This section is **mandatory** and forms the base configuration for all downstream analyses.

```toml
[network]
article_file_path = "data/scopus/articles_2013-01-01_2022-12-31.parquet"
affiliation_file_path = "data/scopus/affiliations_2013-01-01_2022-12-31.gpkg"
output_path = "data/output"
year_gap_stable_links = 2
download_if_missing = true
```

**Key concepts**

* `article_file_path` — publication-level metadata
* `affiliation_file_path` — institutional affiliation records (including geometry)
* `output_path` — root directory for all derived outputs
* `year_gap_stable_links` — minimum temporal persistence for stable co-affiliation links
* `download_if_missing` — when enabled, missing input files are automatically retrieved from Zenodo and stored at the configured locations



---

### `[gravity]` — ZNIB Gravity Model Estimation

Controls the estimation of **zero-inflated negative binomial gravity models** on co-affiliation networks.

This section **inherits all network settings** and does not require a pre-existing network.

```toml
[gravity]
routes_file_path = "data/valhalla/enriched_edges_2013-01-01_2022-12-31.parquet"
fit_models = true
```

When enabled, this stage:

1. Creates all required co-affiliation networks (if missing)
2. Constructs gravity model input datasets
3. Fits ZNIB models and writes diagnostics to disk

---

### `[impact]` — Top Performer Identification

Defines parameters for identifying and analysing **top-performing research organisations**.

This stage inherits the network configuration, automatically triggers network creation,
computes institutional impact measures, generates plots, and stores all outputs at the configured output path..

```toml
[impact]
impact_file_path = "data/impact/impact_2013-01-01_2022-12-31.csv"
min_samples = 300
max_groups = 10
```

---

### `[routing]` — Travel Distance and Time Enrichment

Controls enrichment of institutional pairs with **routing-based distance and travel-time metrics**.

This stage operates on the unfiltered co-affiliation network, automatically creating the required
network, generating all unique affiliation combinations, and enriching them with travel distance and travel-time data

```toml
[routing]
output_file_path_routes = "data/valhalla/enriched_edges_2013-01-01_2022-12-31.parquet"
valhalla_base_url = "http://localhost:8002"
```

---

## Conceptual Overview

CAA analyses **co-affiliation**, defined as the simultaneous presence of two or more institutional affiliations on a single author record within the same publication instance.

Example:

* Author A lists *Institution 1* and *Institution 2* on the same publication
* This induces a co-affiliation edge between Institution 1 and Institution 2

Aggregating these instances across publications and time yields **institutional co-affiliation networks**.

---

## Main Processing Pipeline

### 1. Create Co-Affiliation Networks

**Script:** `create_co_affiliation_networks.py`
**Notebook:** `notebooks/create_co_affiliation_networks.ipynb`

Builds all co-affiliation network variants defined in the configuration file.



---

### 2. Fit ZNIB Gravity Models

**Script:** `create_gravity_models.py`
**Notebook:** `notebooks/create_gravity_models.ipynb`

This script is **self-contained** and performs the following steps:

1. Creates all required co-affiliation networks (if not already present)
2. Builds ZNIB model input datasets
3. Fits zero-inflated negative binomial gravity models
4. Writes fitted models and diagnostics to disk

> Note: This stage automatically resolves all upstream dependencies, including network creation,
> so you do not need to run earlier steps manually.
---

### 3. Generate Plots

**Script:** `create_plots.py`
**Notebook:** `notebooks/create_plots.ipynb`

This step automatically ensures that all required networks and model outputs exist before generating figures.

---

### 4. Enrich Edges with Travel Routing Data

**Script:** `enrich_edges_with_travel_data.py`
**Notebook:** `notebooks/enrich_edges_with_travel_data.ipynb`

This script:

* Creates an unfiltered co-affiliation network if required
* Generates all institutional pairs
* Attaches distance and travel-time metrics

---

### 5. Create Networks for Top Performers

**Script:** `create_top_performers_networks.py`

Automatically builds the full-range co-affiliation network (if required), computes mwPR, and filters network connections accordingly.

---

## Running Scripts Directly

Any pipeline step can be executed independently:

```bash
python scripts/create_gravity_models.py --config config/config.toml
```

This single command is sufficient to generate networks, model inputs, and fitted ZNIB models.

---

## Command Line Interface (CLI)

All CLI commands support:

* `--config <path>`
* `--validate-paths`
* `--debug`

### Available Commands

```bash
create-default-config
create-network
create-znib-gravity-model
create-plots
enrich-edges
create-top-performers-network
```

Each command is dependency-aware and can be run independently.

---

## Docker Usage

The Docker image `metalabvienna/co-affiliation-network` provides a fully containerised execution environment including all CLI tools.

### Example: Fit Gravity Models Only

```bash
docker run --rm -it   -v "$PWD/config:/app/config"   -v "$PWD/data:/app/data"   metalabvienna/co-affiliation-network:latest   create-znib-gravity-model
```

No prior pipeline steps are required.

---

## Citation

If you use CAA in your research, please cite:

```bibtex
@software{caa_2025_v090,
  author    = {Schlager, Christoph},
  title     = {Co-Affiliation Analysis (CAA)},
  year      = {2025},
  version   = {v0.9.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17972957},
  url       = {https://doi.org/10.5281/zenodo.17972957}
}
```
