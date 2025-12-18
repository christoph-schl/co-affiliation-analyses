# Co-Affiliation Analysis (CAA)

**Co-Affiliation Analysis (CAA)** is a Python package for analysing **institutional pairing effects induced by authors holding multiple institutional affiliations simultaneously**.

Rather than modelling collaboration or co-authorship, CAA focuses explicitly on **concurrent multi-affiliation at the author level** and the resulting **co-affiliation structures between institutions**. These structures arise when a single author lists more than one affiliation on the same publication.

CAA provides a **reproducible, configuration-driven analysis pipeline** to identify, quantify, and model these co-affiliation patterns, enabling systematic investigation of **institutional co-affiliation structures and their structural properties** within the research ecosystem.

---

## Features

* Construction of co-affiliation networks induced by simultaneous multi-affiliation
* Multiple network variants (full-range, stable)
* Estimation of Zero-Inflated Negative Binomial (ZNIB) gravity models
* Separate modelling of intra- and inter-institutional co-affiliation intensity
* Enrichment of institutional pairs with travel distance and travel time metrics
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

CAA is organised as a **hierarchical and dependency-aware pipeline**.

Each processing step:

* Automatically executes all required upstream steps
* Reuses existing intermediate outputs where available
* Can be run independently without manual preparation

### Important Implication

You **do not need** to run pipeline steps sequentially by hand.

For example:

* Running `create_gravity_models.py` will:

  1. Create all required co-affiliation networks
  2. Generate ZNIB model input datasets
  3. Fit the specified ZNIB gravity models

The same principle applies to later stages such as plotting or routing enrichment:
**each script ensures that all required inputs exist before continuing**.

This design simplifies execution, reduces user error, and ensures reproducibility.

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

> Note: This step is automatically executed by downstream pipeline stages if required.

---

### 2. Fit ZNIB Gravity Models

**Script:** `create_gravity_models.py`
**Notebook:** `notebooks/create_gravity_models.ipynb`

This script is **self-contained** and performs the following steps:

1. Creates all required co-affiliation networks (if not already present)
2. Builds ZNIB model input datasets
3. Fits zero-inflated negative binomial gravity models
4. Writes fitted models and diagnostics to disk

```python
from maa.znib.znib import create_znib_gravity_models_from_config
from maa.config.constants import CONFIGURATION_PATH

models = create_znib_gravity_models_from_config(
    config_path=CONFIGURATION_PATH,
    write_outputs_to_file=True,
)
```

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

## Docker Usagenotebooks/create_gravity_models.ipynb i

The Docker image `metalabvienna/co-affiliation-network` provides a fully containerised execution environment including all CLI tools.

### Example: Fit Gravity Models Only

```bash
docker run --rm -it \
  -v "$PWD/config:/app/config" \
  -v "$PWD/data:/app/data" \
  metalabvienna/co-affiliation-network:latest \
  create-znib-gravity-model
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

---
