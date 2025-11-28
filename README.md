# Co-Affiliation Analysis (CAA)

Co-Affiliation Analysis (CAA) is a Python package for analysing
institutional pairing effects arising from co-affiliated
authors---researchers who hold two or more institutional affiliations
simultaneously. CAA provides tools for identifying, quantifying, and
analysing these affiliation patterns to better understand collaboration
dynamics within the research ecosystem.

------------------------------------------------------------------------

## Installation

CAA can be installed using either
[**Poetry**](https://python-poetry.org/) or
[**Conda**](https://docs.conda.io/), depending on your workflow.

### Using Poetry

``` bash
pip install poetry
poetry install
poetry shell
```

### Using Conda

``` bash
conda env create -f environment.yml
conda activate caa
```

Update:

``` bash
conda env update -f environment.yml --prune
```

------------------------------------------------------------------------

## Quick Start

After installing the package, the main workflow can be executed using
the CLI scripts in `scripts/`.\
Each script corresponds to a Jupyter notebook in `notebooks/`, providing
a full worked example of the pipeline.

### Step 1 --- Create a default configuration file

``` bash
create-default-config
```

**Options**

-   `--output, -o <path>`
-   `--force, -f`

**Examples**

``` bash
create-default-config
create-default-config --output ./config.toml
create-default-config --force
```

------------------------------------------------------------------------

## Main processing pipeline

### **1. create_co_affiliation_networks.py**

Builds all co-affiliation network variants defined in the configuration.

-   Reads network settings from the configuration file.
-   Builds **year-gap**, **full-range**, and **stable** co-affiliation
    networks.
-   Optionally validates input/output paths.
-   Writes all generated network files to the configured output folder.

Notebook: `notebooks/create_co_affiliation_networks.ipynb`

```python
from maa.network.network import create_networks_from_config
from maa.config.constants import CONFIGURATION_PATH

networks = create_networks_from_config(
        config_path=CONFIGURATION_PATH,
        write_outputs_to_file=True,
    )
```

------------------------------------------------------------------------

### **2. create_gravity_models.py**

Constructs and fits all ZNIB gravity models.

-   Loads model settings from the project configuration.
-   Builds the required ZNIB model input datasets.
-   Fits zero-inflated negative binomial model variants.
-   Outputs fitted models and diagnostic files.

Notebook: `notebooks/create_gravity_models.ipynb`

```python
from maa.znib.znib import create_znib_gravity_models_from_config
from maa.config.constants import CONFIGURATION_PATH

models = create_znib_gravity_models_from_config(
        config_path=CONFIGURATION_PATH,
        write_outputs_to_file=True,
    )

# intra model results
print(models.all.znib_intra_model.summary)
print(models.stable.znib_intra_model.summary)

# inter model results
print(models.all.znib_inter_model.summary)
print(models.stable.znib_inter_model.summary)
```

------------------------------------------------------------------------

### **3. create_plots.py**

Generates all configured visualisations.

-   Loads plot definitions from the configuration file.
-   Validates dependencies (optional).
-   Produces network figures, model plots, and supplementary
    visualisations.
-   Writes all plots to the configured plot directory.

Notebook: `notebooks/create_plots.ipynb

```python
from maa.plot.plot import create_plots_from_config
from maa.config.constants import CONFIGURATION_PATH

create_plots_from_config(
        config_path=CONFIGURATION_PATH,
        write_outputs_to_file=True,
    )
````

------------------------------------------------------------------------

### **4. enrich_edges_with_travel_data.py**

Builds and enriches affiliation edges with travel routing information.

-   Constructs an unfiltered co-affiliation network.
-   Generates all possible affiliation pairs.
-   Queries the Valhalla routing engine to attach distance/time
    metrics.
-   Saves enriched edges to the routing output folder.
-   Supports `--override-edges` to overwrite existing results.

Notebook: `notebooks/enrich_edges_with_travel_data.ipynb`

```python
from maa.routing.routing import create_and_enrich_edges_from_config
from maa.config.constants import CONFIGURATION_PATH

edges = create_and_enrich_edges_from_config(
        config_path=CONFIGURATION_PATH,
        write_outputs_to_file=True,
    )
print(edges.head())
````

------------------------------------------------------------------------

## Running Scripts Directly

``` bash
python scripts/create_co_affiliation_networks.py --config config/config.toml
python scripts/create_gravity_models.py --config config/config.toml
python scripts/create_plots.py --config config/config.toml
python scripts/enrich_edges_with_travel_data.py --config config/config.toml
```

------------------------------------------------------------------------

## Command line interface (CLI)

CAA provides several CLI entry points after installation.

All commands accept:

-   `--config <path>`
-   `--validate-paths`
-   `--debug`

------------------------------------------------------------------------

### 1. Create a Default Configuration File

Generate a ready-to-use config:

``` bash
create-default-config
```

#### Options

-   `--output, -o <path>` --- where to write the TOML file\
-   `--force, -f` --- overwrite existing config

### Examples

``` bash
create-default-config
create-default-config --output ./config.toml
create-default-config --force
```

------------------------------------------------------------------------

## 2. Create Co-Affiliation Networks

``` bash
create-network --config path/to/config.yml
```

------------------------------------------------------------------------

## 3. Fit ZNIB Gravity Models

``` bash
create-znib-gravity-model --config path/to/config.yml
```

------------------------------------------------------------------------

## 4. Generate Plots

``` bash
create-plots --config path/to/config.yml
```

------------------------------------------------------------------------

## 5. Enrich Edges with Travel Routing Data

``` bash
enrich-edges --config path/to/config.yml --override-edges
```

------------------------------------------------------------------------


## Citation

If you use CAA in your research, please cite it as:

**ToDo: Add BibTex**
