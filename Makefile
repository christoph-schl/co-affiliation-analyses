# -------------------------------
# Configuration
# -------------------------------

IMAGE := co-affiliation-network
GIT_TAG := $(shell git describe --tags --exact-match 2>/dev/null)

CONTAINER_NAME := co-affiliation-network

UTAG := metalabvienna

CONFIG_DIR := $(PWD)/config
DATA_DIR := $(PWD)/data

# Default target
.DEFAULT_GOAL := help



# -------------------------------
# Helper Functions
# -------------------------------

define run_docker
	docker run --rm -it \
		--name $(CONTAINER_NAME) \
		-v $(CONFIG_DIR):/app/config \
		-v $(DATA_DIR):/app/data \
		$(UTAG)/$(IMAGE) $(1)
endef


# -------------------------------
# Targets
# -------------------------------

build:
ifndef GIT_TAG
	@echo "No Git tag found, building image without version tag..."
	docker build -t $(UTAG)/$(IMAGE):latest .
else
	@echo "Building image for Git tag $(GIT_TAG)..."
	docker build \
		-t $(UTAG)/$(IMAGE):$(GIT_TAG) \
		-t $(UTAG)/$(IMAGE):latest .
endif

## Open a shell inside the container
shell:
	$(call run_docker, /bin/bash)

## Run the create-network CLI inside the container
network:
	$(call run_docker, create-network)

## Run the create-network CLI inside the container
gravity:
	$(call run_docker, create-znib-gravity-model)

## Run the create-default-config CLI inside the container
default-config:
	$(call run_docker, create-default-config)

## Run the enrich-edges CLI inside the container
enrich-edges:
	$(call run_docker, enrich-edges)

## Run with additional arguments passed like: make network-args ARGS="--dry-run --debug"
network-args:
	$(call run_docker, create-network $(ARGS))

## Clean local outputs
clean:
	rm -rf $(DATA_DIR)/output

## Push docker image to Docker Hub (make sure you are logged in)
push:
ifndef GIT_TAG
	$(error Not on a Git tag. Please create a Git tag (e.g. v0.9.0) before pushing.)
endif
	@echo "Pushing versioned image..."
	docker push $(UTAG)/$(IMAGE):$(GIT_TAG)

	@echo "Pushing latest image..."
	docker push $(UTAG)/$(IMAGE):latest


## Show available commands
help:
	@echo ""
	@echo "Available commands:"
	@echo "  make build          Build docker image"
	@echo "  make shell          Drop into container shell"
	@echo "  make network        Run create-network inside docker"
	@echo "  make gravity        Run create-znib-gravity-model inside docker"
	@echo "  make network-args ARGS=\"--dry-run\""
	@echo "  make clean          Remove generated output"
	@echo ""
