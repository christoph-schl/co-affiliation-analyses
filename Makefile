# -------------------------------
# Configuration
# -------------------------------
IMAGE := co-affiliation-network:latest
CONTAINER_NAME := co-affiliation-network

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
		$(IMAGE) $(1)
endef


# -------------------------------
# Targets
# -------------------------------

## Build docker image
build:
	docker build -t $(IMAGE) .

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

## Run with additional arguments passed like: make network-args ARGS="--dry-run --debug"
network-args:
	$(call run_docker, create-network $(ARGS))

## Clean local outputs
clean:
	rm -rf $(DATA_DIR)/output

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
