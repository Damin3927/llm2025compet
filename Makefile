.DEFAULT_GOAL := help

DOCKER_RUNTIME := $(shell if nvidia-smi >/dev/null 2>&1; then echo "nvidia"; else echo "runc"; fi)
BASE_IMAGE_TYPE := $(shell if nvidia-smi >/dev/null 2>&1; then echo "gpu"; else echo "cpu"; fi)

.PHONY: build
build: ## Build the Docker image (auto-detects GPU support)
	docker build \
		--build-arg BASE_IMAGE_TYPE=$(BASE_IMAGE_TYPE) \
		-t llm2025compet .

.PHONY: up
up: ## Start the Docker container in the background
	DOCKER_RUNTIME=$(DOCKER_RUNTIME) \
		docker compose up -d

.PHONY: run
run: up ## Execute bash in the running container
	docker exec -it llm2025compet-app-1 bash

.PHONY: down
down: ## Stop and remove the Docker container
	docker compose down

.PHONY: help
help:
	@cat Makefile | grep -E '^[a-zA-Z_-]+:.*?## .*$$' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
