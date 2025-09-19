# Dependencies Management with UV
.DEFAULT_GOAL	:= help
PROJECT_NAME	:= math_rag
PYTHON_VERSION	:= 3.12
PATH_TO_ROOT 	:= $(shell git rev-parse --show-toplevel)

.PHONY: help
help:  ## this help section
	@awk 'BEGIN {FS = ":.*##"; printf "\nusage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Dependencies Management with UV

.PHONY: install
install: ## Install all dependencies using uv
	uv sync --extra dev

.PHONY: install-dev
install-dev: ## Install development dependencies only
	uv sync --extra dev

.PHONY: update
update: ## Update all dependencies to latest compatible versions
	uv sync --upgrade

.PHONY: lock
lock: ## Generate uv.lock file
	uv lock

.PHONY: clean-deps
clean-deps: ## Clean UV cache and virtual environment
	uv cache clean
	rm -rf .venv


##@ Manage Neo4j
# Start Neo4j container with persistent storage using docker compose
run:
	docker compose up -d

# Stop Neo4j container
stop:
	docker compose down

# Stop Neo4j container and remove volumes
destroy:
	docker compose down -v

# View Neo4j logs
logs:
	docker compose logs -f neo4j
