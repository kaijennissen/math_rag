# Solution as described in:
# https://dida.do/blog/managing-layered-requirements-with-pip-tools
.DEFAULT_GOAL	:= help
PROJECT_NAME	:= math_rag
PYTHON_VERSION	:= 3.12
PATH_TO_ROOT 	:= $(shell git rev-parse --show-toplevel)

.PHONY: help
help:  ## this help section
	@awk 'BEGIN {FS = ":.*##"; printf "\nusage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Manage Requirements as described in https://dida.do/blog/managing-layered-requirements-with-pip-tools
# Configuration for pip-compile command
PIP_COMPILE := pip-compile --quiet --no-header --allow-unsafe --resolver=backtracking

# Directory where .in and .txt files are located.
REQ_DIR := requirements
C_FILE := constraints.txt


$(REQ_DIR)/constraints.txt: $(REQ_DIR)/*.in ## Generate the constraints.txt file without any constraints.
	CONSTRAINTS=/dev/null $(PIP_COMPILE) --strip-extras --output-file $(REQ_DIR)/$(C_FILE) $^

$(REQ_DIR)/%.txt: $(REQ_DIR)/%.in $(REQ_DIR)/$(C_FILE) ## Generate .txt requirement files from .in files using constraints.txt as constraints.
	CONSTRAINTS=$(C_FILE) $(PIP_COMPILE) --no-annotate --output-file $@ $<

.PHONY: all
all: $(REQ_DIR)/constraints.txt $(addprefix $(REQ_DIR)/, $(addsuffix .txt, $(basename $(notdir $(wildcard $(REQ_DIR)/*.in))))) ## Main target to generate all .txt requirement files.

.PHONY: clean
clean: ## Clean up generated .txt files and constraints.txt.
	rm -rf $(REQ_DIR)/constraints.txt $(addprefix $(REQ_DIR)/, $(addsuffix .txt, $(basename $(notdir $(wildcard $(REQ_DIR)/*.in)))))

.PHONY: update-requirements
update-requirements: clean all ## Update all .txt requirement files by first cleaning then regenerating them.


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
