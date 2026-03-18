SHELL := /bin/bash

PYTHON ?= python
PIP ?= pip
COMPOSE_FILE ?= docker/docker-compose.yaml
MKDOCS_ADDR ?= 127.0.0.1:8000
DOCS_ARGS ?=

# Optional image inputs for test commands
IMG ?=
IMG1 ?=
IMG2 ?=

.PHONY: help install install-docs run compose-up compose-down compose-build compose-logs \
	proto check docs-serve docs-build clean hf-login download-models upload-models \
	test-http-detect test-http-compare test-grpc-detect test-grpc-compare

help:
	@echo "Useful commands:"
	@echo "  make install            - Install runtime dependencies"
	@echo "  make install-docs       - Install docs dependencies"
	@echo "  make hf-login           - Log in to HuggingFace (required before upload)"
	@echo "  make download-models    - Download built-in models from HuggingFace"
	@echo "  make upload-models      - Upload local models/ to HuggingFace"
	@echo "  make run                - Run InferPack locally"
	@echo "  make compose-up         - Start docker compose stack"
	@echo "  make compose-down       - Stop docker compose stack"
	@echo "  make compose-build      - Rebuild docker compose images"
	@echo "  make compose-logs       - Tail inferpack runtime logs"
	@echo "  make proto              - Regenerate gRPC Python stubs"
	@echo "  make check              - Compile-check key Python modules"
	@echo "  make docs-serve         - Serve docs locally (MkDocs)"
	@echo "  make docs-build         - Build docs site"
	@echo "  make clean              - Remove Python cache files"
	@echo ""
	@echo "API test helpers (set IMG/IMG1/IMG2):"
	@echo "  make test-http-detect   IMG=/path/image.jpg"
	@echo "  make test-http-compare  IMG1=/path/a.jpg IMG2=/path/b.jpg"
	@echo "  make test-grpc-detect   IMG=/path/image.jpg"
	@echo "  make test-grpc-compare  IMG1=/path/a.jpg IMG2=/path/b.jpg"

install:
	$(PIP) install -r requirements.txt

hf-login:
	$(PYTHON) -c "from huggingface_hub import login; login()"

download-models:
	$(PYTHON) scripts/download_models.py download

upload-models:
	$(PYTHON) scripts/download_models.py upload

install-docs:
	$(PIP) install -r requirements-docs.txt

run:
	PYTHONPATH=src $(PYTHON) -m inferpack.main

compose-up:
	docker compose -f $(COMPOSE_FILE) up -d --build

compose-down:
	docker compose -f $(COMPOSE_FILE) down

compose-build:
	docker compose -f $(COMPOSE_FILE) build

compose-logs:
	docker logs -f inferpack-runtime

proto:
	$(PYTHON) scripts/gen_proto.sh

check:
	$(PYTHON) -m py_compile \
		src/inferpack/main.py \
		src/inferpack/api/http/routes_pipelines.py \
		src/inferpack/api/grpc/servicer.py \
		src/inferpack/api/shared/pipeline_service.py \
		src/inferpack/stages/face/scrfd_detection.py \
		src/inferpack/stages/face/arcface_extraction.py

docs-serve: install-docs
	$(PYTHON) -m mkdocs serve -a $(MKDOCS_ADDR) $(DOCS_ARGS)

docs-build: install-docs
	$(PYTHON) -m mkdocs build

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

test-http-detect:
	@if [[ -z "$(IMG)" ]]; then echo "Usage: make test-http-detect IMG=/path/image.jpg"; exit 1; fi
	$(PYTHON) tools/test_pipeline_api.py --protocol http --face-detection "$(IMG)"

test-http-compare:
	@if [[ -z "$(IMG1)" || -z "$(IMG2)" ]]; then echo "Usage: make test-http-compare IMG1=/path/a.jpg IMG2=/path/b.jpg"; exit 1; fi
	$(PYTHON) tools/test_pipeline_api.py --protocol http --face-comparison "$(IMG1)" "$(IMG2)"

test-grpc-detect:
	@if [[ -z "$(IMG)" ]]; then echo "Usage: make test-grpc-detect IMG=/path/image.jpg"; exit 1; fi
	$(PYTHON) tools/test_pipeline_api.py --protocol grpc --face-detection "$(IMG)"

test-grpc-compare:
	@if [[ -z "$(IMG1)" || -z "$(IMG2)" ]]; then echo "Usage: make test-grpc-compare IMG1=/path/a.jpg IMG2=/path/b.jpg"; exit 1; fi
	$(PYTHON) tools/test_pipeline_api.py --protocol grpc --face-comparison "$(IMG1)" "$(IMG2)"
