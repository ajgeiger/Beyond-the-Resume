# Common tasks. Run `make help` for the list.

PYTHON ?= python
PORT ?= 8070
IMAGE ?= rag-chatbot
SPACE_ID ?=

.PHONY: help install install-client install-torch download-model sources index index-force eval eval-rerank serve serve-fake smoke test lint format docker-build docker-run space-push

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install: ## Install the server, the client and the dev tools
	$(PYTHON) -m pip install -e ".[server,dev]"

install-torch: ## Optional: sentence-transformers and the reranker (CPU torch)
	$(PYTHON) -m pip install --index-url https://download.pytorch.org/whl/cpu torch
	$(PYTHON) -m pip install -e ".[server,dev,torch]"

download-model: ## Download the official MiniLM ONNX export into models/
	$(PYTHON) scripts/download_embedding_model.py --dir models/all-MiniLM-L6-v2

install-client: ## Install only the HTTP client
	$(PYTHON) -m pip install -e .

sources: ## Copy the source documents into data/raw
	$(PYTHON) -m rag_chatbot sources

index: ## Rebuild the index when a source or setting changed
	$(PYTHON) -m rag_chatbot index

index-force: ## Rebuild the index even when nothing changed
	$(PYTHON) -m rag_chatbot index --force

eval: ## Retrieval recall at k and MRR for bm25, dense and hybrid
	$(PYTHON) -m rag_chatbot eval

eval-rerank: ## The same, plus hybrid with the cross encoder reranker
	$(PYTHON) -m rag_chatbot eval --rerank

serve: ## Run the API on PORT (provider from LLM_PROVIDER and the keys in the environment)
	PORT=$(PORT) $(PYTHON) -m rag_chatbot serve

serve-fake: ## Run the API with the offline fake model, no key needed
	LLM_PROVIDER=fake PORT=$(PORT) $(PYTHON) -m rag_chatbot serve

smoke: ## Ask the running API one question through the client
	$(PYTHON) scripts/smoke_test.py --url http://127.0.0.1:$(PORT)

test: ## Unit and integration tests
	$(PYTHON) -m pytest

lint: ## Ruff lint and format check
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## Format and fix with ruff
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

docker-build: ## Build the image (same image the Hugging Face Space builds)
	docker build -t $(IMAGE) .

docker-run: ## Run the image on port 7860 with keys from .env
	docker run --rm -p 7860:7860 --env-file .env $(IMAGE)

space-push: ## Upload the committed tree to a Hugging Face Space: make space-push SPACE_ID=user/name
	SPACE_ID=$(SPACE_ID) bash scripts/push_to_space.sh
