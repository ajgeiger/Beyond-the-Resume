---
title: RAG Chatbot
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Answers questions about Alex Geiger's projects and experience
---

# RAG Chatbot API

This is the Space card. `scripts/push_to_space.sh` copies it over the root
README.md when it uploads the repository, and Hugging Face reads the front
matter above to build the root Dockerfile and route traffic to port 7860.

The Space serves the API only:

| Route | What it does |
| --- | --- |
| GET /health | Index version and chunk count. Always open. |
| GET /info | Embedding model, language model, chunking and retrieval settings. |
| POST /chat | An answer with numbered citations and the source documents. |
| POST /retrieve | The ranked passages for a query, without a model call. |

## Secrets and variables

Set these under Settings, Variables and secrets. They reach the container as
environment variables and are never stored in the repository.

| Name | Kind | Purpose |
| --- | --- | --- |
| ANTHROPIC_API_KEY | secret | Claude Sonnet 5.5 writes the answers. |
| OPENAI_API_KEY | secret | Optional. Used when LLM_PROVIDER is openai, or when no Anthropic key is set. |
| CHATBOT_API_KEY | secret | Optional. Callers must send it in X-API-Key. |
| LLM_PROVIDER | variable | auto (default), anthropic, openai or fake. |
| LLM_MODEL | variable | Optional. A different model for the active provider. |
| MONTHLY_BUDGET_USD | variable | Spend cap per month. Default 20. |

With no key set the Space still runs, using the offline fake model that quotes
the retrieved passages.

## Hardware

The free CPU basic tier (2 vCPU, 16 GB) is enough: the embedding model is 90 MB
and a query takes about 20 ms before the model call. The reranker is off by
default; set RERANK=true to turn it on. Free Spaces sleep after about 48 hours
without traffic, so the first call after a quiet spell waits for a cold start.
The spend tracker lives in /tmp and resets when the Space restarts, which is
why the cap is also worth setting in the provider's billing console.
