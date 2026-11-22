# Deployment

Beyond the Resume runs on Render as a Docker web service. Render is the main
host and deploys every merge to `main`. A Hugging Face Docker Space is an
optional second host that runs the same image.

## Render (main host)

### 1. Put the repository on GitHub

```bash
cd rag-chatbot
gh repo create ajgeiger/Beyond-the-Resume --public --source . --remote origin
git push -u origin main
```

Changes go through a pull request into `main`.

### 2. Create the service

`render.yaml` defines a Starter plan web service named `rag-chatbot` that
builds the Dockerfile, with `GET /health` as the health check. In the Render
dashboard, choose New, Blueprint, and pick the repository. Render reads
`render.yaml` and creates the service with auto deploy on for `main`.

### 3. Add the secrets

Render asks for the values marked `sync: false` when the Blueprint is applied.
They can also be set later under the service's Environment tab.

| Name | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | Claude Sonnet writes the answers and Claude Haiku rewrites follow ups. |
| `CHATBOT_API_KEY` | Optional. Callers must send it in `X-API-Key`. |
| `MONTHLY_BUDGET_USD` | Spend cap per month. Set to 20 in `render.yaml`. |

Render injects `PORT`, so the image's 7860 fallback is not used there. Spend
tracking is written to `/tmp` and resets when the service restarts, so also
set a monthly limit in the Anthropic console.

### 4. Check it

```bash
curl https://rag-chatbot-v8qn.onrender.com/health
python scripts/smoke_test.py --url https://rag-chatbot-v8qn.onrender.com --api-key "$CHATBOT_API_KEY"
```

### 5. Point the portfolio site at it

Set these on the portfolio site's service:

```bash
CHATBOT_API_URL=https://rag-chatbot-v8qn.onrender.com
CHATBOT_API_KEY=...                 # the same key as the chatbot service
```

### Updating

Change sources or settings, run `make sources index eval test`, commit on a
branch, open a pull request and merge it. Render rebuilds the image from the
new commit and switches over once the health check passes.

## Hugging Face Space (optional)

The same image runs as a Docker Space on the free CPU basic tier.

```bash
pip install huggingface_hub
hf auth login                      # or export HF_TOKEN=...
make space-push SPACE_ID=<hf-user>/beyond-the-resume
```

`scripts/push_to_space.sh` creates the Space if it does not exist, uploads only
the files committed at HEAD, and swaps in `deploy/huggingface/README.md` as the
Space card. Add the secrets in the Space settings, or from a shell where the
keys are already exported:

```bash
hf spaces secrets add <hf-user>/beyond-the-resume -s ANTHROPIC_API_KEY
hf spaces secrets add <hf-user>/beyond-the-resume -s CHATBOT_API_KEY      # optional
hf spaces variables add <hf-user>/beyond-the-resume -e MONTHLY_BUDGET_USD=20
```

Passing only the name reads the value from your environment, so it never
lands in shell history. Restart the Space after changing secrets. A Space does
not redeploy on merge, so run `make space-push` again after each change.

## Larger models

- A different hosted model: set `LLM_MODEL` (and `LLM_PROVIDER`).
- A self hosted open model on a Hugging Face Inference Endpoint, vLLM or any
  OpenAI compatible server: `LLM_PROVIDER=openai`, `LLM_BASE_URL=<endpoint>/v1`,
  `OPENAI_API_KEY=<endpoint token>`, `LLM_MODEL=<model id>`. Add its price to
  `[llm.pricing]` in `configs/settings.toml` so the spend cap stays accurate.
