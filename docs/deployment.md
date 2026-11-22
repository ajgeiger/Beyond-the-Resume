# Deployment

## 1. Put the repository on GitHub

```bash
cd rag-chatbot
gh repo create ajgeiger/Beyond-the-Resume --public --source . --remote origin   # or --private
git push -u origin main
git push -u origin feature/rag-chatbot
gh pr create --base main --head feature/rag-chatbot --title "rag chatbot service" --fill
```

Then protect `main` (Settings, Branches): require a pull request and the `ci`
checks, and allow squash merging only. Merge the pull request with squash.

## 2. Create the Hugging Face Space

```bash
pip install huggingface_hub
hf auth login                      # or export HF_TOKEN=...
make space-push SPACE_ID=<hf-user>/rag-chatbot
```

`scripts/push_to_space.sh` creates the Docker Space on the free CPU basic tier
if it does not exist, uploads only the files committed at HEAD, and swaps in
`deploy/huggingface/README.md` as the Space card.

## 3. Add the secrets

In the Space, Settings, Variables and secrets, or from a shell where the keys
are already exported:

```bash
hf spaces secrets add <hf-user>/rag-chatbot -s ANTHROPIC_API_KEY
hf spaces secrets add <hf-user>/rag-chatbot -s CHATBOT_API_KEY      # optional
hf spaces variables add <hf-user>/rag-chatbot -e MONTHLY_BUDGET_USD=20
```

Passing only the name reads the value from your environment, so it never
lands in shell history. Restart the Space after changing secrets.

## 4. Check it

```bash
curl https://<hf-user>-rag-chatbot.hf.space/health
python scripts/smoke_test.py --url https://<hf-user>-rag-chatbot.hf.space --api-key "$CHATBOT_API_KEY"
```

## 5. Point the dashboard at it

Set these where the dashboard runs:

```bash
CHATBOT_API_URL=https://<hf-user>-rag-chatbot.hf.space
CHATBOT_API_KEY=...                 # if set on the Space
```

## Updating

Change sources or settings, run `make sources index eval test`, commit, merge,
then `make space-push SPACE_ID=...` again. The Space rebuilds from the new
commit.

## Larger models

- A different hosted model: set `LLM_MODEL` (and `LLM_PROVIDER`).
- A self hosted open model on a Hugging Face Inference Endpoint, vLLM or any
  OpenAI compatible server: `LLM_PROVIDER=openai`, `LLM_BASE_URL=<endpoint>/v1`,
  `OPENAI_API_KEY=<endpoint token>`, `LLM_MODEL=<model id>`. Add its price to
  `[llm.pricing]` in `configs/settings.toml` so the spend cap stays accurate.
