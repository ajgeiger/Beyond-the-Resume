#!/usr/bin/env bash
# Upload the committed tree to a Hugging Face Docker Space.
#
#   SPACE_ID=your-user/rag-chatbot bash scripts/push_to_space.sh
#
# Needs: huggingface_hub (pip install huggingface_hub), and either `hf auth login`
# or HF_TOKEN in the environment. Uploads only files tracked by git at HEAD, with
# deploy/huggingface/README.md as the Space card. Secrets are set in the Space
# settings, never uploaded.
set -euo pipefail

if [[ -z "${SPACE_ID:-}" ]]; then
    echo "Set SPACE_ID, for example: make space-push SPACE_ID=your-user/rag-chatbot" >&2
    exit 1
fi

ROOT="$(git rev-parse --show-toplevel)"
STAGING="$ROOT/.space"
rm -rf "$STAGING"
mkdir -p "$STAGING"

# Only committed files, so nothing local or secret leaks into the Space.
git -C "$ROOT" archive HEAD | tar -x -C "$STAGING"
cp "$ROOT/deploy/huggingface/README.md" "$STAGING/README.md"
rm -rf "$STAGING/tests" "$STAGING/docs/images" "$STAGING/.github"

# Create the Space on first use (no error when it exists), then upload.
hf repos create "$SPACE_ID" --repo-type space --space-sdk docker --flavor cpu-basic --exist-ok
hf upload "$SPACE_ID" "$STAGING" . --repo-type space --commit-message "deploy $(git -C "$ROOT" rev-parse --short HEAD)"

echo "Uploaded. The Space builds at https://huggingface.co/spaces/$SPACE_ID"
echo "Health check once built: https://${SPACE_ID/\//-}.hf.space/health"
