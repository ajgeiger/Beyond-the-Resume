# The chatbot API image. Hugging Face Docker Spaces build this file as is.
# The index is committed, so the image needs no source documents.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Spaces run the container as user 1000.
RUN useradd --create-home --uid 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# CPU only torch keeps the image far smaller than the default CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install ".[server]"

USER user

# Download the embedding model at build time so start up works offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user configs ./configs
COPY --chown=user data/processed ./data/processed

# Spend tracking lives in /tmp and resets on restart, so keep the monthly cap conservative.
# Hosting platforms inject PORT. 7860 is the fallback when it is unset.
ENV HF_HUB_OFFLINE=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    USAGE_FILE=/tmp/usage.json

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"

CMD ["python", "-m", "rag_chatbot", "serve"]
