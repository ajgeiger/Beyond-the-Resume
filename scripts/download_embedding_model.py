"""Download the official ONNX export of the embedding model into a local folder.

python scripts/download_embedding_model.py --dir models/all-MiniLM-L6-v2
"""

import argparse
from pathlib import Path

from rag_chatbot.embeddings.OnnxModelCache import OnnxModelCache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="models/all-MiniLM-L6-v2")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    path = OnnxModelCache().fetch(args.model, Path(args.dir))
    print(f"Embedding model is ready in {path}")


if __name__ == "__main__":
    main()
