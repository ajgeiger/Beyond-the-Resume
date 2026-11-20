"""Download the official ONNX export and tokenizer for a sentence-transformers model."""

from pathlib import Path

from huggingface_hub import snapshot_download


class OnnxModelCache:
    """Fetch only the ONNX graph and tokenizer files, not the PyTorch weights."""

    PATTERNS = (
        "onnx/model.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "sentence_bert_config.json",
        "config.json",
        "1_Pooling/config.json",
        "modules.json",
    )

    def is_complete(self, directory: Path) -> bool:
        directory = Path(directory)
        return (directory / "onnx" / "model.onnx").is_file() and (directory / "tokenizer.json").is_file()

    def fetch(self, model_name: str, directory: Path) -> Path:
        directory = Path(directory)
        if self.is_complete(directory):
            return directory
        snapshot_download(repo_id=model_name, allow_patterns=list(self.PATTERNS), local_dir=str(directory))
        if not self.is_complete(directory):
            raise FileNotFoundError(f"The ONNX model or tokenizer was not downloaded to {directory} for {model_name}.")
        return directory
