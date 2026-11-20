"""A local ONNX Runtime model, such as all-MiniLM-L6-v2."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from rag_chatbot.config.EmbeddingSettings import EmbeddingSettings
from rag_chatbot.embeddings.AbstractEmbeddingModel import AbstractEmbeddingModel
from rag_chatbot.embeddings.OnnxModelCache import OnnxModelCache


class OnnxEmbeddingModel(AbstractEmbeddingModel):
    """Runs on CPU. Mean pooling and L2 norm match the sentence-transformers MiniLM pipeline."""

    DEFAULT_MAX_TOKENS = 256
    DEFAULT_MODEL_DIR = Path("models") / "all-MiniLM-L6-v2"

    def __init__(self, settings: EmbeddingSettings):
        super().__init__(settings.model_name)
        self.settings = settings
        directory = self.resolve_directory(settings)
        self.max_tokens = self.read_max_tokens(directory)
        self.tokenizer = self.load_tokenizer(directory, truncate=True)
        self.counter = self.load_tokenizer(directory, truncate=False)
        self.session = self.load_session(directory / "onnx" / "model.onnx", settings.torch_threads)
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.size = int(self.session.get_outputs()[0].shape[-1])

    @property
    def dimension(self) -> int:
        return self.size

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.size), dtype=np.float32)
        rows = [
            self.encode_batch(texts[start : start + self.settings.batch_size])
            for start in range(0, len(texts), self.settings.batch_size)
        ]
        return self.normalize(np.vstack(rows))

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_documents([self.settings.query_prefix + text])[0]

    def count_tokens(self, text: str) -> int:
        return sum(1 for flag in self.counter.encode(text).special_tokens_mask if not flag)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        attention_mask = np.array([item.attention_mask for item in encodings], dtype=np.int64)
        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array([item.ids for item in encodings], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = attention_mask
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array([item.type_ids for item in encodings], dtype=np.int64)
        hidden = self.session.run(None, feed)[0]
        return self.mean_pool(hidden, attention_mask)

    @staticmethod
    def mean_pool(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        weights = attention_mask.astype(np.float32)[:, :, None]
        summed = (hidden * weights).sum(axis=1)
        counts = np.maximum(weights.sum(axis=1), 1e-9)
        return (summed / counts).astype(np.float32)

    def resolve_directory(self, settings: EmbeddingSettings) -> Path:
        cache = OnnxModelCache()
        candidates = []
        if settings.model_dir:
            candidates.append(Path(settings.model_dir))
        candidates.append(self.DEFAULT_MODEL_DIR)
        for directory in candidates:
            if cache.is_complete(directory):
                return directory
        return cache.fetch(settings.model_name, candidates[0])

    def load_tokenizer(self, directory: Path, truncate: bool) -> Tokenizer:
        tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        if truncate:
            tokenizer.enable_truncation(max_length=self.max_tokens)
            pad_id = tokenizer.token_to_id("[PAD]")
            tokenizer.enable_padding(pad_id=0 if pad_id is None else pad_id, pad_token="[PAD]")
        return tokenizer

    def load_session(self, model_path: Path, threads: int) -> ort.InferenceSession:
        options = ort.SessionOptions()
        if threads > 0:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])

    def read_max_tokens(self, directory: Path) -> int:
        path = directory / "sentence_bert_config.json"
        if not path.is_file():
            return self.DEFAULT_MAX_TOKENS
        values = json.loads(path.read_text(encoding="utf-8"))
        return int(values.get("max_seq_length", self.DEFAULT_MAX_TOKENS))
