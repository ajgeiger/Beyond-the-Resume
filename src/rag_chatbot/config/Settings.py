"""All settings, read from configs/settings.toml with environment overrides."""

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from rag_chatbot.config.ApiSettings import ApiSettings
from rag_chatbot.config.ChunkSettings import ChunkSettings
from rag_chatbot.config.EmbeddingSettings import EmbeddingSettings
from rag_chatbot.config.GuardrailSettings import GuardrailSettings
from rag_chatbot.config.LlmSettings import LlmSettings
from rag_chatbot.config.RerankerSettings import RerankerSettings
from rag_chatbot.config.RetrievalSettings import RetrievalSettings


@dataclass(frozen=True)
class Settings:
    """Settings for every part of the pipeline.

    The config file is found from CONFIG_PATH, then configs/settings.toml
    under the current folder. Relative paths in it are read from the
    repository root, the folder above configs. Secrets only ever come from
    the environment.
    """

    chunking: ChunkSettings = field(default_factory=ChunkSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    reranker: RerankerSettings = field(default_factory=RerankerSettings)
    llm: LlmSettings = field(default_factory=LlmSettings)
    guardrails: GuardrailSettings = field(default_factory=GuardrailSettings)
    api: ApiSettings = field(default_factory=ApiSettings)
    index_dir: Path = Path("data/processed/index")
    sources_file: Path = Path("configs/sources.toml")
    root: Path = Path(".")

    DEFAULT_CONFIG = Path("configs") / "settings.toml"

    @classmethod
    def load(cls, config_path: Path | None = None, env=None) -> "Settings":
        env = os.environ if env is None else env
        path = Path(config_path or env.get("CONFIG_PATH") or cls.DEFAULT_CONFIG).resolve()
        values = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        root = path.parent.parent
        paths = values.get("paths", {})
        guardrails = dict(values.get("guardrails", {}))
        guardrails["usage_file"] = root / guardrails.get("usage_file", "data/usage.json")
        settings = cls(
            chunking=ChunkSettings(**values.get("chunking", {})),
            embedding=EmbeddingSettings(**values.get("embedding", {})),
            retrieval=RetrievalSettings(**values.get("retrieval", {})),
            reranker=RerankerSettings(**values.get("reranker", {})),
            llm=LlmSettings(**values.get("llm", {})),
            guardrails=GuardrailSettings(**guardrails),
            api=ApiSettings(**values.get("api", {})),
            index_dir=root / paths.get("index_dir", "data/processed/index"),
            sources_file=root / paths.get("sources_file", "configs/sources.toml"),
            root=root,
        )
        return settings.with_env(env)

    def with_env(self, env) -> "Settings":
        """Apply environment overrides. Hosting platforms set PORT and the secrets. Blank values count as unset."""
        env = {name: value for name, value in env.items() if str(value).strip()}
        llm = replace(
            self.llm,
            provider=env.get("LLM_PROVIDER", self.llm.provider).strip().lower(),
            base_url=env.get("LLM_BASE_URL", self.llm.base_url),
            max_output_tokens=int(env.get("MAX_OUTPUT_TOKENS", self.llm.max_output_tokens)),
            anthropic_api_key=env.get("ANTHROPIC_API_KEY", ""),
            openai_api_key=env.get("OPENAI_API_KEY", ""),
        )
        if env.get("LLM_MODEL"):
            model_field = "openai_model" if llm.resolved_provider == "openai" else "anthropic_model"
            llm = replace(llm, **{model_field: env["LLM_MODEL"]})
        guardrails = replace(
            self.guardrails,
            monthly_budget_usd=float(env.get("MONTHLY_BUDGET_USD", self.guardrails.monthly_budget_usd)),
            usage_file=Path(env["USAGE_FILE"]) if "USAGE_FILE" in env else self.guardrails.usage_file,
        )
        api = replace(
            self.api,
            host=env.get("HOST", self.api.host),
            port=int(env.get("PORT", self.api.port)),
            api_key=env.get("CHATBOT_API_KEY", self.api.api_key),
        )
        reranker = self.reranker
        if "RERANK" in env:
            reranker = replace(reranker, enabled=env["RERANK"].strip().lower() in ("1", "true", "yes"))
        index_dir = Path(env["INDEX_DIR"]) if "INDEX_DIR" in env else self.index_dir
        return replace(self, llm=llm, guardrails=guardrails, api=api, reranker=reranker, index_dir=index_dir)
