"""Loading the config file and the environment overrides."""

from pathlib import Path

from rag_chatbot.config.Settings import Settings

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "settings.toml"


def test_default_config_loads_with_paths_from_the_repository_root():
    settings = Settings.load(CONFIG, env={})
    assert settings.index_dir == ROOT / "data" / "processed" / "index"
    assert settings.chunking.chunk_overlap_tokens < settings.chunking.chunk_size_tokens
    assert settings.llm.anthropic_model == "claude-sonnet-5-5"
    assert settings.llm.rewrite_model == "claude-haiku-5-5"
    assert settings.llm.resolved_provider == "fake"
    assert settings.guardrails.history_turns == 10
    assert settings.guardrails.max_history_chars == 2000
    assert settings.retrieval.default_top_k == 8


def test_environment_overrides_the_file():
    env = {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "key",
        "LLM_MODEL": "bigger-model",
        "LLM_REWRITE_MODEL": "claude-haiku-5-5",
        "PORT": "7860",
        "MONTHLY_BUDGET_USD": "5",
        "CHATBOT_API_KEY": "secret",
    }
    settings = Settings.load(CONFIG, env=env)
    assert settings.llm.resolved_provider == "openai"
    assert settings.llm.model == "bigger-model"
    assert settings.llm.rewrite_model == "claude-haiku-5-5"
    assert settings.api.port == 7860
    assert settings.api.api_key == "secret"
    assert settings.guardrails.monthly_budget_usd == 5.0


def test_an_anthropic_key_switches_auto_to_claude():
    settings = Settings.load(CONFIG, env={"ANTHROPIC_API_KEY": "key"})
    assert settings.llm.resolved_provider == "anthropic"
    assert settings.llm.model == "claude-sonnet-5-5"


def test_blank_values_from_an_env_file_are_ignored():
    env = {"LLM_PROVIDER": "", "PORT": "", "LLM_MODEL": "", "MONTHLY_BUDGET_USD": " "}
    settings = Settings.load(CONFIG, env=env)
    assert settings.llm.provider == "auto" and settings.api.port == 8070
