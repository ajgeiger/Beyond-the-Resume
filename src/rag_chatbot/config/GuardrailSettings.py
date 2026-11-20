"""Limits on questions, conversation length and monthly spend."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GuardrailSettings:
    monthly_budget_usd: float = 20.0
    max_question_chars: int = 500
    history_turns: int = 10
    max_history_chars: int = 2000
    usage_file: Path = Path("data/usage.json")
