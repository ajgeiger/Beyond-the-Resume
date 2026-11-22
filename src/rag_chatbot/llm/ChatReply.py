"""The raw answer from a provider, before guardrails."""

from dataclasses import dataclass, field

from rag_chatbot.llm.TokenUsage import TokenUsage


@dataclass(frozen=True)
class ChatReply:
    text: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    # True when the model stopped at the output token limit instead of finishing.
    truncated: bool = False
