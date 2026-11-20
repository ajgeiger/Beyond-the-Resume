"""Rewrite a follow up into a standalone question with a cheap Claude call."""

import anthropic

from rag_chatbot.llm.ModelPricing import ModelPricing
from rag_chatbot.llm.TokenUsage import TokenUsage
from rag_chatbot.rewrite.AbstractQueryRewriter import AbstractQueryRewriter
from rag_chatbot.rewrite.RewriteResult import RewriteResult


class ClaudeQueryRewriter(AbstractQueryRewriter):
    """Claude Haiku by default. The rewrite prompt is short and only the query is kept."""

    DEFAULT_MODEL = "claude-haiku-5-5"
    MAX_OUTPUT_TOKENS = 80
    HISTORY_MESSAGES = 4
    MAX_MESSAGE_CHARS = 500
    RULES = (
        "Rewrite the latest question as one standalone search query about Alex Geiger's work "
        "and experience. Use earlier turns only to resolve references such as 'that' or 'he'. "
        "Output only the rewritten question."
    )

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, client=None, extra_pricing=None):
        if not api_key and client is None:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        self.model = model
        self.client = client or anthropic.Anthropic(api_key=api_key)
        self.pricing = ModelPricing.for_model(model, extra_pricing)

    def rewrite(self, question: str, history: list[dict]) -> RewriteResult:
        response = self.client.messages.create(**self.payload(question, history))
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        query = " ".join(text.split()).strip().strip('"')
        usage = response.usage
        cost = TokenUsage(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        ).cost(self.pricing)
        if not query:
            raise ValueError("The rewrite model returned an empty query.")
        return RewriteResult(query, cost)

    def payload(self, question: str, history: list[dict]) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.MAX_OUTPUT_TOKENS,
            "system": self.RULES,
            "messages": [{"role": "user", "content": self.user_turn(question, history)}],
        }

    def user_turn(self, question: str, history: list[dict]) -> str:
        lines = ["Earlier turns:"]
        recent = [item for item in history if item.get("role") in ("user", "assistant") and item.get("content")]
        if not recent:
            lines.append("(none)")
        for item in recent[-self.HISTORY_MESSAGES :]:
            text = str(item["content"]).strip()[: self.MAX_MESSAGE_CHARS]
            lines.append(f"{item['role']}: {text}")
        lines.append(f"Latest question: {question}")
        return "\n".join(lines)
