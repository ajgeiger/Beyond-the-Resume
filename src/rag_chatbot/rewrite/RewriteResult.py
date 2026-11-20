"""The rewritten retrieval query and what it cost."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewriteResult:
    """query is what retrieval should search for. cost_usd is 0 when no model was called."""

    query: str
    cost_usd: float = 0.0
