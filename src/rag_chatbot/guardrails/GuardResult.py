"""The outcome of a guardrail check."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardResult:
    """allowed is False when the text must not be shown. reason says which rule fired."""

    allowed: bool
    text: str
    reason: str = ""
