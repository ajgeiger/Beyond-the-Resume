"""The optional cross encoder that reorders the fused results."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RerankerSettings:
    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    def to_dict(self) -> dict:
        return asdict(self)
