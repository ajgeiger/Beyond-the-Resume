"""Scores for one retrieval setup over the evaluation set."""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class EvalReport:
    """recall_at_k is the share of on topic questions with a relevant chunk in the top k.

    mrr is the mean of 1 / rank of the first relevant chunk, counting 0 when
    none is found in the top max(k). off_topic_empty is the share of off topic
    questions that correctly came back with no results.
    """

    label: str
    questions: int
    recall_at_k: dict[int, float]
    mrr: float
    off_topic_questions: int
    off_topic_empty: float
    misses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        values = asdict(self)
        values["recall_at_k"] = {str(k): value for k, value in self.recall_at_k.items()}
        return values

    def line(self) -> str:
        recalls = "  ".join(f"R@{k} {value:.3f}" for k, value in self.recall_at_k.items())
        return f"{self.label:<16} {recalls}  MRR {self.mrr:.3f}  off topic empty {self.off_topic_empty:.2f}"
