"""Merge ranked lists by reciprocal rank fusion."""


class ReciprocalRankFusion:
    """Each list adds 1 / (k + rank) to a chunk's score, with rank starting at 1.

    Fusion uses only positions, so dense cosine scores and BM25 scores never
    need to be put on the same scale.
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, rankings: list[list[int]]) -> list[tuple[int, float]]:
        """Positions with their fused score, best first. Ties keep the order of first appearance."""
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, position in enumerate(ranking, start=1):
                scores[position] = scores.get(position, 0.0) + 1.0 / (self.k + rank)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)
