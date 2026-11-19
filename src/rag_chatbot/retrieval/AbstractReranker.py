"""The contract for a model that rescores query and passage pairs."""

from abc import ABC, abstractmethod


class AbstractReranker(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def scores(self, query: str, passages: list[str]) -> list[float]:
        """One relevance score per passage. Higher is better."""
