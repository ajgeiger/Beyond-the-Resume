"""Track spending per calendar month and enforce the monthly cap."""

import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path


class CostTracker:
    """Keep a running dollar total per month in a small JSON file.

    Once the month's total reaches the budget, the assistant stops calling
    the model until the next month starts. The file survives restarts.
    """

    def __init__(self, path: Path, budget_usd: float, clock: Callable[[], datetime] | None = None):
        self.path = Path(path)
        self.budget_usd = budget_usd
        self.clock = clock or (lambda: datetime.now(UTC))
        self.lock = threading.Lock()

    def month_key(self) -> str:
        return self.clock().strftime("%Y-%m")

    def read(self) -> dict:
        if not self.path.is_file():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def spent(self) -> float:
        return float(self.read().get(self.month_key(), {}).get("usd", 0.0))

    def remaining(self) -> float:
        return max(self.budget_usd - self.spent(), 0.0)

    def is_exhausted(self) -> bool:
        return self.spent() >= self.budget_usd

    def record(self, cost_usd: float) -> float:
        """Add one call's cost and return the new month total."""
        with self.lock:
            totals = self.read()
            month = totals.setdefault(self.month_key(), {"usd": 0.0, "questions": 0})
            month["usd"] = round(month["usd"] + cost_usd, 6)
            month["questions"] += 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(totals, indent=2) + "\n", encoding="utf-8")
            return month["usd"]
