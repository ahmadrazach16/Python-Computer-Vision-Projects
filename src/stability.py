from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional

@dataclass
class StablePrediction:
    label: Optional[int]
    confidence: float
    changed: bool

class PredictionSmoother:
    """Majority-vote smoother that suppresses single-frame flicker."""

    def __init__(self, window: int = 7, min_votes: int = 4):
        self.window = deque(maxlen=window)
        self.min_votes = min_votes
        self.current: Optional[int] = None

    def update(self, label: Optional[int], confidence: float) -> StablePrediction:
        self.window.append((label, confidence))
        valid = [(n, c) for n, c in self.window if n is not None]
        if not valid:
            changed = self.current is not None
            self.current = None
            return StablePrediction(None, 0.0, changed)

        counts = Counter(n for n, _ in valid)
        winner, votes = counts.most_common(1)[0]
        if votes < self.min_votes:
            return StablePrediction(self.current, 0.0, False)

        confs = [c for n, c in valid if n == winner]
        smoothed_conf = sum(confs) / len(confs)
        changed = winner != self.current
        self.current = winner
        return StablePrediction(winner, smoothed_conf, changed)
