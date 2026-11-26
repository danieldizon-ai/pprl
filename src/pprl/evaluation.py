"""Linkage quality and blocking metrics.

Precision and recall are reported separately as well as combined. Hand and
Christen (2018) show that the F-measure weights precision and recall
differently for different classifiers, so comparing methods by F alone can
mislead; it is included because the literature reports it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .matching import Links, one_to_one

__all__ = ["BlockingQuality", "LinkageQuality", "blocking_quality", "linkage_quality", "sweep"]


@dataclass(frozen=True)
class LinkageQuality:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 1.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def linkage_quality(
    links: Links | set[tuple[int, int]], truth: set[tuple[int, int]]
) -> LinkageQuality:
    predicted = links.as_set() if isinstance(links, Links) else links
    tp = len(predicted & truth)
    return LinkageQuality(tp, len(predicted) - tp, len(truth) - tp)


@dataclass(frozen=True)
class BlockingQuality:
    candidates: int
    total_pairs: int
    true_pairs_retained: int
    true_pairs: int

    @property
    def reduction_ratio(self) -> float:
        """Share of the full comparison space removed by blocking."""
        return 1.0 - self.candidates / self.total_pairs if self.total_pairs else 0.0

    @property
    def pair_completeness(self) -> float:
        """Share of true matches that survive blocking (an upper bound on recall)."""
        return self.true_pairs_retained / self.true_pairs if self.true_pairs else 1.0


def blocking_quality(
    ia: npt.NDArray[np.int64],
    ib: npt.NDArray[np.int64],
    n_a: int,
    n_b: int,
    truth: set[tuple[int, int]],
) -> BlockingQuality:
    cands = set(zip(ia.tolist(), ib.tolist(), strict=True))
    return BlockingQuality(len(cands), n_a * n_b, len(cands & truth), len(truth))


def sweep(
    ia: npt.NDArray[np.int64],
    ib: npt.NDArray[np.int64],
    scores: npt.NDArray[np.float64],
    truth: set[tuple[int, int]],
    thresholds: npt.NDArray[np.float64] | None = None,
) -> list[tuple[float, LinkageQuality]]:
    """Linkage quality under one-to-one assignment at each threshold."""
    ts = thresholds if thresholds is not None else np.round(np.arange(0.50, 1.0001, 0.02), 2)
    return [(float(t), linkage_quality(one_to_one(ia, ib, scores, float(t)), truth)) for t in ts]
