"""Turning similarity scores into match decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["Links", "one_to_one", "threshold"]

IndexArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Links:
    """A set of predicted links ``(a[i], b[i])`` with their scores."""

    a: IndexArray
    b: IndexArray
    score: FloatArray

    def __len__(self) -> int:
        return int(self.a.size)

    def as_set(self) -> set[tuple[int, int]]:
        return set(zip(self.a.tolist(), self.b.tolist(), strict=True))


def threshold(ia: IndexArray, ib: IndexArray, scores: FloatArray, t: float) -> Links:
    """Every candidate pair scoring at least ``t``. May link one record many times."""
    keep = scores >= t
    return Links(ia[keep], ib[keep], scores[keep])


def one_to_one(ia: IndexArray, ib: IndexArray, scores: FloatArray, t: float) -> Links:
    """Greedy one-to-one assignment above threshold ``t``.

    Pairs are taken in descending score order, skipping any whose records are
    already linked. This is the usual choice when each data set holds at most
    one record per person. Ties are broken by index order, so the result is
    deterministic. Greedy assignment is not globally optimal (the Hungarian
    algorithm is), but on linkage score matrices, which are sparse and
    strongly diagonal, the difference is small and greedy is ``O(m log m)``.
    """
    keep = scores >= t
    ia, ib, scores = ia[keep], ib[keep], scores[keep]
    order = np.lexsort((ib, ia, -scores))
    used_a: set[int] = set()
    used_b: set[int] = set()
    chosen: list[int] = []
    for idx in order.tolist():
        x, y = int(ia[idx]), int(ib[idx])
        if x in used_a or y in used_b:
            continue
        used_a.add(x)
        used_b.add(y)
        chosen.append(idx)
    sel = np.asarray(chosen, dtype=np.int64)
    return Links(ia[sel], ib[sel], scores[sel])
