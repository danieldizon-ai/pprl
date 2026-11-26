"""Similarity between packed Bloom filters.

All functions are vectorised over aligned pairs of rows, so a linkage unit can
score millions of candidate pairs without Python-level loops.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .encoding import BitMatrix

__all__ = ["dice", "dice_pairs", "hamming_pairs", "jaccard_pairs", "popcount", "qgram_dice"]

FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.int64]


def popcount(bits: BitMatrix) -> npt.NDArray[np.int64]:
    """Number of set bits in each row."""
    return np.bitwise_count(bits).sum(axis=1, dtype=np.int64)


def _check(a: BitMatrix, b: BitMatrix) -> None:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("encodings must be 2-D with the same number of words")


_CHUNK = 1 << 18


def _intersections(a: BitMatrix, b: BitMatrix, ia: IndexArray, ib: IndexArray) -> IndexArray:
    """``|a[ia[i]] & b[ib[i]]|`` computed in chunks to bound peak memory."""
    out = np.empty(ia.shape, dtype=np.int64)
    for s in range(0, ia.size, _CHUNK):
        e = s + _CHUNK
        out[s:e] = np.bitwise_count(a[ia[s:e]] & b[ib[s:e]]).sum(axis=1, dtype=np.int64)
    return out


def dice_pairs(a: BitMatrix, b: BitMatrix, ia: IndexArray, ib: IndexArray) -> FloatArray:
    """Sørensen-Dice coefficient ``2|A∩B| / (|A|+|B|)`` for pairs ``(a[ia[i]], b[ib[i]])``.

    Two empty filters are given similarity 0, not 1: a record with no usable
    identifiers must never match anything.
    """
    _check(a, b)
    if ia.shape != ib.shape:
        raise ValueError("index arrays must have the same shape")
    if ia.size == 0:
        return np.zeros(0, dtype=np.float64)
    pa = popcount(a)[ia]
    pb = popcount(b)[ib]
    inter = _intersections(a, b, ia, ib)
    denom = pa + pb
    out = np.zeros(ia.shape, dtype=np.float64)
    nz = denom > 0
    out[nz] = 2.0 * inter[nz] / denom[nz]
    return out


def jaccard_pairs(a: BitMatrix, b: BitMatrix, ia: IndexArray, ib: IndexArray) -> FloatArray:
    """Jaccard coefficient ``|A∩B| / |A∪B|`` for aligned pairs."""
    _check(a, b)
    if ia.size == 0:
        return np.zeros(0, dtype=np.float64)
    inter = _intersections(a, b, ia, ib)
    union = popcount(a)[ia] + popcount(b)[ib] - inter
    out = np.zeros(ia.shape, dtype=np.float64)
    nz = union > 0
    out[nz] = inter[nz] / union[nz]
    return out


def hamming_pairs(a: BitMatrix, b: BitMatrix, ia: IndexArray, ib: IndexArray) -> IndexArray:
    """Hamming distance for aligned pairs."""
    _check(a, b)
    if ia.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.bitwise_count(a[ia] ^ b[ib]).sum(axis=1, dtype=np.int64)


def dice(a: BitMatrix, b: BitMatrix) -> float:
    """Dice coefficient of two single packed filters (1-D or ``(1, words)``)."""
    a2 = a.reshape(1, -1)
    b2 = b.reshape(1, -1)
    zero = np.zeros(1, dtype=np.int64)
    return float(dice_pairs(a2, b2, zero, zero)[0])


def qgram_dice(x: set[str], y: set[str]) -> float:
    """Dice coefficient of two token sets in the clear (the unencoded reference)."""
    if not x and not y:
        return 0.0
    return 2.0 * len(x & y) / (len(x) + len(y))
