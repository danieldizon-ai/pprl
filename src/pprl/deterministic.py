"""Deterministic linkage on keyed hashes, the baseline that Bloom filters improve on.

Each record is reduced to ``HMAC-SHA256(key, normalised identifier)``. Two
records link only when their fingerprints are identical, so any variation in
the identifier causes a missed link. This is the construction described in the
Always Foundation PPRL case study (keyed hash of a normalised mobile number);
here it is generalised to a concatenation of normalised fields so that it can
be compared with CLKs on the same synthetic data.
"""

from __future__ import annotations

import hmac
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from .encoding import Record
from .keys import SecretKey
from .matching import Links

__all__ = ["KeyedHasher"]


class KeyedHasher:
    """HMAC-SHA256 fingerprints over a fixed list of normalised fields."""

    def __init__(
        self,
        fields: Sequence[tuple[str, Callable[[str | None], str]]],
        key: SecretKey,
    ) -> None:
        if not fields:
            raise ValueError("at least one field is required")
        self._fields = list(fields)
        self._key = key.derive("deterministic/fingerprint")

    def fingerprint(self, record: Record) -> bytes | None:
        """The fingerprint, or ``None`` when any field is empty after normalisation."""
        parts = []
        for name, norm in self._fields:
            value = norm(record.get(name))
            if not value:
                return None
            parts.append(value.encode("utf-8"))
        # Unit separator prevents ambiguity between ("ab","c") and ("a","bc").
        return hmac.digest(self._key, b"\x1f".join(parts), "sha256")

    def link(self, records_a: Iterable[Record], records_b: Iterable[Record]) -> Links:
        """Exact-match links. Fingerprints shared by several records link all combinations."""
        index: dict[bytes, list[int]] = defaultdict(list)
        for j, r in enumerate(records_b):
            fp = self.fingerprint(r)
            if fp is not None:
                index[fp].append(j)
        pa: list[int] = []
        pb: list[int] = []
        for i, r in enumerate(records_a):
            fp = self.fingerprint(r)
            if fp is None:
                continue
            for j in index.get(fp, ()):
                pa.append(i)
                pb.append(j)
        a = np.asarray(pa, dtype=np.int64)
        return Links(a, np.asarray(pb, dtype=np.int64), np.ones(a.size, dtype=np.float64))
