"""Blocking: reduce the ``|A| x |B|`` comparison space to plausible candidates.

Two families are provided, matching the two places blocking can happen in the
three-party protocol:

* :class:`KeyedBlocker` runs at each custodian, in the clear, and emits only
  HMAC-protected block keys. It is accurate on clean data and cheap, but leaks
  the block-size distribution to the linkage unit.
* :class:`HammingLSHBlocker` runs at the linkage unit on the encodings alone,
  using bit-sampling locality-sensitive hashing for Hamming distance
  (Indyk and Motwani 1998; applied to PPRL by Durham 2012 and
  Karapiperis and Verykios 2015).

Both return candidate pairs as two aligned ``int64`` index arrays.
"""

from __future__ import annotations

import hmac
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .encoding import BitMatrix, Record, unpack_bits
from .keys import SecretKey

__all__ = [
    "BlockingKey",
    "HammingLSHBlocker",
    "KeyedBlocker",
    "candidate_pairs_from_blocks",
    "full_comparison",
    "join_on_keys",
]

IndexArray = npt.NDArray[np.int64]
BlockingKey = Callable[[Record], str | None]
"""Clear-text blocking key function; ``None`` or ``""`` means the record is not blocked by it."""


def join_on_keys(
    rows_a: IndexArray,
    keys_a: npt.NDArray[np.int64],
    rows_b: IndexArray,
    keys_b: npt.NDArray[np.int64],
    n_b: int,
    *,
    max_block: int | None = None,
) -> tuple[IndexArray, IndexArray]:
    """Equi-join two ``(row, key)`` lists into de-duplicated candidate pairs.

    Fully vectorised: rows are sorted by key, block sizes are found with
    ``searchsorted``, and the cross product of every shared block is
    materialised with index arithmetic rather than Python loops. Blocks
    larger than ``max_block`` on either side are skipped, which bounds the cost
    of very common keys at the price of recall.
    """
    empty = np.zeros(0, dtype=np.int64)
    if keys_a.size == 0 or keys_b.size == 0:
        return empty, empty.copy()
    oa = np.argsort(keys_a, kind="stable")
    ob = np.argsort(keys_b, kind="stable")
    ka, kb = keys_a[oa], keys_b[ob]
    common = np.intersect1d(ka, kb)
    if common.size == 0:
        return empty, empty.copy()
    sa = np.searchsorted(ka, common, "left")
    ca = np.searchsorted(ka, common, "right") - sa
    sb = np.searchsorted(kb, common, "left")
    cb = np.searchsorted(kb, common, "right") - sb
    if max_block is not None:
        ok = (ca <= max_block) & (cb <= max_block)
        sa, ca, sb, cb = sa[ok], ca[ok], sb[ok], cb[ok]
    sizes = ca * cb
    total = int(sizes.sum())
    if total == 0:
        return empty, empty.copy()
    gid = np.repeat(np.arange(sizes.size), sizes)
    offset = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    width = cb[gid]
    ia = rows_a[oa[sa[gid] + offset // width]]
    ib = rows_b[ob[sb[gid] + offset % width]]
    flat = np.unique(ia * np.int64(n_b) + ib)
    return flat // n_b, flat % n_b


def candidate_pairs_from_blocks(
    blocks_a: dict[bytes, list[int]],
    blocks_b: dict[bytes, list[int]],
    n_b: int,
    max_block: int | None = None,
) -> tuple[IndexArray, IndexArray]:
    """All cross pairs that share at least one block, de-duplicated."""
    codes: dict[bytes, int] = {}

    def flatten(blocks: dict[bytes, list[int]]) -> tuple[IndexArray, npt.NDArray[np.int64]]:
        rows: list[int] = []
        keys: list[int] = []
        for k, members in blocks.items():
            code = codes.setdefault(k, len(codes))
            rows.extend(members)
            keys.extend([code] * len(members))
        return np.asarray(rows, dtype=np.int64), np.asarray(keys, dtype=np.int64)

    ra, ka = flatten(blocks_a)
    rb, kb = flatten(blocks_b)
    return join_on_keys(ra, ka, rb, kb, n_b, max_block=max_block)


@dataclass
class KeyedBlocker:
    """Union of several keyed blocking passes computed by the custodians.

    Each blocking key function is evaluated on the clear-text record; its value
    is protected with HMAC-SHA256 under a sub-key of the shared secret, so the
    linkage unit sees opaque block identifiers only.

    Example passes: Soundex of family name plus year of birth; first two letters
    of given name plus postcode. Several passes recover pairs that one corrupted
    field would otherwise lose.
    """

    keys: Sequence[BlockingKey]
    secret: SecretKey
    max_block: int | None = 1000

    def __post_init__(self) -> None:
        if not self.keys:
            raise ValueError("at least one blocking key is required")
        self._hmac_key = self.secret.derive("blocking/keyed")

    def block_ids(self, records: Iterable[Record]) -> dict[bytes, list[int]]:
        """Map each protected block identifier to the row indices it contains."""
        blocks: dict[bytes, list[int]] = defaultdict(list)
        for row, record in enumerate(records):
            for pass_no, fn in enumerate(self.keys):
                value = fn(record)
                if not value:
                    continue
                msg = pass_no.to_bytes(2, "big") + value.encode("utf-8")
                blocks[hmac.digest(self._hmac_key, msg, "sha256")[:16]].append(row)
        return dict(blocks)

    def candidates(
        self, records_a: Iterable[Record], records_b: Iterable[Record]
    ) -> tuple[IndexArray, IndexArray]:
        list_b = list(records_b)
        return candidate_pairs_from_blocks(
            self.block_ids(records_a), self.block_ids(list_b), len(list_b), self.max_block
        )


@dataclass
class HammingLSHBlocker:
    """Bit-sampling LSH over encodings.

    Each of ``tables`` hash tables samples ``bits_per_table`` bit positions;
    two encodings collide in a table when they agree on all sampled bits.
    For encodings at normalised Hamming distance *d*, the probability of
    becoming a candidate is ``1 - (1 - (1 - d)**r)**t`` for ``r`` bits per
    table and ``t`` tables, so ``r`` controls precision and ``t`` recall.

    Sampling positions need not be secret; they are drawn from a seeded
    generator so that runs are reproducible.
    """

    length: int
    bits_per_table: int = 24
    tables: int = 30
    seed: int = 0
    max_block: int | None = 1000

    def __post_init__(self) -> None:
        if not 0 < self.bits_per_table <= min(self.length, 56):
            raise ValueError("bits_per_table must be in (0, min(length, 56)]")
        if not 0 < self.tables < 128:
            raise ValueError("tables must be in (0, 128)")
        rng = np.random.default_rng(self.seed)
        self._samples = np.stack(
            [
                rng.choice(self.length, size=self.bits_per_table, replace=False)
                for _ in range(self.tables)
            ]
        )

    def collision_probability(self, normalised_distance: float) -> float:
        """Theoretical probability that a pair at this distance becomes a candidate."""
        agree = (1.0 - normalised_distance) ** self.bits_per_table
        return float(1.0 - (1.0 - agree) ** self.tables)

    def table_keys(self, encodings: BitMatrix) -> npt.NDArray[np.int64]:
        """Hash key of every row in every table, shape ``(tables, n)``.

        Keys are the sampled bits read as an integer, offset by table so that
        keys from different tables never collide.
        """
        bits = unpack_bits(encodings, self.length)
        weights = np.left_shift(np.int64(1), np.arange(self.bits_per_table, dtype=np.int64))
        keys = np.stack([bits[:, s].astype(np.int64) @ weights for s in self._samples])
        table_offset = np.arange(self.tables, dtype=np.int64)[:, None] << np.int64(
            self.bits_per_table
        )
        out: npt.NDArray[np.int64] = keys + table_offset
        return out

    def candidates(self, a: BitMatrix, b: BitMatrix) -> tuple[IndexArray, IndexArray]:
        ka = self.table_keys(a)
        kb = self.table_keys(b)
        rows_a = np.tile(np.arange(a.shape[0], dtype=np.int64), self.tables)
        rows_b = np.tile(np.arange(b.shape[0], dtype=np.int64), self.tables)
        return join_on_keys(
            rows_a, ka.ravel(), rows_b, kb.ravel(), b.shape[0], max_block=self.max_block
        )


def full_comparison(n_a: int, n_b: int) -> tuple[IndexArray, IndexArray]:
    """Every pair, for small inputs and for measuring blocking loss."""
    ia, ib = np.meshgrid(
        np.arange(n_a, dtype=np.int64), np.arange(n_b, dtype=np.int64), indexing="ij"
    )
    return ia.ravel(), ib.ravel()
