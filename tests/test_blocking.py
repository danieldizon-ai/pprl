from itertools import pairwise

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from pprl.blocking import HammingLSHBlocker, KeyedBlocker, full_comparison, join_on_keys
from pprl.encoding import CLKEncoder, Record
from pprl.experiments import DEFAULT_SCHEMA
from pprl.keys import SecretKey
from pprl.normalise import normalise_date, soundex

KEY = SecretKey(b"k" * 32)

keys_strategy = st.lists(st.integers(0, 6), max_size=25)


@given(keys_strategy, keys_strategy, st.one_of(st.none(), st.integers(1, 4)))
def test_join_matches_naive_nested_loop(
    ka: list[int], kb: list[int], max_block: int | None
) -> None:
    ra = np.arange(len(ka), dtype=np.int64)
    rb = np.arange(len(kb), dtype=np.int64)
    ia, ib = join_on_keys(
        ra,
        np.array(ka, dtype=np.int64),
        rb,
        np.array(kb, dtype=np.int64),
        max(len(kb), 1),
        max_block=max_block,
    )
    expected = {
        (i, j)
        for i, x in enumerate(ka)
        for j, y in enumerate(kb)
        if x == y and (max_block is None or (ka.count(x) <= max_block and kb.count(y) <= max_block))
    }
    got = set(zip(ia.tolist(), ib.tolist(), strict=True))
    assert got == expected
    assert len(got) == ia.size  # de-duplicated


def _family_yob(r: Record) -> str | None:
    s, d = soundex(r.get("family_name") or ""), normalise_date(r.get("dob"))
    return f"{s}|{d[:4]}" if s and d else None


def test_keyed_blocker_groups_by_protected_key() -> None:
    a = [{"family_name": "Smith", "dob": "1980-01-01"}, {"family_name": "Lee", "dob": "1990-01-01"}]
    b = [
        {"family_name": "Smyth", "dob": "1980-06-30"},
        {"family_name": "Li", "dob": "1990-02-02"},
        {"family_name": "Smith", "dob": "1999-01-01"},
    ]
    blocker = KeyedBlocker([_family_yob], KEY)
    ia, ib = blocker.candidates(a, b)
    assert set(zip(ia.tolist(), ib.tolist(), strict=True)) == {(0, 0), (1, 1)}
    ids = blocker.block_ids(a)
    assert all(len(k) == 16 for k in ids)  # opaque, fixed-size identifiers


def test_lsh_always_pairs_identical_encodings() -> None:
    recs = [
        {"given_name": "Ava", "family_name": "Kim", "dob": "2001-01-01", "postcode": "3000"},
        {"given_name": "Leo", "family_name": "Rossi", "dob": "1950-12-31", "postcode": "4000"},
    ]
    enc = CLKEncoder(DEFAULT_SCHEMA, KEY)
    e = enc.encode(recs)
    lsh = HammingLSHBlocker(enc.output_length, 24, 10)
    ia, ib = lsh.candidates(e, e)
    pairs = set(zip(ia.tolist(), ib.tolist(), strict=True))
    assert {(0, 0), (1, 1)} <= pairs


def test_lsh_collision_probability_is_monotone() -> None:
    lsh = HammingLSHBlocker(1024, 24, 30)
    ps = [lsh.collision_probability(d) for d in (0.0, 0.05, 0.1, 0.3)]
    assert ps[0] == 1.0
    assert all(x >= y for x, y in pairwise(ps))


def test_full_comparison() -> None:
    ia, ib = full_comparison(2, 3)
    assert list(zip(ia.tolist(), ib.tolist(), strict=True)) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
