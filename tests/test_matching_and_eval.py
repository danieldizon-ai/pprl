import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from pprl.attacks import frequency_alignment_attack
from pprl.deterministic import KeyedHasher
from pprl.evaluation import blocking_quality, linkage_quality, sweep
from pprl.keys import SecretKey
from pprl.matching import one_to_one, threshold
from pprl.normalise import normalise_date, normalise_name

pair_lists = st.lists(
    st.tuples(st.integers(0, 8), st.integers(0, 8), st.floats(0, 1, allow_nan=False)),
    max_size=40,
    unique_by=lambda x: (x[0], x[1]),
)


@given(pair_lists, st.floats(0, 1))
def test_one_to_one_properties(pairs: list[tuple[int, int, float]], t: float) -> None:
    ia = np.array([p[0] for p in pairs], dtype=np.int64)
    ib = np.array([p[1] for p in pairs], dtype=np.int64)
    s = np.array([p[2] for p in pairs], dtype=np.float64)
    links = one_to_one(ia, ib, s, t)
    assert len(set(links.a.tolist())) == len(links)
    assert len(set(links.b.tolist())) == len(links)
    assert np.all(links.score >= t)
    # Maximality: every unchosen pair above threshold conflicts with a chosen one.
    chosen_a, chosen_b = set(links.a.tolist()), set(links.b.tolist())
    for x, y, sc in pairs:
        if sc >= t:
            assert x in chosen_a or y in chosen_b
    assert len(threshold(ia, ib, s, t)) >= len(links)


def test_one_to_one_prefers_higher_scores() -> None:
    ia = np.array([0, 0, 1], dtype=np.int64)
    ib = np.array([0, 1, 1], dtype=np.int64)
    s = np.array([0.9, 0.95, 0.99])
    assert one_to_one(ia, ib, s, 0.5).as_set() == {(1, 1), (0, 0)}


def test_linkage_and_blocking_quality() -> None:
    q = linkage_quality({(0, 0), (1, 2)}, {(0, 0), (1, 1)})
    assert (q.true_positives, q.false_positives, q.false_negatives) == (1, 1, 1)
    assert q.precision == q.recall == q.f1 == 0.5
    bq = blocking_quality(np.array([0, 1]), np.array([0, 0]), 2, 2, {(0, 0), (1, 1)})
    assert bq.reduction_ratio == 0.5
    assert bq.pair_completeness == 0.5


def test_sweep_is_monotone_in_link_count() -> None:
    ia = np.array([0, 1, 2], dtype=np.int64)
    ib = np.array([0, 1, 2], dtype=np.int64)
    s = np.array([0.6, 0.8, 0.95])
    res = sweep(ia, ib, s, {(0, 0), (1, 1), (2, 2)})
    counts = [q.true_positives + q.false_positives for _, q in res]
    assert counts == sorted(counts, reverse=True)


def test_keyed_hasher_exact_only() -> None:
    h = KeyedHasher([("n", normalise_name), ("d", normalise_date)], SecretKey(b"k" * 32))
    a: list[dict[str, str | None]] = [{"n": "Anna Lee", "d": "1990-01-01"}, {"n": "Bob", "d": None}]
    b = [{"n": "ANNA  LEE", "d": "1990-01-01"}, {"n": "Ana Lee", "d": "1990-01-01"}]
    assert h.link(a, b).as_set() == {(0, 0)}
    assert h.fingerprint(a[1]) is None
    h2 = KeyedHasher([("x", normalise_name), ("y", normalise_name)], SecretKey(b"k" * 32))
    assert h2.fingerprint({"x": "ab", "y": "c"}) != h2.fingerprint({"x": "a", "y": "bc"})


def test_frequency_attack_recovers_deterministic_encodings() -> None:
    values = ["smith"] * 50 + ["jones"] * 30 + ["lee"] * 10 + ["kim"] * 5
    encodings = [hash(("secret", v)) for v in values]
    res = frequency_alignment_attack(encodings, values, ["smith", "jones", "lee"], top=3)
    assert res.reidentified == 90
    assert 0.0 <= res.reidentification_rate <= 1.0
    unique = frequency_alignment_attack(list(range(len(values))), values, ["smith"], top=1)
    assert unique.reidentified <= 1
