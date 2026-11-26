"""End-to-end linkage on synthetic data, as the three parties would run it."""

import numpy as np

from pprl.blocking import HammingLSHBlocker
from pprl.deterministic import KeyedHasher
from pprl.encoding import CLKEncoder
from pprl.evaluation import linkage_quality
from pprl.experiments import DEFAULT_SCHEMA
from pprl.keys import SecretKey
from pprl.matching import one_to_one
from pprl.normalise import normalise_date, normalise_name
from pprl.similarity import dice_pairs
from pprl.synthetic import generate_linkage_dataset


def test_synthetic_data_is_reproducible() -> None:
    a = generate_linkage_dataset(200, 150, 100, corruption="medium", seed=5)
    b = generate_linkage_dataset(200, 150, 100, corruption="medium", seed=5)
    assert a.a == b.a and a.b == b.b and a.true_links == b.true_links
    assert len(a.true_links) == 100
    for i, j in a.true_links:
        assert a.entity_a[i] == a.entity_b[j]


def test_clk_linkage_beats_exact_matching() -> None:
    ds = generate_linkage_dataset(800, 800, 400, corruption="low", seed=11)
    key = SecretKey.generate()  # shared by the custodians only

    # Custodians encode locally and send encodings to the linkage unit.
    enc = CLKEncoder(DEFAULT_SCHEMA, key)
    a, b = enc.encode(ds.a), enc.encode(ds.b)

    # The linkage unit blocks and scores without the key.
    ia, ib = HammingLSHBlocker(enc.output_length, 32, 40).candidates(a, b)
    links = one_to_one(ia, ib, dice_pairs(a, b, ia, ib), 0.8)
    q = linkage_quality(links, ds.true_links)
    assert q.precision > 0.97
    assert q.recall > 0.90

    exact = KeyedHasher(
        [("given_name", normalise_name), ("family_name", normalise_name), ("dob", normalise_date)],
        key,
    )
    q_exact = linkage_quality(exact.link(ds.a, ds.b), ds.true_links)
    assert q.recall > q_exact.recall + 0.1
    assert np.all(links.score >= 0.8)
