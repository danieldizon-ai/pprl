"""Reproducible evaluation on synthetic data.

Run ``pprl-experiments --out docs/results.md`` (or ``python -m pprl.experiments``)
to regenerate every number in ``docs/evaluation.md``. All randomness is seeded.

Two experiments:

1. **Linkage quality.** Deterministic keyed hashing against CLKs with and
   without hardening, across four corruption levels. Each method's threshold
   is chosen on a separate *calibration* data set (different seed) and then
   applied unchanged to the *test* data sets, so reported precision and recall
   are not tuned on the data they describe.
2. **Frequency attack.** How often an honest-but-curious linkage unit,
   holding encodings and a public name-frequency list, recovers family names.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .attacks import frequency_alignment_attack, rows_as_tokens
from .blocking import HammingLSHBlocker
from .deterministic import KeyedHasher
from .encoding import CLKEncoder, FieldSpec, Hardening, Record
from .evaluation import LinkageQuality, blocking_quality, linkage_quality, sweep
from .keys import SecretKey
from .matching import one_to_one
from .normalise import (
    DateParts,
    Positional,
    QGrams,
    normalise_date,
    normalise_digits,
    normalise_name,
)
from .similarity import dice_pairs
from .synthetic import FAMILY_NAMES, LinkageDataset, generate_linkage_dataset

__all__ = ["DEFAULT_SCHEMA", "main"]

DEFAULT_SCHEMA: tuple[FieldSpec, ...] = (
    FieldSpec("given_name", normalise_name, QGrams(2), hashes=10),
    FieldSpec("family_name", normalise_name, QGrams(2), hashes=10),
    FieldSpec("dob", normalise_date, DateParts(), hashes=12),
    FieldSpec("postcode", normalise_digits, Positional(), hashes=6),
)

# LSH parameters per hardening. Sparse filters need more sampled bits per table
# for the same selectivity; folded filters are denser and need fewer.
_LSH = {"none": (32, 40), "balance": (32, 40), "xor_fold": (20, 40), "rule_90": (20, 40)}


@dataclass(frozen=True)
class Method:
    name: str
    hardening: tuple[Hardening, ...]
    lsh_key: str


METHODS: tuple[Method, ...] = (
    Method("CLK", (), "none"),
    Method("CLK + balance", (Hardening.BALANCE,), "balance"),
    Method("CLK + XOR fold", (Hardening.XOR_FOLD,), "xor_fold"),
    Method("CLK + rule 90", (Hardening.RULE_90,), "rule_90"),
)


@dataclass
class Row:
    method: str
    level: str
    precision: list[float]
    recall: list[float]
    f1: list[float]
    reduction_ratio: list[float]
    pair_completeness: list[float]
    threshold: float | None
    seconds: list[float]


def _fmt(xs: Sequence[float]) -> str:
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{statistics.fmean(xs):.3f} ± {statistics.stdev(xs):.3f}"


def _score(method: Method, key: SecretKey, ds: LinkageDataset) -> tuple[np.ndarray, ...]:
    enc = CLKEncoder(DEFAULT_SCHEMA, key, 1024, method.hardening)
    a = enc.encode(ds.a)
    b = enc.encode(ds.b)
    r, t = _LSH[method.lsh_key]
    ia, ib = HammingLSHBlocker(enc.output_length, r, t, seed=7).candidates(a, b)
    return ia, ib, dice_pairs(a, b, ia, ib)


def _calibrate(method: Method, key: SecretKey, level: str, n: int, seed: int) -> float:
    ds = generate_linkage_dataset(n, n, n // 2, corruption=level, seed=seed)
    ia, ib, s = _score(method, key, ds)
    best_t, best_q = 0.8, LinkageQuality(0, 0, len(ds.true_links))
    for t, q in sweep(ia, ib, s, ds.true_links):
        if q.f1 > best_q.f1:
            best_t, best_q = t, q
    return best_t


def linkage_experiment(n: int, seeds: Sequence[int], levels: Sequence[str]) -> list[Row]:
    key = SecretKey.generate()
    rows: list[Row] = []
    det = KeyedHasher(
        [("given_name", normalise_name), ("family_name", normalise_name), ("dob", normalise_date)],
        key,
    )
    for level in levels:
        test_sets = [
            generate_linkage_dataset(n, n, n // 2, corruption=level, seed=s) for s in seeds
        ]
        r = Row("Keyed hash (exact)", level, [], [], [], [], [], None, [])
        for ds in test_sets:
            t0 = time.perf_counter()
            q = linkage_quality(det.link(ds.a, ds.b), ds.true_links)
            r.seconds.append(time.perf_counter() - t0)
            r.precision.append(q.precision)
            r.recall.append(q.recall)
            r.f1.append(q.f1)
        rows.append(r)
        for m in METHODS:
            t_star = _calibrate(m, key, level, n, seed=10_000 + len(level))
            r = Row(m.name, level, [], [], [], [], [], t_star, [])
            for ds in test_sets:
                t0 = time.perf_counter()
                ia, ib, s = _score(m, key, ds)
                q = linkage_quality(one_to_one(ia, ib, s, t_star), ds.true_links)
                r.seconds.append(time.perf_counter() - t0)
                bq = blocking_quality(ia, ib, n, n, ds.true_links)
                r.precision.append(q.precision)
                r.recall.append(q.recall)
                r.f1.append(q.f1)
                r.reduction_ratio.append(bq.reduction_ratio)
                r.pair_completeness.append(bq.pair_completeness)
            rows.append(r)
    return rows


def _with_yob(records: Sequence[Record]) -> list[dict[str, str | None]]:
    out = []
    for r in records:
        d = dict(r)
        dob = normalise_date(r.get("dob"))
        d["yob"] = dob[:4] if dob else None
        out.append(d)
    return out


Encoder = Callable[[list[dict[str, str | None]]], list[bytes]]


def _no_tokens(_value: str) -> list[str]:
    return []


def attack_experiment(n: int, seeds: Sequence[int]) -> list[tuple[str, list[float], list[int]]]:
    """Re-identification rate of the top-20 frequency alignment attack on family names."""
    key = SecretKey.generate()
    configs: list[tuple[str, Encoder]] = []

    def field_bf(hardening: tuple[Hardening, ...], salt: str | None) -> Encoder:
        specs = [FieldSpec("family_name", normalise_name, QGrams(2), 20, salt_field=salt)]
        if salt:
            # The salt field sets no bits of its own; it only keys the family-name field.
            specs.append(FieldSpec(salt, normalise_digits, _no_tokens, 1))
        enc = CLKEncoder(specs, key, 512, hardening)
        return lambda recs: rows_as_tokens(enc.encode(recs))

    hasher = KeyedHasher([("family_name", normalise_name)], key)
    configs.append(
        ("Keyed hash of family name", lambda recs: [hasher.fingerprint(r) or b"" for r in recs])
    )
    configs.append(("Field-level Bloom filter", field_bf((), None)))
    configs.append(("Field-level Bloom filter + balance", field_bf((Hardening.BALANCE,), None)))
    configs.append(("Field-level Bloom filter + XOR fold", field_bf((Hardening.XOR_FOLD,), None)))
    configs.append(("Field-level Bloom filter salted by year of birth", field_bf((), "yob")))
    clk = CLKEncoder(DEFAULT_SCHEMA, key, 1024)
    configs.append(("CLK (all four fields)", lambda recs: rows_as_tokens(clk.encode(recs))))

    public = list(FAMILY_NAMES)
    results: list[tuple[str, list[float], list[int]]] = []
    for name, fn in configs:
        rates: list[float] = []
        distinct: list[int] = []
        for s in seeds:
            ds = generate_linkage_dataset(n, n, n // 2, corruption="clean", seed=s)
            recs = _with_yob(ds.a)
            res = frequency_alignment_attack(
                fn(recs), [r["family_name"] for r in recs], public, top=20
            )
            rates.append(res.reidentification_rate)
            distinct.append(res.distinct_encodings)
        results.append((name, rates, distinct))
    return results


def _threshold(t: float | None) -> str:
    return "exact" if t is None else f"{t:.2f}"


def render(
    rows: list[Row], attacks: list[tuple[str, list[float], list[int]]], n: int, seeds: Sequence[int]
) -> str:
    lines = [
        "# Evaluation results",
        "",
        "Generated by `pprl-experiments`. Do not edit by hand.",
        "",
        f"Data: synthetic, {n} records per data set, {n // 2} true matches, "
        f"{len(seeds)} test seeds ({', '.join(map(str, seeds))}); values are mean ± sample s.d.",
        "Encoding: 1024-bit CLK over given name, family name, date of birth and postcode "
        "(schema in `pprl.experiments.DEFAULT_SCHEMA`). Matching: greedy one-to-one at a "
        "threshold calibrated on a separate data set. Blocking: bit-sampling LSH.",
        "",
        "## 1. Linkage quality",
        "",
        "| Corruption | Method | Threshold | Precision | Recall | F1 "
        "| Reduction ratio | Pair completeness |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.level} | {r.method} | {_threshold(r.threshold)} | "
            f"{_fmt(r.precision)} | {_fmt(r.recall)} | {_fmt(r.f1)} | "
            f"{_fmt(r.reduction_ratio) if r.reduction_ratio else 'n/a'} | "
            f"{_fmt(r.pair_completeness) if r.pair_completeness else 'n/a'} |"
        )
    lines += [
        "",
        "## 2. Frequency alignment attack on family names",
        "",
        "Attacker: holds the encodings of one clean data set and the public frequency ranking "
        "of family names; aligns the 20 most frequent encodings with the 20 most frequent names.",
        "",
        "| Encoding | Records re-identified | Distinct encodings |",
        "|---|---|---|",
    ]
    for name, rates, distinct in attacks:
        lines.append(f"| {name} | {_fmt(rates)} | {round(statistics.fmean(distinct))} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=5000, help="records per data set")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--levels", nargs="+", default=["clean", "low", "medium", "high"])
    ap.add_argument("--out", default="-", help="output Markdown file, or - for stdout")
    args = ap.parse_args(argv)
    rows = linkage_experiment(args.n, args.seeds, args.levels)
    attacks = attack_experiment(args.n, args.seeds)
    text = render(rows, attacks, args.n, args.seeds)
    if args.out == "-":
        sys.stdout.write(text)
    else:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
