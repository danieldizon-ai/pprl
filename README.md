# pprl

Privacy-preserving record linkage in Python: keyed Bloom-filter (CLK)
encodings with optional hardening, blocking, similarity scoring and
one-to-one matching, plus a synthetic-data evaluation and an attack baseline
that measure what the encodings cost in accuracy and what they leak.

Record linkage finds records that belong to the same person across data sets
held by different organisations. The privacy-preserving form does this
without the organisations disclosing names, dates of birth or other
identifiers to each other or to the party that does the matching.

```
pip install -e ".[dev]"
pytest
pprl-experiments --out docs/results.md   # regenerates every number in docs/evaluation.md
```

## Contents

- [Why this exists](#why-this-exists)
- [Protocol](#protocol)
- [Architecture](#architecture)
- [Usage](#usage)
- [Results](#results)
- [Design decisions](#design-decisions)
- [Conclusion](#conclusion)
- [Responsible use](#responsible-use)
- [Documentation](#documentation)
- [References](#references)

## Why this exists

Always Foundation published a case study, *Privacy-Preserving Record Linkage
for Gospel Impact Research* (2026), on whether independent ministries could
recognise the same consenting adult without exchanging databases. Its design
used a deterministic keyed hash (HMAC-SHA256) of a normalised mobile number.
The study left two technical questions open:

1. **Accuracy was not evaluated.** No precision, recall or false-match rate
   was measured.
2. **Bloom filters were not adopted**, partly because of published
   cryptanalysis of Bloom-filter encodings.

This library is independent work that takes up both questions in general
terms: it implements the standard error-tolerant alternative (CLKs), the
hardening techniques proposed in response to the cryptanalysis, and an
evaluation that measures both linkage quality and leakage on synthetic data.

It does not revisit Always Foundation's institutional decision (ADR 0013)
not to operate cross-organisation linkage of people; see
[Responsible use](#responsible-use).

## Protocol

The library implements the three-party model used by population health
linkage units:

```
 Custodian A ──┐  encodings, keyed block ids        ┌──▶ link ids for A's rows
               ├──────────────────▶ Linkage unit ────┤
 Custodian B ──┘  (no key, no clear text)           └──▶ link ids for B's rows
      ▲
      └── shared 256-bit key, custodians only, destroyed after the project
```

1. Custodians agree a fresh key `K` and an encoder configuration.
2. Each custodian normalises and encodes its own records locally.
3. The linkage unit blocks, scores and matches encodings. It never holds `K`.
4. The linkage unit returns project-specific link identifiers to each
   custodian; researchers receive de-identified content joined on those.

Assumptions, adversaries and what is and is not withstood are in
[`docs/threat-model.md`](docs/threat-model.md).

## Architecture

```
src/pprl/
  keys.py           SecretKey, HKDF-SHA256 (RFC 5869), keyed permutation
  normalise.py      name/date/digit normalisation, q-grams, Soundex
  encoding.py       CLKEncoder, FieldSpec, Hardening; packed uint64 bit matrices
  similarity.py     vectorised Dice / Jaccard / Hamming over candidate pairs
  blocking.py       KeyedBlocker (custodian side), HammingLSHBlocker (linkage unit)
  matching.py       threshold and greedy one-to-one assignment
  deterministic.py  keyed-hash exact matching (the baseline)
  synthetic.py      seeded person generator with a corruption model
  evaluation.py     precision, recall, F1, reduction ratio, pair completeness
  attacks.py        frequency alignment attack
  experiments.py    the reproducible evaluation behind docs/results.md
```

Data flows left to right; only `encoding.py` and `blocking.KeyedBlocker`
touch clear text and the key, and they run at the custodians.

```
records ─▶ normalise ─▶ tokenise ─▶ HMAC positions ─▶ Bloom filter ─▶ harden ─▶ pack
                                                                                 │
                    links ◀─ one_to_one ◀─ dice_pairs ◀─ blocking candidates ◀───┘
```

## Usage

```python
from pprl import CLKEncoder, FieldSpec, HammingLSHBlocker, SecretKey, dice_pairs, one_to_one
from pprl.normalise import (
    DateParts,
    Positional,
    QGrams,
    normalise_date,
    normalise_digits,
    normalise_name,
)

schema = [
    FieldSpec("given_name", normalise_name, QGrams(2), hashes=10),
    FieldSpec("family_name", normalise_name, QGrams(2), hashes=10),
    FieldSpec("dob", normalise_date, DateParts(), hashes=12),
    FieldSpec("postcode", normalise_digits, Positional(), hashes=6),
]

# At each custodian, with the shared key:
key = SecretKey.from_hex(shared_key_hex)
encoder = CLKEncoder(schema, key, length=1024)
encodings_a = encoder.encode(records_a)  # numpy uint64, shape (n, 16)

# At the linkage unit, with encodings only:
ia, ib = HammingLSHBlocker(encoder.output_length, bits_per_table=32, tables=40).candidates(
    encodings_a, encodings_b
)
links = one_to_one(ia, ib, dice_pairs(encodings_a, encodings_b, ia, ib), t=0.82)
```

Choose the threshold on synthetic or clerically reviewed data with the same
configuration, as `experiments.py` does; it depends on the schema, the
hardening and the data quality.

## Results

5,000 × 5,000 synthetic records, 2,500 true matches, thresholds calibrated on
a separate data set, mean of three seeds. Full method and caveats in
[`docs/evaluation.md`](docs/evaluation.md).

| Corruption | Keyed hash recall | CLK precision | CLK recall |
|---|---|---|---|
| low | 0.740 | 0.984 | 0.953 |
| medium | 0.469 | 0.970 | 0.880 |
| high | 0.237 | 0.900 | 0.821 |

Keyed-hash precision is 0.999 at every level. Against a frequency alignment
attack on family names, keyed hashes and field-level Bloom filters leak
equally (49.9 % of records re-identified on this synthetic vocabulary), and
**balancing, XOR folding and rule 90 make no difference**, because they are
deterministic. Salting by year of birth (0.4 %) and multi-field CLKs (0.0 %)
remove that signal; they do not, by that, defeat stronger published attacks.

## Design decisions

**Random hashing over double hashing.** Bit positions come from
HMAC-SHA256 under a per-field sub-key rather than `h1 + i·h2`. Double hashing
was the basis of the 2014 cryptanalysis by Niedermeyer et al.; random hashing
costs one HMAC per token per 8 positions and is cached per token.

**One master key, many sub-keys.** HKDF gives each field, the blocker, the
fingerprinting baseline and the balancing permutation independent keys, so
leaked block identifiers cannot be used against encodings and vice versa.
`SecretKey` refuses keys under 32 bytes and never prints its material.

**Packed `uint64` matrices and vectorised scoring.** Encodings are stored as
`(n, ceil(l/64))` arrays and scored with `numpy.bitwise_count`, chunked to
bound memory. The blocking join is vectorised (sort, `searchsorted`, index
arithmetic for the cross product), so neither scoring nor blocking has a
Python loop over pairs.

**Two blocking strategies.** Keyed phonetic blocking is computed by the
custodians and is accurate on clean keys but reveals block-size
distributions. LSH runs on encodings at the linkage unit and needs no extra
input from custodians. Both are provided because the right choice depends on
who is trusted with what.

**Empty means no match.** Two records with no usable identifiers have Dice
0, not 1. Records with nothing to compare must never link.

**Calibrate thresholds out of sample.** The evaluation picks each threshold on
a different seed from the one it reports on. Picking it on the test data
would overstate quality.

**Deterministic baseline included.** Keyed-hash exact matching is in the
library so that the cost and benefit of Bloom filters are measured against
the design they would replace, not against nothing.

**Hardening is opt-in and documented as partial.** Each transform is tied to
the specific attack it targets, and the evaluation shows what it does not do.

## Conclusion

**The technique works; the deployment conditions are the problem.**

On synthetic data, keyed Bloom-filter linkage does what it claims. It recovers
most true matches that exact keyed hashing misses (88 % against 47 % recall at
medium corruption, at 97 % precision). Linkage runs on encodings alone, and
known hardening and salting measures reduce specific, documented leaks. As an
engineering question, privacy-preserving linkage across independent
organisations is feasible.

Whether to operate it is a different question. Always Foundation, and the
author as a founding contributor, decided not to deploy it (ADR 0013). The
reasons are about trust and operations, not algorithms:

1. **The security reduces to key custody.** Without the key, encodings resist
   dictionary attack. With it, every low-entropy field (date of birth, postcode,
   phone number, common names) can be enumerated and every record re-identified
   (threat T2). Some party must hold that key for the life of the project, and
   the protection of every record then depends on that party's operational
   security, staff and future decisions.
2. **The three-party model needs an intermediary that is actually independent.**
   The protocol assumes a linkage unit that never sees the key, does not collude
   with a custodian, and keeps linkage separate from analysis (assumptions A1
   to A5). Health systems meet these with statutory linkage units, enabling
   legislation, ethics review and audit. A voluntary network of organisations
   has none of these, and an operator that is also a participant cannot be its
   own independent party.
3. **The threat environment makes a linked dataset a target.** A link map of
   sensitive participation is valuable to attackers and subject to compelled
   disclosure (T6). Its sensitivity grows with every organisation that joins
   (T7). The Australian Signals Directorate reported more than 1,200 incidents
   in 2024–25, up 11 % year on year (as cited in the Always Foundation case
   study). Cryptography withstands theft of encodings. It does not withstand
   compromise of the key holder, and it cannot make a dataset safe that should
   not exist.
4. **No formal guarantee bounds the leakage.** Bloom-filter encodings leak
   similarity and structure by design. Hardening raises the cost of known
   attacks but proves nothing about unknown ones (`docs/privacy.md`).

This repository is therefore a **reference artefact**. It shows what is
technically achievable and measures the accuracy and leakage trade-offs, for
researchers and organisations examining the same question. Anyone taking the
work further should do so only in a tightly controlled setting, with these
minimum conditions:

- synthetic or ethics-approved sample data, never live operational records;
- an independent linkage party with no other role in the project;
- a key generated per study, held in a key management system (ideally split
  between custodians), and destroyed on a fixed date with the destruction
  recorded;
- linked outputs released only in aggregate, with small cells suppressed;
- a defined end date after which link maps and encodings are deleted.

For production use across independent organisations, the trust and custody
requirements are, in our assessment, too difficult to meet reliably.

## Responsible use

This is a research library for settings that already have the governance
linkage requires: an independent linkage unit, separation of linkage and
analysis roles, a legal basis or ethics approval, and output controls. It is
not a turnkey system and gives no formal privacy guarantee.

Some linkage should not be built. Always Foundation's ADR 0013 records its
decision not to link people across independent ministries by any technique,
on the grounds that the capability itself, however protected, concentrates
risk that its mission does not need. This library is not designed for, and
should not be used for, tracking religious participation or reconstructing
people's activity across organisations without their specific, informed
consent and independent oversight.

## Documentation

- [`docs/threat-model.md`](docs/threat-model.md): parties, assumptions, adversaries, residual risk
- [`docs/privacy.md`](docs/privacy.md): what encodings protect and leak, hardening, configuration
- [`docs/evaluation.md`](docs/evaluation.md): method, results, findings, limitations
- [`docs/results.md`](docs/results.md): generated results table

## References

- Always Foundation. *Privacy-Preserving Record Linkage for Gospel Impact Research*, case study, 2026; and ADR 0013, *No linking of people across organisations*, [always-platform](https://github.com/alwaysfoundation/always-platform). The framing of the open questions and part of the threat model build on this work.
- Boyd JH et al. Data linkage infrastructure for cross-jurisdictional health-related research in Australia. *BMC Health Services Research* 2012;12:480.
- Christen P, Vatsalan D. Flexible and extensible generation and corruption of personal data. *CIKM* 2013.
- Christen P, Vidanage A, Ranbaduge T, Schnell R. Pattern-mining based cryptanalysis of Bloom filters for privacy-preserving record linkage. *PAKDD* 2018.
- Christen P, Ranbaduge T, Vatsalan D, Schnell R. Precise and fast cryptanalysis for Bloom filter based privacy-preserving record linkage. *IEEE TKDE* 2019;31(11):2164–2177.
- Durham EA. *A framework for accurate, efficient private record linkage.* PhD thesis, Vanderbilt University, 2012.
- Durham EA, Kantarcioglu M, Xue Y, Toth C, Kuzu M, Malin B. Composite Bloom filters for secure record linkage. *IEEE TKDE* 2014;26(12):2956–2968.
- Hand D, Christen P. A note on using the F-measure for evaluating record linkage algorithms. *Statistics and Computing* 2018;28:539–547.
- Indyk P, Motwani R. Approximate nearest neighbors: towards removing the curse of dimensionality. *STOC* 1998.
- Karapiperis D, Verykios VS. An LSH-based blocking approach with a homomorphic matching technique for privacy-preserving record linkage. *IEEE TKDE* 2015;27(4):909–921.
- Krawczyk H, Bellare M, Canetti R. HMAC: keyed-hashing for message authentication. RFC 2104, 1997.
- Krawczyk H, Eronen P. HMAC-based extract-and-expand key derivation function (HKDF). RFC 5869, 2010.
- Niedermeyer F, Steinmetzer S, Kroll M, Schnell R. Cryptanalysis of basic Bloom filters used for privacy preserving record linkage. *Journal of Privacy and Confidentiality* 2014;6(2).
- Randall SM, Ferrante AM, Boyd JH, Bauer JK, Semmens JB. Privacy-preserving record linkage on large real world datasets. *Journal of Biomedical Informatics* 2014;50:205–212.
- Ranbaduge T, Schnell R. Securing Bloom filters for privacy-preserving record linkage. *CIKM* 2020.
- Schnell R, Bachteler T, Reiher J. Privacy-preserving record linkage using Bloom filters. *BMC Medical Informatics and Decision Making* 2009;9:41.
- Schnell R, Bachteler T, Reiher J. A novel error-tolerant anonymous linking code. German Record Linkage Center, WP-GRLC-2011-02, 2011.
- Schnell R, Borgs C. Randomized response and balanced Bloom filters for privacy preserving record linkage. *IEEE ICDM Workshops* 2016.
- Schnell R, Borgs C, Encarnação J. XOR-folding for hardening Bloom filter-based encryptions for privacy-preserving record linkage. German Record Linkage Center, WP-GRLC-2016-03, 2016.
- Schnell R, Borgs C, Encarnação J. Hardening encrypted patient names against cryptographic attacks using cellular automata. *IEEE ICDM Workshops* 2018.
- Vidanage A, Christen P, Ranbaduge T, Schnell R. A graph matching attack on privacy-preserving record linkage. *CIKM* 2020.

## Licence and attribution

Code under the Apache License 2.0 (`LICENSE`). Written by Daniel Dizon with
AI assistance (Claude); see the commit trailers. Not affiliated with or
endorsed by Always Foundation.
