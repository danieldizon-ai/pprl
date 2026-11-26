# Threat model

This document states who the parties are, what each is trusted with, which
attacks the design withstands and which it does not. It is written so that a
reviewer can disagree with a specific assumption rather than with the whole.

Part of the structure here, in particular the emphasis on key custody,
inference from outputs, compelled disclosure and scope growth, follows the
threat analysis in Always Foundation's case study *Privacy-Preserving Record
Linkage for Gospel Impact Research* (2026), which examined a deterministic
keyed-hash design. This library generalises that analysis to Bloom-filter
encodings.

## 1. Parties and protocol

The library implements the **three-party model** used by population health
linkage units in Australia and elsewhere (Boyd et al. 2012):

```
 Custodian A ──┐                                  ┌── Custodian A
 (clear text)  │  encodings (+ keyed block ids)   │   receives link keys
               ├────────────────▶ Linkage unit ───┤   for its own rows
 Custodian B ──┘   no key, no clear text          └── Custodian B
 (clear text)
        ▲
        └── shared secret key K, agreed between custodians only
```

1. The custodians agree a fresh 256-bit key `K` for this project, out of band.
   The linkage unit never receives it.
2. Each custodian normalises its records, encodes them with the same
   `CLKEncoder` configuration and `K`, and optionally computes keyed block
   identifiers with `KeyedBlocker`.
3. The linkage unit receives encodings (and block identifiers), blocks, scores
   candidate pairs and decides matches. It sees neither clear text nor `K`.
4. The linkage unit returns, to each custodian, a project-specific link
   identifier for each of that custodian's matched rows. Researchers receive
   de-identified content data joined on link identifiers, never encodings.
5. At the end of the project `K` is destroyed and the destruction recorded.

## 2. Assets

| Asset | Held by | Why it matters |
|---|---|---|
| Clear-text identifiers | Each custodian, for its own rows | The thing being protected |
| Key `K` | Custodians | With `K`, encodings can be recomputed for guessed values |
| Encodings | Linkage unit | Leak similarity, equality and frequency by design (see `privacy.md`) |
| Link map | Linkage unit | Reveals which rows in A and B are the same person |
| Linked content | Researchers | The output; sensitive even without names |

## 3. Assumptions

- **A1. Key secrecy.** `K` is generated with a CSPRNG, shared only between
  custodians, never reused across projects, and destroyed afterwards.
- **A2. No collusion between the linkage unit and a custodian.** A custodian
  that gives `K` to the linkage unit collapses the model to "linkage unit
  holds clear text".
- **A3. Honest-but-curious linkage unit.** It follows the protocol but may
  analyse everything it receives.
- **A4. HMAC-SHA256 is a pseudorandom function.** Without `K`, bit positions
  are unpredictable.
- **A5. Separation of outputs.** Researchers never receive encodings or the
  link map, only link identifiers and content.

## 4. Adversaries and outcomes

| # | Adversary | Capability | Outcome |
|---|---|---|---|
| T1 | External attacker steals encodings | Encodings only, no `K` | Cannot compute encodings for guessed values (A4). Can still run frequency and pattern attacks (T4). **Partly withstood.** |
| T2 | Attacker or insider obtains `K` | `K` plus encodings | Dictionary attack: encode every plausible value and compare. Dates of birth (~36,500 values), postcodes and common names fall immediately. **Not withstood.** |
| T3 | Curious linkage unit | Encodings, block ids, link map | Learns which records match and how similar non-matches are. Can attempt T4. **Partly withstood.** |
| T4 | Cryptanalysis without `K` | Encodings plus public frequency tables | Frequency alignment re-identifies common values when an encoding is deterministic per value; pattern-mining and graph-matching attacks recover q-grams from bit co-occurrence (Christen et al. 2018, 2019; Vidanage et al. 2020). **Depends on configuration**; see `evaluation.md` §2. |
| T5 | Inference from outputs | Linked, de-identified content | Rare combinations of attributes can identify a person in a small population even with no identifiers. **Not addressed by encoding**; needs output controls. |
| T6 | Compelled disclosure | Legal process against any party | A link map, once created, can be demanded. **Not addressed by cryptography**; only by not creating or not retaining it. |
| T7 | Scope growth | Organisational | More data sets, longer retention, finer outputs: each defensible, together they change the sensitivity of the linked data. **Governance, not code.** |

## 5. What the design withstands, precisely

Under A1 to A5, a linkage unit or thief holding encodings **without `K`**:

- cannot test whether a specific person's identifiers are in the data by
  encoding them, because encoding needs `K`;
- cannot learn identifiers by brute force over their domain, for the same
  reason;
- **can** learn equality and similarity between encodings, because that is
  what linkage requires;
- **can** learn value frequencies when the same value always produces the same
  encoding (single-field encodings, deterministic hashes);
- **can**, with enough records and a good frequency model, mount published
  cryptanalyses that partially recover q-grams from unhardened Bloom filters.

## 6. Mitigations implemented

| Mitigation | Addresses | Where |
|---|---|---|
| Random hashing with HMAC-SHA256 per field, not double hashing | Linear-structure attack on double hashing (Niedermeyer et al. 2014); T1 dictionary | `encoding.py` |
| Independent sub-keys via HKDF for every component | Compromise of one output (e.g. block ids) not reusable against another | `keys.py` |
| Multi-field CLK (record-level encoding) | Frequency alignment: combined values are near-unique | `encoding.py` |
| Salting a field with another field | Frequency alignment on single fields | `FieldSpec.salt_field` |
| Balancing (constant Hamming weight) | Weight-based inference of value length | `Hardening.BALANCE` |
| XOR folding, rule 90 | Bit-to-q-gram mapping in pattern attacks | `Hardening.XOR_FOLD`, `Hardening.RULE_90` |
| Keyed block identifiers | Clear-text blocking keys reaching the linkage unit | `KeyedBlocker` |
| Key material redacted from `repr` | Accidental logging of `K` | `SecretKey` |

## 7. Residual risks and operational requirements

These are not solved by this library and must be handled by whoever operates it:

1. **Key custody is the single point of failure.** Whoever holds `K` can
   reverse the encodings of low-entropy fields. Hold it in a key management
   system, split it between custodians if possible (for example a threshold
   scheme), and destroy it on a fixed date.
2. **No formal privacy guarantee.** Bloom-filter PPRL is not semantically
   secure and does not provide differential privacy. Hardening raises the cost
   of known attacks; it does not prove resistance to unknown ones.
3. **Outputs re-identify.** Apply small-cell suppression and output checking
   to anything released from linked data.
4. **Legal and ethical basis.** Linkage of sensitive data should run under
   ethics approval, an explicit legal basis or individual consent, and with
   separation between linkage and analysis roles.
5. **Purpose.** Some linkage should not be built at all. Always Foundation's
   ADR 0013 records an institutional decision not to link people across
   independent ministries, whatever the technique, because the capability
   itself carries risk its mission does not need. That reasoning applies
   beyond its original setting and is worth reading before deploying any
   linkage system.
