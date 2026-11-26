# Privacy properties and limits

A short, precise account of what the encodings in this library protect, what
they leak by design, and what each hardening option changes. Read
`threat-model.md` first for the parties and assumptions.

## 1. The core trade-off

Linkage needs a function `E` such that similar identifiers give similar
encodings. Any such function leaks similarity, and therefore cannot be
semantically secure: an observer learns *something* about the plaintext from
the relationships between encodings. PPRL designs differ in how much
structure they leak and to whom, not in whether they leak.

| Encoding | Tolerates typos | Leaks equality | Leaks similarity | Leaks value frequency | Without key, brute force |
|---|---|---|---|---|---|
| Unkeyed hash (SHA-256 of value) | No | Yes | No | Yes | **Trivial**: hash every value |
| Keyed hash (HMAC of value) | No | Yes | No | Yes | Infeasible (needs key) |
| Field-level Bloom filter | Yes | Yes | Yes | Yes | Infeasible (needs key) |
| Multi-field CLK | Yes | Yes | Yes | Weakly (combinations are rare) | Infeasible (needs key) |
| CLK with salting | Yes | Yes | Yes | Weakly, per salt group | Infeasible (needs key) |

"Infeasible" assumes HMAC-SHA256 is a PRF and the key is secret. **With the
key, every row of this table becomes "trivial"** for fields whose domain can
be enumerated: dates of birth, postcodes, phone numbers (~10⁸ Australian
mobiles), and names from a census list.

## 2. What each component guarantees

**`SecretKey` and HKDF sub-keys.** Each component (each field's bit
positions, blocking, deterministic fingerprints, the balancing permutation)
uses an independent 256-bit sub-key. Knowing one component's outputs gives no
advantage against another's, under the standard HKDF assumptions (RFC 5869).

**Random hashing.** Each token's *k* bit positions are taken from
`HMAC-SHA256(K_field, salt ‖ 0x1F ‖ token ‖ counter)`. Positions for
different tokens are independent and unpredictable without `K_field`. This
removes the linear relation between positions that double hashing
(`h1 + i·h2 mod l`) creates, which the 2014 attack of Niedermeyer et al.
exploited. Modulo bias is below `l / 2³²`, under 10⁻⁶ for `l ≤ 4096`.

**Salting.** Mixing another field into a field's key (for example year of
birth into family name) makes equal names in different salt groups encode
independently. This defeats frequency alignment across groups. It costs
recall whenever the salt field itself is wrong, because the salted field then
contributes nothing to similarity.

**Multi-field CLKs.** Combining fields makes encodings close to unique per
person, so the frequency of an encoding says little. Pattern-mining attacks
work on bit co-occurrence across many records and are not prevented by
combination alone.

## 3. Hardening options

None of these is a proof. Each targets a specific published weakness.

| Option | What it does | Removes | Does not remove | Cost |
|---|---|---|---|---|
| `BALANCE` | Concatenate filter and complement, permute under key | Hamming weight as a signal for value length | Equality, frequency, co-occurrence | 2× length; Dice values shift up, so thresholds must be recalibrated |
| `XOR_FOLD` | XOR the two halves | One-to-one bit-to-q-gram mapping | Equality, frequency | Half length; some similarity resolution lost |
| `RULE_90` | Each bit becomes XOR of its two neighbours | Direct bit-to-q-gram mapping | Equality, frequency | Small recall loss |

The evaluation (`evaluation.md` §2) shows a result that is easy to miss:
**hardening transforms are deterministic functions of the filter, so they do
nothing against frequency alignment.** Identical values still give identical
encodings. Only salting or combining fields changes that.

## 4. What this library does not provide

- **No differential privacy.** Encodings are deterministic. Randomised
  response on Bloom filter bits (Schnell and Borgs 2016) and differentially
  private Bloom filters exist in the literature and are not implemented.
- **No protection from a key holder.** See §1.
- **No secure computation.** Matching happens in the clear on encodings at the
  linkage unit. Private set intersection and secure multiparty computation
  avoid a central matcher at higher cost and are out of scope.
- **No output privacy.** Linked results can re-identify people through rare
  attribute combinations; suppression and output checking are the operator's
  responsibility.

## 5. Choosing a configuration

A defensible default for research linkage of person records:

1. Multi-field CLK over given name, family name, date of birth and one
   geographic field, 1024 bits, *k* = 10 to 12 per token.
2. `RULE_90` or `XOR_FOLD` if the threat model includes pattern-mining by the
   linkage unit; recalibrate the threshold on synthetic data first.
3. A fresh key per project, destroyed at the end.
4. Keyed blocking computed by the custodians, or LSH at the linkage unit.

Do not use single-field encodings of low-entropy values (date of birth alone,
sex, postcode) as linkage keys: they are equivalent to clear text for anyone
who learns the key and leak frequency to anyone who does not.
