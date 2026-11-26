# Evaluation

All numbers in this document come from [`results.md`](results.md), generated
by `pprl-experiments` (`src/pprl/experiments.py`). Rerun it to reproduce them;
every random choice is seeded except the project key, which is fresh for each
run and does not affect the statistics.

## Questions

1. How much recall does error-tolerant Bloom-filter linkage gain over exact
   matching on keyed hashes, at what cost in precision, as data quality falls?
2. What do the hardening transforms cost in linkage quality?
3. Does hardening protect against the simplest frequency attack, and what does?

Question 1 is the one the Always Foundation case study explicitly left open
for its deterministic design ("Linkage accuracy. Not evaluated."). Question 3
addresses its stated reason for not adopting Bloom filters, the published
cryptanalysis.

## Method

**Data.** Synthetic only. Two data sets of 5,000 records each sharing 2,500
people. Data set A holds clean records; B holds independently corrupted
copies of the shared people plus 2,500 different people. Fields: given name,
family name, date of birth, postcode. Names follow a Zipf-like distribution
over built-in lists (~120 given and ~120 family names). Four corruption
levels (`CorruptionProfile.named`): *clean*, *low*, *medium*, *high*, mixing
character edits, nicknames, family-name change, missing values, day/month
swaps, digit errors and change of postcode. Three test seeds.

**Methods.**

- *Keyed hash (exact)*: `HMAC-SHA256(K, given ‖ family ‖ dob)` after
  normalisation; links on equality. This is the deterministic construction
  from the case study, extended from one identifier to three.
- *CLK*: 1024-bit record-level Bloom filter, random hashing, *k* = 10 per name
  bigram, 12 per date part, 6 per postcode digit.
- *CLK + balance / XOR fold / rule 90*: the same CLK with one hardening step.

**Blocking.** Bit-sampling LSH at the linkage unit, 40 tables, 32 sampled bits
(20 for the denser folded encodings).

**Classification.** Dice similarity, greedy one-to-one assignment. The
threshold for each method and corruption level is the F1-maximising value on a
**separate calibration data set** with a different seed, then applied
unchanged to the three test sets. Reported precision and recall are therefore
not tuned on the data they describe.

**Attack.** A linkage unit holding the encodings of data set A (clean) and the
true frequency ranking of family names aligns the 20 most frequent encodings
with the 20 most frequent names.

## Results

### Linkage quality

Mean over three test seeds (s.d. ≤ 0.008 throughout; full table in
`results.md`).

| Corruption | Method | Precision | Recall | F1 | Pair completeness |
|---|---|---|---|---|---|
| clean | Keyed hash | 0.999 | 1.000 | 1.000 | n/a |
| clean | CLK | 1.000 | 1.000 | 1.000 | 1.000 |
| low | Keyed hash | 0.999 | 0.740 | 0.850 | n/a |
| low | CLK | 0.984 | 0.953 | 0.969 | 0.991 |
| medium | Keyed hash | 0.999 | 0.469 | 0.639 | n/a |
| medium | CLK | 0.970 | 0.880 | 0.922 | 0.968 |
| medium | CLK + balance | 0.949 | 0.908 | 0.928 | 0.969 |
| medium | CLK + rule 90 | 0.977 | 0.871 | 0.921 | 0.948 |
| high | Keyed hash | 0.999 | 0.237 | 0.383 | n/a |
| high | CLK | 0.900 | 0.821 | 0.859 | 0.909 |
| high | CLK + XOR fold | 0.939 | 0.771 | 0.847 | 0.862 |

LSH removed 99.4 to 99.7 % of the 25 million possible pairs in every
configuration.

### Frequency attack on family names

| Encoding | Records re-identified |
|---|---|
| Keyed hash of family name | 49.9 % |
| Field-level Bloom filter | 49.9 % |
| Field-level Bloom filter + balance | 49.9 % |
| Field-level Bloom filter + XOR fold | 49.9 % |
| Field-level Bloom filter salted by year of birth | 0.4 % |
| CLK (four fields) | 0.0 % |

## Findings

1. **Exact matching is precise and brittle.** Its precision stays at 0.999
   at every level (the shortfall is synthetic people who share name and date
   of birth), but its recall falls from 1.00 to 0.74 with low corruption and
   to 0.24 with high corruption. Every missed link is silent.
2. **Bloom-filter linkage trades a little precision for a lot of recall.** At
   low corruption the CLK recovers 95 % of true links at 98 % precision; at
   high corruption, 82 % at 90 %. Pair completeness bounds recall: at high
   corruption, 9 % of true pairs never reach the classifier, so blocking,
   not scoring, is the next thing to improve.
3. **Hardening costs a few points at most.** Balancing shifts the
   precision/recall balance (it needs a much higher threshold, around 0.9 to
   0.98) with similar F1. Folding and rule 90 cost one to five points of pair
   completeness under LSH because denser filters make sampled bits less
   selective.
4. **Hardening does not stop frequency alignment.** Balancing, folding and
   rule 90 are deterministic functions of the filter, so the same name still
   produces the same encoding and the attack succeeds exactly as often as
   against a keyed hash. Salting by year of birth (0.4 %) or encoding several
   fields together (0.0 %) is what removes the signal.

## Limitations

- **Synthetic data.** Real data has correlated errors, households, twins,
  transliteration variants and duplicate records within a data set that this
  generator does not model. Absolute numbers will not transfer; the relative
  ordering of methods is the more robust result.
- **Small name vocabulary.** With ~120 family names the attack's success rate
  is far higher than against a real population with tens of thousands of
  names, and the attacker here knows the true frequency ranking exactly. The
  comparison between encodings is the point, not the 49.9 %.
- **One weak attack.** Frequency alignment is a lower bound on leakage.
  Pattern-mining and graph-matching attacks (Christen et al. 2018;
  Vidanage et al. 2020) can recover q-grams from multi-field CLKs given enough
  records, and are not evaluated here. A CLK scoring 0.0 % above is not
  thereby shown to be secure.
- **Greedy assignment and a single threshold.** A Fellegi-Sunter model or a
  clerical-review band between two thresholds would change the trade-off.
- **F-measure.** Reported for comparability with the literature; Hand and
  Christen (2018) explain why it should not be the only criterion.

## Conclusion

The evaluation answers the technical question the Always Foundation case
study left open. Error-tolerant Bloom-filter linkage substantially outperforms
exact keyed hashing on imperfect data, and the cost of hardening is a few
points of recall. It does not change the deployment question. The residual
risks are key custody, dependence on an independent intermediary, the
threat environment, and the absence of a formal leakage bound. Those
risks sit outside the algorithm, and they are why the approach is presented
here as a reference artefact rather than a recommendation to deploy. See the
README's Conclusion section and `threat-model.md` §7.
