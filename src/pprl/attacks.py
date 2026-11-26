"""Attack baselines used to evaluate what an encoding leaks.

These are deliberately simple, well-known attacks. They give a *lower bound*
on leakage: an encoding that resists them is not thereby secure, because
stronger published attacks exist (pattern mining, Christen et al. 2018;
graph matching, Vidanage et al. 2020) and are not implemented here.

The attacker model is an honest-but-curious linkage unit holding the
encodings of one data set and a public frequency table of the plaintext
values (for example a census list of common family names). It does not hold
the key.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np

from .encoding import BitMatrix

__all__ = ["FrequencyAttackResult", "frequency_alignment_attack", "rows_as_tokens"]


@dataclass(frozen=True)
class FrequencyAttackResult:
    records: int
    reidentified: int
    distinct_encodings: int
    guesses_made: int

    @property
    def reidentification_rate(self) -> float:
        """Share of all records whose plaintext value the attacker guessed correctly."""
        return self.reidentified / self.records if self.records else 0.0


def rows_as_tokens(encodings: BitMatrix) -> list[bytes]:
    """Turn each packed row into a hashable token, so identical encodings compare equal."""
    return [row.tobytes() for row in np.ascontiguousarray(encodings)]


def frequency_alignment_attack(
    encodings: Sequence[Hashable],
    true_values: Sequence[str | None],
    public_ranking: Sequence[str],
    top: int = 20,
) -> FrequencyAttackResult:
    """Align the most frequent encodings with the most frequent public values.

    The attacker sorts distinct encodings by frequency and guesses that the
    *i*-th most frequent encoding is the *i*-th most frequent public value,
    for the ``top`` most frequent encodings. Against a deterministic
    encoding of a single field with a skewed value distribution this
    re-identifies the commonest values. Ties are broken by first appearance,
    as an attacker without more information would.

    ``true_values`` are used only to score the attack.
    """
    if len(encodings) != len(true_values):
        raise ValueError("encodings and true_values must be the same length")
    counts = Counter(encodings)
    ranked = [enc for enc, _ in counts.most_common(top)]
    guess = {enc: public_ranking[i] for i, enc in enumerate(ranked) if i < len(public_ranking)}
    hits = sum(
        1
        for enc, truth in zip(encodings, true_values, strict=True)
        if enc in guess and truth is not None and guess[enc] == truth
    )
    return FrequencyAttackResult(len(encodings), hits, len(counts), len(guess))
