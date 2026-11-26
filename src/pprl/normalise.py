"""Field normalisation and tokenisation.

Linkage quality depends more on consistent normalisation than on the encoding.
Both custodians MUST apply the same normalisation, so it is defined here once,
deterministically, with no locale dependence.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

__all__ = [
    "DateParts",
    "Positional",
    "QGrams",
    "Tokeniser",
    "normalise_date",
    "normalise_digits",
    "normalise_name",
    "soundex",
]

_WS = re.compile(r"\s+")


def normalise_name(value: str | None) -> str:
    """Normalise a personal name.

    Unicode NFKD decomposition with combining marks removed, case folding,
    apostrophes dropped, hyphens and other separators turned into spaces, and
    whitespace collapsed. Letters from non-Latin scripts are kept, so names in
    those scripts are not silently erased.
    """
    if not value:
        return ""
    decomposed = unicodedata.normalize("NFKD", value)
    out: list[str] = []
    for ch in decomposed:
        if unicodedata.combining(ch):
            continue
        if ch in "'’`":
            continue
        out.append(ch.casefold() if ch.isalpha() else " ")
    return _WS.sub(" ", "".join(out)).strip()


def normalise_digits(value: str | None) -> str:
    """Keep ASCII digits only (postcodes, phone numbers)."""
    if not value:
        return ""
    return "".join(ch for ch in value if "0" <= ch <= "9")


def normalise_date(value: str | date | None) -> str:
    """Return ``YYYYMMDD`` for a date or an ISO 8601 date string, or ``""``."""
    if value is None or value == "":
        return ""
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    try:
        return date.fromisoformat(value.strip()[:10]).strftime("%Y%m%d")
    except ValueError:
        return ""


_SOUNDEX_CODES = {
    **dict.fromkeys("bfpv", "1"),
    **dict.fromkeys("cgjkqsxz", "2"),
    **dict.fromkeys("dt", "3"),
    "l": "4",
    **dict.fromkeys("mn", "5"),
    "r": "6",
}


def soundex(value: str) -> str:
    """American Soundex of a normalised Latin-script name, or ``""``.

    Used only for blocking. ``h`` and ``w`` do not separate letters with the
    same code; vowels do.
    """
    letters = [c for c in normalise_name(value) if "a" <= c <= "z"]
    if not letters:
        return ""
    first = letters[0]
    code = [first.upper()]
    prev = _SOUNDEX_CODES.get(first, "")
    for c in letters[1:]:
        digit = _SOUNDEX_CODES.get(c, "")
        if digit and digit != prev:
            code.append(digit)
        if c not in "hw":
            prev = digit
        if len(code) == 4:
            break
    return "".join(code).ljust(4, "0")


Tokeniser = Callable[[str], list[str]]


@dataclass(frozen=True)
class QGrams:
    """Padded character q-grams, the standard tokenisation for names.

    ``"anna"`` with ``q=2`` gives ``_a an nn na a_``. Duplicates are kept once,
    because a Bloom filter cannot count.
    """

    q: int = 2
    pad: bool = True

    def __call__(self, value: str) -> list[str]:
        if not value:
            return []
        s = f"_{value}_" if self.pad else value
        if len(s) < self.q:
            return [s]
        return sorted({s[i : i + self.q] for i in range(len(s) - self.q + 1)})


@dataclass(frozen=True)
class Positional:
    """Each character tagged with its position, for fixed-format codes.

    ``"2000"`` gives ``0:2 1:0 2:0 3:0``, so one wrong digit changes one token.
    """

    def __call__(self, value: str) -> list[str]:
        return [f"{i}:{c}" for i, c in enumerate(value)]


@dataclass(frozen=True)
class DateParts:
    """Year, month and day of a ``YYYYMMDD`` string as separate tokens.

    A day/month transposition or a one-unit error then changes only one or two
    tokens instead of most character q-grams.
    """

    def __call__(self, value: str) -> list[str]:
        if len(value) != 8:
            return []
        y, m, d = value[:4], value[4:6], value[6:]
        return [f"y:{y}", f"m:{m}", f"d:{d}", f"md:{m}{d}"]
