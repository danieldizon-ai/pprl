"""Synthetic person records with controlled corruption and known ground truth.

No real person's data is used anywhere in this repository. Names are drawn
from built-in lists with a Zipf-like frequency distribution, so that common
names are common, which matters both for blocking cost and for frequency
attacks. Dates of birth and postcodes are uniform over plausible ranges.

The corruption model follows the error types used in linkage evaluation
generators such as GeCo (Christen and Vatsalan 2013): keyboard-agnostic
character edits, nickname substitution, missing values, day/month
transposition, digit errors and family-name change.
"""

from __future__ import annotations

import string
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import numpy.typing as npt

__all__ = ["CorruptionProfile", "LinkageDataset", "Person", "generate_linkage_dataset"]

# fmt: off
GIVEN_NAMES: tuple[str, ...] = (
    "james", "olivia", "william", "charlotte", "jack", "amelia", "noah", "isla", "thomas",
    "mia", "oliver", "ava", "henry", "grace", "lucas", "chloe", "leo", "emily", "alexander",
    "sophie", "ethan", "ruby", "samuel", "harper", "daniel", "zoe", "joshua", "evie", "liam",
    "matilda", "benjamin", "lily", "harrison", "ella", "max", "sienna", "isaac", "willow",
    "hudson", "georgia", "levi", "hannah", "archie", "lucy", "elijah", "ivy", "michael",
    "sarah", "david", "jessica", "andrew", "rebecca", "christopher", "elizabeth", "matthew",
    "katherine", "nicholas", "margaret", "robert", "jennifer", "anthony", "patricia", "joseph",
    "catherine", "wei", "mei", "jun", "ling", "hao", "yan", "minh", "linh", "duc", "thao",
    "arjun", "priya", "rohan", "ananya", "vikram", "kavya", "mohammed", "fatima", "omar",
    "aisha", "ali", "zainab", "yusuf", "mariam", "giorgos", "eleni", "nikos", "maria",
    "giuseppe", "giulia", "marco", "francesca", "jose", "lucia", "carlos", "sofia", "kwame",
    "ama", "tane", "aroha", "sione", "mele", "hiroshi", "yuki", "seo-yeon", "min-jun",
    "siobhan", "declan", "aoife", "ciaran", "ngaire", "bronwyn", "dylan", "megan",
)

FAMILY_NAMES: tuple[str, ...] = (
    "smith", "jones", "williams", "brown", "wilson", "taylor", "johnson", "white", "martin",
    "anderson", "thompson", "nguyen", "thomas", "walker", "harris", "lee", "ryan", "robinson",
    "kelly", "king", "davis", "wright", "evans", "roberts", "green", "hall", "wood", "jackson",
    "clarke", "patel", "khan", "lewis", "james", "phillips", "mitchell", "turner", "campbell",
    "edwards", "murphy", "young", "hughes", "moore", "allen", "scott", "baker", "cooper",
    "morris", "ward", "watson", "stewart", "morgan", "cook", "bell", "bennett", "murray",
    "price", "mcdonald", "gray", "o'brien", "o'connor", "fitzgerald", "macleod", "chen",
    "wang", "li", "zhang", "liu", "huang", "wu", "tran", "le", "pham", "hoang", "vu", "singh",
    "sharma", "kumar", "gupta", "reddy", "iyer", "ahmed", "hussain", "rahman", "haddad",
    "nasser", "papadopoulos", "georgiou", "nikolaidis", "rossi", "russo", "ferrari",
    "esposito", "bianchi", "garcia", "rodriguez", "martinez", "lopez", "fernandez", "silva",
    "santos", "mensah", "owusu", "ngata", "parata", "tuilagi", "fifita", "tanaka", "suzuki",
    "kim", "park", "choi", "o'sullivan", "byrne", "doherty", "van der berg", "de souza",
    "schmidt", "muller", "novak", "kowalski",
)

NICKNAMES: dict[str, tuple[str, ...]] = {
    "william": ("will", "bill", "liam"), "james": ("jim", "jamie"), "thomas": ("tom", "tommy"),
    "alexander": ("alex", "xander"), "samuel": ("sam",), "daniel": ("dan", "danny"),
    "joshua": ("josh",), "benjamin": ("ben",), "isaac": ("ike",), "michael": ("mike", "mick"),
    "david": ("dave",), "andrew": ("andy", "drew"), "christopher": ("chris",),
    "matthew": ("matt",), "nicholas": ("nick",), "robert": ("rob", "bob", "bobby"),
    "anthony": ("tony",), "joseph": ("joe",), "elizabeth": ("liz", "beth", "eliza"),
    "katherine": ("kate", "katie", "kath"), "margaret": ("maggie", "meg", "peggy"),
    "jennifer": ("jen", "jenny"), "patricia": ("pat", "trish"), "catherine": ("cath", "cate"),
    "rebecca": ("becky", "bec"), "jessica": ("jess",), "charlotte": ("charlie", "lottie"),
    "amelia": ("millie",), "matilda": ("tilly",), "olivia": ("liv",), "sophie": ("soph",),
    "hannah": ("han",), "francesca": ("fran",), "giuseppe": ("joe", "pino"),
}
# fmt: on


def _zipf_weights(n: int, s: float) -> npt.NDArray[np.float64]:
    w: npt.NDArray[np.float64] = 1.0 / np.arange(1, n + 1, dtype=np.float64) ** s
    out: npt.NDArray[np.float64] = w / w.sum()
    return out


@dataclass(frozen=True)
class Person:
    entity_id: int
    given_name: str
    family_name: str
    dob: str  # ISO 8601
    sex: str
    postcode: str

    def as_record(self) -> dict[str, str | None]:
        return {
            "given_name": self.given_name,
            "family_name": self.family_name,
            "dob": self.dob,
            "sex": self.sex,
            "postcode": self.postcode,
        }


@dataclass(frozen=True)
class CorruptionProfile:
    """Per-record probabilities of each error type, applied independently."""

    name_typo: float = 0.10
    nickname: float = 0.05
    family_name_change: float = 0.02
    missing_field: float = 0.03
    dob_swap_day_month: float = 0.02
    dob_digit: float = 0.03
    postcode_digit: float = 0.03
    postcode_move: float = 0.05

    @classmethod
    def named(cls, level: str) -> CorruptionProfile:
        """``clean``, ``low``, ``medium`` or ``high``."""
        if level == "clean":
            return cls(0, 0, 0, 0, 0, 0, 0, 0)
        if level == "low":
            return cls()
        if level == "medium":
            return cls(0.25, 0.10, 0.04, 0.06, 0.04, 0.06, 0.06, 0.10)
        if level == "high":
            return cls(0.45, 0.15, 0.08, 0.10, 0.06, 0.10, 0.10, 0.20)
        raise ValueError(f"unknown corruption level {level!r}")


@dataclass
class LinkageDataset:
    """Two record sets and the true links between them."""

    a: list[dict[str, str | None]]
    b: list[dict[str, str | None]]
    entity_a: npt.NDArray[np.int64]
    entity_b: npt.NDArray[np.int64]
    true_links: set[tuple[int, int]] = field(default_factory=set)

    @property
    def true_values_a(self) -> list[str | None]:
        return [r["family_name"] for r in self.a]


def _typo(rng: np.random.Generator, s: str) -> str:
    if not s:
        return s
    letters = string.ascii_lowercase
    op = rng.integers(4)
    i = int(rng.integers(len(s)))
    if op == 0:  # substitute
        return s[:i] + letters[rng.integers(26)] + s[i + 1 :]
    if op == 1:  # insert
        return s[:i] + letters[rng.integers(26)] + s[i:]
    if op == 2 and len(s) > 2:  # delete
        return s[:i] + s[i + 1 :]
    if len(s) > 1:  # transpose
        i = min(i, len(s) - 2)
        return s[:i] + s[i + 1] + s[i] + s[i + 2 :]
    return s


def _digit(rng: np.random.Generator, s: str) -> str:
    if not s:
        return s
    i = int(rng.integers(len(s)))
    if not s[i].isdigit():
        return s
    d = str((int(s[i]) + int(rng.integers(1, 10))) % 10)
    return s[:i] + d + s[i + 1 :]


def _corrupt(rng: np.random.Generator, p: Person, prof: CorruptionProfile) -> dict[str, str | None]:
    r: dict[str, str | None] = p.as_record()
    given = r["given_name"] or ""
    if rng.random() < prof.nickname and given in NICKNAMES:
        opts = NICKNAMES[given]
        given = opts[int(rng.integers(len(opts)))]
    if rng.random() < prof.name_typo:
        given = _typo(rng, given)
    r["given_name"] = given
    family = r["family_name"] or ""
    if rng.random() < prof.family_name_change:
        family = FAMILY_NAMES[int(rng.integers(len(FAMILY_NAMES)))]
    if rng.random() < prof.name_typo:
        family = _typo(rng, family)
    r["family_name"] = family
    dob = r["dob"] or ""
    if rng.random() < prof.dob_swap_day_month:
        y, m, d = dob.split("-")
        if int(d) <= 12:
            dob = f"{y}-{d}-{m}"
    if rng.random() < prof.dob_digit:
        candidate = _digit(rng, dob)
        try:
            date.fromisoformat(candidate)
            dob = candidate
        except ValueError:
            pass
    r["dob"] = dob
    pc = r["postcode"] or ""
    if rng.random() < prof.postcode_move:
        pc = f"{int(rng.integers(2000, 8000)):04d}"
    elif rng.random() < prof.postcode_digit:
        pc = _digit(rng, pc)
    r["postcode"] = pc
    if rng.random() < prof.missing_field:
        victim = ("given_name", "dob", "postcode")[int(rng.integers(3))]
        r[victim] = None
    return r


def _people(rng: np.random.Generator, n: int, start_id: int) -> list[Person]:
    gw = _zipf_weights(len(GIVEN_NAMES), 0.9)
    fw = _zipf_weights(len(FAMILY_NAMES), 1.0)
    gi = rng.choice(len(GIVEN_NAMES), size=n, p=gw)
    fi = rng.choice(len(FAMILY_NAMES), size=n, p=fw)
    base = date(1940, 1, 1)
    span = (date(2007, 12, 31) - base).days
    days = rng.integers(0, span, size=n)
    sexes = rng.choice(["f", "m", "x"], size=n, p=[0.495, 0.495, 0.01])
    pcs = rng.integers(2000, 8000, size=n)
    return [
        Person(
            entity_id=start_id + k,
            given_name=GIVEN_NAMES[gi[k]],
            family_name=FAMILY_NAMES[fi[k]],
            dob=(base + timedelta(days=int(days[k]))).isoformat(),
            sex=str(sexes[k]),
            postcode=f"{pcs[k]:04d}",
        )
        for k in range(n)
    ]


def generate_linkage_dataset(
    n_a: int,
    n_b: int,
    overlap: int,
    *,
    corruption: CorruptionProfile | str = "low",
    seed: int = 0,
    corrupt_a: bool = False,
) -> LinkageDataset:
    """Generate two data sets sharing ``overlap`` people.

    Data set A holds clean originals (unless ``corrupt_a``); data set B holds
    corrupted copies of the shared people plus distinct non-matching people.
    Rows are shuffled so that position carries no information.
    """
    if overlap > min(n_a, n_b):
        raise ValueError("overlap cannot exceed either data set size")
    prof = CorruptionProfile.named(corruption) if isinstance(corruption, str) else corruption
    rng = np.random.default_rng(seed)
    shared = _people(rng, overlap, 0)
    only_a = _people(rng, n_a - overlap, overlap)
    only_b = _people(rng, n_b - overlap, n_a)
    people_a: Sequence[Person] = shared + only_a
    people_b: Sequence[Person] = shared + only_b
    order_a = rng.permutation(n_a)
    order_b = rng.permutation(n_b)
    a = [
        _corrupt(rng, people_a[k], prof) if corrupt_a else people_a[k].as_record() for k in order_a
    ]
    b = [_corrupt(rng, people_b[k], prof) for k in order_b]
    ent_a = np.array([people_a[k].entity_id for k in order_a], dtype=np.int64)
    ent_b = np.array([people_b[k].entity_id for k in order_b], dtype=np.int64)
    pos_b = {int(e): j for j, e in enumerate(ent_b)}
    truth = {(i, pos_b[int(e)]) for i, e in enumerate(ent_a) if int(e) in pos_b}
    return LinkageDataset(a, b, ent_a, ent_b, truth)
