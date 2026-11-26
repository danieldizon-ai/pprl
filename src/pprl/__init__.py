"""Privacy-preserving record linkage with keyed Bloom-filter encodings.

See the README for the protocol, ``docs/threat-model.md`` for the attacker
model and ``docs/privacy.md`` for what the encodings do and do not protect.
"""

from .blocking import HammingLSHBlocker, KeyedBlocker
from .deterministic import KeyedHasher
from .encoding import CLKEncoder, FieldSpec, Hardening, pack_bits, unpack_bits
from .keys import SecretKey
from .matching import Links, one_to_one, threshold
from .similarity import dice, dice_pairs

__all__ = [
    "CLKEncoder",
    "FieldSpec",
    "HammingLSHBlocker",
    "Hardening",
    "KeyedBlocker",
    "KeyedHasher",
    "Links",
    "SecretKey",
    "dice",
    "dice_pairs",
    "one_to_one",
    "pack_bits",
    "threshold",
    "unpack_bits",
]

__version__ = "0.1.0"
