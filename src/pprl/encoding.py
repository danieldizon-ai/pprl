"""Keyed Bloom-filter encodings: field-level Bloom filters and CLKs.

A cryptographic long-term key (CLK, Schnell, Bachteler and Reiher 2011) maps
the q-grams of several identifying fields into one Bloom filter of ``length``
bits. Two records with similar identifiers have similar bit patterns, so
similarity can be estimated without revealing the identifiers.

Bit positions are chosen by *random hashing* (Niedermeyer et al. 2014):
each token's positions come from HMAC-SHA256 under a field-specific sub-key,
not from the double-hashing scheme of the original proposal, whose linear
structure was exploited by the 2014 attack.

Hardening transformations (Ranbaduge and Schnell 2020 survey) are optional and
compose in the order given. None of them is a proof of security; see
``docs/privacy.md`` for what each one does and does not achieve.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt

from .keys import SecretKey, keyed_permutation
from .normalise import Tokeniser

__all__ = [
    "BitMatrix",
    "CLKEncoder",
    "FieldSpec",
    "Hardening",
    "Record",
    "pack_bits",
    "unpack_bits",
]

Record = Mapping[str, str | None]
BitMatrix = npt.NDArray[np.uint64]
"""Bloom filters packed row-wise into 64-bit words, shape ``(n, ceil(bits / 64))``."""


class Hardening(Enum):
    """Post-processing applied to each Bloom filter after encoding."""

    BALANCE = "balance"
    """Concatenate the filter with its complement and permute under a key
    (Schnell and Borgs 2016). Every encoding then has Hamming weight
    exactly ``length``, removing weight as a signal, at the cost of doubling size."""

    XOR_FOLD = "xor_fold"
    """XOR the two halves (Schnell, Borgs and Encarnação 2016). Halves length;
    each output bit mixes two input bits, which hinders bit-to-q-gram mapping."""

    RULE_90 = "rule_90"
    """One step of the elementary cellular automaton rule 90 with wrap-around
    (Schnell, Borgs and Encarnação 2018): bit *i* becomes bit *i-1* XOR bit *i+1*."""


@dataclass(frozen=True)
class FieldSpec:
    """How one identifying field contributes to the encoding.

    Attributes:
        name: Key in the input records.
        normalise: Normalisation applied before tokenising.
        tokenise: Tokeniser producing the set of tokens to insert.
        hashes: Number of bit positions per token (*k*). Fields that are more
            discriminating or more reliable may be given more positions,
            which raises their weight in the similarity (Durham et al. 2014).
        salt_field: Optional name of another field whose normalised value is
            mixed into this field's keys. Identical names in records with
            different salt values then encode differently, which defeats
            frequency alignment across salt groups, but a salt error turns a
            true match into a non-match for this field.
    """

    name: str
    normalise: Callable[[str | None], str]
    tokenise: Tokeniser
    hashes: int = 10
    salt_field: str | None = None

    def __post_init__(self) -> None:
        if self.hashes < 1:
            raise ValueError("hashes must be at least 1")


def pack_bits(bits: npt.NDArray[np.bool_]) -> BitMatrix:
    """Pack a boolean matrix ``(n, length)`` into ``(n, ceil(length/64))`` uint64 words."""
    n, length = bits.shape
    words = -(-length // 64)
    padded = np.zeros((n, words * 64), dtype=np.uint8)
    padded[:, :length] = bits
    packed = np.packbits(padded, axis=1, bitorder="little")
    return np.ascontiguousarray(packed).view("<u8").astype(np.uint64, copy=False)


def unpack_bits(packed: BitMatrix, length: int) -> npt.NDArray[np.bool_]:
    """Inverse of :func:`pack_bits`."""
    as_bytes = np.ascontiguousarray(packed.astype("<u8", copy=False)).view(np.uint8)
    bits = np.unpackbits(as_bytes, axis=1, bitorder="little")[:, :length]
    return bits.astype(bool)


@dataclass
class CLKEncoder:
    """Encode records into keyed Bloom filters.

    One encoder (same schema, key, length and hardening) MUST be used by every
    custodian in a linkage project; encodings from different configurations
    are not comparable.

    Example:
        >>> from pprl.normalise import QGrams, normalise_name
        >>> key = SecretKey(b"k" * 32)
        >>> enc = CLKEncoder([FieldSpec("name", normalise_name, QGrams(2))], key, length=256)
        >>> enc.encode([{"name": "Anna"}]).shape
        (1, 4)
    """

    schema: Sequence[FieldSpec]
    key: SecretKey
    length: int = 1024
    hardening: Sequence[Hardening] = ()
    _field_keys: dict[str, bytes] = field(init=False, repr=False)
    _cache: dict[tuple[str, str, str], tuple[int, ...]] = field(init=False, repr=False)
    _perm: npt.NDArray[np.intp] | None = field(init=False, repr=False, default=None)
    _specs: dict[str, FieldSpec] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.schema:
            raise ValueError("schema must contain at least one field")
        names = [f.name for f in self.schema]
        if len(set(names)) != len(names):
            raise ValueError("field names must be unique")
        if self.length < 64:
            raise ValueError("length must be at least 64 bits")
        if Hardening.XOR_FOLD in self.hardening and self.length % 2:
            raise ValueError("XOR folding needs an even length")
        self._field_keys = {f.name: self.key.derive(f"bloom/field/{f.name}") for f in self.schema}
        self._cache = {}
        self._specs = {f.name: f for f in self.schema}
        for f in self.schema:
            if f.salt_field is not None and f.salt_field == f.name:
                raise ValueError(f"field {f.name!r} cannot salt itself")
        if Hardening.BALANCE in self.hardening:
            perm_key = self.key.derive("bloom/balance-permutation")
            self._perm = np.asarray(keyed_permutation(perm_key, 2 * self.length), dtype=np.intp)

    @property
    def output_length(self) -> int:
        """Length in bits of encodings after hardening."""
        n = self.length
        for h in self.hardening:
            if h is Hardening.BALANCE:
                n *= 2
            elif h is Hardening.XOR_FOLD:
                if n % 2:
                    raise ValueError("XOR folding needs an even length at that stage")
                n //= 2
        return n

    def _positions(self, spec: FieldSpec, token: str, salt: str) -> tuple[int, ...]:
        cache_key = (spec.name, salt, token)
        hit = self._cache.get(cache_key)
        if hit is not None:
            return hit
        key = self._field_keys[spec.name]
        message = salt.encode("utf-8") + b"\x1f" + token.encode("utf-8")
        out: list[int] = []
        counter = 0
        while len(out) < spec.hashes:
            block = hmac.digest(key, message + counter.to_bytes(4, "big"), "sha256")
            for i in range(0, 32, 4):
                if len(out) == spec.hashes:
                    break
                # Modulo bias is below length / 2**32, negligible for practical lengths.
                out.append(int.from_bytes(block[i : i + 4], "big") % self.length)
            counter += 1
        result = tuple(out)
        self._cache[cache_key] = result
        return result

    def tokens(self, record: Record) -> dict[str, list[str]]:
        """The tokens each field contributes (for inspection and tests)."""
        return {
            spec.name: spec.tokenise(spec.normalise(record.get(spec.name))) for spec in self.schema
        }

    def encode_unpacked(self, records: Iterable[Record]) -> npt.NDArray[np.bool_]:
        """Encode to an unpacked boolean matrix of shape ``(n, output_length)``."""
        rows: list[npt.NDArray[np.bool_]] = []
        for record in records:
            bits = np.zeros(self.length, dtype=bool)
            for spec in self.schema:
                value = spec.normalise(record.get(spec.name))
                if not value:
                    continue
                salt = ""
                if spec.salt_field is not None:
                    salt_spec = self._specs.get(spec.salt_field)
                    raw_salt = record.get(spec.salt_field)
                    salt = salt_spec.normalise(raw_salt) if salt_spec else (raw_salt or "")
                for token in spec.tokenise(value):
                    bits[list(self._positions(spec, token, salt))] = True
            rows.append(bits)
        matrix = np.vstack(rows) if rows else np.zeros((0, self.length), dtype=bool)
        return self._harden(matrix)

    def encode(self, records: Iterable[Record]) -> BitMatrix:
        """Encode records into packed Bloom filters."""
        return pack_bits(self.encode_unpacked(records))

    def _harden(self, bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        for h in self.hardening:
            if h is Hardening.BALANCE:
                if bits.shape[1] != self.length:
                    raise ValueError("BALANCE must be applied before any length-changing step")
                if self._perm is None:  # pragma: no cover - set in __post_init__
                    raise RuntimeError("balance permutation missing")
                bits = np.concatenate([bits, ~bits], axis=1)[:, self._perm]
            elif h is Hardening.XOR_FOLD:
                half = bits.shape[1] // 2
                bits = bits[:, :half] ^ bits[:, half : 2 * half]
            elif h is Hardening.RULE_90:
                bits = np.roll(bits, 1, axis=1) ^ np.roll(bits, -1, axis=1)
        return bits
