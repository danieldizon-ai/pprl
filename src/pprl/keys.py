"""Secret key handling and key derivation.

Every keyed operation in this library (Bloom-filter positions, blocking keys,
deterministic fingerprints, permutations) uses a sub-key derived from one
master secret with HKDF-SHA256 (RFC 5869). Distinct labels give independent
sub-keys, so compromising the output of one component (for example a blocking
key table) does not reveal the key used by another.

The master secret is shared by the data custodians and MUST NOT be given to
the linkage unit. See ``docs/threat-model.md``.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Iterator

__all__ = ["MIN_KEY_BYTES", "SecretKey", "hkdf_sha256", "keyed_permutation", "keyed_stream"]

MIN_KEY_BYTES = 32
_HASH_LEN = 32


def hkdf_sha256(ikm: bytes, info: bytes, length: int = _HASH_LEN, salt: bytes = b"") -> bytes:
    """HKDF-SHA256 extract-and-expand (RFC 5869)."""
    if length <= 0 or length > 255 * _HASH_LEN:
        raise ValueError("HKDF output length out of range")
    prk = hmac.digest(salt or b"\x00" * _HASH_LEN, ikm, "sha256")
    okm = b""
    block = b""
    counter = 1
    while len(okm) < length:
        block = hmac.digest(prk, block + info + bytes([counter]), "sha256")
        okm += block
        counter += 1
    return okm[:length]


class SecretKey:
    """A master secret of at least 32 bytes.

    The raw bytes are never included in ``repr`` or ``str`` so that keys do not
    leak into logs or tracebacks by accident.
    """

    __slots__ = ("_raw",)

    def __init__(self, raw: bytes) -> None:
        if not isinstance(raw, bytes | bytearray):
            raise TypeError("key material must be bytes")
        if len(raw) < MIN_KEY_BYTES:
            raise ValueError(f"key material must be at least {MIN_KEY_BYTES} bytes")
        self._raw = bytes(raw)

    @classmethod
    def generate(cls) -> SecretKey:
        """Generate a fresh 256-bit key from the operating system CSPRNG."""
        return cls(secrets.token_bytes(MIN_KEY_BYTES))

    @classmethod
    def from_hex(cls, text: str) -> SecretKey:
        return cls(bytes.fromhex(text))

    def derive(self, label: str, length: int = _HASH_LEN) -> bytes:
        """Derive an independent sub-key for one purpose, named by ``label``."""
        return hkdf_sha256(self._raw, b"pprl/v1/" + label.encode("utf-8"), length)

    def __repr__(self) -> str:
        fingerprint = hashlib.sha256(b"pprl/key-id/" + self._raw).hexdigest()[:8]
        return f"SecretKey(id={fingerprint}, <redacted>)"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SecretKey):
            return NotImplemented
        return hmac.compare_digest(self._raw, other._raw)

    def __hash__(self) -> int:
        return hash(hashlib.sha256(self._raw).digest())


def keyed_stream(key: bytes, label: bytes) -> Iterator[int]:
    """An endless stream of 32-bit unsigned integers from HMAC-SHA256 in counter mode."""
    counter = 0
    while True:
        block = hmac.digest(key, label + counter.to_bytes(8, "big"), "sha256")
        for i in range(0, _HASH_LEN, 4):
            yield int.from_bytes(block[i : i + 4], "big")
        counter += 1


def keyed_permutation(key: bytes, n: int, label: bytes = b"perm") -> list[int]:
    """A permutation of ``range(n)`` determined by ``key`` (Fisher-Yates, unbiased).

    Uses rejection sampling on an HMAC-SHA256 stream so that every permutation
    is equally likely under a uniformly random key.
    """
    perm = list(range(n))
    stream = keyed_stream(key, label)
    for i in range(n - 1, 0, -1):
        bound = i + 1
        limit = (1 << 32) - ((1 << 32) % bound)
        while True:
            r = next(stream)
            if r < limit:
                break
        j = r % bound
        perm[i], perm[j] = perm[j], perm[i]
    return perm
