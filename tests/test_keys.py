import pytest

from pprl.keys import SecretKey, hkdf_sha256, keyed_permutation


def test_hkdf_rfc5869_case_1() -> None:
    okm = hkdf_sha256(
        ikm=bytes.fromhex("0b" * 22),
        salt=bytes.fromhex("000102030405060708090a0b0c"),
        info=bytes.fromhex("f0f1f2f3f4f5f6f7f8f9"),
        length=42,
    )
    assert okm.hex() == (
        "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
    )


def test_hkdf_rejects_bad_length() -> None:
    with pytest.raises(ValueError):
        hkdf_sha256(b"k", b"i", 0)
    with pytest.raises(ValueError):
        hkdf_sha256(b"k", b"i", 255 * 32 + 1)


def test_short_keys_rejected() -> None:
    with pytest.raises(ValueError):
        SecretKey(b"x" * 31)
    with pytest.raises(TypeError):
        SecretKey("x" * 32)  # type: ignore[arg-type]


def test_repr_never_contains_key_material() -> None:
    raw = bytes(range(32))
    key = SecretKey(raw)
    assert raw.hex() not in repr(key)
    assert "redacted" in str(key)


def test_derived_keys_are_independent_and_stable() -> None:
    key = SecretKey(b"a" * 32)
    assert key.derive("x") == key.derive("x")
    assert key.derive("x") != key.derive("y")
    assert SecretKey(b"b" * 32).derive("x") != key.derive("x")


def test_generate_gives_distinct_keys() -> None:
    assert SecretKey.generate() != SecretKey.generate()


def test_keyed_permutation_is_a_permutation() -> None:
    p = keyed_permutation(b"k" * 32, 1000)
    assert sorted(p) == list(range(1000))
    assert p == keyed_permutation(b"k" * 32, 1000)
    assert p != keyed_permutation(b"j" * 32, 1000)
