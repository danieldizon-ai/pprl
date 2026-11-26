import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pprl.encoding import CLKEncoder, FieldSpec, Hardening, pack_bits, unpack_bits
from pprl.experiments import DEFAULT_SCHEMA
from pprl.keys import SecretKey
from pprl.normalise import Positional, QGrams, normalise_digits, normalise_name
from pprl.similarity import dice, popcount

KEY = SecretKey(b"k" * 32)
REC = {"given_name": "Olivia", "family_name": "Nguyen", "dob": "1990-03-14", "postcode": "2042"}


@given(arrays(np.bool_, st.tuples(st.integers(0, 5), st.integers(1, 300))))
def test_pack_unpack_roundtrip(bits: npt.NDArray[np.bool_]) -> None:
    packed = pack_bits(bits)
    assert packed.dtype == np.uint64
    assert packed.shape == (bits.shape[0], -(-bits.shape[1] // 64))
    assert np.array_equal(unpack_bits(packed, bits.shape[1]), bits)


def test_encoding_is_deterministic_and_keyed() -> None:
    a = CLKEncoder(DEFAULT_SCHEMA, KEY).encode([REC])
    b = CLKEncoder(DEFAULT_SCHEMA, KEY).encode([REC])
    c = CLKEncoder(DEFAULT_SCHEMA, SecretKey(b"j" * 32)).encode([REC])
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert dice(a, c) < 0.6


def test_similarity_tracks_identifier_similarity() -> None:
    enc = CLKEncoder(DEFAULT_SCHEMA, KEY)
    typo = {**REC, "family_name": "Nguyn"}
    other = {"given_name": "James", "family_name": "Smith", "dob": "1971-11-02", "postcode": "6000"}
    e = enc.encode([REC, typo, other])
    assert dice(e[0], e[0]) == 1.0
    assert dice(e[0], e[1]) > 0.85
    assert dice(e[0], e[2]) < 0.4


def test_empty_record_encodes_to_zero_bits() -> None:
    e = CLKEncoder(DEFAULT_SCHEMA, KEY).encode([{}])
    assert popcount(e)[0] == 0


def test_tokens_exposes_field_tokens() -> None:
    toks = CLKEncoder(DEFAULT_SCHEMA, KEY).tokens(REC)
    assert "_o" in toks["given_name"]
    assert toks["postcode"] == ["0:2", "1:0", "2:4", "3:2"]


@pytest.mark.parametrize(
    ("hardening", "length"),
    [
        ((), 1024),
        ((Hardening.BALANCE,), 2048),
        ((Hardening.XOR_FOLD,), 512),
        ((Hardening.RULE_90,), 1024),
        ((Hardening.BALANCE, Hardening.XOR_FOLD), 1024),
    ],
)
def test_hardening_output_length(hardening: tuple[Hardening, ...], length: int) -> None:
    enc = CLKEncoder(DEFAULT_SCHEMA, KEY, 1024, hardening)
    assert enc.output_length == length
    assert enc.encode_unpacked([REC]).shape == (1, length)


@settings(max_examples=25)
@given(st.text(alphabet="abcdefghij", min_size=1, max_size=12))
def test_balanced_filters_have_constant_weight(name: str) -> None:
    enc = CLKEncoder([FieldSpec("n", normalise_name, QGrams(2))], KEY, 256, (Hardening.BALANCE,))
    assert enc.encode_unpacked([{"n": name}]).sum() == 256


def test_salt_changes_encoding_of_equal_values() -> None:
    yob = FieldSpec("yob", normalise_digits, Positional(), 1)
    salted = CLKEncoder(
        [FieldSpec("n", normalise_name, QGrams(2), salt_field="yob"), yob], KEY, 512
    )
    plain = CLKEncoder([FieldSpec("n", normalise_name, QGrams(2))], KEY, 512)
    recs = [{"n": "smith", "yob": "1980"}, {"n": "smith", "yob": "1981"}]
    a, b = salted.encode(recs)
    assert not np.array_equal(a, b)
    c, d = plain.encode(recs)
    assert np.array_equal(c, d)


def test_invalid_configurations() -> None:
    with pytest.raises(ValueError):
        CLKEncoder([], KEY)
    with pytest.raises(ValueError):
        CLKEncoder([*DEFAULT_SCHEMA, DEFAULT_SCHEMA[0]], KEY)
    with pytest.raises(ValueError):
        CLKEncoder(DEFAULT_SCHEMA, KEY, 32)
    with pytest.raises(ValueError):
        CLKEncoder(DEFAULT_SCHEMA, KEY, 1001, (Hardening.XOR_FOLD,))
    with pytest.raises(ValueError):
        FieldSpec("x", normalise_name, QGrams(2), hashes=0)
    with pytest.raises(ValueError):
        CLKEncoder([FieldSpec("x", normalise_name, QGrams(2), salt_field="x")], KEY)
