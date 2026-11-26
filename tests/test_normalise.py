import pytest

from pprl.normalise import (
    DateParts,
    Positional,
    QGrams,
    normalise_date,
    normalise_digits,
    normalise_name,
    soundex,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  José  ", "jose"),
        ("O'Brien", "obrien"),
        ("O’Brien", "obrien"),
        ("Smith-Jones", "smith jones"),
        ("MÜLLER", "muller"),
        ("van  der   Berg", "van der berg"),
        ("Nguyễn", "nguyen"),
        ("李", "李"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalise_name(raw: str | None, expected: str) -> None:
    assert normalise_name(raw) == expected


def test_normalise_digits_and_dates() -> None:
    assert normalise_digits("+61 (2) 9999-0000") == "61299990000"
    assert normalise_date("1987-05-12") == "19870512"
    assert normalise_date("1987-05-12T10:00:00Z") == "19870512"
    assert normalise_date("12/05/1987") == ""
    assert normalise_date(None) == ""


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("Robert", "R163"),
        ("Rupert", "R163"),
        ("Ashcraft", "A261"),
        ("Tymczak", "T522"),
        ("Pfister", "P236"),
        ("Lee", "L000"),
        ("", ""),
    ],
)
def test_soundex(name: str, code: str) -> None:
    assert soundex(name) == code


def test_qgrams() -> None:
    assert QGrams(2)("anna") == sorted({"_a", "an", "nn", "na", "a_"})
    assert QGrams(2)("") == []
    assert QGrams(3, pad=False)("ab") == ["ab"]


def test_positional_and_date_parts() -> None:
    assert Positional()("2000") == ["0:2", "1:0", "2:0", "3:0"]
    assert DateParts()("19870512") == ["y:1987", "m:05", "d:12", "md:0512"]
    assert DateParts()("1987") == []
