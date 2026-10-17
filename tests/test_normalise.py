import pytest

from crypticbench.normalise import (
    display_from_grid,
    enum_parts,
    enum_total,
    enumeration_from_display,
    non_alpha_residue,
    norm_answer,
    strip_enumeration,
)


def test_norm_answer_drops_separators():
    assert norm_answer("ice-cream van") == "ICECREAMVAN"
    assert norm_answer(" O'Neill ") == "ONEILL"
    assert norm_answer("medic") == "MEDIC"


def test_non_alpha_residue():
    assert non_alpha_residue("ICE-CREAM") == ""
    assert non_alpha_residue("Catch-22") == "22"
    assert non_alpha_residue("AT&T") == "&"


@pytest.mark.parametrize(
    "clue,rest,enum",
    [
        ("Severe, partly in eastern region (5)", "Severe, partly in eastern region", "5"),
        ("Take a bow, first (3,4)", "Take a bow, first", "3,4"),
        ("A clue (5-3)", "A clue", "5-3"),
        ("Odd one ( 2, 3 ,2 )", "Odd one", "2,3,2"),
        ("No enumeration here", "No enumeration here", None),
    ],
)
def test_strip_enumeration(clue, rest, enum):
    assert strip_enumeration(clue) == (rest, enum)


def test_enumeration_from_display():
    assert enumeration_from_display("MEDIC") == "5"
    assert enumeration_from_display("ICE CREAM") == "3,5"
    assert enumeration_from_display("ICE-CREAM") == "3-5"
    assert enumeration_from_display("ICE-CREAM VAN") == "3-5,3"
    assert enumeration_from_display("O'NEILL") == "6"


def test_enum_total_and_parts():
    assert enum_total("3-5,3") == 11
    assert enum_parts("3-5,3") == [(3, "-"), (5, ","), (3, "")]
    assert enum_parts("7") == [(7, "")]


def test_display_round_trip():
    assert display_from_grid("ICECREAMVAN", "3-5,3") == "ICE-CREAM VAN"
    assert display_from_grid("sausage", "7") == "SAUSAGE"
    with pytest.raises(ValueError):
        display_from_grid("MEDIC", "3,3")
