import pytest
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.atomic_unit import AtomicUnit, GERMAN_TO_ENGLISH_TYPE

# --- Valid Test Cases ---

valid_test_data = [
    (
        {
            "section": 7,
            "section_title": "Perfekte Abbildungen",
            "subsection": 4,
            "subsection_title": "Satz",
            "subsubsection": 9,
            "type": "Theorem",
            "identifier": "Satz 7.4.9",
            "text": "...",
            "proof": "...",
        },
        "7.4.9",
    ),
    (
        {
            "section": 5,
            "section_title": "Title",
            "subsection": 1,
            "subsection_title": "Def",
            "subsubsection": 2,
            "type": "Definition",
            "identifier": "Definition 5.1.2",
            "text": "...",
        },
        "5.1.2",
    ),
    (
        {
            "section": 1,
            "section_title": "Intro",
            "subsection": 2,
            "subsection_title": "Ex",
            "subsubsection": 3,
            "type": "Example",
            "identifier": "Beispiel 1.2.3",
            "text": "Text",
            "proof": None,
        },
        "1.2.3",
    ),
]


@pytest.mark.parametrize("data, expected_number", valid_test_data)
def test_atomic_unit_valid(data, expected_number):
    """Tests successful creation of AtomicUnit with valid data."""
    unit = AtomicUnit.from_dict(data)
    assert unit.section == data["section"]
    assert unit.subsection == data["subsection"]
    assert unit.subsubsection == data["subsubsection"]
    assert unit.type == data["type"]
    assert unit.identifier == data["identifier"]
    assert unit.text == data["text"]
    assert unit.proof == data.get("proof")
    assert unit.get_full_number() == expected_number
    # Check type consistency (implicitly tested in __post_init__)
    german_type = data["identifier"].split()[0]
    assert GERMAN_TO_ENGLISH_TYPE.get(german_type) == data["type"]


# --- Invalid Test Cases ---

invalid_test_data_for_warnings = [
    # Number mismatch -> Expect Warning
    pytest.param(
        {
            "section": 7,
            "subsection": 4,
            "subsubsection": 9,
            "type": "Theorem",
            "identifier": "Satz 7.4.8",
            "text": "...",
            "section_title": "T",
            "subsection_title": "S",
        },
        logging.WARNING,  # Expect a warning
        "Identifier number mismatch",
        id="number_mismatch_warn",
    ),
    # Type mismatch (Identifier: Satz -> Theorem, Type: Definition) -> Expect Warning
    pytest.param(
        {
            "section": 7,
            "subsection": 4,
            "subsubsection": 9,
            "type": "Definition",
            "identifier": "Satz 7.4.9",
            "text": "...",
            "section_title": "T",
            "subsection_title": "S",
        },
        logging.WARNING,  # Expect a warning
        "Type mismatch",
        id="type_mismatch_warn",
    ),
    # Type mismatch (Identifier: Lemma -> Lemma, Type: theorem lowercase) - should NOT warn (case-insensitive)
    pytest.param(
        {
            "section": 2,
            "subsection": 1,
            "subsubsection": 3,
            "type": "lemma",
            "identifier": "Lemma 2.1.3",
            "text": "...",
            "section_title": "T",
            "subsection_title": "S",
        },
        None,  # Expect NO warning/error
        None,
        id="type_mismatch_case_insensitive_ok",
    ),
    # Missing number in identifier -> Expect Warning
    pytest.param(
        {
            "section": 7,
            "subsection": 4,
            "subsubsection": 9,
            "type": "Theorem",
            "identifier": "Satz",
            "text": "...",
            "section_title": "T",
            "subsection_title": "S",
        },
        logging.WARNING,  # Expect a warning
        "Could not extract number",
        id="missing_number_warn",
    ),
    # Invalid number format in identifier -> Expect Warning
    pytest.param(
        {
            "section": 7,
            "subsection": 4,
            "subsubsection": 9,
            "type": "Theorem",
            "identifier": "Satz 7.4",
            "text": "...",
            "section_title": "T",
            "subsection_title": "S",
        },
        logging.WARNING,  # Expect a warning
        "Could not extract number",
        id="invalid_number_format_warn",
    ),
    # Missing type in identifier (just number) -> Expect Warning
    pytest.param(
        {
            "section": 7,
            "subsection": 4,
            "subsubsection": 9,
            "type": "Theorem",
            "identifier": "7.4.9",
            "text": "...",
            "section_title": "T",
            "subsection_title": "S",
        },
        logging.WARNING,  # Expect a warning
        "Could not extract type",
        id="missing_type_warn",
    ),
]

# Test cases that should still raise ValueError
invalid_test_data_for_errors = [
    # Unknown German type in identifier -> Expect Error
    pytest.param(
        {
            "section": 7,
            "subsection": 4,
            "subsubsection": 9,
            "type": "Theorem",
            "identifier": "Unbekannt 7.4.9",
            "text": "...",
            "section_title": "T",
            "subsection_title": "S",
        },
        ValueError,
        "Unknown German type",
        id="unknown_german_type_error",  # Renamed ID slightly
    ),
]


@pytest.mark.parametrize(
    "data, expected_log_level, log_message_contains", invalid_test_data_for_warnings
)
def test_atomic_unit_warnings(data, expected_log_level, log_message_contains, caplog):
    """Tests creation of AtomicUnit logs the expected warnings for specific inconsistencies."""
    with caplog.at_level(logging.WARNING):
        AtomicUnit.from_dict(data)

    if expected_log_level is None:
        assert not caplog.records, f"Expected no logs, but got: {caplog.text}"
    else:
        assert len(caplog.records) >= 1, (
            f"Expected a warning log, but none was captured for: {data['identifier']}"
        )
        found_match = False
        for record in caplog.records:
            if (
                record.levelno == expected_log_level
                and log_message_contains in record.message
            ):
                found_match = True
                break
        assert found_match, (
            f"Expected log level {expected_log_level} with message containing '{log_message_contains}' not found in logs: {caplog.text}"
        )


@pytest.mark.parametrize(
    "data, expected_exception, error_message_contains", invalid_test_data_for_errors
)
def test_atomic_unit_errors(data, expected_exception, error_message_contains):
    """Tests creation of AtomicUnit still raises errors for critical issues."""
    with pytest.raises(expected_exception) as excinfo:
        AtomicUnit.from_dict(data)
    assert error_message_contains in str(excinfo.value)
