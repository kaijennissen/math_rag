import logging
import re
from typing import Dict, Any, Optional

import pytest
from math_rag.core.atomic_unit import AtomicUnit, GERMAN_TO_ENGLISH_TYPE

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
    (
        {
            "section": 2,
            "section_title": "Intro",
            "subsection": 3,
            "subsection_title": "Ex",
            "subsubsection": 4,
            "type": "Example",
            "identifier": "Beispiele 2.3.4",
            "text": "Text",
            "proof": None,
        },
        "2.3.4",  # Plural identifier, singular type -> OK
    ),
    (
        {
            "section": 4,
            "section_title": "Intro",
            "subsection": 5,
            "subsection_title": "Ex",
            "subsubsection": 6,
            "type": "Exercise",
            "identifier": "Aufgaben 4.5.6",
            "text": "Text",
            "proof": None,
        },
        "4.5.6",  # Plural identifier, singular type -> OK
    ),
    (
        {
            "section": 9,
            "section_title": "Main",
            "subsection": 1,
            "subsection_title": "Intro",
            "subsubsection": 1,  # Must be an integer
            "type": "Introduction",
            "identifier": "Einleitung 9.1.1",  # Using German type that's in the mapping
            "text": "Chapter intro",
        },
        "9.1.1",  # Updated to match the subsubsection number
    ),
]


@pytest.mark.parametrize("data, expected_number", valid_test_data)
def test_atomic_unit_valid(data, expected_number):
    """Tests successful creation of AtomicUnit with valid data."""
    unit = AtomicUnit.from_dict(data)
    # Add assertions for potentially None fields with defaults
    assert unit.section == data["section"]
    assert unit.subsection == data.get("subsection")
    assert unit.subsubsection == data.get("subsubsection")
    assert unit.type == data["type"]
    assert unit.identifier == data.get("identifier")
    assert unit.text == data["text"]
    assert unit.proof == data.get("proof")
    assert unit.get_full_number() == expected_number
    # Check type consistency (if identifier exists)
    if unit.identifier:
        german_type_match = re.match(r"([a-zA-ZäöüÄÖÜß]+)", unit.identifier)
        if german_type_match:
            german_type = german_type_match.group(1)
            expected_english_type = GERMAN_TO_ENGLISH_TYPE.get(german_type)
            assert expected_english_type is not None  # Should be known if valid
            assert expected_english_type.lower() == unit.type.lower()


# --- Test Cases For Warnings ---
# Test cases for validation warnings
invalid_test_data_for_warnings = [
    # Number mismatch -> Expect Warning
    pytest.param(
        {
            "section": 7,
            "subsection": 4,
            "subsubsection": 9,
            "type": "Theorem",
            "identifier": "Satz 7.5.9",  # Wrong subsection number
            "text": "...",
        },
        logging.WARNING,
        r"Section number mismatch in identifier. Expected 7.4.9, found 7.5.9",
        id="number_mismatch",
    ),
    # Type mismatch -> Expect Warning
    pytest.param(
        {
            "section": 1,
            "subsection": 2,
            "subsubsection": 3,
            "type": "Theorem",
            "identifier": "Definition 1.2.3",  # Type doesn't match
            "text": "...",
        },
        logging.WARNING,
        r"Type mismatch. Identifier suggests 'Definition', but type is 'Theorem'",
        id="type_mismatch",
    ),
    # Type mismatch case-insensitive -> Should not warn
    pytest.param(
        {
            "section": 2,
            "subsection": 1,
            "subsubsection": 3,
            "type": "lemma",
            "identifier": "Lemma 2.1.3",
            "text": "...",
        },
        None,
        None,
        id="type_case_insensitive_ok",
    ),
    # Missing number in identifier -> No warning (just a type is valid)
    pytest.param(
        {
            "section": 7,
            "subsection": 4,
            "subsubsection": 9,
            "type": "Theorem",
            "identifier": "Satz",
            "text": "...",
        },
        None,
        None,
        id="missing_number_ok",
    ),
    # Invalid number format -> No warning (only warns if numbers don't match)
    pytest.param(
        {
            "section": 7,
            "subsection": 4,
            "subsubsection": 9,
            "type": "Theorem",
            "identifier": "Satz 7.4.9.1",  # Extra number
            "text": "...",
        },
        None,
        None,
        id="invalid_number_format_ok",
    ),
    # Just a number -> Valid
    pytest.param(
        {
            "section": 7,
            "subsection": 4,
            "subsubsection": 9,
            "type": "Theorem",
            "identifier": "7.4.9",
            "text": "...",
        },
        None,
        None,
        id="just_number_ok",
    ),
    # Unknown German type -> Expect Warning
    pytest.param(
        {
            "section": 3,
            "subsection": 1,
            "subsubsection": 4,
            "type": "Theorem",
            "identifier": "Foo 3.1.4",
            "text": "...",
        },
        logging.WARNING,
        r"Unknown type 'Foo' in identifier",
        id="unknown_type_warn",
    ),
    # Type mismatch (plural form) -> Expect Warning
    pytest.param(
        {
            "section": 2,
            "subsection": 3,
            "subsubsection": 4,
            "type": "Theorem",
            "identifier": "Beispiele 2.3.4",
            "text": "...",
        },
        logging.WARNING,
        r"Type mismatch. Identifier suggests 'Example', but type is 'Theorem'",
        id="plural_type_mismatch_warn",
    ),
]


# Keep only the consolidated warning test function
@pytest.mark.parametrize(
    "data, expected_log_level, log_message_contains", invalid_test_data_for_warnings
)
def test_atomic_unit_all_warnings(
    data, expected_log_level, log_message_contains, caplog
):
    """Tests creation of AtomicUnit logs the expected warnings (or no warnings)."""
    caplog.set_level(logging.WARNING)
    AtomicUnit.from_dict(data)

    if expected_log_level is not None:
        assert any(
            record.levelno == expected_log_level
            and log_message_contains in record.message
            for record in caplog.records
        ), f"Expected warning containing '{log_message_contains}' not found in logs"
    else:
        assert not caplog.records, f"Unexpected warnings in logs: {caplog.records}"


# Test data for property methods
PROPERTY_TEST_DATA = [
    (
        {
            "section": 1,
            "subsection": 2,
            "subsubsection": 3,
            "type": "Theorem",
            "identifier": "Satz 1.2.3",
            "text": "Test",
        },
        "1.2.3",  # get_full_number
        "1.2",  # get_subsection_number
    ),
    (
        {
            "section": 4,
            "subsection": 5,
            "subsubsection": 1,  # Must be an integer
            "type": "Definition",
            "identifier": "Definition 4.5",
            "text": "Test",
        },
        "4.5.1",  # get_full_number with subsubsection
        "4.5",  # get_subsection_number
    ),
]


@pytest.mark.parametrize("data,expected_full,expected_subsection", PROPERTY_TEST_DATA)
def test_properties(data: Dict[str, Any], expected_full: str, expected_subsection: str):
    """Test the property methods of AtomicUnit."""
    unit = AtomicUnit.from_dict(data)
    assert unit.get_full_number() == expected_full
    assert unit.get_subsection_number() == expected_subsection


# Test data for identifier validation
IDENTIFIER_VALIDATION_DATA = [
    (
        {
            "section": 1,
            "subsection": 2,
            "subsubsection": 3,
            "type": "Theorem",
            "identifier": "Satz 1.2.3",
            "text": "Test",
        },
        None,  # No warning expected
        "",  # No message expected
    ),
    (
        {
            "section": 1,
            "subsection": 2,
            "subsubsection": 3,
            "type": "Theorem",
            "identifier": "Satz 9.9.9",  # Mismatched numbers
            "text": "Test",
        },
        logging.WARNING,
        "Section number mismatch in identifier. Expected 1.2.3, found 9.9.9",
    ),
    (
        {
            "section": 1,
            "subsection": 2,
            "subsubsection": 3,
            "type": "Theorem",
            "identifier": "InvalidType 1.2.3",  # Unknown type
            "text": "Test",
        },
        logging.WARNING,
        "Unknown type 'InvalidType' in identifier",
    ),
]


@pytest.mark.parametrize("data,log_level,log_message", IDENTIFIER_VALIDATION_DATA)
def test_identifier_validation(
    data: Dict[str, Any], log_level: Optional[int], log_message: str, caplog
):
    """Test identifier validation logic."""
    caplog.set_level(logging.WARNING)
    AtomicUnit.from_dict(data)

    if log_level is not None:
        assert any(
            record.levelno == log_level and log_message in str(record.message)
            for record in caplog.records
        ), f"Expected {log_level} with '{log_message}' in logs"
    else:
        assert not caplog.records, f"Unexpected warnings in logs: {caplog.records}"


# Test data for type validation
TYPE_VALIDATION_DATA = [
    (
        {
            "section": 1,
            "subsection": 2,
            "subsubsection": 3,
            "type": "Theorem",
            "identifier": "Satz 1.2.3",
            "text": "Test",
        },
        True,  # Valid
    ),
    (
        {
            "section": 1,
            "subsection": 2,
            "subsubsection": 3,
            "type": "InvalidType",  # Not in GERMAN_TO_ENGLISH_TYPE
            "identifier": "Satz 1.2.3",
            "text": "Test",
        },
        True,  # Still valid, just logs a warning
    ),
]


@pytest.mark.parametrize("data,is_valid", TYPE_VALIDATION_DATA)
def test_type_validation(data: Dict[str, Any], is_valid: bool, caplog):
    """Test type validation logic."""
    caplog.set_level(logging.WARNING)
    try:
        AtomicUnit.from_dict(data)
        assert is_valid
        if not is_valid:
            pytest.fail("Expected validation error but none occurred")
    except Exception as e:
        if is_valid:
            pytest.fail(f"Unexpected validation error: {e}")
        assert not is_valid
