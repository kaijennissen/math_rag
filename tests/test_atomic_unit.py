import pytest
import logging
import re  # Added for regex search in warnings

from math_rag.atomic_unit import AtomicUnit, GERMAN_TO_ENGLISH_TYPE

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
            "subsubsection": None,
            "type": "Introduction",
            "identifier": None,
            "text": "Chapter intro",
        },
        "9.1",  # Expected number is None
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
# Consolidating into one list for test_atomic_unit_all_warnings

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
        logging.WARNING,
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
        logging.WARNING,
        "Type mismatch",
        id="type_mismatch_warn",
    ),
    # Type mismatch (Identifier: Lemma -> Lemma, Type: theorem lowercase) - should NOT warn
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
        None,
        None,
        id="type_mismatch_case_insensitive_ok",
    ),
    # Missing number in identifier (but subsubsection exists) -> Expect Warning
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
        logging.WARNING,
        r"Could not extract number.*for subsubsection 7\.4\.9",
        id="missing_number_with_subsubsection_warn",
    ),  # Added regex pattern
    # Invalid number format in identifier (but subsubsection exists) -> Expect Warning
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
        logging.WARNING,
        r"Could not extract number.*for subsubsection 7\.4\.9",
        id="invalid_number_format_with_subsubsection_warn",
    ),  # Added regex pattern
    # Missing type in identifier (but subsubsection exists) -> Expect Warning
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
        logging.WARNING,
        r"Could not extract type.*for subsubsection 7\.4\.9",
        id="missing_type_with_subsubsection_warn",
    ),  # Added regex pattern
    # Unknown German type in identifier -> Expect Warning
    pytest.param(
        {
            "section": 3,
            "subsection": 1,
            "subsubsection": 4,
            "type": "Unknown",
            "identifier": "Foo 3.1.4",
            "text": "...",
            "section_title": "T",
            "subsection_title": "S",
        },
        logging.WARNING,
        "Unknown German type 'Foo'",
        id="unknown_german_type_warn",
    ),
    # Plural identifier mismatch (Beispiele -> Example, Type: Theorem) -> Expect Warning
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
        r"Type mismatch.*Identifier implies 'Example'",
        id="plural_type_mismatch_warn",
    ),  # Added regex pattern
    # Missing Identifier, Missing Subsubsection -> Expect NO Warning
    pytest.param(
        {
            "section": 9,
            "subsection": 1,
            "subsubsection": None,
            "type": "Introduction",
            "identifier": None,
            "text": "Intro",
        },
        None,
        None,
        id="missing_identifier_no_subsubsection_ok",
    ),
    # Missing Identifier, PRESENT Subsubsection -> Expect Warning
    pytest.param(
        {
            "section": 9,
            "subsection": 1,
            "subsubsection": 5,
            "type": "Remark",
            "identifier": None,
            "text": "Something",
        },
        logging.WARNING,
        r"Missing identifier for content in subsubsection 9\.1\.5",
        id="missing_identifier_with_subsubsection_warn",
    ),  # Added regex pattern
]


# Keep only the consolidated warning test function
@pytest.mark.parametrize(
    "data, expected_log_level, log_message_contains", invalid_test_data_for_warnings
)
def test_atomic_unit_all_warnings(
    data, expected_log_level, log_message_contains, caplog
):
    """Tests creation of AtomicUnit logs the expected warnings (or no warnings)."""
    with caplog.at_level(logging.WARNING):
        AtomicUnit.from_dict(data)  # Create the unit, triggering __post_init__

    if expected_log_level is None:
        # Assert that NO warning messages were logged
        assert len(caplog.records) == 0, f"Expected no warnings, but got: {caplog.text}"
    else:
        # Assert that AT LEAST one warning message was logged
        assert len(caplog.records) > 0, (
            f"Expected warning containing '{log_message_contains}', but no warnings were logged."
        )
        # Assert that the specific expected message part is present in the logs
        found_message = False
        for record in caplog.records:
            # Use re.search for flexible matching of the warning message
            if (
                record.levelno == expected_log_level
                and log_message_contains
                and re.search(log_message_contains, record.message)
            ):
                found_message = True
                break
        assert found_message, (
            f"Expected log message pattern '{log_message_contains}' not found in warnings: {caplog.text}"
        )
