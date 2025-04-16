import pytest
import tempfile
import os

from math_rag.section_headers import SectionHeaders, Subsection  # noqa: E402

MOCK_YAML = """
1 Topologische Strukturen, Grundbegriffe und Beispiele:
  - 1.0 Einleitung
  - 1.1 Topologien auf einer Menge
  - 1.2 Punkte und Mengen in topologischen Räumen
  - 1.3 Mengensysteme und Überdeckungen
  - 1.4 Beispiele von topologischen Räumen
2 Konvergenz:
  - 2.0 Einleitung
  - 2.1 Konvergenz- und Verdichtungspunkte von Folgen
  - 2.2 Konvergenz- und Berührpunkte von Mengensystemen
  - 2.3 Stapel, Filter, Grills und Ultrafilter
  - 2.4 Existenz und Eindeutigkeit von Konvergenz- und Berührpunkten
"""


@pytest.fixture(scope="module")
def mock_yaml_file():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
        f.write(MOCK_YAML)
        temp_path = f.name
    yield temp_path
    os.remove(temp_path)


@pytest.fixture(scope="module")
def section_headers(mock_yaml_file):
    return SectionHeaders(mock_yaml_file)


@pytest.mark.parametrize(
    "section_number,expected_title",
    [
        (1, "Topologische Strukturen, Grundbegriffe und Beispiele"),
        (2, "Konvergenz"),
    ],
)
def test_get_section_title(section_headers, section_number, expected_title):
    assert section_headers.get_section_title(section_number) == expected_title


@pytest.mark.parametrize(
    "section_number,expected_subs",
    [
        (
            1,
            [
                Subsection(1.0, "Einleitung"),
                Subsection(1.1, "Topologien auf einer Menge"),
                Subsection(1.2, "Punkte und Mengen in topologischen Räumen"),
                Subsection(1.3, "Mengensysteme und Überdeckungen"),
                Subsection(1.4, "Beispiele von topologischen Räumen"),
            ],
        ),
        (
            2,
            [
                Subsection(2.0, "Einleitung"),
                Subsection(2.1, "Konvergenz- und Verdichtungspunkte von Folgen"),
                Subsection(2.2, "Konvergenz- und Berührpunkte von Mengensystemen"),
                Subsection(2.3, "Stapel, Filter, Grills und Ultrafilter"),
                Subsection(
                    2.4, "Existenz und Eindeutigkeit von Konvergenz- und Berührpunkten"
                ),
            ],
        ),
    ],
)
def test_get_subsections(section_headers, section_number, expected_subs):
    assert section_headers.get_subsections(section_number) == expected_subs


@pytest.mark.parametrize(
    "section_number,subsection_number,expected_title",
    [
        (1, 0, "Einleitung"),
        (1, 2, "Punkte und Mengen in topologischen Räumen"),
        (2, 3, "Stapel, Filter, Grills und Ultrafilter"),
    ],
)
def test_get_subsection_title(
    section_headers, section_number, subsection_number, expected_title
):
    assert (
        section_headers.get_subsection_title(section_number, subsection_number)
        == expected_title
    )


@pytest.mark.parametrize(
    "section_number,subsection_number,expected_full_title",
    [
        (1, None, "1 Topologische Strukturen, Grundbegriffe und Beispiele"),
        (2, None, "2 Konvergenz"),
        (1, 1, "1.1 Topologien auf einer Menge"),
        (2, 4, "2.4 Existenz und Eindeutigkeit von Konvergenz- und Berührpunkten"),
    ],
)
def test_full_title(
    section_headers, section_number, subsection_number, expected_full_title
):
    assert (
        section_headers.full_title(section_number, subsection_number)
        == expected_full_title
    )
