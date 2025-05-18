import yaml
from typing import Optional, List
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Subsection:
    number: float
    title: str


@dataclass(frozen=True)
class Section:
    number: int
    title: str
    subsections: List[Subsection] = field(default_factory=list)


class SectionHeaders:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            self.raw_data = yaml.safe_load(f)
        self.sections: List[Section] = []
        self._parse()

    def _parse(self):
        for section_key, subsections in self.raw_data.items():
            match = re.match(r"^(\d+)\s+(.+)$", section_key)
            if not match:
                continue
            section_number = int(match.group(1))
            section_title = match.group(2)
            subsection_objs = []
            for subsection in subsections:
                sub_match = re.match(r"^(\d+)\.(\d+)\s+(.+)$", subsection)
                if not sub_match:
                    continue
                # sub_section_number = int(sub_match.group(1))  # Not used, but could check for consistency
                sub_number = float(f"{section_number}.{int(sub_match.group(2))}")
                sub_title = sub_match.group(3)
                subsection_objs.append(Subsection(number=sub_number, title=sub_title))
            self.sections.append(
                Section(
                    number=section_number,
                    title=section_title,
                    subsections=subsection_objs,
                )
            )

    def get_section_title(self, section_number: int) -> Optional[str]:
        section = next((s for s in self.sections if s.number == section_number), None)
        return section.title if section else None

    def get_subsections(self, section_number: int) -> List[Subsection]:
        section = next((s for s in self.sections if s.number == section_number), None)
        return section.subsections if section else []

    def get_subsection_title(
        self, section_number: int, subsection_number: int
    ) -> Optional[str]:
        subs = self.get_subsections(section_number)
        # Convert subsection number to float for comparison (e.g., 1.2 for section 1, subsection 2)
        target_number = float(f"{section_number}.{subsection_number}")
        sub = next((sub for sub in subs if sub.number == target_number), None)
        return sub.title if sub else None

    def get_all_sections(self) -> List[int]:
        return [s.number for s in self.sections]

    def all_sections(self) -> List[Section]:
        return self.sections

    def all_subsections(self, section_number: int) -> List[Subsection]:
        return self.get_subsections(section_number)

    def full_title(
        self, section_number: int, subsection_number: Optional[int] = None
    ) -> Optional[str]:
        if subsection_number is None:
            title = self.get_section_title(section_number)
            if title:
                return f"{section_number} {title}"
            return None
        else:
            title = self.get_subsection_title(section_number, subsection_number)
            if title:
                return f"{section_number}.{subsection_number} {title}"
            return None


if __name__ == "__main__":
    SECTION_HEADERS_PATH = "docs/section_headers.yaml"
    section_headers = SectionHeaders(SECTION_HEADERS_PATH)

    for section in section_headers.all_sections():
        print(f"Section {section.number} with title '{section.title}'")
        for subsection in section.subsections:
            print(f"   Subsection {subsection.number} with title '{subsection.title}'")
