"""
SQLModel definitions for the SQLite database storing atomic units.
"""

import datetime
from pathlib import Path
from typing import Optional
from sqlmodel import Field, SQLModel, create_engine, Session, select
from pydantic import validator
from math_rag.core.atomic_unit import AtomicUnit
from math_rag.core.project_root import ROOT


class AtomicUnitBase(SQLModel):
    """Base model for atomic units with common fields."""

    section: int = Field(index=True)
    section_title: Optional[str] = Field(default="")
    subsection: int = Field(index=True)
    subsection_title: Optional[str] = Field(default=None)
    subsubsection: int = Field(default=None, index=True)
    type: str = Field(index=True)
    identifier: Optional[str] = Field(default=None, index=True)
    text: str
    proof: Optional[str] = None
    summary: Optional[str] = None

    @validator("identifier", pre=True)
    def validate_identifier(cls, v):
        """Allow None for identifier but convert empty string to None."""
        if v == "":
            return None
        return v


class AtomicUnitDB(AtomicUnitBase, table=True):
    """SQLModel class for atomic units stored in the database."""

    __tablename__ = "atomicunit"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now, nullable=False
    )
    updated_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now, nullable=False
    )
    source_file: str = Field(index=True)

    def to_core_atomic_unit(self) -> AtomicUnit:
        """Convert to the core AtomicUnit dataclass for compatibility."""

        return AtomicUnit(
            section=self.section,
            subsection=self.subsection,
            subsubsection=self.subsubsection,
            type=self.type,
            identifier=self.identifier or "",
            text=self.text,
            section_title=self.section_title,
            subsection_title=self.subsection_title,
            proof=self.proof,
            summary=self.summary,
        )

    @classmethod
    def from_core_atomic_unit(cls, unit: "AtomicUnit", source_file: str):
        """Create from a core AtomicUnit instance."""
        return cls(
            section=unit.section,
            section_title=unit.section_title,
            subsection=unit.subsection,
            subsection_title=unit.subsection_title,
            subsubsection=unit.subsubsection,
            type=unit.type,
            identifier=unit.identifier,
            text=unit.text,
            proof=unit.proof,
            source_file=source_file,
        )


class DatabaseManager:
    """Manager for the SQLite database operations."""

    def __init__(self, db_path: Path):
        """Initialize database manager with configurable path.

        Args:
            db_path: Path to the database file (required).
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{str(self.db_path)}", echo=False)
        self._create_db_and_tables()

    def _create_db_and_tables(self):
        """Create database tables if they don't exist."""
        SQLModel.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return Session(self.engine)

    def add_atomic_unit(self, unit: AtomicUnitDB) -> int:
        """Add an atomic unit to the database and return its ID."""
        with self.get_session() as session:
            session.add(unit)
            session.commit()
            session.refresh(unit)
            return unit.id

    def add_atomic_units_batch(self, units: list[AtomicUnitDB]) -> list[int]:
        """Add multiple atomic units in a batch and return their IDs."""
        with self.get_session() as session:
            session.add_all(units)
            session.commit()
            for unit in units:
                session.refresh(unit)
            return [unit.id for unit in units]

    def get_atomic_unit(self, unit_id: int) -> Optional[AtomicUnitDB]:
        """Get an atomic unit by ID."""
        with self.get_session() as session:
            return session.get(AtomicUnitDB, unit_id)

    def get_atomic_units_by_section(self, section: int) -> list[AtomicUnitDB]:
        """Get all atomic units for a section."""
        with self.get_session() as session:
            statement = select(AtomicUnitDB).where(AtomicUnitDB.section == section)
            return session.exec(statement).all()

    def get_atomic_units_by_subsection(
        self, section: int, subsection: int
    ) -> list[AtomicUnitDB]:
        """Get all atomic units for a specific subsection."""
        with self.get_session() as session:
            statement = select(AtomicUnitDB).where(
                AtomicUnitDB.section == section, AtomicUnitDB.subsection == subsection
            )
            return session.exec(statement).all()

    def get_atomic_units_without_summary(self, limit: int = 100) -> list[AtomicUnitDB]:
        """Get atomic units without summaries, for batch processing."""
        with self.get_session() as session:
            statement = (
                select(AtomicUnitDB).where(AtomicUnitDB.summary == None).limit(limit)  # noqa: E711
            )
            return list(session.exec(statement))

    def update_summary(self, unit_id: int, summary: str) -> bool:
        """Update the summary for an atomic unit and return success status."""
        with self.get_session() as session:
            unit = session.get(AtomicUnitDB, unit_id)
            if not unit:
                return False
            unit.summary = summary
            unit.updated_at = datetime.datetime.now()
            session.add(unit)
            session.commit()
            return True

    def get_units_by_source_file(self, source_file: str) -> list[AtomicUnitDB]:
        """Get all units from a specific source file."""
        with self.get_session() as session:
            statement = select(AtomicUnitDB).where(
                AtomicUnitDB.source_file == source_file
            )
            return session.exec(statement).all()

    def delete_units_by_source_file(self, source_file: str) -> int:
        """Delete all units from a specific source file and return count."""
        with self.get_session() as session:
            units = select(AtomicUnitDB).where(AtomicUnitDB.source_file == source_file)
            count = len(session.exec(units).all())

            if count > 0:
                session.exec(
                    f"DELETE FROM atomicunit WHERE source_file = '{source_file}'"
                )
                session.commit()

            return count

    def get_all_source_files(self) -> list[str]:
        """Get a list of all source files in the database."""
        with self.get_session() as session:
            statement = select(AtomicUnitDB.source_file).distinct()
            return list(session.exec(statement))

    def count_total_units(self) -> int:
        """Count the total number of atomic units in the database."""
        with self.get_session() as session:
            statement = select(AtomicUnitDB)
            return len(session.exec(statement).all())

    def count_units_with_summary(self) -> int:
        """Count atomic units with summaries."""
        with self.get_session() as session:
            statement = select(AtomicUnitDB).where(AtomicUnitDB.summary != None)  # noqa: E711
            return len(session.exec(statement).all())


if __name__ == "__main__":
    DB_PATH = ROOT / "data" / "atomic_units.sqlite"
    engine = create_engine(f"sqlite:///{str(DB_PATH)}", echo=False)
    with Session(engine) as session:
        db_rows = session.exec(
            select(AtomicUnitDB)
        ).all()  # This returns a list of AtomicUnitDB
        units = [row.to_core_atomic_unit() for row in db_rows]
    print(len(units))
