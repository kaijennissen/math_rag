"""
Script to migrate existing atomic units from JSON/pickle files to SQLite database.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from math_rag.core.atomic_unit import AtomicItem as CoreAtomicItem
from math_rag.core.db_models import AtomicItem, DatabaseManager
from math_rag.core.project_root import ROOT

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_file(file_path: Path, db_manager: DatabaseManager) -> int:
    """Process a single atomic unit file and add its contents to the database."""
    logger.info(f"Processing file: {file_path}")

    # Determine if it's JSON or pickle
    if file_path.suffix == ".json":
        with file_path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON from {file_path}")
                return 0
    elif file_path.suffix == ".pkl":
        # For pickle files, we'll skip them and just use the corresponding JSON file
        # This avoids pickle deserialization issues when class definitions change
        json_file = file_path.with_suffix(".json")
        if json_file.exists():
            logger.info(f"Skipping pickle file {file_path}, will use JSON file instead")
            return 0
        else:
            logger.warning(
                f"Cannot process pickle file {file_path} and no JSON alternative found"
            )
            return 0
    else:
        logger.warning(f"Unsupported file type: {file_path}")
        return 0

    # Process atomic units
    # Use the full filename as the source identifier
    file_identifier = str(file_path.name)

    # Check if file has already been processed
    existing_units = db_manager.get_units_by_source_file(file_identifier)
    if existing_units:
        logger.info(
            f"File {file_identifier} already processed with {len(existing_units)} units"
        )
        return 0

    # Extract chunks from the data structure
    # The files contain either:
    # 1. {"chunks": [list_of_atomic_units]} format (from extract_atomic_units.py)
    # 2. Direct list of atomic units (legacy format)
    chunks = []

    if isinstance(data, dict) and "chunks" in data:
        # Standard format from extract_atomic_units.py
        chunks = data["chunks"]
    elif isinstance(data, list):
        # Legacy direct list format
        chunks = data
    else:
        logger.warning(f"Unexpected data format in {file_path}: {type(data)}")
        return 0

    # Convert chunks to database units
    atomic_units = []
    for chunk_data in chunks:
        try:
            if isinstance(chunk_data, CoreAtomicItem):
                core_unit = chunk_data
            elif isinstance(chunk_data, dict):
                core_unit = CoreAtomicItem.from_dict(chunk_data)
            else:
                logger.warning(f"Unknown chunk type in {file_path}: {type(chunk_data)}")
                continue

            db_unit = AtomicItem.from_core_atomic_unit(
                core_unit, source_file=file_identifier
            )
            atomic_units.append(db_unit)
        except Exception as e:
            logger.error(f"Error processing chunk in {file_path}: {e}")
            continue

    # Add all units in a single batch
    if atomic_units:
        ids = db_manager.add_atomic_units_batch(atomic_units)
        units_added = len(ids)
        logger.info(f"Added {units_added} units from {file_path}")
        return units_added
    else:
        logger.warning(f"No valid atomic units found in {file_path}")
        return 0


def migrate_all_files(
    db_manager: DatabaseManager, source_dir: Optional[str] = None, num_workers: int = 4
) -> int:
    """
    Migrate all atomic unit files to the SQLite database.

    Args:
        db_manager: DatabaseManager instance (required)
        source_dir: Directory containing atomic unit files,
                   defaults to 'docs/atomic_units'
        num_workers: Number of worker threads for parallel processing

    Returns:
        Total number of units migrated
    """
    if source_dir is None:
        source_dir = ROOT / "docs" / "atomic_units"
    else:
        source_dir = Path(source_dir)

    # Get only JSON files matching subsection_*_units pattern
    file_paths = [p for p in source_dir.rglob("subsection_*_units.json") if p.is_file()]

    logger.info(f"Found {len(file_paths)} files to process")

    # Process files in parallel
    total_units = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(process_file, file_path, db_manager): file_path
            for file_path in file_paths
        }

        for future in future_to_file:
            try:
                units_added = future.result()
                total_units += units_added
            except Exception as e:
                logger.error(f"Error processing {future_to_file[future]}: {e}")

    logger.info(f"Migration complete. Total units added: {total_units}")
    return total_units


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate atomic units to SQLite database"
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to the SQLite database (required)",
    )
    parser.add_argument(
        "--source-dir",
        help="Directory containing atomic unit files (default: docs/atomic_units)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads (default: 4)",
    )

    args = parser.parse_args()

    # Create database manager
    db_manager = DatabaseManager(Path(args.db_path))

    migrate_all_files(db_manager, args.source_dir, args.workers)
