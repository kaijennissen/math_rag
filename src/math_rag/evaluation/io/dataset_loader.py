"""
Dataset loading utilities for evaluation.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List

import dspy

logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str) -> List[Dict[str, str]]:
    """
    Load evaluation dataset from CSV file.

    Args:
        dataset_path: Path to CSV file with columns: difficulty,question,answer,source

    Returns:
        List of dictionaries with dataset rows

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If required columns are missing
    """
    if not Path(dataset_path).exists():
        logger.warning(f"Dataset file not found: {dataset_path}")
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Check required columns
        required_columns = {"difficulty", "question", "answer", "source"}
        if not required_columns.issubset(reader.fieldnames or []):
            missing = required_columns - set(reader.fieldnames or [])
            raise ValueError(f"Missing required columns: {missing}")

        examples = []
        for i, row in enumerate(reader):
            try:
                # Validate difficulty is integer
                example = dspy.Example(
                    question=row["question"],
                    answer=row["answer"],
                    source=row["source"],
                    difficulty=int(row["difficulty"]),
                )
                examples.append(example)
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping malformed row {i + 1}: {e}")
                continue

        logger.info(f"Loaded {len(examples)} questions from {dataset_path}")
        return examples
