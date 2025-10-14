"""
Input/output functionality for Math-RAG evaluation system.

This submodule contains I/O operations including:
- Dataset loading from CSV files
- Output writing to JSON files
- Logging and summary formatting
"""

from math_rag.evaluation.io.dataset_loader import load_dataset
from math_rag.evaluation.io.output_writer import (
    log_summary_results,
    write_results,
    write_summary,
)

__all__ = [
    # Dataset loading
    "load_dataset",
    # Output handling
    "log_summary_results",
    "write_results",
    "write_summary",
]
