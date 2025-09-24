"""
Module for generating summaries for atomic units using LLMs.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from math_rag.core import ROOT
from math_rag.core.db_models import AtomicItem, DatabaseManager

# Load environment variables
load_dotenv()

# Load LLM configuration
config_path = ROOT / "config" / "config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

LLM_CONFIG = config.get("llm", {})
MODEL_NAME = LLM_CONFIG.get("model", "gpt-4.1")
TEMPERATURE = LLM_CONFIG.get("temperature", 0.2)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Timeout settings
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2

# RAG-optimized summary prompt template
SUMMARY_PROMPT_TEMPLATE = """Erstelle eine suchoptimierte deutsche Zusammenfassung
(max. 80 Wörter) für ein RAG-System des folgenden mathematischen Inhalts.

OPTIMIERE FÜR RETRIEVAL:
1. Beginne mit dem Konzepttyp (Definition, Satz, Beweis, Beispiel)
2. Nenne das mathematische Gebiet (z.B. Topologie, Analysis, Algebra)
3. Liste die wichtigsten mathematischen Begriffe und Schlüsselwörter auf
4. Erwähne Voraussetzungen oder verwandte Konzepte
5. Verwende sowohl formale als auch intuitive Beschreibungen
6. Schließe alternative Bezeichnungen oder Synonyme ein
7. Antizipiere typische Suchfragen zu diesem Inhalt

INHALT:
{identifier}: {text}{proof_text}

SUCHOPTIMIERTE ZUSAMMENFASSUNG:"""


def generate_summary_prompt(unit: AtomicItem) -> str:
    """
    Generate a prompt for summarizing an atomic unit.

    Args:
        unit: The atomic unit to summarize

    Returns:
        Prompt string for the LLM
    """
    unit_type = unit.type.capitalize()
    identifier = unit.identifier or f"Unit {unit.id}"

    # Create section number reference
    section_ref = f"{unit.section}.{unit.subsection}"
    if unit.subsubsection is not None:
        section_ref += f".{unit.subsubsection}"

    # Include proof if available
    proof_text = f"\n\nProof:\n{unit.proof}" if unit.proof else ""

    return SUMMARY_PROMPT_TEMPLATE.format(
        unit_type=unit_type,
        identifier=identifier,
        text=unit.text,
        proof_text=proof_text,
    )


def generate_summary(unit: AtomicItem, model_name: str) -> Optional[str]:
    """
    Generate a summary for an atomic unit using the configured LLM.

    Args:
        unit: The atomic unit to summarize
        model_name: Name of the LLM model to use

    Returns:
        Generated summary or None if an error occurred
    """
    llm = ChatOpenAI(
        model=model_name,
        temperature=TEMPERATURE,
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=DEFAULT_TIMEOUT,
    )
    prompt = generate_summary_prompt(unit)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            messages = [
                SystemMessage(
                    content="Du bist ein Experte für Mathematik und spezialisiert "
                    "auf das Erstellen suchoptimierter Zusammenfassungen "
                    "für RAG-Systeme. Deine Aufgabe ist es, mathematische "
                    "Inhalte so zusammenzufassen, dass sie optimal von "
                    "Nutzern gefunden werden können, die nach verwandten "
                    "Konzepten suchen."
                ),
                HumanMessage(content=prompt),
            ]

            response = llm.invoke(messages)
            summary = response.content.strip()
            return summary

        except Exception as e:
            logger.warning(
                f"Error generating summary for unit {unit.id} "
                f"(attempt {attempt}/{MAX_RETRIES}): {e}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)  # Exponential backoff
            else:
                logger.error(
                    f"Failed to generate summary for unit {unit.id} "
                    f"after {MAX_RETRIES} attempts"
                )
                return None


def process_summaries(
    db_manager: DatabaseManager,
    model_name: str,
    batch_size: int = 100,
    max_units: Optional[int] = None,
) -> dict:
    """
    Process all atomic units without summaries and generate summaries for them.

    Args:
        db_path: Path to the SQLite database file
        model_name: Name of the LLM model to use
        batch_size: Number of units to process in each batch
        max_units: Optional maximum number of units to process

    Returns:
        Statistics dict with counts of processed, successful, and failed units
    """
    stats = {"processed": 0, "success": 0, "failed": 0}
    units_remaining = max_units

    while True:
        # Adjust batch size for last batch if max_units specified
        current_batch_size = batch_size
        if units_remaining is not None:
            if units_remaining <= 0:
                break
            current_batch_size = min(batch_size, units_remaining)

        # Get a batch of units without summaries
        units = db_manager.get_atomic_units_without_summary(limit=current_batch_size)

        if not units:
            logger.info("No more units to process")
            break

        logger.info(f"Processing batch of {len(units)} units")

        for unit in tqdm(units, desc="Generating summaries"):
            stats["processed"] += 1

            # Generate summary
            summary = generate_summary(unit, model_name)

            if summary:
                # Update database
                success = db_manager.update_summary(unit.id, summary)

                if success:
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                    logger.warning(f"Failed to update summary for unit {unit.id}")
            else:
                stats["failed"] += 1

        # Update remaining count if needed
        if units_remaining is not None:
            units_remaining -= len(units)

        logger.info(
            f"Batch complete. Processed {len(units)} units "
            f"({stats['success']} successful, {stats['failed']} failed). "
            f"Total: {stats['processed']} processed"
        )

    logger.info(
        f"Summary generation complete. Processed {stats['processed']} units "
        f"({stats['success']} successful, {stats['failed']} failed)"
    )

    return stats


def get_progress_stats(db_manager: DatabaseManager) -> dict:
    """Get current progress statistics."""

    # Get database stats
    total_units = db_manager.count_total_units()
    units_with_summary = db_manager.count_units_with_summary()
    units_without_summary = total_units - units_with_summary

    # Calculate progress percentage
    progress_pct = 0
    if total_units > 0:
        progress_pct = (units_with_summary / total_units) * 100

    return {
        "total_units": total_units,
        "units_with_summary": units_with_summary,
        "units_without_summary": units_without_summary,
        "progress_percentage": progress_pct,
    }


def main():
    """Main function for CLI usage."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate summaries for atomic units")
    parser.add_argument(
        "--model",
        default=None,
        help=f"LLM model to use (default: {MODEL_NAME} from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)",
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        help="Maximum number of units to process",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show progress stats without processing",
    )

    args = parser.parse_args()

    db_manager = DatabaseManager(Path(args.db_path))

    if args.stats_only:
        stats = get_progress_stats(db_manager)
        print(json.dumps(stats, indent=2))
    else:
        process_summaries(
            db_manager=db_manager,
            model_name=args.model,
            batch_size=args.batch_size,
            max_units=args.max_units,
        )


if __name__ == "__main__":
    main()
