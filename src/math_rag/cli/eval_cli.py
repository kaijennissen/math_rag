"""
Command-line interface for evaluation.

Provides CLI entrypoint for running baseline evaluation over the RAG system
using Pydantic models for type safety and structured data handling.
"""

import logging
import os
from pathlib import Path

from math_rag.config.evaluation_settings import get_config
from math_rag.evaluation import (
    calculate_evaluation_summary,
    load_dataset,
    log_summary_results,
    run_evaluation,
    setup_rag_system,
    write_results,
    write_summary,
)

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(
    neo4j_config,
    openai_config,
    difficulty: list[int],
    dataset_path: str,
    output_path: str,
    sample: int,
    log_level: str,
):
    """Main CLI entry point using Pydantic models."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = load_dataset(dataset_path)
        if not dataset:
            logger.error("No valid questions loaded from dataset")
            return 1

        # Apply filters (I/O logic at CLI level)
        if difficulty and len(difficulty) < 4:
            dataset = [q for q in dataset if q["difficulty"] in difficulty]
            logger.info(f"Filtered to difficulty levels: {difficulty}")

        if sample:
            import random

            dataset = random.sample(dataset, sample)
            logger.info(f"Sampled {sample} questions")

        # Setup RAG system (I/O logic at CLI level)
        rag_system = setup_rag_system(neo4j_config)

        # Run evaluation (pure logic, no I/O)
        logger.info("Starting evaluation...")
        evaluation_results = run_evaluation(
            rag_system=rag_system, dataset=dataset, openai_config=openai_config
        )

        # Calculate summary using new Pydantic-based functions
        logger.info("Calculating evaluation summary...")
        summary = calculate_evaluation_summary(evaluation_results)

        # Write output (I/O at CLI level)
        output_path_obj = Path(output_path)
        write_results(evaluation_results, output_path_obj)

        # Write summary to separate file
        summary_path = output_path_obj.with_suffix(".summary.json")
        write_summary(summary, summary_path)

        # Log summary statistics using new structured logging
        if evaluation_results:
            log_summary_results(summary, logger)

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    settings = get_config()
    main(
        neo4j_config=settings.neo4j_config,
        openai_config=settings.openai_config,
        difficulty=settings.difficulty_levels,
        dataset_path=settings.dataset_path,
        output_path=settings.output_path,
        sample=settings.sample_size,
        log_level=settings.log_level,
    )
