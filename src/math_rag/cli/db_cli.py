"""
CLI for SQLite database operations for atomic units.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from tabulate import tabulate

from math_rag.core.db_models import DatabaseManager
from math_rag.data_processing.migrate_to_sqlite import migrate_all_files
from math_rag.data_processing.summarize import process_summaries, get_progress_stats
from math_rag.core.project_root import ROOT

# Hardcoded database path
SQLITE_DB_PATH = ROOT / "data" / "atomic_units.sqlite"

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_db_command(args):
    """Initialize the SQLite database."""
    db_manager = DatabaseManager(SQLITE_DB_PATH)
    logger.info(f"Database initialized at {db_manager.db_path}")


def migrate_command(args):
    """Migrate atomic units from files to database."""
    start_time = datetime.now()

    # Create database manager once
    db_manager = DatabaseManager(SQLITE_DB_PATH)

    source_dir = Path(args.source_dir) if args.source_dir else None
    total_units = migrate_all_files(
        db_manager=db_manager, source_dir=source_dir, num_workers=args.workers
    )

    duration = datetime.now() - start_time
    logger.info(f"Migration complete. {total_units} units added in {duration}")


def stats_command(args):
    """Show database statistics."""
    db_manager = DatabaseManager(SQLITE_DB_PATH)

    # Get counts
    total_units = db_manager.count_total_units()
    units_with_summary = db_manager.count_units_with_summary()
    units_without_summary = total_units - units_with_summary

    # Get all source files
    source_files = db_manager.get_all_source_files()

    # Print summary
    print("\n=== Database Statistics ===")
    print(f"Database path: {db_manager.db_path}")

    stats_table = [
        ["Total units", total_units],
        ["Units with summary", units_with_summary],
        ["Units without summary", units_without_summary],
        ["Source files", len(source_files)],
    ]

    print(tabulate(stats_table, tablefmt="plain"))

    # Print source file details if requested
    if args.verbose:
        print("\n=== Source Files ===")
        file_stats = []

        for file in source_files:
            units = db_manager.get_units_by_source_file(file)
            with_summary = sum(1 for u in units if u.summary is not None)
            file_stats.append([file, len(units), with_summary])

        print(
            tabulate(
                file_stats,
                headers=["Source File", "Units", "With Summary"],
                tablefmt="grid",
            )
        )


def summarize_command(args):
    """Generate summaries for atomic units."""
    # Create database manager once
    db_manager = DatabaseManager(SQLITE_DB_PATH)

    if args.stats_only:
        progress = get_progress_stats(db_manager)

        # Print progress
        print("\n=== Summary Generation Progress ===")
        print(f"Total units: {progress['total_units']}")
        print(f"Units with summary: {progress['units_with_summary']}")
        print(
            f"Progress: {progress['progress_percentage']:.2f}% "
            f"({progress['units_with_summary']}/{progress['total_units']})"
        )
        return

    # Run summary generation
    start_time = datetime.now()
    total_stats = process_summaries(
        db_manager=db_manager,
        model_name=args.model,
        batch_size=args.batch_size,
        max_units=args.max_units,
    )
    duration = datetime.now() - start_time

    print("\n=== Summary Generation Complete ===")
    summary_table = [
        ["Units processed", total_stats["processed"]],
        ["Successful", total_stats["success"]],
        ["Failed", total_stats["failed"]],
        ["Duration", duration],
    ]

    print(tabulate(summary_table, tablefmt="plain"))


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Atomic Unit Database Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Database init command
    init_parser = subparsers.add_parser(  # noqa: F841
        "init", help="Initialize the SQLite database"
    )

    # Migrate command
    migrate_parser = subparsers.add_parser(
        "migrate", help="Migrate atomic units from files to database"
    )
    migrate_parser.add_argument(
        "--source-dir",
        help="Directory containing atomic unit files (default: docs/atomic_units)",
    )
    migrate_parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker threads (default: 4)"
    )

    # Database stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed statistics"
    )

    # Summarize command
    summarize_parser = subparsers.add_parser(
        "summarize", help="Generate summaries for atomic units"
    )
    summarize_parser.add_argument(
        "--model", default="gpt-4-turbo", help="LLM model to use (default: gpt-4-turbo)"
    )
    summarize_parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing (default: 10)",
    )
    summarize_parser.add_argument("--checkpoint", help="Path to checkpoint file")
    summarize_parser.add_argument(
        "--max-units", type=int, help="Maximum number of units to process"
    )
    summarize_parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show progress stats without processing",
    )

    # Parse args
    args = parser.parse_args()

    # Execute command
    if args.command == "init":
        init_db_command(args)
    elif args.command == "migrate":
        migrate_command(args)
    elif args.command == "stats":
        stats_command(args)
    elif args.command == "summarize":
        summarize_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
