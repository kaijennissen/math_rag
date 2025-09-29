"""
Simple CLI to view summaries alongside original content for quick comparison.
"""

import argparse

from math_rag.config import MathRagBaseSettings, settings_provider
from math_rag.core.db_models import DatabaseManager


def show_summary_comparison(args):
    """Show original content and summary side by side."""
    settings = settings_provider.get_settings(MathRagBaseSettings)
    db_manager = DatabaseManager(settings.db_path)

    with db_manager.get_session() as session:
        from sqlmodel import select

        from math_rag.core.db_models import AtomicItem

        statement = select(AtomicItem).where(AtomicItem.summary.is_not(None))
        if args.limit:
            statement = statement.limit(args.limit)

        units = list(session.exec(statement))

    for unit in units:
        print(f"\n{'=' * 60}")
        print(
            f"Unit {unit.id} | {unit.type} | Section {unit.section}.{unit.subsection}"
        )
        if unit.identifier:
            print(f"ID: {unit.identifier}")
        print(f"{'=' * 60}")

        print("ORIGINAL:")
        print(unit.text)
        if unit.proof:
            print(f"\nProof: {unit.proof}")

        print(f"\n{'-' * 30}")
        print("SUMMARY:")
        print(unit.summary)
        print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="View summaries with original content")
    parser.add_argument(
        "--limit", type=int, default=5, help="Number of units to show (default: 5)"
    )

    args = parser.parse_args()
    show_summary_comparison(args)


if __name__ == "__main__":
    main()
