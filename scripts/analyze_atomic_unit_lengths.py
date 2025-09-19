#!/usr/bin/env python
"""
Analyze character lengths of atomic units to identify potential issues with vector
embeddings.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.math_rag.atomic_unit import AtomicUnit

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DOCS_PATH = Path("docs")
ATOMIC_UNITS_PATH = DOCS_PATH / "atomic_units"
OUTPUT_PATH = Path("analysis_results")


def load_atomic_units() -> List[AtomicUnit]:
    """
    Load all atomic units from JSON files in the atomic_units directory.

    Returns:
        List[AtomicUnit]: List of AtomicUnit objects
    """
    atomic_units = []
    json_files = list(ATOMIC_UNITS_PATH.glob("*.json"))

    logger.info(f"Found {len(json_files)} JSON files to process")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Process each chunk in the JSON file
            for chunk_data in data.get("chunks", []):
                atomic_unit = AtomicUnit.from_dict(chunk_data)
                atomic_units.append(atomic_unit)

        except Exception as e:
            logger.error(f"Error loading {json_file}: {str(e)}")

    logger.info(f"Loaded {len(atomic_units)} atomic units in total")
    return atomic_units


def calculate_character_lengths(atomic_units: List[AtomicUnit]) -> pd.DataFrame:
    """
    Calculate character lengths for each atomic unit.

    Args:
        atomic_units: List of AtomicUnit objects

    Returns:
        pd.DataFrame: DataFrame with character length information
    """
    data = []

    for unit in atomic_units:
        text_length = len(unit.text) if unit.text else 0
        proof_length = len(unit.proof) if unit.proof else 0
        total_length = text_length + proof_length

        data.append(
            {
                "section": unit.section,
                "subsection": unit.subsection,
                "subsubsection": unit.subsubsection,
                "identifier": unit.identifier,
                "type": unit.type,
                "text_length": text_length,
                "proof_length": proof_length,
                "total_length": total_length,
                "has_proof": proof_length > 0,
            }
        )

    return pd.DataFrame(data)


def calculate_statistics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calculate statistics for the entire dataset and by type.

    Args:
        df: DataFrame with character length information

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with statistics
    """
    # Overall statistics
    overall_stats = df[["text_length", "proof_length", "total_length"]].describe()

    # Statistics by type
    type_stats = df.groupby("type").agg(
        {
            "text_length": [
                "count",
                "mean",
                "std",
                "min",
                lambda x: x.quantile(0.25),
                lambda x: x.quantile(0.5),
                lambda x: x.quantile(0.75),
                "max",
            ],
            "proof_length": ["mean", "max"],
            "total_length": [
                "mean",
                "std",
                "min",
                lambda x: x.quantile(0.25),
                lambda x: x.quantile(0.5),
                lambda x: x.quantile(0.75),
                "max",
            ],
            "has_proof": "sum",
        }
    )

    # Rename the lambda functions to more readable names
    type_stats = type_stats.rename(
        columns={"<lambda_0>": "25%", "<lambda_1>": "50%", "<lambda_2>": "75%"}
    )

    # Sort by mean total length, descending
    type_stats = type_stats.sort_values(by=("total_length", "mean"), ascending=False)

    return {"overall": overall_stats, "by_type": type_stats}


def create_visualizations(
    df: pd.DataFrame, stats: Dict[str, pd.DataFrame]
) -> Dict[str, go.Figure]:
    """
    Create visualizations for the analysis.

    Args:
        df: DataFrame with character length information
        stats: Dictionary with statistics

    Returns:
        Dict[str, go.Figure]: Dictionary with plotly figures
    """
    figures = {}

    # 1. Overall distribution - Boxplot
    fig_boxplot = px.box(
        df,
        y="total_length",
        x="type",
        title="Distribution of Character Lengths by Type",
        labels={
            "total_length": "Total Characters (text + proof)",
            "type": "Mathematical Type",
        },
        color="type",
        points="all",
    )
    fig_boxplot.update_layout(
        showlegend=False, xaxis={"categoryorder": "total ascending"}
    )
    figures["boxplot"] = fig_boxplot

    # 2. Histogram of total lengths
    fig_hist = px.histogram(
        df,
        x="total_length",
        nbins=50,
        title="Histogram of Total Character Lengths",
        labels={
            "total_length": "Total Characters (text + proof)",
            "count": "Number of Units",
        },
        color="type",
    )
    fig_hist.update_layout(bargap=0.1)
    figures["histogram"] = fig_hist

    # 3. Text vs Proof length scatter plot
    fig_scatter = px.scatter(
        df,
        x="text_length",
        y="proof_length",
        title="Text Length vs Proof Length",
        labels={"text_length": "Text Characters", "proof_length": "Proof Characters"},
        color="type",
        hover_data=["identifier", "total_length"],
        opacity=0.7,
    )
    # Add diagonal reference line (x=y)
    max_val = max(df["text_length"].max(), df["proof_length"].max())  # noqa:F841
    figures["scatter"] = fig_scatter

    # 4. Summary statistics bar chart
    type_means = stats["by_type"][("total_length", "mean")].reset_index()
    type_means.columns = ["type", "mean_total_length"]

    fig_bar = px.bar(
        type_means,
        x="type",
        y="mean_total_length",
        title="Average Character Length by Type",
        labels={
            "mean_total_length": "Average Total Characters",
            "type": "Mathematical Type",
        },
        color="type",
    )
    fig_bar.update_layout(showlegend=False, xaxis={"categoryorder": "total descending"})
    figures["bar_chart"] = fig_bar

    return figures


def save_results(
    df: pd.DataFrame, stats: Dict[str, pd.DataFrame], figures: Dict[str, go.Figure]
):
    """
    Save analysis results to files.

    Args:
        df: DataFrame with character length information
        stats: Dictionary with statistics
        figures: Dictionary with plotly figures
    """
    # Create output directory if it doesn't exist
    OUTPUT_PATH.mkdir(exist_ok=True)

    # Save raw data
    df.to_csv(OUTPUT_PATH / "atomic_unit_lengths.csv", index=False)

    # Save statistics
    with open(OUTPUT_PATH / "overall_statistics.txt", "w") as f:
        f.write("Overall Statistics\n")
        f.write("=================\n\n")
        f.write(stats["overall"].to_string())

    # Save type statistics to CSV
    stats["by_type"].to_csv(OUTPUT_PATH / "statistics_by_type.csv")

    # Save figures
    for name, fig in figures.items():
        fig.write_html(OUTPUT_PATH / f"{name}.html")
        fig.write_image(OUTPUT_PATH / f"{name}.png")

    logger.info(f"Results saved to {OUTPUT_PATH}")


def generate_summary_report(stats: Dict[str, pd.DataFrame], df: pd.DataFrame) -> str:
    """
    Generate a summary report of the analysis.

    Args:
        stats: Dictionary with statistics
        df: DataFrame with character length information

    Returns:
        str: Summary report text
    """
    overall = stats["overall"]
    by_type = stats["by_type"]

    # Calculate percentiles for thresholds
    percentile_90 = np.percentile(df["total_length"], 90)
    percentile_95 = np.percentile(df["total_length"], 95)

    # Count units exceeding typical embedding limits
    exceed_1k = (df["total_length"] > 1000).sum()
    exceed_2k = (df["total_length"] > 2000).sum()
    exceed_4k = (df["total_length"] > 4000).sum()

    # Generate report
    report = f"""# Atomic Unit Character Length Analysis Report

## Summary Statistics

- Total Units Analyzed: {len(df)}
- Average Total Length: {overall.loc["mean", "total_length"]:.1f} characters
- Median Total Length: {df["total_length"].median():.1f} characters
- Minimum Length: {overall.loc["min", "total_length"]:.0f} characters
- Maximum Length: {overall.loc["max", "total_length"]:.0f} characters
- Standard Deviation: {overall.loc["std", "total_length"]:.1f} characters

## Distribution Analysis

- 25th Percentile: {df["total_length"].quantile(0.25):.1f} characters
- 75th Percentile: {df["total_length"].quantile(0.75):.1f} characters
- 90th Percentile: {percentile_90:.1f} characters
- 95th Percentile: {percentile_95:.1f} characters

## Embedding Threshold Analysis

- Units exceeding 1,000 characters: {exceed_1k} ({exceed_1k / len(df) * 100:.1f}%)
- Units exceeding 2,000 characters: {exceed_2k} ({exceed_2k / len(df) * 100:.1f}%)
- Units exceeding 4,000 characters: {exceed_4k} ({exceed_4k / len(df) * 100:.1f}%)

## Types with Longest Average Length

"""

    # Add top 5 types by average length
    top_types = by_type.head(5)
    for type_name, row in zip(top_types.index, top_types[("total_length", "mean")]):
        report += f"- {type_name}: {row:.1f} characters on average\n"

    report += "\n## Recommendations\n\n"

    # Add recommendations based on findings
    if exceed_4k > 0:
        report += "- Consider chunking units exceeding 4,000 characters, "
        report += "as they exceed most embedding models' limits.\n"
    if exceed_2k / len(df) > 0.1:  # If more than 10% exceed 2000 chars
        report += "- Implement a chunking strategy for units above 2,000 characters.\n"

    report += "- Use hierarchical embedding approaches for very long units rather than "
    report += "embedding entire text at once.\n"
    report += "- Consider text summarization for particularly long proofs "
    report += "while preserving key mathematical notation.\n"

    return report


def main():
    """Main function to run the analysis"""
    logger.info("Starting analysis of atomic unit character lengths")

    # Load atomic units
    atomic_units = load_atomic_units()

    # Calculate character lengths
    df = calculate_character_lengths(atomic_units)

    # Calculate statistics
    stats = calculate_statistics(df)

    # Create visualizations
    figures = create_visualizations(df, stats)

    # Generate summary report
    report = generate_summary_report(stats, df)

    # Save results
    save_results(df, stats, figures)

    # Save report
    with open(OUTPUT_PATH / "summary_report.md", "w") as f:
        f.write(report)

    logger.info(
        f"Analysis complete. Report saved to {OUTPUT_PATH / 'summary_report.md'}"
    )


if __name__ == "__main__":
    main()
