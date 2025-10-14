#!/usr/bin/env python3
"""
Visualization script for evaluation results.

This script creates a simple plotly visualization of evaluation results,
showing scores across different dimensions for each question.

Usage:
    python visualize_results.py input_file output_file

Example:
    python visualize_results.py results.json evaluation_viz.html
"""

import argparse
import json
import random
from typing import Any, Dict, List

import plotly.graph_objects as go


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load evaluation results from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_visualization(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Create a visualization of evaluation results.

    Args:
        results: List of evaluation result dictionaries
        output_path: Path to save the HTML output file
    """
    # Extract the dimensions we want to visualize
    dimensions = [
        "mathematical_correctness_score",
        "context_relevance_score",
        "context_utilization_score",
        "completeness_score",
    ]

    # More readable labels for dimensions
    dimension_labels = {
        "mathematical_correctness_score": "Mathematical Correctness",
        "context_relevance_score": "Context Relevance",
        "context_utilization_score": "Context Utilization",
        "completeness_score": "Completeness",
    }

    # Create figure
    fig = go.Figure()

    # Set random seed for reproducible jitter
    random.seed(42)

    # Calculate jitter amount based on score range
    all_scores = [
        result.get(dim, 0)
        for result in results
        for dim in dimensions
        if result.get(dim) is not None
    ]
    score_range = max(all_scores) - min(all_scores) if all_scores else 5
    jitter_amount = score_range * 0.02  # 2% of the score range

    # Add a trace for each question
    for i, result in enumerate(results):
        # Skip entries that don't have scores (e.g., if predicted_answer is null)
        if any(result.get(dim) is None for dim in dimensions):
            continue

        # Extract scores and add jitter
        base_scores = [result.get(dim, 0) for dim in dimensions]
        # Add small random offset to make overlapping points distinguishable
        scores = [
            score + random.uniform(-jitter_amount, jitter_amount)
            for score in base_scores
        ]

        # Create a short label for the question (truncate if too long)
        question = result.get("question", f"Question {i + 1}")
        if len(question) > 50:
            question = question[:47] + "..."

        # Add trace for this question
        fig.add_trace(
            go.Scatter(
                x=list(dimension_labels.values()),
                y=scores,
                mode="lines+markers",
                name=question,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Score: %{customdata}<br>"
                    f"Question: {question}<br>"
                    f"Overall Pass: {result.get('overall_pass', 'N/A')}"
                ),
                customdata=base_scores,  # Show original scores in hover
            )
        )

    # Update layout
    fig.update_layout(
        title="Evaluation Results by Dimension",
        xaxis_title="Dimension",
        yaxis_title="Score",
        yaxis=dict(
            range=[
                0,
                max(
                    5,
                    max(
                        result.get(dim, 0)
                        for result in results
                        for dim in dimensions
                        if result.get(dim) is not None
                    )
                    + 0.5,
                ),
            ]
        ),
        legend_title="Questions",
        hovermode="closest",
    )

    # Save figure
    fig.write_html(output_path)
    print(f"Visualization saved to {output_path}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Visualize evaluation results from JSON file."
    )
    parser.add_argument(
        "input_file", help="Path to the JSON file containing evaluation results"
    )
    parser.add_argument("output_file", help="Path to save the HTML visualization")

    args = parser.parse_args()

    try:
        results = load_results(args.input_file)
        create_visualization(results, args.output_file)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
