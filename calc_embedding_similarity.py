#!/usr/bin/env python
"""
Simple script to calculate cosine similarity between two text embeddings.
"""

import numpy as np
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# User-editable text variables - modify these to compare different texts
text1 = "Wie lautet die Definition für einen T_4-Raum?"  # Natural language question
text2 = "(1) $\\underline{X}$ heißt $T_{4}$-Raum, wenn zu je zwei disjunkten abgeschlossenen Mengen $A$ und $B$ in $\\underline{X}$ (offene) Umgebungen $U$ von $A$ und $V$ von $B$ mit $U \\cap V=\\emptyset$ existieren.\n\n(2) $\\underline{X}$ heißt normal, wenn $\\underline{X}$ gleichzeitig $\\mathrm{T}_{4}$-Raum und $\\mathrm{T}_{1}$-Raum ist."  # LaTeX formatting

# Initialize embedding model
embedding_model = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
)


def calculate_cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embedding vectors.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        float: Cosine similarity score between 0 and 1
    """
    # Calculate dot product
    dot_product = np.dot(embedding1, embedding2)

    # Calculate magnitudes
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    # Calculate cosine similarity
    similarity = dot_product / (norm1 * norm2)

    return similarity


def main():
    # Print input texts
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print("-" * 50)

    # Generate embeddings
    print("Generating embeddings...")
    embedding1 = embedding_model.embed_query(text1)
    embedding2 = embedding_model.embed_query(text2)

    # Calculate similarity
    similarity = calculate_cosine_similarity(embedding1, embedding2)

    # Print results
    print(f"Cosine Similarity: {similarity:.4f}")
    print(f"Similarity Percentage: {similarity * 100:.2f}%")


if __name__ == "__main__":
    main()
