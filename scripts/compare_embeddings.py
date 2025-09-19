#!/usr/bin/env python
"""
Script to compare similarity scores across different embedding models for mathematical content.
"""

import importlib.util
import os
import time
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Same texts as in calc_embedding_similarity.py
# text1 = "Wie lautet die Definition für einen T_4-Raum?"  # Natural language question
text1 = "Was ist ein T_{4}-Raum?"  # Natural language question
text2 = "(1) $\\underline{X}$ heißt $T_{4}$-Raum, wenn zu je zwei disjunkten abgeschlossenen Mengen $A$ und $B$ in $\\underline{X}$ (offene) Umgebungen $U$ von $A$ und $V$ von $B$ mit $U \\cap V=\\emptyset$ existieren.\n\n(2) $\\underline{X}$ heißt normal, wenn $\\underline{X}$ gleichzeitig $\\mathrm{T}_{4}$-Raum und $\\mathrm{T}_{1}$-Raum ist."  # LaTeX formatting


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


def check_package_installed(package_name: str) -> bool:
    """Check if a Python package is installed."""
    return importlib.util.find_spec(package_name) is not None


def get_available_models() -> List[Tuple[str, str, str]]:
    """
    Returns a list of available embedding models.

    Returns:
        List of tuples: (model_name, library_name, model_id)
    """
    models = []

    # OpenAI models
    models.extend(
        [
            ("OpenAI Small", "openai", "text-embedding-3-small"),
            ("OpenAI Large", "openai", "text-embedding-3-large"),
        ]
    )

    # Models using sentence-transformers (installed)
    models.extend(
        [
            (
                "MXBAI German",
                "huggingface",
                "mixedbread-ai/deepset-mxbai-embed-de-large-v1",
            ),
            ("BGE-M3", "bge", "BAAI/bge-m3"),
            ("BGE German", "bge", "BAAI/bge-large-de-v1.5"),
            ("E5 Multilingual", "huggingface", "intfloat/multilingual-e5-large"),
            ("MathBERT", "huggingface", "tbs17/MathBERT"),
        ]
    )

    return models


def get_embeddings(text: str, model_info: Tuple[str, str, str]):
    """
    Generate embeddings for the text using the specified model.

    Args:
        text: Input text
        model_info: Tuple of (model_name, library_name, model_id)

    Returns:
        List: Embedding vector
    """
    model_name, library, model_id = model_info

    if library == "openai":
        from langchain_openai import OpenAIEmbeddings

        model = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"), model=model_id
        )
        return model.embed_query(text)

    elif library == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        model = HuggingFaceEmbeddings(model_name=model_id)
        return model.embed_query(text)

    elif library == "bge":
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        # Get HuggingFace API token from environment variables
        hf_api_token = os.getenv("HUGGINGFACE_API_KEY")

        if not hf_api_token:
            raise ValueError("HUGGINGFACE_API_KEY environment variable not set")

        # Configure the endpoint embeddings
        model = HuggingFaceEndpointEmbeddings(
            model=model_id,
            task="feature-extraction",
            huggingfacehub_api_token=hf_api_token,
        )
        try:
            embeddings = model.embed_query(text)
        except Exception as e:
            raise ValueError(f"Failed to generate embeddings: {e}")
        return embeddings
    else:
        raise ValueError(f"Unknown library: {library}")


def main():
    # Print input texts
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print("-" * 80)

    # Get available models
    available_models = get_available_models()

    if not available_models:
        print("No models available. Please install required packages.")
        return

    # Print header
    print(f"{'Model':<20} | {'Similarity':<10} | {'Time (s)':<10}")
    print("-" * 80)

    # Test each model
    for model_info in available_models:
        model_name, _, _ = model_info

        try:
            # Time the embedding process
            start_time = time.time()

            # Generate embeddings
            embedding1 = get_embeddings(text1, model_info)
            embedding2 = get_embeddings(text2, model_info)

            # Calculate similarity
            similarity = calculate_cosine_similarity(embedding1, embedding2)

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            # Print results
            print(f"{model_name:<20} | {similarity:.4f}     | {elapsed_time:.2f}")

        except Exception as e:
            print(f"{model_name:<20} | Error: {str(e)}")


if __name__ == "__main__":
    main()
