"""
CLI for extracting reference relationships from PDF documents.

This module provides a command-line interface to extract internal references
(e.g., when Theorem 2.3.1 references Definition 1.2.4) from mathematical PDFs
and save them as tuples for use in the knowledge graph.
"""

from math_rag.config import ReferenceExtractionSettings, settings_provider
from math_rag.data_processing.extract_references import main

if __name__ == "__main__":
    settings = settings_provider.get_settings(ReferenceExtractionSettings)

    main(
        pdf_path=settings.pdf_path,
        start_page=settings.start_page,
        output_path=settings.output_path,
    )
