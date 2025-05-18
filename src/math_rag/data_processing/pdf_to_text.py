import argparse
import logging
import os
import time
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import pickle

from langchain_community.document_loaders import MathpixPDFLoader
from langchain.schema import Document
from dotenv import load_dotenv
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pdf_processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

load_dotenv()

DOCS_PATH = Path("docs/")
CHECKPOINT_DIR = DOCS_PATH / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Maximum retry attempts
MAX_RETRIES = 3
# Initial backoff time in seconds
INITIAL_BACKOFF = 2


def get_pdf_page_count(pdf_path: Path) -> int:
    """Get the number of pages in a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        doc.close()
        return page_count
    except Exception as e:
        logger.error(f"Error getting page count for {pdf_path}: {e}")
        return 0


def load_checkpoint(pdf_path: Path) -> Dict:
    """Load processing checkpoint for a PDF file."""
    checkpoint_file = CHECKPOINT_DIR / f"{pdf_path.stem}_checkpoint.json"
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")

    # Initialize a new checkpoint
    return {
        "filename": pdf_path.name,
        "total_pages": get_pdf_page_count(pdf_path),
        "processed_pages": [],
        "failed_pages": [],
        "processing_complete": False,
        "last_updated": time.time(),
    }


def save_checkpoint(pdf_path: Path, checkpoint_data: Dict) -> None:
    """Save processing checkpoint for a PDF file."""
    checkpoint_file = CHECKPOINT_DIR / f"{pdf_path.stem}_checkpoint.json"
    checkpoint_data["last_updated"] = time.time()

    try:
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")


def save_page_result(pdf_path: Path, page_num: int, content: Document) -> None:
    """Save the processed content of a page."""
    result_dir = CHECKPOINT_DIR / pdf_path.stem
    result_dir.mkdir(exist_ok=True)

    try:
        with open(result_dir / f"page_{page_num}.pkl", "wb") as f:
            pickle.dump(content, f)
    except Exception as e:
        logger.error(f"Error saving page result: {e}")


def load_page_results(pdf_path: Path) -> List[Document]:
    """Load all processed page results for a PDF."""
    result_dir = CHECKPOINT_DIR / pdf_path.stem
    if not result_dir.exists():
        return []

    documents = []
    try:
        for page_file in sorted(
            result_dir.glob("page_*.pkl"), key=lambda x: int(x.stem.split("_")[1])
        ):
            with open(page_file, "rb") as f:
                page_doc = pickle.load(f)
                documents.extend(page_doc)
    except Exception as e:
        logger.error(f"Error loading page results: {e}")

    return documents


def process_pdf_page(pdf_path: Path, page_num: int) -> Optional[List[Document]]:
    """Process a single page of a PDF with MathPix."""
    retries = 0
    backoff = INITIAL_BACKOFF

    # Check if the file exists for better error messages in tests
    if not pdf_path.exists():
        logger.error(f"Attempt failed for page {page_num}: no such file: '{pdf_path}'")
        return None

    while retries < MAX_RETRIES:
        try:
            logger.info(f"Processing {pdf_path.name} - page {page_num}")
            # Create a temporary PDF with only the target page
            temp_dir = Path(tempfile.mkdtemp())
            temp_pdf_path = temp_dir / f"page_{page_num}.pdf"

            try:
                # Extract the single page using PyMuPDF
                doc = fitz.open(pdf_path)
                new_doc = fitz.open()
                new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                new_doc.save(temp_pdf_path)
                new_doc.close()
                doc.close()

                loader = MathpixPDFLoader(
                    str(temp_pdf_path),
                    processed_file_format="md",
                    mathpix_api_id=os.environ.get("MATHPIX_API_ID"),
                    mathpix_api_key=os.environ.get("MATHPIX_API_KEY"),
                )
                result = loader.load()

                # Clean image URLs from the page content
                for doc in result:
                    doc.page_content = remove_image_urls(doc.page_content)

                logger.info(f"Successfully processed page {page_num}")
                return result
            finally:
                # Clean up temporary files
                try:
                    shutil.rmtree(temp_dir)
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to clean up temporary files: {cleanup_error}"
                    )
        except Exception as e:
            retries += 1
            logger.warning(f"Attempt {retries} failed for page {page_num}: {e}")

            if retries < MAX_RETRIES:
                logger.info(f"Retrying in {backoff} seconds...")
                time.sleep(backoff)
                # Exponential backoff
                backoff *= 2
            else:
                logger.error(
                    f"Failed to process page {page_num} after {MAX_RETRIES} attempts"
                )
                return None


def process_pdf(pdf_path: Path) -> List[Document]:
    """Process a PDF file page by page with checkpoints and error handling."""
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return []

    # Load or create checkpoint
    checkpoint = load_checkpoint(pdf_path)
    logger.info(f"Processing {pdf_path.name} - {checkpoint['total_pages']} pages total")

    # If already completed, just load the results
    if checkpoint["processing_complete"]:
        logger.info(f"PDF {pdf_path.name} already fully processed, loading results")
        return load_page_results(pdf_path)

    # Process each page that hasn't been processed yet
    for page_num in range(checkpoint["total_pages"]):
        if page_num in checkpoint["processed_pages"]:
            logger.info(f"Page {page_num} already processed, skipping")
            continue

        if page_num in checkpoint["failed_pages"]:
            logger.info(f"Page {page_num} previously failed, skipping")
            continue

        page_result = process_pdf_page(pdf_path, page_num)

        if page_result:
            save_page_result(pdf_path, page_num, page_result)
            checkpoint["processed_pages"].append(page_num)
            save_checkpoint(pdf_path, checkpoint)
        else:
            checkpoint["failed_pages"].append(page_num)
            save_checkpoint(pdf_path, checkpoint)

    # Check if all pages were processed successfully
    if len(checkpoint["processed_pages"]) == checkpoint["total_pages"]:
        logger.info(f"All pages in {pdf_path.name} processed successfully")
        checkpoint["processing_complete"] = True
        save_checkpoint(pdf_path, checkpoint)
    else:
        logger.warning(
            f"Processing of {pdf_path.name} incomplete. "
            f"Processed {len(checkpoint['processed_pages'])} of {checkpoint['total_pages']} pages. "
            f"Failed pages: {checkpoint['failed_pages']}"
        )

    # Return all successfully processed documents
    return load_page_results(pdf_path)


def concatenate_docs(docs: List[Document]) -> List[Document]:
    """Concatenate documents from the same source."""
    concat_docs = {}
    for doc in docs:
        source = doc.metadata["source"]
        if source not in concat_docs:
            concat_docs[source] = doc
        else:
            concat_docs[source].page_content += "\n\n" + doc.page_content
    return list(concat_docs.values())


def remove_image_urls(text: str) -> str:
    """Remove image URLs from markdown text.

    Specifically targets image references from MathPix that look like:
    ![](https://cdn.mathpix.com/cropped/...)
    """
    import re

    # Pattern to match markdown image syntax: ![alt text](url)
    # Focusing on mathpix.com URLs but general enough to catch all images
    pattern = r"!\[\]\(https?:\/\/cdn\.mathpix\.com\/[^)]+\)"
    return re.sub(pattern, "", text)


def save_processed_document(docs: List[Document], output_file: str) -> List[Document]:
    """Save the final processed document as a single text file and return the documents."""
    output_dir = DOCS_PATH / "processed"
    output_dir.mkdir(exist_ok=True)

    # Save as a single text file
    text_file = output_dir / f"{output_file}.txt"
    try:
        with open(text_file, "w", encoding="utf-8") as f:
            # Combine all document contents into a single text file
            full_text = ""
            for i, doc in enumerate(docs):
                # Clean the content by removing image URLs
                cleaned_content = remove_image_urls(doc.page_content)
                full_text += cleaned_content
            f.write(full_text)
        logger.info(f"Saved processed document as text file to {text_file}")

        # Also save as pickle for the test
        pkl_file = output_dir / f"{output_file}.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump(docs, f)

        return docs
    except Exception as e:
        logger.error(f"Error saving processed document as text: {e}")
        return docs


def process_single_pdf(pdf_path):
    """Process a single PDF file and save the results."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return False

    try:
        logger.info(f"Processing PDF file: {pdf_path}")
        docs = process_pdf(pdf_path)

        if docs:
            # Concatenate pages from the same source
            docs = concatenate_docs(docs)
            logger.info(
                f"Successfully processed {pdf_path.name}: {len(docs)} documents"
            )

            # Save the final processed document
            save_processed_document(docs, pdf_path.stem)
            return True
        else:
            logger.warning(f"No documents extracted from {pdf_path.name}")
            return False

    except Exception as e:
        logger.error(f"Error processing {pdf_path.name}: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a PDF file with MathPix API, page by page"
    )
    parser.add_argument("--input-file", help="Path to the PDF file to process")

    args = parser.parse_args()

    process_single_pdf(args.input_file)
