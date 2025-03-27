import logging
import os
from pathlib import Path

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    LatexTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.document_loaders import MathpixPDFLoader
from langchain_core.documents import Document
from typing_extensions import Iterable
from rag_chat.project_root import ROOT


def concatenate_docs(docs: Iterable):
    concat_docs = {}
    for doc in docs:
        source = doc.metadata["source"]
        if source not in concat_docs:
            concat_docs[source] = doc
        else:
            concat_docs[source].page_content += "\n\n" + doc.page_content
    return list(concat_docs.values())


def load_and_process_pdfs(docs_path: Path):
    pdf_files = list(docs_path.glob("*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files")

    docs = []
    for pdf_file in pdf_files:
        loader = MathpixPDFLoader(
            pdf_file,
            processed_file_format="md",
            mathpix_api_id=os.getenv("MATHPIX_API_ID"),
            mathpix_api_key=os.getenv("MATHPIX_API_KEY"),
        )
        docs.extend(loader.load())

    docs = concatenate_docs(docs)
    logging.info(f"Loaded {len(docs)} documents")

    text_splitter = LatexTextSplitter()
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = header_splitter.split_text(docs[0].page_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    text_splits = text_splitter.split_documents(md_header_splits)

    logging.info(f"Split {len(docs)} documents into {len(text_splits)} chunks.")

    return text_splits


if __name__ == "__main__":
    docs = load_and_process_pdfs(ROOT / "docs")
    print(f"Loaded {len(docs)} documents")
