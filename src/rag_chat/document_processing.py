import glob
import logging
import os
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import MathpixPDFLoader, PyPDFLoader
from typing_extensions import Iterable


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
        # loader = PyPDFLoader(pdf_file)
        loader = MathpixPDFLoader(
            pdf_file,
            mathpix_api_id=os.environ["MATHPIX_API_ID"],
            mathpix_api_key=os.environ["MATHPIX_API_KEY"],
        )
        docs.extend(loader.load())

    docs = concatenate_docs(docs)
    logging.info(f"Loaded {len(docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs)
    logging.info(f"Split {len(docs)} documents into {len(doc_splits)} chunks.")

    return doc_splits
