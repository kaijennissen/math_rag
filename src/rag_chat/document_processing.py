import logging
import os
from pathlib import Path

from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import MathpixPDFLoader
from typing_extensions import Iterable

from rag_chat.project_root import ROOT
from dotenv import load_dotenv
import pickle
import tiktoken

load_dotenv()


def concatenate_docs(docs: Iterable):
    concat_docs = {}
    for doc in docs:
        source = doc.metadata["source"]
        if source not in concat_docs:
            concat_docs[source] = doc
        else:
            concat_docs[source].page_content += "\n\n" + doc.page_content
    return list(concat_docs.values())


def save_documents(documents, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(documents, f)


def load_documents(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None


def load_and_process_pdfs(docs_path: Path):
    pdf_files = list(docs_path.glob("*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files")

    docs = []
    for pdf_file in pdf_files:
        loader = MathpixPDFLoader(
            pdf_file,
            processed_file_format="md",
            mathpix_api_id=os.environ.get("MATHPIX_API_ID"),
            mathpix_api_key=os.environ.get("MATHPIX_API_KEY"),
        )
        docs.extend(loader.load())

    docs = concatenate_docs(docs)
    logging.info(f"Loaded {len(docs)} documents")

    # Serialize documents to disk for caching
    # cache_path =  "docs/cached_KE_5.pkl"
    # save_documents(docs, cache_path)
    # cached_docs = load_documents(cache_path)

    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = header_splitter.split_text(docs[0].page_content)

    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    for chunk_num, chunk in enumerate(md_header_splits):
        # Print metadata and content length for each chunk
        if chunk.metadata:
            headers = ", ".join([f"{k}: {v}" for k, v in chunk.metadata.items()])
            print(f"Chunk {chunk_num + 1} " + "=" * 120)
            print(f"Metadata: {headers}")
            print(f"Content length: {len(chunk.page_content)} characters")
            # Count tokens using tiktoken
            tokens = encoding.encode(chunk.page_content)
            token_count = len(tokens)
            print(f"Token count: {token_count}")

        # Print first 50 characters of each chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200, separators=["\n\n", "\nBeweis", " ", ""]
    )
    text_splits = text_splitter.split_documents(md_header_splits)

    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    for chunk_num, chunk in enumerate(text_splits):
        # Print metadata and content length for each chunk
        if chunk.metadata:
            headers = ", ".join([f"{k}: {v}" for k, v in chunk.metadata.items()])
            print(f"Chunk {chunk_num + 1} " + "=" * 120)
            print(f"Metadata: {headers}")
            print(f"Content length: {len(chunk.page_content)} characters")
            print(f"Content: {chunk.page_content[:20]}")
            tokens = encoding.encode(chunk.page_content)
            token_count = len(tokens)
            print(f"Token count: {token_count}")

    logging.info(f"Split {len(docs)} documents into {len(text_splits)} chunks.")

    return text_splits


if __name__ == "__main__":
    docs = load_and_process_pdfs(ROOT / "docs")
    print(f"Loaded {len(docs)} documents")
