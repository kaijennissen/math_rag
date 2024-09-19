import glob
import logging
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def load_and_process_pdfs(folder_path):
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files")

    docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        docs.extend(loader.load())

    # df_docs = pd.DataFrame(
    #    [
    #        {
    #            "page_content": doc.page_content,
    #            "source": doc.metadata["source"],
    #            "metadata": doc.metadata,
    #        }
    #        for doc in docs
    #    ]
    # )

    # Concatenate pages belonging to the same document
    concat_docs = []
    current_doc = None
    for doc in docs:
        if current_doc is None:
            current_doc = doc
        elif current_doc.metadata["source"] == doc.metadata["source"]:
            current_doc.page_content += "\n\n" + doc.page_content
        else:
            concat_docs.append(current_doc)
            current_doc = doc

    concat_docs.append(current_doc)
    docs = concat_docs
    logging.info(f"Loaded {len(docs)} documents")

    return docs


def setup_document_processing(docs):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs)
    logging.info(f"Split {len(docs)} documents into {len(doc_splits)} chunks")

    return doc_splits
