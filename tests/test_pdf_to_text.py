import json
import pickle
import shutil
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from langchain.schema import Document

# Filter out PyMuPDF/SWIG deprecation warnings
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="importlib._bootstrap"
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sys")

import sys  # noqa: E402

sys.path.append(str(Path(__file__).parent.parent))
from src.pdf_to_text import (  # noqa: E402
    get_pdf_page_count,
    load_checkpoint,
    save_checkpoint,
    save_page_result,
    load_page_results,
    process_pdf_page,
    process_pdf,
    concatenate_docs,
    save_processed_document,
    main,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture
def mock_pdf_path(temp_dir):
    """Create a mock PDF path."""
    return temp_dir / "test.pdf"


@pytest.fixture
def mock_checkpoint_dir(temp_dir):
    """Create a mock checkpoint directory."""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def mock_document():
    """Create a mock Document object."""
    return [Document(page_content="Test content", metadata={"source": "test.pdf"})]


@pytest.fixture
def mock_checkpoint_data():
    """Create mock checkpoint data."""
    return {
        "filename": "test.pdf",
        "total_pages": 3,
        "processed_pages": [0, 1],
        "failed_pages": [],
        "processing_complete": False,
        "last_updated": 1714447211.0,
    }


def test_get_pdf_page_count():
    """Test getting PDF page count."""
    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_doc.page_count = 10
        mock_open.return_value = mock_doc

        result = get_pdf_page_count(Path("test.pdf"))

        assert result == 10
        mock_open.assert_called_once_with(Path("test.pdf"))


def test_get_pdf_page_count_error():
    """Test error handling when getting PDF page count fails."""
    with patch("fitz.open", side_effect=Exception("Test error")):
        result = get_pdf_page_count(Path("test.pdf"))
        assert result == 0


def test_load_checkpoint_new(mock_pdf_path, mock_checkpoint_dir):
    """Test loading a new checkpoint when none exists."""
    with patch("src.pdf_to_text.CHECKPOINT_DIR", mock_checkpoint_dir):
        with patch("src.pdf_to_text.get_pdf_page_count", return_value=5):
            result = load_checkpoint(mock_pdf_path)

            assert result["filename"] == mock_pdf_path.name
            assert result["total_pages"] == 5
            assert result["processed_pages"] == []
            assert result["failed_pages"] == []
            assert result["processing_complete"] is False
            assert "last_updated" in result


def test_load_checkpoint_existing(
    mock_pdf_path, mock_checkpoint_dir, mock_checkpoint_data
):
    """Test loading an existing checkpoint."""
    checkpoint_file = mock_checkpoint_dir / f"{mock_pdf_path.stem}_checkpoint.json"
    with open(checkpoint_file, "w") as f:
        json.dump(mock_checkpoint_data, f)

    with patch("src.pdf_to_text.CHECKPOINT_DIR", mock_checkpoint_dir):
        result = load_checkpoint(mock_pdf_path)

        assert result == mock_checkpoint_data


def test_save_checkpoint(mock_pdf_path, mock_checkpoint_dir, mock_checkpoint_data):
    """Test saving a checkpoint."""
    with patch("src.pdf_to_text.CHECKPOINT_DIR", mock_checkpoint_dir):
        save_checkpoint(mock_pdf_path, mock_checkpoint_data)

        checkpoint_file = mock_checkpoint_dir / f"{mock_pdf_path.stem}_checkpoint.json"
        assert checkpoint_file.exists()

        with open(checkpoint_file, "r") as f:
            saved_data = json.load(f)

        # Compare most fields, but allow for last_updated to be different
        assert saved_data["filename"] == mock_checkpoint_data["filename"]
        assert saved_data["total_pages"] == mock_checkpoint_data["total_pages"]
        assert saved_data["processed_pages"] == mock_checkpoint_data["processed_pages"]
        assert saved_data["failed_pages"] == mock_checkpoint_data["failed_pages"]
        assert (
            saved_data["processing_complete"]
            == mock_checkpoint_data["processing_complete"]
        )
        assert "last_updated" in saved_data


def test_save_page_result(mock_pdf_path, mock_checkpoint_dir, mock_document):
    """Test saving a page result."""
    with patch("src.pdf_to_text.CHECKPOINT_DIR", mock_checkpoint_dir):
        save_page_result(mock_pdf_path, 1, mock_document)

        result_dir = mock_checkpoint_dir / mock_pdf_path.stem
        result_file = result_dir / "page_1.pkl"

        assert result_dir.exists()
        assert result_file.exists()

        with open(result_file, "rb") as f:
            loaded_doc = pickle.load(f)

        assert loaded_doc[0].page_content == mock_document[0].page_content
        assert loaded_doc[0].metadata == mock_document[0].metadata


def test_load_page_results(mock_pdf_path, mock_checkpoint_dir, mock_document):
    """Test loading page results."""
    # Save some test data first
    result_dir = mock_checkpoint_dir / mock_pdf_path.stem
    result_dir.mkdir(exist_ok=True)

    for i in range(3):
        with open(result_dir / f"page_{i}.pkl", "wb") as f:
            pickle.dump(mock_document, f)

    with patch("src.pdf_to_text.CHECKPOINT_DIR", mock_checkpoint_dir):
        result = load_page_results(mock_pdf_path)

        assert len(result) == 3
        for doc in result:
            assert doc.page_content == mock_document[0].page_content
            assert doc.metadata == mock_document[0].metadata


def test_process_pdf_page_success(mock_pdf_path):
    """Test successful processing of a PDF page."""
    mock_result = [
        Document(page_content="Test content", metadata={"source": "test.pdf"})
    ]

    with patch("src.pdf_to_text.MathpixPDFLoader") as mock_loader:
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = mock_result
        mock_loader.return_value = mock_loader_instance

        result = process_pdf_page(mock_pdf_path, 0)

        assert result == mock_result
        mock_loader.assert_called_once()
        mock_loader_instance.load.assert_called_once()


def test_process_pdf_page_retry_then_success(mock_pdf_path):
    """Test retrying after initial failure when processing a PDF page."""
    mock_result = [
        Document(page_content="Test content", metadata={"source": "test.pdf"})
    ]

    with patch("src.pdf_to_text.MathpixPDFLoader") as mock_loader:
        mock_loader_instance = MagicMock()
        # First call raises an exception, second call succeeds
        mock_loader_instance.load.side_effect = [Exception("API error"), mock_result]
        mock_loader.return_value = mock_loader_instance

        with patch("src.pdf_to_text.time.sleep"):  # Don't actually sleep in tests
            result = process_pdf_page(mock_pdf_path, 0)

            assert result == mock_result
            assert mock_loader.call_count == 2
            assert mock_loader_instance.load.call_count == 2


def test_process_pdf_page_all_retries_fail(mock_pdf_path):
    """Test when all retries fail when processing a PDF page."""
    with patch("src.pdf_to_text.MathpixPDFLoader") as mock_loader:
        mock_loader_instance = MagicMock()
        # All calls raise an exception
        mock_loader_instance.load.side_effect = Exception("API error")
        mock_loader.return_value = mock_loader_instance

        with patch("src.pdf_to_text.time.sleep"):  # Don't actually sleep in tests
            with patch(
                "src.pdf_to_text.MAX_RETRIES", 2
            ):  # Limit to 2 retries for faster testing
                result = process_pdf_page(mock_pdf_path, 0)

                assert result is None
                assert mock_loader.call_count == 2
                assert mock_loader_instance.load.call_count == 2


def test_process_pdf_already_complete(
    mock_pdf_path, mock_checkpoint_dir, mock_document
):
    """Test processing a PDF that is already marked as complete."""
    checkpoint_data = {
        "filename": mock_pdf_path.name,
        "total_pages": 3,
        "processed_pages": [0, 1, 2],
        "failed_pages": [],
        "processing_complete": True,
        "last_updated": 1714447211.0,
    }

    # Create the checkpoint file
    checkpoint_file = mock_checkpoint_dir / f"{mock_pdf_path.stem}_checkpoint.json"
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f)

    # Create the mock PDF file so it exists
    mock_pdf_path.touch()

    # Create and save some mock results
    with patch("src.pdf_to_text.CHECKPOINT_DIR", mock_checkpoint_dir):
        with patch("src.pdf_to_text.load_page_results", return_value=mock_document):
            result = process_pdf(mock_pdf_path)

            assert result == mock_document
            # Should not attempt to process any pages


def test_process_pdf_with_new_pages(mock_pdf_path, mock_checkpoint_dir, mock_document):
    """Test processing a PDF with some pages already processed and some new pages."""
    checkpoint_data = {
        "filename": mock_pdf_path.name,
        "total_pages": 3,
        "processed_pages": [0],
        "failed_pages": [],
        "processing_complete": False,
        "last_updated": 1714447211.0,
    }

    # Create the checkpoint file
    checkpoint_file = mock_checkpoint_dir / f"{mock_pdf_path.stem}_checkpoint.json"
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f)

    # Create the mock PDF file so it exists
    mock_pdf_path.touch()

    with patch("src.pdf_to_text.CHECKPOINT_DIR", mock_checkpoint_dir):
        with patch("src.pdf_to_text.process_pdf_page", return_value=mock_document):
            with patch("src.pdf_to_text.save_page_result"):
                with patch(
                    "src.pdf_to_text.load_page_results", return_value=mock_document * 3
                ):
                    result = process_pdf(mock_pdf_path)

                    assert result == mock_document * 3

                    # Should have updated the checkpoint file with newly processed pages
                    with open(checkpoint_file, "r") as f:
                        updated_checkpoint = json.load(f)

                    assert updated_checkpoint["processed_pages"] == [0, 1, 2]
                    assert updated_checkpoint["processing_complete"] is True


def test_concatenate_docs():
    """Test concatenating documents from the same source."""
    docs = [
        Document(page_content="Content 1", metadata={"source": "doc1"}),
        Document(page_content="Content 2", metadata={"source": "doc1"}),
        Document(page_content="Content 3", metadata={"source": "doc2"}),
    ]

    result = concatenate_docs(docs)

    assert len(result) == 2

    # Find doc1 and doc2 in the results
    doc1 = next((doc for doc in result if doc.metadata["source"] == "doc1"), None)
    doc2 = next((doc for doc in result if doc.metadata["source"] == "doc2"), None)

    assert doc1 is not None
    assert doc2 is not None
    assert doc1.page_content == "Content 1\n\nContent 2"
    assert doc2.page_content == "Content 3"


def test_save_processed_document(mock_pdf_path, temp_dir, mock_document):
    """Test saving a processed document."""
    output_dir = temp_dir / "processed"

    with patch("src.pdf_to_text.DOCS_PATH", temp_dir):
        save_processed_document(mock_document, mock_pdf_path)

        assert output_dir.exists()
        output_file = output_dir / f"{mock_pdf_path.stem}.pkl"
        assert output_file.exists()

        with open(output_file, "rb") as f:
            loaded_doc = pickle.load(f)

        assert loaded_doc == mock_document


def test_main(temp_dir):
    """Test the main function."""
    pdf_file = temp_dir / "test.pdf"
    pdf_file.touch()  # Create an empty file

    with patch("src.pdf_to_text.DOCS_PATH", temp_dir):
        with patch(
            "src.pdf_to_text.process_pdf",
            return_value=[
                Document(page_content="Test", metadata={"source": "test.pdf"})
            ],
        ):
            with patch("src.pdf_to_text.save_processed_document"):
                main()
                # Success if no exception is raised
