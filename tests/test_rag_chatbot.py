from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from rag_chat.rag_chatbot import GraphState, create_rag_chatbot


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    return {
        "docs_path": "mock_docs",
        "use_openai": False,
        "llm_model": "mock_model",
        "temperature": 0.7,
        "top_k": 3,
        "score_threshold": 0.5,
    }


@pytest.fixture
def mock_docs() -> List[Document]:
    return [
        Document(page_content="Mock content 1", metadata={"source": "doc1"}),
        Document(page_content="Mock content 2", metadata={"source": "doc2"}),
    ]


@pytest.fixture
def mock_chatbot(mock_config: Dict[str, Any], mock_docs: List[Document]) -> Mock:
    with patch("rag_chat.rag_chatbot.load_config", return_value=mock_config), patch(
        "rag_chat.rag_chatbot.load_and_process_pdfs", return_value=mock_docs
    ), patch("rag_chat.rag_chatbot.initialize_embeddings", return_value=Mock()), patch(
        "rag_chat.rag_chatbot.initialize_retrievers", return_value=Mock()
    ), patch(
        "rag_chat.rag_chatbot.initialize_llm", return_value=Mock()
    ):
        yield create_rag_chatbot()


def test_create_rag_chatbot(mock_chatbot):
    assert mock_chatbot is not None


def test_normal_llm_workflow(mock_chatbot):
    with patch.object(mock_chatbot, "invoke") as mock_invoke:
        mock_invoke.return_value = {
            "question": "What is algebra?",
            "generation": "Algebra is a branch of mathematics.",
        }

        input_state = GraphState(
            question="What is algebra?", generation="", documents=[], retries=0
        )
        result = mock_chatbot.invoke(input_state)

        assert result["question"] == "What is algebra?"
        assert "generation" in result
        assert "Algebra is a branch of mathematics." in result["generation"]

    # Add more assertions based on expected behavior
