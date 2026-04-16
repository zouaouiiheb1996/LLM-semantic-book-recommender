import pytest
from unittest.mock import MagicMock, patch

from src.llm.explanation import generate_explanation


def test_generate_explanation_returns_string():
    mock_response = MagicMock()
    mock_response.content = "This book matches your interest in adventure."

    with patch("src.llm.explanation.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        result = generate_explanation("adventure story", "A tale of epic journeys.")

    assert isinstance(result, str)
    assert result == "This book matches your interest in adventure."


def test_generate_explanation_includes_query_in_prompt():
    mock_response = MagicMock()
    mock_response.content = "Great match."

    with patch("src.llm.explanation.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        generate_explanation("mystery thriller", "A whodunit set in London.")
        call_args = mock_llm.invoke.call_args[0][0]

    assert "mystery thriller" in call_args
