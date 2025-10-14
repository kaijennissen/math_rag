"""Simple tests for helper functions."""

from helpers import format_error_message, sanitize_user_input


def test_sanitize_input():
    """Test input sanitization."""
    # Normal input
    assert sanitize_user_input("  Hello   world  ") == "Hello world"

    # Empty input
    assert sanitize_user_input("") == ""
    assert sanitize_user_input(None) == ""

    # Long input
    long_input = "a" * 3000
    result = sanitize_user_input(long_input)
    assert len(result) <= 2003
    assert result.endswith("...")


def test_error_formatting():
    """Test error message formatting."""
    error = ValueError("Test error")
    result = format_error_message(error)
    assert "Test error" in result
    assert "Sorry" in result


if __name__ == "__main__":
    test_sanitize_input()
    test_error_formatting()
    print("✅ All tests passed!")
