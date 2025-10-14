"""Basic utility functions for the Streamlit Math-RAG interface."""


def sanitize_user_input(user_input: str) -> str:
    """
    Basic input sanitization.

    Args:
        user_input: Raw user input

    Returns:
        Cleaned input string
    """
    if not user_input or not isinstance(user_input, str):
        return ""

    # Basic cleanup
    sanitized = user_input.strip()
    sanitized = " ".join(sanitized.split())  # Remove excessive whitespace

    # Prevent extremely long inputs
    if len(sanitized) > 2000:
        sanitized = sanitized[:2000] + "..."

    return sanitized


def format_error_message(error: Exception) -> str:
    """Format error messages for user display."""
    return f"Sorry, I encountered an error: {str(error)}"
