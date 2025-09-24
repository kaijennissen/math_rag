# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Development Guidelines for math_rag

## Environment Setup
- Use `uv` for Python environment and dependency management
- IMPORTANT: Use `uv` commands for all package operations to ensure fast and reliable dependency resolution

## Build and Test Commands
- Sync all dependencies: `uv sync --all-extras`
- Add new dependencies: `uv add <package-name>`
- Add development dependencies: `uv add --dev <package-name>`
- Run all tests: `pytest`
- Run a single test: `pytest tests/test_rag_chatbot.py::test_create_rag_chatbot -v`
- Lint code: `ruff format src/ tests/` and `ruff check src/ tests/`

## Code Style
- Formatting: Use ruff with 88 character line length
- Imports: Use ruff to sort imports, arranged by standard library → third-party → local
- Type annotations: Use proper typing (List, Dict, etc. from typing module)
- Naming: snake_case for variables/functions, PascalCase for classes
- Error handling: Use try/except blocks with specific exceptions
- Docstrings: Document functions, classes, and modules
- Logging: Use the logging module instead of print statements
- Command-line argument parsing: Define ArgumentParser and parse arguments inside `if __name__ == '__main__'` but outside of the main function. Then pass the parsed arguments to the main function as parameters. This keeps the main function reusable and testable.

## Project Structure
- Code in `src/math_rag/`
- Tests in `tests/`
- Configuration in `config/`
- Documentation in `docs/`
