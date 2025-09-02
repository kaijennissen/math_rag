# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Development Guidelines for math_rag

## Environment Setup
- All commands must be run in the micromamba environment: `micromamba run -n math_rag <command>`
- IMPORTANT: Never use pip directly. Always use `python -m pip` to ensure the correct pip is used

## Build and Test Commands
- Install dependencies: `micromamba run -n math_rag python -m pip install -r requirements/dev.txt`
- Run all tests: `micromamba run -n math_rag pytest`
- Run a single test: `micromamba run -n math_rag pytest tests/test_rag_chatbot.py::test_create_rag_chatbot -v`
- Lint code: `micromamba run -n math_rag ruff format src/ tests/` and `micromamba run -n math_rag ruff check src/ tests/`

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
