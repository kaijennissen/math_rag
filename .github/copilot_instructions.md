# GitHub Copilot Instructions

This file provides general guidance for GitHub Copilot when working with this repository.

## Development Environment

- **Package Management**: Use `uv` for Python environment and dependency management
- **Python Version**: Ensure compatibility with Python 3.12+
- **Virtual Environment**: Use `uv` to manage virtual environments automatically

## Code Quality & Standards

### Linting and Formatting
- **Code Formatter**: Use `ruff format` for code formatting (88 character line length)
- **Linter**: Use `ruff check` for linting and code quality checks
- **Import Sorting**: Use ruff for import organization (standard library → third-party → local)
- **Type Annotations**: Include proper type hints using typing module

### Pre-commit Hooks
- **Always use pre-commit**: Run `pre-commit run --all-files` before committing
- **Install pre-commit**: Use `pre-commit install` to set up automatic checks
- **Pre-commit includes**:
  - Syntax validation (AST checks)
  - YAML syntax validation
  - Trailing whitespace removal
  - End-of-file newline fixes
  - Large file detection
  - Secret detection
  - Ruff linting and formatting

## Testing

### Test Framework
- **Test Runner**: Use `pytest` for running tests
- **Test Discovery**: Tests are located in `tests/` directory
- **Test Files**: Follow `test_*.py` naming convention
- **Run Tests**: Use `pytest` or `uv run pytest` to execute tests

### Testing Best Practices
- Write tests for new functionality
- Maintain existing test coverage
- Use descriptive test names and docstrings
- Follow existing test patterns in the codebase

## Build and Development Commands

```bash
# Install dependencies
uv sync --extra dev

# Run tests
pytest
# or
uv run pytest

# Format code
ruff format src/ tests/

# Lint code
ruff check src/ tests/

# Run all pre-commit checks
pre-commit run --all-files
```

## Code Style Guidelines

- **Naming Conventions**:
  - Variables and functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`
- **Error Handling**: Use specific exception types in try/except blocks
- **Documentation**: Include docstrings for functions, classes, and modules
- **Logging**: Use the `logging` module instead of print statements
- **Command-line Arguments**: Use `ArgumentParser` and pass parsed arguments to main functions

## Dependencies

- **Adding Dependencies**: Use `uv add <package-name>` for runtime dependencies
- **Development Dependencies**: Use `uv add --dev <package-name>` for dev dependencies
- **Lock File**: Keep `uv.lock` updated when adding/removing dependencies

## General Guidelines

- Keep changes minimal and focused
- Follow existing code patterns and conventions
- Ensure all CI checks pass before submitting changes
- Write clear commit messages
- Update documentation when necessary
