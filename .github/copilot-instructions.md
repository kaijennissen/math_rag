# GitHub Copilot Instructions

This file provides comprehensive guidance for GitHub Copilot (and other coding agents) when working with this repository.

---

## a) Project Goals

**Math-RAG** is a **POC/MVP/Learning Project** focused on:

- **Retrieval-Augmented Generation (RAG)** on mathematical documents
- **Knowledge Graphs** for capturing mathematical concept hierarchies and relationships
- **Graph-enhanced retrieval** combining structured and unstructured knowledge
- **Experimental architecture** for understanding how GraphRAG can improve mathematical QA systems

**Key Philosophy**: This is a learning project. The goal is to explore, experiment, and understand RAG techniques rather than build production-grade software. Prioritize clarity, modularity, and educational value.

---

## b) Setup Instructions

**Focus on Python-based setup** (not Docker/Neo4j, which are handled separately):

### Initial Setup

```bash
# 1. Clone the repository
git clone https://github.com/kajen3/math_rag.git
cd math_rag

# 2. Install uv (if not already installed)
pip install uv

# 3. Create virtual environment and install dependencies
uv sync --extra dev

# 4. Set up pre-commit hooks
pre-commit install

# 5. Create .env file with required API keys
cp .env.example .env
# Edit .env and add your keys (OPENAI_API_KEY, etc.)
```

### Development Workflow

```bash
# Install/update dependencies
uv sync --extra dev           # Sync all dependencies including dev extras
uv add <package-name>         # Add a new runtime dependency
uv add --dev <package-name>   # Add a new development dependency
uv sync --upgrade             # Update all dependencies

# Run tests
pytest                        # Run all tests
pytest tests/path/test_file.py::test_function -v  # Run specific test

# Code quality checks
ruff format src/ tests/       # Format code
ruff check src/ tests/        # Lint code
pre-commit run --all-files    # Run all pre-commit checks

# Lock dependencies
uv lock                       # Generate/update uv.lock file
```

**Important**: Always use `uv` commands for package operations to ensure fast and reliable dependency resolution.

---

## c) Code Style

### Design Principles

Follow these software design principles rigorously:

- **KISS (Keep It Simple, Stupid)**: Prefer simple, straightforward solutions over clever ones
- **YAGNI (You Aren't Gonna Need It)**: Don't add functionality until it's actually needed
- **DRY (Don't Repeat Yourself)**: Avoid code duplication; extract common functionality
- **SOLID Principles**:
  - **S**ingle Responsibility: Each class/function should have one reason to change
  - **O**pen/Closed: Open for extension, closed for modification
  - **L**iskov Substitution: Derived classes must be substitutable for base classes
  - **I**nterface Segregation: Many specific interfaces > one general interface
  - **D**ependency Inversion: Depend on abstractions, not concretions

### Code Standards

#### Formatting
- **Line Length**: 88 characters (enforced by ruff)
- **Import Order**: Standard library → Third-party → Local (use ruff for sorting)
- **Quotes**: Double quotes for strings (enforced by ruff)
- **Type Annotations**: Always use proper type hints from `typing` module

#### Naming Conventions
- **Variables/Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Private members**: `_leading_underscore`

#### Logging vs Print
- **ALWAYS use logging module** instead of `print()` statements
- Use appropriate log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- Example:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  logger.info("Processing document...")
  ```

#### Configuration Management
- **Use Pydantic Settings** (`pydantic_settings.BaseSettings`) for all configuration
- Configuration should read from environment variables and `.env` files
- See `src/math_rag/config/settings.py` for examples
- Example:
  ```python
  from pydantic_settings import BaseSettings

  class MySettings(BaseSettings):
      api_key: str
      model_config = SettingsConfigDict(env_file=".env")
  ```

#### Public-Facing API
- **All public-facing functionality should be in the `cli` module**
- CLI entry points use `ArgumentParser` for argument parsing
- Structure: Parse arguments in `if __name__ == '__main__'`, pass to main function
- This keeps main functions reusable and testable

#### Error Handling
- Use specific exception types in `try/except` blocks
- Don't catch generic `Exception` unless absolutely necessary
- **Exception for CLI modules**: In the `cli` module, it's acceptable to use broad `try/except` blocks to catch all errors at the top level, as this is a POC/MVP project. This allows for graceful error handling and user-friendly error messages at the application boundary.
- Include meaningful error messages with context

#### Documentation
- Write docstrings for all public functions, classes, and modules
- Use Google or NumPy docstring style consistently
- Include parameter types, return types, and raised exceptions

#### File Changes
- **Only change files requested to change** - do not refactor unrelated code
- Make surgical, minimal changes to accomplish the task
- Respect existing code patterns unless explicitly asked to change them

### Pre-commit Hooks

Always run pre-commit checks before committing:
```bash
pre-commit run --all-files
```

Pre-commit automatically handles:
- AST syntax validation
- YAML syntax validation
- Trailing whitespace removal
- End-of-file fixes
- Large file detection
- Secret detection (via detect-secrets)
- Ruff linting and formatting

---

## d) Testing Guidelines

### Test Framework
- **Test Runner**: `pytest`
- **Test Location**: `tests/` directory (mirrors `src/` structure)
- **Test Files**: Must follow `test_*.py` naming convention

### Testing Philosophy

Focus on:
1. **Core functionality** - the critical paths through the code
2. **Edge cases** - boundary conditions, empty inputs, None values, malformed data
3. **Error conditions** - how the code handles failures

**Do NOT focus on**:
- Happy-path testing in loops (repetitive positive scenarios)
- Testing external libraries (assume they work)
- Over-testing simple getters/setters

### Test Organization

Use **test classes** to group related tests:

```python
class TestFormatDocument:
    """Test document formatting for edge cases that could cause runtime errors."""

    def test_empty_content(self):
        """Test handling of empty document content."""
        # test implementation

    def test_missing_metadata(self):
        """Test handling of missing metadata fields."""
        # test implementation
```

### Parametrized Testing

Use `pytest.parametrize` to **reduce boilerplate** and test multiple scenarios:

```python
@pytest.mark.parametrize(
    "page_content,metadata,expected_output,description",
    [
        ("Test", {}, ["Document 1", "Test"], "empty metadata"),
        ("", {"title": "Empty"}, ["Document 2"], "empty content"),
        ("Math", {"type": "theorem"}, ["Math", "None"], "missing text_nl"),
    ],
)
def test_format_document_edge_cases(
    page_content, metadata, expected_output, description
):
    """Test formatting edge cases that could cause runtime errors."""
    doc = Document(page_content=page_content, metadata=metadata)
    result = format_document(doc, 1)

    for expected in expected_output:
        assert expected in result, f"Failed for case: {description}"
```

### Test Structure

- **Arrange**: Set up test data and preconditions
- **Act**: Execute the code under test
- **Assert**: Verify the outcome

### Test Coverage

Focus on:
- **Type safety**: Test with different input types (None, empty, invalid)
- **Boundary conditions**: Min/max values, empty collections
- **Error paths**: Exception handling, validation failures
- **Integration points**: Where modules interact

---

## e) Code Review Guidelines

Code reviews should be split into two levels:

### High-Level Review (Architecture & Design)

Focus on the big picture:

- **Design Patterns**: Are appropriate patterns being used?
- **Architecture**: Does this fit well into the existing structure?
- **Abstractions**: Are abstractions at the right level? Too complex or too simple?
- **Modularity**: Is the code properly modular and loosely coupled?
- **SOLID Principles**: Are SOLID principles being followed?
- **Performance**: Any obvious performance issues? (Not premature optimization)
- **Security**: Any security concerns (API keys, injection risks, etc.)?
- **Scalability**: Will this approach work as the system grows?
- **Dependencies**: Are new dependencies necessary? Are they lightweight?

Questions to ask:
- Does this solution align with project goals (POC/MVP/Learning)?
- Is this the simplest solution that could work (KISS)?
- Are we building features we actually need (YAGNI)?
- Is there duplicate code that should be abstracted (DRY)?

### Low-Level Review (Code Quality & Correctness)

Focus on implementation details:

- **Syntax Errors**: Check for typos, missing imports, incorrect syntax
- **Type Errors**: Verify type annotations are correct and consistent
- **Logic Errors**: Look for off-by-one errors, incorrect conditionals, wrong operators
- **Null/None Handling**: Are None values handled correctly?
- **Return Values**: Are all code paths returning appropriate values?
- **Exception Handling**: Are exceptions caught and handled properly?
- **Edge Cases**: Are edge cases handled (empty lists, zero values, etc.)?
- **Resource Management**: Are files/connections closed properly (use context managers)?
- **Variable Naming**: Are names clear, descriptive, and follow conventions?
- **Code Duplication**: Any repeated code blocks that should be extracted?
- **Logging**: Are appropriate log statements included?
- **Comments**: Are complex sections documented? Are comments necessary and up-to-date?

Questions to ask:
- Could this code throw an unhandled exception?
- Are there any missing return statements?
- Are type hints accurate?
- Is error handling specific enough?
- Are there any potential bugs hiding in edge cases?

### Review Checklist

Before approving changes, verify:
- [ ] All tests pass (`pytest`)
- [ ] Code is formatted (`ruff format src/ tests/`)
- [ ] No linting errors (`ruff check src/ tests/`)
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] Type hints are present and correct
- [ ] Logging is used instead of print statements
- [ ] Configuration uses Pydantic Settings
- [ ] Public APIs are in CLI module
- [ ] Tests cover core functionality and edge cases
- [ ] Documentation is updated if needed
- [ ] Changes are minimal and focused

---

## Build and Development Commands

Quick reference for common commands:

```bash
# Dependencies
uv sync --extra dev           # Install all dependencies
uv add <package>              # Add runtime dependency
uv add --dev <package>        # Add dev dependency
uv sync --upgrade             # Update all dependencies
uv lock                       # Update lock file

# Testing
pytest                        # Run all tests
pytest tests/test_file.py -v # Run specific test file with verbose output
pytest -k "test_name"         # Run tests matching pattern
pytest --collect-only         # List all tests without running

# Code Quality
ruff format src/ tests/       # Format code
ruff check src/ tests/        # Lint code
ruff check --fix src/ tests/  # Auto-fix linting issues
pre-commit run --all-files    # Run all pre-commit checks

# Project Management
git status                    # Check git status
git diff                      # See uncommitted changes
```

---

## Project Structure

- `src/math_rag/` - Main source code
  - `cli/` - Command-line interfaces (public-facing API)
  - `config/` - Configuration management (Pydantic Settings)
  - `core/` - Core domain models and entities
  - `data_processing/` - Document processing pipeline
  - `graph_construction/` - Knowledge graph building
  - `graph_indexing/` - Graph indexing and embeddings
  - `graph_tools/` - Graph utilities and queries
  - `rag_agents/` - RAG agent implementations
  - `utils/` - Shared utilities
- `tests/` - Test suite (mirrors src structure)
- `config/` - Configuration files (YAML, etc.)
- `docs/` - Documentation and processed documents

---

## General Guidelines

- **Keep changes minimal and focused** - only change what's necessary
- **Follow existing patterns** - maintain consistency with existing code
- **Ensure CI passes** - all checks must pass before merging
- **Write clear commit messages** - explain what and why, not how
- **Update documentation** - keep docs in sync with code changes
- **Ask questions** - if something is unclear, ask before implementing
- **Respect project goals** - this is a learning/POC project, not production software
