# Development Setup Guide

This guide will help you set up the development environment for ScriptGen.

## Prerequisites

- Python 3.10 or higher
- Git
- A Google API key (for Gemini models)
- A Tavily API key (for search)

## Initial Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Chinmay-Bakhale/scriptgen.git
cd scriptgen
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# For development (includes testing tools)
pip install -e ".[dev]"
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# .env
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**Getting API Keys:**
- **Google API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Tavily API Key**: Get from [Tavily](https://tavily.com/)

## Running the Application

### Basic Usage

```bash
python workflow.py
```

You'll be prompted to either:
1. Enter a topic manually
2. Let the AI find a trending topic

### Using as a Module

```python
from workflow import MultiAgentResearchSystem

system = MultiAgentResearchSystem()
result = system.run(topic="Your research topic here")
print(result)
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest -m integration

# Skip integration tests
pytest -m "not integration"
```

### Code Quality Checks

```bash
# Format code with black
black .

# Check formatting without changes
black --check .

# Lint with flake8
flake8 .

# Sort imports
isort .

# Check import sorting
isort --check-only .
```

### Running All Checks (Pre-commit)

```bash
# Run this before committing
black . && isort . && flake8 . && pytest
```

## Git Workflow

### Creating a Feature Branch

```bash
# Update main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
```

### Making Changes

```bash
# Make your changes, then:
git add .
git commit -m "feat: descriptive commit message"

# Push to remote
git push origin feature/your-feature-name
```

### Commit Message Convention

Use conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `style:` - Formatting changes
- `chore:` - Maintenance tasks

### Creating a Pull Request

1. Push your branch to GitHub
2. Go to the repository on GitHub
3. Click "Pull Request"
4. Fill out the PR template
5. Wait for CI/CD checks to pass
6. Request review

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration. On every push/PR:

1. **Linting**: Code is checked with flake8 and black
2. **Testing**: All tests run on Python 3.10 and 3.11
3. **Coverage**: Code coverage is calculated and reported
4. **Security**: Bandit scans for security issues

### Viewing CI/CD Results

- Go to the "Actions" tab on GitHub
- Click on the latest workflow run
- View logs and test results
- Download coverage reports from artifacts

## Project Structure

```
scriptgen/
├── .github/
│   └── workflows/
│       └── ci.yml           # CI/CD configuration
├── tests/
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── conftest.py          # Test fixtures
├── workflow.py              # Main workflow logic
├── image_prompt.py          # Image prompt generator
├── requirements.txt         # Dependencies
├── pytest.ini              # Pytest configuration
├── pyproject.toml          # Project metadata
├── .flake8                 # Linting rules
├── .gitignore              # Git ignore patterns
└── README.md               # Project documentation
```

## Troubleshooting

### API Key Issues

**Problem**: `GOOGLE_API_KEY not found`

**Solution**: 
```bash
# Check if .env file exists
ls -la .env

# Verify it contains your keys
cat .env

# Make sure python-dotenv is installed
pip install python-dotenv
```

### Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Test Failures

**Problem**: Tests fail locally

**Solution**:
```bash
# Clear pytest cache
pytest --cache-clear

# Run with verbose output
pytest -vv

# Check if you're in the right directory
pwd
```

### Rate Limiting

**Problem**: API rate limit errors

**Solution**: The workflow includes sleep delays. If you hit limits:
- Use smaller test datasets
- Mock API calls in tests
- Wait before retrying

## Adding New Features

### Adding a New Agent

1. Create agent class in `workflow.py`
2. Add agent node method (e.g., `_new_agent_node`)
3. Register in `_build_workflow`
4. Write unit tests in `tests/unit/`
5. Update documentation

### Adding New Tests

1. Create test file in appropriate directory
2. Import necessary fixtures from `conftest.py`
3. Write test class and methods
4. Run tests to verify
5. Check coverage

## Resources

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Tavily API Docs](https://docs.tavily.com/)
- [Google Gemini API](https://ai.google.dev/)
- [Pytest Documentation](https://docs.pytest.org/)
- [GitHub Actions](https://docs.github.com/en/actions)

## Getting Help

- Open an issue on GitHub
- Check existing issues for solutions
- Review the documentation
- Ask in pull request comments
