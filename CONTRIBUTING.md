# Contributing to Yahoo Finance API Wrapper

Thank you for your interest in contributing to the Yahoo Finance API Wrapper! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/yourusername/yahoo-finance-api.git
cd yahoo-finance-api
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the development server:
```bash
python main.py
```

## Code Style Guidelines

- Follow PEP 8 style guide for Python code
- Use meaningful variable and function names
- Include docstrings for functions and classes
- Keep functions focused and single-purpose
- Add type hints where applicable

## Pull Request Process

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit them:
```bash
git commit -m "Description of changes"
```

3. Push to your fork:
```bash
git push origin feature/your-feature-name
```

4. Create a Pull Request from your fork to our main repository

### PR Requirements

- Clear description of changes
- Tests for new features
- Documentation updates if needed
- No breaking changes without discussion
- Passes all existing tests

## Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Include both positive and negative test cases

## Documentation

- Update README.md for new features
- Include docstrings for new functions/classes
- Update API documentation if endpoints change

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Follow the project's license terms

## Questions or Need Help?

- Open an issue for bug reports
- Use discussions for feature requests
- Tag maintainers for urgent issues

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
