# Contributing to Fugatto Audio Lab

Thank you for your interest in contributing to Fugatto Audio Lab! This document provides guidelines for contributing to the project.

## 🚀 Quick Start

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/yourusername/fugatto-audio-lab.git`
3. **Install** development dependencies: `pip install -e ".[dev]"`
4. **Create** a feature branch: `git checkout -b feature/your-feature-name`
5. **Make** your changes and test them
6. **Submit** a pull request

## 🛠️ Development Setup

### Prerequisites

- Python 3.10+ 
- CUDA 12.0+ (for GPU acceleration)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fugatto-audio-lab.git
cd fugatto-audio-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## 📝 Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting  
- **Ruff**: Fast Python linter
- **MyPy**: Type checking

Run all checks:
```bash
# Format code
black fugatto_lab tests
isort fugatto_lab tests

# Lint code
ruff check fugatto_lab tests

# Type check
mypy fugatto_lab
```

## 🧪 Testing

We use pytest for testing:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fugatto_lab

# Run specific test file
pytest tests/test_model.py

# Run tests in parallel
pytest -n auto
```

### Writing Tests

- Place tests in the `tests/` directory
- Follow the naming convention `test_*.py`
- Use descriptive test names
- Include both unit and integration tests
- Mock external dependencies (NVIDIA models, etc.)

## 🎯 Priority Areas

We welcome contributions in these areas:

### High Priority
- **Audio Effects**: New transformation algorithms
- **Model Optimizations**: Faster inference, memory efficiency
- **Evaluation Metrics**: Better quality assessment tools
- **Documentation**: Tutorials, examples, API docs

### Medium Priority  
- **UI Improvements**: Better Gradio/Streamlit interfaces
- **Dataset Tools**: Data preprocessing utilities
- **Export Formats**: Additional audio format support
- **Performance**: Benchmarking and optimization

### Future Enhancements
- **Multi-GPU Training**: Distributed training support
- **Real-time Streaming**: Live audio generation
- **Community Models**: Model zoo expansion
- **Mobile Support**: Edge deployment tools

## 📋 Pull Request Process

1. **Check existing issues** or create one to discuss your idea
2. **Fork** the repository and create a feature branch
3. **Write code** following our style guidelines
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Run the test suite** to ensure everything works
7. **Submit** a pull request with a clear description

### PR Title Format
- `feat: add new audio effect algorithm`
- `fix: resolve memory leak in model loading`
- `docs: update installation instructions`
- `test: add coverage for voice cloning module`

### PR Description Template
```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for changes
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## 🐛 Bug Reports

When reporting bugs, please include:

- **Environment**: OS, Python version, CUDA version
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **Error messages** and stack traces
- **Minimal code example** if possible

Use the bug report template in GitHub Issues.

## 💡 Feature Requests

For new features:

- **Check existing issues** to avoid duplicates
- **Describe the use case** and motivation
- **Provide examples** of desired API/behavior
- **Consider implementation complexity**

## 📖 Documentation

Documentation improvements are always welcome:

- **API Documentation**: Docstrings and type hints
- **Tutorials**: Jupyter notebooks and examples
- **Architecture**: System design documentation
- **Performance**: Benchmarks and optimization guides

## 🎵 Audio Content Guidelines

When contributing audio examples or datasets:

- **Copyright**: Only use royalty-free or original content
- **Quality**: High-quality samples (48kHz preferred)
- **Diversity**: Include varied content (music, speech, SFX)
- **Metadata**: Provide clear descriptions and tags
- **Size**: Keep example files under 10MB

## 🔒 Security

- **Never commit** API keys, tokens, or credentials
- **Report security issues** privately via email
- **Follow responsible disclosure** practices
- **Use environment variables** for sensitive config

## 📞 Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and community chat
- **Discord**: Real-time community support
- **Email**: Direct contact for sensitive issues

## 🏆 Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes for significant contributions
- Annual contributor spotlight
- Community Discord roles

## 📄 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

**Thank you for contributing to Fugatto Audio Lab!** 🎵

Your contributions help make generative audio more accessible to everyone.