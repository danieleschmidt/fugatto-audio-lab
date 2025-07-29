# Testing Guide

## Overview

This document outlines the testing strategy and procedures for Fugatto Audio Lab.

## Test Structure

```
tests/
├── unit/                  # Unit tests
│   ├── test_core.py      # Core functionality
│   ├── test_models.py    # Model loading/inference
│   └── test_utils.py     # Utility functions
├── integration/          # Integration tests
│   ├── test_pipeline.py  # End-to-end pipelines
│   └── test_api.py       # API integration
├── performance/          # Performance benchmarks
│   ├── test_inference.py # Inference speed/memory
│   └── test_training.py  # Training performance
├── security/            # Security tests
│   └── test_security.py # Security validations
└── fixtures/            # Test data
    ├── audio/           # Sample audio files
    └── models/          # Mock model weights
```

## Test Categories

### Unit Tests
- **Purpose**: Test individual functions and classes in isolation
- **Scope**: Single functions, methods, or small components
- **Speed**: Fast (<1s per test)
- **Coverage**: Aim for >90% line coverage

### Integration Tests
- **Purpose**: Test component interactions and workflows
- **Scope**: Multiple components working together
- **Speed**: Medium (1-10s per test)
- **Coverage**: Critical user journeys

### Performance Tests
- **Purpose**: Validate performance characteristics
- **Scope**: Inference speed, memory usage, throughput
- **Speed**: Slow (10s+ per test)
- **Coverage**: Performance-critical paths

### Security Tests
- **Purpose**: Validate security controls and detect vulnerabilities
- **Scope**: Input validation, authentication, data protection
- **Speed**: Medium (1-10s per test)
- **Coverage**: Security-sensitive functionality

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
pytest tests/security/

# Run with coverage
pytest --cov=fugatto_lab --cov-report=html

# Run specific test file
pytest tests/unit/test_core.py

# Run specific test function
pytest tests/unit/test_core.py::test_audio_loading
```

### Test Markers

```bash
# Run only fast tests
pytest -m "not slow"

# Run only GPU tests (requires GPU)
pytest -m gpu

# Run unit tests only
pytest -m unit

# Run integration tests only
pytest -m integration

# Skip network-dependent tests
pytest -m "not network"
```

### Parallel Testing

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Run with specific number of workers
pytest -n 4
```

### Environment-Specific Testing

```bash
# Test with different Python versions using tox
tox -e py310,py311,py312

# Test linting
tox -e lint

# Test security
tox -e security

# Test documentation build
tox -e docs
```

## Test Configuration

### pytest.ini Configuration

Key settings in `pytest.ini`:
- Minimum coverage threshold: 80%
- Strict marker enforcement
- Warning filters for external libraries
- Test discovery patterns

### Coverage Configuration

Coverage settings in `pyproject.toml`:
- Source packages: `fugatto_lab`
- Excluded files: tests, setup.py
- Report formats: terminal, HTML, XML

## Writing Tests

### Test Naming Conventions

- Test files: `test_*.py` or `*_test.py`
- Test classes: `Test*`
- Test functions: `test_*`
- Use descriptive names: `test_audio_loading_with_invalid_file`

### Test Structure (AAA Pattern)

```python
def test_audio_generation():
    # Arrange
    model = FugattoModel.from_pretrained("test-model")
    prompt = "ocean waves"
    
    # Act
    audio = model.generate(prompt, duration_seconds=5)
    
    # Assert
    assert audio is not None
    assert len(audio) > 0
    assert audio.sample_rate == 48000
```

### Fixtures and Test Data

```python
import pytest
from fugatto_lab import FugattoModel

@pytest.fixture
def sample_model():
    """Provide a test model instance."""
    return FugattoModel.from_pretrained("test-model", cache_dir="tests/fixtures/models")

@pytest.fixture
def sample_audio():
    """Provide sample audio data."""
    return torch.randn(1, 48000)  # 1 second at 48kHz

def test_with_fixtures(sample_model, sample_audio):
    result = sample_model.transform(sample_audio, "add reverb")
    assert result is not None
```

### Mocking External Dependencies

```python
from unittest.mock import patch, MagicMock

@patch('fugatto_lab.core.torch.load')
def test_model_loading(mock_torch_load):
    # Mock external dependencies
    mock_torch_load.return_value = {"model_state": {}}
    
    model = FugattoModel.from_pretrained("test-model")
    assert model is not None
    mock_torch_load.assert_called_once()
```

## Performance Testing

### Benchmark Tests

```python
import pytest
from fugatto_lab import FugattoModel

@pytest.mark.benchmark
def test_inference_speed(benchmark):
    model = FugattoModel.from_pretrained("nvidia/fugatto-base")
    prompt = "piano music"
    
    result = benchmark(model.generate, prompt, duration_seconds=5)
    assert result is not None

@pytest.mark.performance
def test_memory_usage():
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Perform memory-intensive operation
    model = FugattoModel.from_pretrained("nvidia/fugatto-base")
    audio = model.generate("test", duration_seconds=30)
    
    final_memory = process.memory_info().rss
    memory_used = final_memory - initial_memory
    
    # Assert memory usage is within acceptable bounds
    assert memory_used < 2 * 1024 * 1024 * 1024  # 2GB limit
```

## Security Testing

### Input Validation Tests

```python
import pytest
from fugatto_lab import FugattoModel

def test_malicious_prompt_handling():
    model = FugattoModel.from_pretrained("test-model")
    
    # Test various malicious inputs
    malicious_prompts = [
        "../../../etc/passwd",
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "\x00\x01\x02\x03",  # Binary data
    ]
    
    for prompt in malicious_prompts:
        with pytest.raises((ValueError, SecurityError)):
            model.generate(prompt)

def test_file_path_validation():
    from fugatto_lab.utils import validate_audio_path
    
    # Test path traversal attempts
    malicious_paths = [
        "../../../etc/passwd",
        "/root/.ssh/id_rsa",
        "C:\\Windows\\System32\\config\\SAM",
    ]
    
    for path in malicious_paths:
        with pytest.raises(SecurityError):
            validate_audio_path(path)
```

## Continuous Integration

### GitHub Actions Integration

Tests run automatically on:
- Pull requests
- Push to main branch
- Scheduled daily runs
- Manual workflow dispatch

### Test Matrix

Tests run across:
- Python versions: 3.10, 3.11, 3.12
- Operating systems: Ubuntu, macOS, Windows
- Hardware: CPU-only, GPU-enabled

### Quality Gates

Pull requests must pass:
- All unit tests
- Integration tests
- Security tests
- Lint checks
- Coverage threshold (80%)
- Performance benchmarks

## Test Data Management

### Audio Fixtures

Store test audio files in `tests/fixtures/audio/`:
- Keep files small (<1MB each)
- Use multiple formats (WAV, MP3, FLAC)
- Include edge cases (silence, clipping, unusual sample rates)

### Model Fixtures

Store mock model weights in `tests/fixtures/models/`:
- Use minimal model architectures
- Include different model versions
- Mock external model downloads

### Data Generation

```python
# Generate test audio programmatically
import torch
import torchaudio

def generate_test_sine_wave(frequency=440, duration=1.0, sample_rate=48000):
    """Generate a sine wave for testing."""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * torch.pi * frequency * t)
    return audio.unsqueeze(0)  # Add channel dimension
```

## Debugging Failed Tests

### Common Issues

1. **Model Loading Failures**
   - Check model cache directory
   - Verify network connectivity for downloads
   - Ensure sufficient disk space

2. **GPU Test Failures**
   - Verify CUDA availability
   - Check GPU memory usage
   - Use CPU fallback in CI

3. **Audio Processing Errors**
   - Validate audio file formats
   - Check sample rate compatibility
   - Verify audio duration requirements

### Debug Commands

```bash
# Run tests with verbose output
pytest -v

# Run tests with debug prints
pytest -s

# Run specific failing test with maximum verbosity
pytest -vvv tests/unit/test_core.py::test_failing_function

# Drop into debugger on failure
pytest --pdb

# Generate test report
pytest --html=report.html --self-contained-html
```

## Best Practices

1. **Test Independence**: Tests should not depend on each other
2. **Deterministic Results**: Use fixed seeds for reproducible results
3. **Resource Cleanup**: Always clean up temporary files and resources
4. **Fast Feedback**: Keep unit tests fast for quick development cycles
5. **Realistic Data**: Use realistic test data that represents actual usage
6. **Error Cases**: Test both success and failure scenarios
7. **Documentation**: Document complex test scenarios and edge cases

## Monitoring and Reporting

### Coverage Reports

- HTML reports: `htmlcov/index.html`
- XML reports: `coverage.xml` (for CI integration)
- Terminal summary during test runs

### Performance Tracking

- Benchmark results in CI artifacts
- Performance regression detection
- Memory usage profiling reports

### Security Scanning

- Bandit security reports
- Dependency vulnerability scanning
- SAST/DAST integration points