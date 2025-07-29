# Development Guide

## Quick Start

Get up and running with Fugatto Audio Lab development in minutes.

### Prerequisites

- Python 3.10+ 
- CUDA 12.0+ (for GPU acceleration)
- Git
- 16GB+ RAM recommended
- 8GB+ GPU memory for training

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/fugatto-audio-lab.git
cd fugatto-audio-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Verify installation
make test
```

## Development Workflow

### Daily Development

```bash
# Start development session
source venv/bin/activate
cd fugatto-audio-lab

# Pull latest changes
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes...

# Run quality checks
make quality

# Run tests
make test

# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

### Code Quality

We maintain high code quality through automated tooling:

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Security scanning  
make security

# Run all checks
make quality
```

### Testing

```bash
# Run full test suite
make test

# Quick tests (no coverage)
make test-quick

# Run specific test file
pytest tests/test_core.py -v

# Test with specific Python version
tox -e py310
```

## Project Structure

```
fugatto-audio-lab/
├── fugatto_lab/          # Main package
│   ├── __init__.py       # Package initialization
│   ├── core.py           # Core model classes
│   ├── processing.py     # Audio processing utilities
│   ├── training/         # Training modules
│   ├── evaluation/       # Evaluation tools
│   ├── ui/              # Web interface components
│   └── utils/           # Utility functions
├── tests/               # Test suite
│   ├── conftest.py      # Pytest configuration
│   ├── test_core.py     # Core functionality tests
│   └── fixtures/        # Test data
├── docs/                # Documentation
│   ├── workflows/       # CI/CD guides
│   ├── ARCHITECTURE.md  # System design
│   └── DEVELOPMENT.md   # This file
├── examples/            # Example notebooks and scripts
├── scripts/             # Development and build scripts
├── configs/             # Configuration files
└── pyproject.toml       # Project configuration
```

## Development Tools

### IDE Configuration

#### VS Code
Recommended extensions:
- Python
- Pylance  
- Black Formatter
- Git Lens
- Thunder Client (API testing)

Settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.testing.pytestEnabled": true
}
```

#### PyCharm
- Enable Black integration
- Configure pytest as test runner
- Set up Ruff for linting
- Enable type checking with MyPy

### Debugging

#### Python Debugging
```python
# Use built-in debugger
import pdb; pdb.set_trace()

# Or IPython debugger (more features)
import ipdb; ipdb.set_trace()

# For async code
import aioipdb; await aioipdb.set_trace()
```

#### Audio Debugging
```python
# Visualize audio waveforms
from fugatto_lab.utils import plot_waveform, plot_spectrogram

plot_waveform(audio, title="Generated Audio")
plot_spectrogram(audio, sr=48000)

# Save intermediate results
processor.save_audio(intermediate_audio, "debug_output.wav")
```

## Testing Guidelines

### Test Organization

```python
# tests/test_feature.py
import pytest
from fugatto_lab.feature import FeatureClass

class TestFeatureClass:
    """Test cases for FeatureClass."""
    
    def test_initialization(self):
        """Test basic initialization."""
        feature = FeatureClass()
        assert isinstance(feature, FeatureClass)
    
    def test_core_functionality(self, sample_data):
        """Test main functionality."""
        result = feature.process(sample_data)
        assert result is not None
        
    @pytest.mark.slow
    def test_performance(self, large_dataset):
        """Test performance with large data."""
        # Performance tests marked as slow
        pass
```

### Fixtures and Mocking

```python
# tests/conftest.py
@pytest.fixture
def sample_audio():
    """Generate test audio data."""
    return np.random.randn(48000).astype(np.float32)

@pytest.fixture
def mock_model(monkeypatch):
    """Mock expensive model loading."""
    def mock_load(*args, **kwargs):
        return MockModel()
    
    monkeypatch.setattr("fugatto_lab.core.load_model", mock_load)
    return MockModel()
```

### Test Categories

- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test component interactions  
- **Performance Tests**: Benchmark critical paths
- **Regression Tests**: Prevent known bugs from returning
- **Property Tests**: Test invariants with random data

## Performance Optimization

### Profiling

```python
# CPU profiling
import cProfile
cProfile.run('your_function()', 'profile_output.prof')

# Memory profiling  
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Your code here
    pass

# GPU profiling
with torch.profiler.profile() as prof:
    model(input_data)
print(prof.key_averages().table())
```

### Optimization Checklist

- [ ] Use torch.compile() for model inference
- [ ] Enable mixed precision training (fp16/bf16)
- [ ] Implement gradient checkpointing for large models
- [ ] Use DataLoader with num_workers > 0
- [ ] Profile memory usage and optimize accordingly
- [ ] Consider model quantization for deployment

## Documentation

### API Documentation

```python
def generate_audio(prompt: str, duration: float = 10.0) -> np.ndarray:
    """Generate audio from text prompt.
    
    Args:
        prompt: Text description of desired audio
        duration: Length of generated audio in seconds
        
    Returns:
        Generated audio as numpy array
        
    Raises:
        ValueError: If duration is negative or too large
        
    Example:
        >>> audio = generate_audio("Ocean waves", duration=5.0)
        >>> print(audio.shape)
        (240000,)  # 5 seconds at 48kHz
    """
```

### README Updates

When adding new features:
1. Update installation instructions if needed
2. Add usage examples  
3. Update feature list
4. Include performance notes
5. Add troubleshooting section

## Release Process

### Version Management

We use semantic versioning (MAJOR.MINOR.PATCH):

```bash
# Patch release (bug fixes)
make release-patch

# Minor release (new features, backward compatible)
make release-minor  

# Major release (breaking changes)
make release-major
```

### Pre-release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped appropriately
- [ ] Security scan passed
- [ ] Performance benchmarks run
- [ ] Example notebooks tested

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```python
# Reduce batch size
trainer.batch_size = 4

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use CPU offloading
model.enable_model_cpu_offload()
```

#### Import Errors
```bash
# Reinstall in development mode
pip install -e ".[dev]"

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -exec rm -rf {} +
```

#### Pre-commit Hook Failures
```bash
# Run hooks manually
pre-commit run --all-files

# Skip hooks temporarily (not recommended)
git commit --no-verify
```

### Getting Help

1. **Check existing issues**: Search GitHub issues first
2. **Run diagnostics**: Use `fugatto-lab doctor` command
3. **Provide details**: Include Python version, OS, GPU info
4. **Minimal example**: Create reproducible test case
5. **Community support**: Join Discord/discussions

## Contributing Best Practices

### Code Style
- Follow PEP 8 (enforced by Black)
- Use type hints for all public APIs
- Write descriptive docstrings
- Keep functions focused and small
- Use meaningful variable names

### Git Workflow
- Create feature branches from `main`
- Write clear commit messages
- Squash commits before merging
- Keep PRs focused and reviewable
- Update tests and docs with changes

### Review Process
- All PRs require approval
- Address all review comments
- Ensure CI passes before merge
- Test locally before pushing
- Be responsive to feedback

## Advanced Topics

### Custom Model Integration

```python
# Implement custom model wrapper
class CustomFugattoModel(FugattoModel):
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        super().__init__(self.config.model_name)
    
    def generate(self, prompt: str, **kwargs) -> np.ndarray:
        # Custom generation logic
        return super().generate(prompt, **kwargs)
```

### Plugin Development

```python
# Create audio effect plugin
from fugatto_lab.plugins import AudioEffectPlugin

class ReverbPlugin(AudioEffectPlugin):
    name = "reverb"
    
    def process(self, audio: np.ndarray, 
                room_size: float = 0.5,
                wet_dry: float = 0.3) -> np.ndarray:
        # Implement reverb effect
        return processed_audio
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "room_size": {"min": 0.0, "max": 1.0, "default": 0.5},
            "wet_dry": {"min": 0.0, "max": 1.0, "default": 0.3}
        }
```

### Distributed Training

```python
# Setup distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

model = DistributedDataParallel(model, device_ids=[local_rank])
```

---

Happy coding! 🎵 Join our community and help make generative audio more accessible to everyone.