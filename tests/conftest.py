"""
Comprehensive pytest configuration and fixtures for Fugatto Audio Lab tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_audio() -> torch.Tensor:
    """Generate sample audio tensor for testing."""
    # 1 second of 48kHz sine wave at 440Hz
    sample_rate = 48000
    duration = 1.0
    frequency = 440.0
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)  # Add batch dim
    
    return audio


@pytest.fixture
def sample_stereo_audio() -> torch.Tensor:
    """Generate sample stereo audio for testing."""
    sample_rate = 48000
    duration = 2.0
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    left = torch.sin(2 * torch.pi * 440 * t)
    right = torch.sin(2 * torch.pi * 880 * t) * 0.7  # Different frequency, lower volume
    
    return torch.stack([left, right], dim=0)


@pytest.fixture
def mock_model_config() -> Dict[str, Any]:
    """Mock model configuration for testing."""
    return {
        "model_name": "test-fugatto",
        "sample_rate": 48000,
        "channels": 1,
        "max_length": 1500,
        "temperature": 0.8,
        "top_p": 0.95,
        "cfg_scale": 3.0,
    }


@pytest.fixture
def mock_fugatto_model():
    """Mock FugattoModel for testing without loading actual weights."""
    with patch("fugatto_lab.core.FugattoModel") as mock:
        model_instance = Mock()
        model_instance.generate.return_value = torch.randn(1, 48000)  # 1 second
        model_instance.transform.return_value = torch.randn(1, 96000)  # 2 seconds
        model_instance.sample_rate = 48000
        mock.from_pretrained.return_value = model_instance
        yield model_instance


@pytest.fixture
def mock_audio_processor():
    """Mock AudioProcessor for testing."""
    with patch("fugatto_lab.core.AudioProcessor") as mock:
        processor_instance = Mock()
        processor_instance.load_audio.return_value = torch.randn(1, 48000)
        processor_instance.save_audio.return_value = None
        mock.return_value = processor_instance
        yield processor_instance


@pytest.fixture(autouse=True)
def mock_cuda_available():
    """Mock CUDA availability for consistent testing."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


# Legacy fixtures for backward compatibility
@pytest.fixture
def fugatto_model():
    """Create test Fugatto model instance (legacy)."""
    with patch("fugatto_lab.core.FugattoModel") as mock:
        model_instance = Mock()
        model_instance.generate.return_value = torch.randn(1, 48000)
        mock.from_pretrained.return_value = model_instance
        return model_instance


@pytest.fixture
def audio_processor():
    """Create test audio processor instance (legacy)."""
    with patch("fugatto_lab.core.AudioProcessor") as mock:
        processor_instance = Mock()
        processor_instance.sample_rate = 48000
        mock.return_value = processor_instance
        return processor_instance


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network access"
    )