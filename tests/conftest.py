"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from fugatto_lab import FugattoModel, AudioProcessor


@pytest.fixture
def sample_audio():
    """Generate sample audio for testing."""
    return np.random.randn(48000).astype(np.float32)


@pytest.fixture
def fugatto_model():
    """Create test Fugatto model instance."""
    return FugattoModel.from_pretrained("test-model")


@pytest.fixture
def audio_processor():
    """Create test audio processor instance."""
    return AudioProcessor(sample_rate=48000)