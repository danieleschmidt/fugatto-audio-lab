"""Tests for core Fugatto functionality."""

import pytest
import numpy as np
from fugatto_lab.core import FugattoModel, AudioProcessor


class TestFugattoModel:
    """Test cases for FugattoModel class."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = FugattoModel("test-model")
        assert model.model_name == "test-model"
    
    def test_from_pretrained(self):
        """Test from_pretrained class method."""
        model = FugattoModel.from_pretrained("test-model")
        assert isinstance(model, FugattoModel)
        assert model.model_name == "test-model"
    
    def test_generate_audio(self, fugatto_model):
        """Test audio generation."""
        audio = fugatto_model.generate("test prompt", duration_seconds=1.0)
        assert isinstance(audio, np.ndarray)
        assert audio.shape == (48000,)  # 1 second at 48kHz
        assert audio.dtype == np.float32
    
    def test_transform_audio(self, fugatto_model, sample_audio):
        """Test audio transformation."""
        transformed = fugatto_model.transform(sample_audio, "test prompt")
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == sample_audio.shape
        assert transformed.dtype == sample_audio.dtype


class TestAudioProcessor:
    """Test cases for AudioProcessor class."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = AudioProcessor(sample_rate=44100)
        assert processor.sample_rate == 44100
    
    def test_load_audio(self, audio_processor):
        """Test audio loading."""
        audio = audio_processor.load_audio("test.wav")
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
    
    def test_save_audio(self, audio_processor, sample_audio, tmp_path):
        """Test audio saving."""
        filepath = tmp_path / "test_output.wav"
        # Should not raise exception
        audio_processor.save_audio(sample_audio, str(filepath))
    
    def test_plot_comparison(self, audio_processor, sample_audio):
        """Test audio comparison plotting."""
        audio2 = np.random.randn(48000).astype(np.float32)
        # Should not raise exception
        audio_processor.plot_comparison(sample_audio, audio2)