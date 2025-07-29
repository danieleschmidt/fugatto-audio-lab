"""
Integration tests for Fugatto Audio Lab.
"""

import pytest
import torch
from pathlib import Path


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete audio generation workflows."""
    
    def test_text_to_audio_pipeline(self, mock_fugatto_model, mock_audio_processor, temp_dir):
        """Test full text-to-audio generation pipeline."""
        from fugatto_lab.core import FugattoModel, AudioProcessor
        
        # Mock the complete pipeline
        model = mock_fugatto_model
        processor = mock_audio_processor
        
        # Test generation
        prompt = "A cat meowing in a cathedral"
        audio = model.generate(prompt=prompt, duration_seconds=5)
        
        # Verify output
        assert audio is not None
        assert audio.shape[-1] == 48000 * 5  # 5 seconds at 48kHz
        
        # Save output
        output_path = temp_dir / "generated_audio.wav"
        processor.save_audio(audio, str(output_path))
        
        # Verify save was called
        processor.save_audio.assert_called_once()


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance and memory benchmarks."""
    
    def test_generation_latency(self, mock_fugatto_model, performance_monitor):
        """Benchmark audio generation latency."""
        model = mock_fugatto_model
        
        performance_monitor.start()
        audio = model.generate(
            prompt="Test prompt",
            duration_seconds=10,
            temperature=0.8
        )
        metrics = performance_monitor.stop()
        
        # Performance assertions
        assert metrics["duration"] < 30.0  # Should complete in under 30s
        assert audio is not None
    
    def test_memory_usage(self, mock_fugatto_model, performance_monitor):
        """Test memory usage during generation."""
        model = mock_fugatto_model
        
        performance_monitor.start()
        
        # Generate multiple samples
        for i in range(3):
            audio = model.generate(
                prompt=f"Test prompt {i}",
                duration_seconds=5
            )
        
        metrics = performance_monitor.stop()
        
        # Memory should not grow excessively
        assert metrics["memory_delta"] < 1024 * 1024 * 1024  # < 1GB growth


@pytest.mark.integration
@pytest.mark.network
class TestModelLoading:
    """Test model loading and initialization."""
    
    def test_model_download_and_cache(self, tmp_path):
        """Test model downloading and caching behavior."""
        # This would test actual model download in real scenario
        cache_dir = tmp_path / "model_cache"
        cache_dir.mkdir()
        
        # Mock the download process
        with pytest.raises(Exception):
            # Should fail gracefully if model not available
            from fugatto_lab.core import FugattoModel
            model = FugattoModel.from_pretrained(
                "invalid-model-name",
                cache_dir=str(cache_dir)
            )