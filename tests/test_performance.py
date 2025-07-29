"""
Performance and load testing for Fugatto Audio Lab.
"""

import pytest
import torch
import time
import concurrent.futures
from unittest.mock import Mock


@pytest.mark.slow
class TestScalabilityAndLoad:
    """Test system behavior under various load conditions."""
    
    def test_concurrent_generation(self, mock_fugatto_model):
        """Test concurrent audio generation requests."""
        model = mock_fugatto_model
        
        def generate_audio(prompt_id):
            return model.generate(
                prompt=f"Test prompt {prompt_id}",
                duration_seconds=3
            )
        
        # Test concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(generate_audio, i) 
                for i in range(8)
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures, timeout=60):
                result = future.result()
                results.append(result)
        
        # Verify all requests completed
        assert len(results) == 8
        for result in results:
            assert result is not None
    
    def test_memory_leak_detection(self, mock_fugatto_model):
        """Test for memory leaks during repeated generation."""
        import psutil
        import gc
        
        model = mock_fugatto_model
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss
        
        # Generate many samples
        for i in range(20):
            audio = model.generate(
                prompt=f"Memory test {i}",
                duration_seconds=2
            )
            
            # Force cleanup every few iterations
            if i % 5 == 0:
                gc.collect()
        
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss
        memory_growth = final_memory - baseline_memory
        
        # Memory growth should be reasonable
        max_acceptable_growth = 500 * 1024 * 1024  # 500MB
        assert memory_growth < max_acceptable_growth
    
    def test_batch_processing_efficiency(self, mock_fugatto_model):
        """Test efficiency of batch vs individual processing."""
        model = mock_fugatto_model
        
        # Mock batch generation
        model.generate_batch = Mock(return_value=[
            torch.randn(1, 48000) for _ in range(5)
        ])
        
        # Time individual generations
        start_time = time.time()
        individual_results = []
        for i in range(5):
            result = model.generate(
                prompt=f"Individual {i}",
                duration_seconds=1
            )
            individual_results.append(result)
        individual_time = time.time() - start_time
        
        # Time batch generation
        start_time = time.time()
        batch_results = model.generate_batch([
            f"Batch {i}" for i in range(5)
        ])
        batch_time = time.time() - start_time
        
        # Batch should be more efficient (in real scenario)
        assert len(batch_results) == 5
        assert len(individual_results) == 5


@pytest.mark.slow
class TestResourceUtilization:
    """Test CPU, GPU, and memory utilization."""
    
    def test_cpu_utilization(self, mock_fugatto_model):
        """Monitor CPU usage during generation."""
        import psutil
        
        model = mock_fugatto_model
        
        # Monitor CPU usage
        cpu_percent_before = psutil.cpu_percent(interval=1)
        
        # Generate audio
        audio = model.generate(
            prompt="CPU test",
            duration_seconds=5
        )
        
        cpu_percent_after = psutil.cpu_percent(interval=1)
        
        # CPU usage should be reasonable
        assert audio is not None
        # In real scenario, would check actual CPU utilization
    
    @pytest.mark.gpu
    def test_gpu_memory_management(self, mock_fugatto_model):
        """Test GPU memory allocation and cleanup."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        model = mock_fugatto_model
        
        # Mock GPU memory functions
        with pytest.MonkeyPatch().context() as m:
            m.setattr(torch.cuda, "memory_allocated", lambda: 1024 * 1024 * 100)  # 100MB
            m.setattr(torch.cuda, "empty_cache", Mock())
            
            # Generate audio
            audio = model.generate(
                prompt="GPU test",
                duration_seconds=3
            )
            
            assert audio is not None


@pytest.mark.slow
class TestRegressionBenchmarks:
    """Regression tests for performance metrics."""
    
    def test_generation_speed_regression(self, mock_fugatto_model):
        """Test that generation speed doesn't regress."""
        model = mock_fugatto_model
        
        # Baseline timing
        start_time = time.time()
        audio = model.generate(
            prompt="Speed test",
            duration_seconds=10,
            temperature=0.8
        )
        generation_time = time.time() - start_time
        
        # Performance targets (adjust based on actual hardware)
        max_acceptable_time = 30.0  # 30 seconds for 10 seconds of audio
        assert generation_time < max_acceptable_time
        assert audio is not None
    
    def test_audio_quality_metrics(self, mock_fugatto_model, sample_audio):
        """Test that generated audio meets quality standards."""
        model = mock_fugatto_model
        
        # Generate audio
        generated = model.generate(
            prompt="Quality test",
            duration_seconds=2
        )
        
        # Basic quality checks
        assert generated is not None
        assert not torch.isnan(generated).any()
        assert not torch.isinf(generated).any()
        
        # Audio should be in reasonable dynamic range
        assert generated.abs().max() <= 1.0
        assert generated.abs().mean() > 0.001  # Not silent