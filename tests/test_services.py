"""Tests for the Fugatto Audio Lab services."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

from fugatto_lab.services import AudioGenerationService, VoiceCloneService, ModelConversionService
from fugatto_lab.core import FugattoModel, AudioProcessor


class TestAudioGenerationService:
    """Test the AudioGenerationService class."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies for the service."""
        with patch('fugatto_lab.services.audio_service.FugattoModel') as mock_model_class, \
             patch('fugatto_lab.services.audio_service.AudioProcessor') as mock_processor_class, \
             patch('fugatto_lab.services.audio_service.get_monitor') as mock_monitor:
            
            # Setup mock model
            mock_model = Mock()
            mock_model.generate.return_value = np.random.randn(48000).astype(np.float32)
            mock_model.transform.return_value = np.random.randn(96000).astype(np.float32)
            mock_model.model_name = "test-model"
            mock_model_class.from_pretrained.return_value = mock_model
            
            # Setup mock processor
            mock_processor = Mock()
            mock_processor.sample_rate = 48000
            mock_processor.normalize_loudness.return_value = np.random.randn(48000).astype(np.float32)
            mock_processor.get_audio_stats.return_value = {'rms': 0.5, 'peak': 0.8}
            mock_processor.save_audio.return_value = None
            mock_processor_class.return_value = mock_processor
            
            # Setup mock monitor
            mock_monitor_instance = Mock()
            mock_monitor_instance.record_generation_metrics.return_value = None
            mock_monitor.return_value = mock_monitor_instance
            
            yield {
                'model': mock_model,
                'processor': mock_processor,
                'monitor': mock_monitor_instance
            }
    
    def test_service_initialization(self, mock_dependencies):
        """Test service initialization."""
        service = AudioGenerationService()
        assert service.model_name == "nvidia/fugatto-base"
        assert service.cache_enabled == True
        assert service.max_cache_size == 100
    
    def test_generate_audio_basic(self, mock_dependencies):
        """Test basic audio generation."""
        service = AudioGenerationService()
        
        result = service.generate_audio(
            prompt="A cat meowing",
            duration_seconds=5.0,
            temperature=0.8
        )
        
        assert isinstance(result, dict)
        assert 'audio_data' in result
        assert 'sample_rate' in result
        assert 'duration_seconds' in result
        assert 'prompt' in result
        assert 'generation_time_ms' in result
        assert result['prompt'] == "A cat meowing"
        
        # Verify model was called
        mock_dependencies['model'].generate.assert_called_once()
    
    def test_generate_audio_with_output_path(self, mock_dependencies):
        """Test audio generation with file output."""
        service = AudioGenerationService()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            result = service.generate_audio(
                prompt="Test audio",
                duration_seconds=3.0,
                output_path=output_path
            )
            
            assert 'output_path' in result
            assert result['output_path'] == output_path
            
            # Verify save was called
            mock_dependencies['processor'].save_audio.assert_called_once()
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_generate_audio_with_caching(self, mock_dependencies):
        """Test audio generation with caching."""
        service = AudioGenerationService()
        cache_key = "test_cache_key"
        
        # First generation
        result1 = service.generate_audio(
            prompt="Cached audio",
            duration_seconds=2.0,
            cache_key=cache_key
        )
        
        # Second generation with same cache key
        result2 = service.generate_audio(
            prompt="Cached audio",
            duration_seconds=2.0,
            cache_key=cache_key
        )
        
        # Model should only be called once due to caching
        assert mock_dependencies['model'].generate.call_count == 1
    
    def test_transform_audio_with_array(self, mock_dependencies):
        """Test audio transformation with numpy array input."""
        service = AudioGenerationService()
        
        input_audio = np.random.randn(48000).astype(np.float32)
        
        result = service.transform_audio(
            input_audio=input_audio,
            prompt="Add echo",
            strength=0.5
        )
        
        assert isinstance(result, dict)
        assert 'audio_data' in result
        assert 'original_audio' in result
        assert 'transformation_time_ms' in result
        assert result['prompt'] == "Add echo"
        assert result['strength'] == 0.5
        
        # Verify model was called
        mock_dependencies['model'].transform.assert_called_once()
    
    def test_transform_audio_with_file_path(self, mock_dependencies):
        """Test audio transformation with file path input."""
        service = AudioGenerationService()
        
        # Mock file loading
        mock_dependencies['processor'].load_audio.return_value = np.random.randn(48000).astype(np.float32)
        
        result = service.transform_audio(
            input_audio="/fake/path/audio.wav",
            prompt="Transform audio",
            strength=0.7
        )
        
        assert 'input_path' in result
        assert result['input_path'] == "/fake/path/audio.wav"
        
        # Verify file was loaded
        mock_dependencies['processor'].load_audio.assert_called_once_with("/fake/path/audio.wav")
    
    def test_batch_generate(self, mock_dependencies):
        """Test batch audio generation."""
        service = AudioGenerationService()
        
        prompts = ["Cat meowing", "Dog barking", "Bird singing"]
        
        results = service.batch_generate(
            prompts=prompts,
            duration_seconds=2.0,
            temperature=0.7
        )
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert 'batch_index' in result
            assert result['batch_index'] == i
            assert 'audio_data' in result
        
        # Verify model was called for each prompt
        assert mock_dependencies['model'].generate.call_count == 3
    
    def test_compare_generations(self, mock_dependencies):
        """Test generation comparison with different temperatures."""
        service = AudioGenerationService()
        
        temperatures = [0.5, 0.8, 1.0]
        
        result = service.compare_generations(
            prompt="Test comparison",
            temperatures=temperatures,
            duration_seconds=3.0
        )
        
        assert 'generations' in result
        assert len(result['generations']) == 3
        assert 'avg_generation_time_ms' in result
        
        # Verify model was called for each temperature
        assert mock_dependencies['model'].generate.call_count == 3
    
    def test_cache_operations(self, mock_dependencies):
        """Test cache management operations."""
        service = AudioGenerationService()
        
        # Generate with cache
        service.generate_audio("Test", cache_key="test1")
        service.generate_audio("Test2", cache_key="test2")
        
        # Check cache stats
        stats = service.get_cache_stats()
        assert stats['cache_enabled'] == True
        assert stats['cache_size'] == 2
        assert 'test1' in stats['cache_keys']
        assert 'test2' in stats['cache_keys']
        
        # Clear cache
        service.clear_cache()
        stats_after = service.get_cache_stats()
        assert stats_after['cache_size'] == 0
    
    def test_service_stats(self, mock_dependencies):
        """Test service statistics retrieval."""
        service = AudioGenerationService()
        
        stats = service.get_service_stats()
        
        assert 'service' in stats
        assert stats['service'] == 'AudioGenerationService'
        assert 'model_name' in stats
        assert 'cache_stats' in stats
        assert 'performance_stats' in stats


class TestVoiceCloneService:
    """Test the VoiceCloneService class."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock dependencies for voice service."""
        with patch('fugatto_lab.services.voice_service.FugattoModel') as mock_model_class, \
             patch('fugatto_lab.services.voice_service.AudioProcessor') as mock_processor_class:
            
            # Setup mock model
            mock_model = Mock()
            mock_model.generate.return_value = np.random.randn(48000).astype(np.float32)
            mock_model.transform.return_value = np.random.randn(48000).astype(np.float32)
            mock_model_class.from_pretrained.return_value = mock_model
            
            # Setup mock processor
            mock_processor = Mock()
            mock_processor.sample_rate = 48000
            mock_processor.preprocess.return_value = np.random.randn(48000).astype(np.float32)
            mock_processor.load_audio.return_value = np.random.randn(48000).astype(np.float32)
            mock_processor.get_audio_stats.return_value = {'rms': 0.4, 'peak': 0.7}
            mock_processor_class.return_value = mock_processor
            
            yield {'model': mock_model, 'processor': mock_processor}
    
    def test_extract_speaker_embedding(self, mock_dependencies):
        """Test speaker embedding extraction."""
        service = VoiceCloneService()
        
        reference_audio = np.random.randn(48000).astype(np.float32)
        
        result = service.extract_speaker_embedding(
            reference_audio=reference_audio,
            speaker_id="test_speaker"
        )
        
        assert isinstance(result, dict)
        assert 'speaker_id' in result
        assert 'embedding' in result
        assert 'extraction_time_ms' in result
        assert result['speaker_id'] == "test_speaker"
        
        # Verify speaker is cached
        assert "test_speaker" in service.speaker_embeddings
    
    def test_clone_voice_basic(self, mock_dependencies):
        """Test basic voice cloning."""
        service = VoiceCloneService()
        
        reference_audio = np.random.randn(48000).astype(np.float32)
        
        result = service.clone_voice(
            reference_audio=reference_audio,
            text="Hello, this is a test",
            speaker_id="test_speaker"
        )
        
        assert isinstance(result, dict)
        assert 'cloned_audio' in result
        assert 'original_text' in result
        assert 'cloning_time_ms' in result
        assert result['original_text'] == "Hello, this is a test"
        
        # Verify model was called
        mock_dependencies['model'].generate.assert_called_once()
    
    def test_clone_voice_with_cached_speaker(self, mock_dependencies):
        """Test voice cloning with cached speaker embedding."""
        service = VoiceCloneService()
        
        # First extract embedding
        reference_audio = np.random.randn(48000).astype(np.float32)
        service.extract_speaker_embedding(reference_audio, "cached_speaker")
        
        # Clone with cached speaker
        result = service.clone_voice(
            reference_audio=reference_audio,  # Will be ignored due to cache
            text="Using cached speaker",
            speaker_id="cached_speaker"
        )
        
        assert result['speaker_id'] == "cached_speaker"
    
    def test_batch_clone_voices(self, mock_dependencies):
        """Test batch voice cloning."""
        service = VoiceCloneService()
        
        reference_audio = np.random.randn(48000).astype(np.float32)
        texts = ["First text", "Second text", "Third text"]
        
        results = service.batch_clone_voices(
            reference_audio=reference_audio,
            texts=texts,
            speaker_id="batch_speaker"
        )
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert 'batch_index' in result
            assert result['batch_index'] == i
            assert 'cloned_audio' in result
    
    def test_convert_voice_realtime(self, mock_dependencies):
        """Test real-time voice conversion."""
        service = VoiceCloneService()
        
        # Setup a target speaker
        service.speaker_embeddings["target"] = {
            'embedding': {'pitch_mean': 300.0, 'formants': [800, 1200, 2400]}
        }
        
        input_audio = np.random.randn(96000).astype(np.float32)  # 2 seconds
        
        result = service.convert_voice_realtime(
            input_audio=input_audio,
            target_speaker_id="target",
            chunk_size_seconds=1.0
        )
        
        assert 'converted_audio' in result
        assert 'target_speaker_id' in result
        assert 'num_chunks' in result
        assert result['target_speaker_id'] == "target"
        assert result['num_chunks'] == 2  # 2 second audio with 1 second chunks
    
    def test_speaker_management(self, mock_dependencies):
        """Test speaker list and removal operations."""
        service = VoiceCloneService()
        
        # Add some speakers
        reference_audio = np.random.randn(48000).astype(np.float32)
        service.extract_speaker_embedding(reference_audio, "speaker1")
        service.extract_speaker_embedding(reference_audio, "speaker2")
        
        # Get speaker list
        speakers = service.get_speaker_list()
        assert len(speakers) == 2
        speaker_ids = [s['speaker_id'] for s in speakers]
        assert "speaker1" in speaker_ids
        assert "speaker2" in speaker_ids
        
        # Remove speaker
        removed = service.remove_speaker("speaker1")
        assert removed == True
        
        # Verify removal
        speakers_after = service.get_speaker_list()
        assert len(speakers_after) == 1
        assert speakers_after[0]['speaker_id'] == "speaker2"
        
        # Try to remove non-existent speaker
        not_removed = service.remove_speaker("nonexistent")
        assert not_removed == False
    
    def test_service_stats(self, mock_dependencies):
        """Test service statistics."""
        service = VoiceCloneService()
        
        # Add a speaker
        reference_audio = np.random.randn(48000).astype(np.float32)
        service.extract_speaker_embedding(reference_audio, "test_speaker")
        
        stats = service.get_service_stats()
        
        assert stats['service'] == 'VoiceCloneService'
        assert stats['cached_speakers'] == 1
        assert 'test_speaker' in stats['speaker_list']


class TestModelConversionService:
    """Test the ModelConversionService class."""
    
    @pytest.fixture
    def mock_torch(self):
        """Mock torch operations."""
        with patch('fugatto_lab.services.conversion_service.torch') as mock_torch:
            mock_torch.save.return_value = None
            yield mock_torch
    
    def test_service_initialization(self):
        """Test service initialization."""
        service = ModelConversionService()
        
        assert 'input' in service.supported_formats
        assert 'output' in service.supported_formats
        assert 'encodec' in service.supported_formats['input']
        assert 'fugatto' in service.supported_formats['output']
    
    def test_convert_encodec_to_fugatto(self, mock_torch):
        """Test EnCodec to Fugatto conversion."""
        service = ModelConversionService()
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            result = service.convert_encodec_to_fugatto(
                encodec_checkpoint="facebook/encodec_32khz",
                output_path=output_path,
                target_sample_rate=48000,
                optimize_for_inference=True
            )
            
            assert isinstance(result, dict)
            assert 'source_model' in result
            assert 'output_path' in result
            assert 'conversion_time_ms' in result
            assert 'validation' in result
            assert result['target_sample_rate'] == 48000
            assert result['optimized'] == True
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_convert_audiocraft_to_fugatto(self, mock_torch):
        """Test AudioCraft to Fugatto conversion."""
        service = ModelConversionService()
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            result = service.convert_audiocraft_to_fugatto(
                audiocraft_model="facebook/musicgen-large",
                output_path=output_path,
                model_type="musicgen"
            )
            
            assert isinstance(result, dict)
            assert 'source_model' in result
            assert 'model_type' in result
            assert 'conversion_time_ms' in result
            assert result['model_type'] == 'musicgen'
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_batch_convert_models(self, mock_torch):
        """Test batch model conversion."""
        service = ModelConversionService()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_list = [
                {
                    'source_type': 'encodec',
                    'source_path': 'facebook/encodec_32khz',
                    'name': 'encodec_model'
                },
                {
                    'source_type': 'audiocraft',
                    'source_path': 'facebook/musicgen-small',
                    'model_type': 'musicgen',
                    'name': 'musicgen_model'
                }
            ]
            
            results = service.batch_convert_models(model_list, tmp_dir)
            
            assert len(results) == 2
            for result in results:
                assert 'batch_index' in result
                assert 'model_name' in result
                if 'error' not in result:
                    assert 'conversion_time_ms' in result
    
    def test_validate_conversion(self, mock_torch):
        """Test model validation."""
        service = ModelConversionService()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            # Create a fake model file
            import pickle
            fake_model = {'model_type': 'fugatto', 'architecture': {}}
            pickle.dump(fake_model, tmp)
            model_path = tmp.name
        
        try:
            validation = service.validate_conversion(model_path)
            
            assert isinstance(validation, dict)
            assert 'structure_valid' in validation
            assert 'forward_pass_valid' in validation
            assert 'overall_valid' in validation
            
        finally:
            Path(model_path).unlink(missing_ok=True)
    
    def test_get_supported_formats(self):
        """Test supported formats retrieval."""
        service = ModelConversionService()
        
        formats = service.get_supported_formats()
        
        assert isinstance(formats, dict)
        assert 'input' in formats
        assert 'output' in formats
        assert isinstance(formats['input'], list)
        assert isinstance(formats['output'], list)
    
    def test_conversion_stats(self):
        """Test conversion statistics."""
        service = ModelConversionService()
        
        stats = service.get_conversion_stats()
        
        assert stats['service'] == 'ModelConversionService'
        assert 'supported_formats' in stats
        assert 'cache_size' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])