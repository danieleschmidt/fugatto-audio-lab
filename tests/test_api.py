"""Tests for the Fugatto Audio Lab API."""

import pytest
import json
import io
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

try:
    from fastapi.testclient import TestClient
    import httpx
    FASTAPI_AVAILABLE = True
except ImportError:
    TestClient = None
    FASTAPI_AVAILABLE = False

from fugatto_lab.api.app import create_app
from fugatto_lab.database.models import AudioRecordData


@pytest.fixture
def mock_services():
    """Mock all external services for API testing."""
    with patch('fugatto_lab.api.routes.get_audio_service') as mock_audio, \
         patch('fugatto_lab.api.routes.get_voice_service') as mock_voice, \
         patch('fugatto_lab.api.routes.get_audio_repository') as mock_repo, \
         patch('fugatto_lab.services.AudioGenerationService') as mock_audio_class, \
         patch('fugatto_lab.database.get_db_manager') as mock_db:
        
        # Mock audio service
        mock_audio_instance = Mock()
        mock_audio_instance.generate_audio.return_value = {
            'duration_seconds': 10.0,
            'sample_rate': 48000,
            'model_name': 'test-model',
            'generation_time_ms': 1000.0,
            'audio_stats': {'rms': 0.5, 'peak': 0.8}
        }
        mock_audio_instance.transform_audio.return_value = {
            'duration_seconds': 5.0,
            'transformation_time_ms': 500.0,
            'audio_stats': {'rms': 0.6, 'peak': 0.9}
        }
        mock_audio_instance.get_service_stats.return_value = {'status': 'healthy'}
        mock_audio.return_value = mock_audio_instance
        mock_audio_class.return_value = mock_audio_instance
        
        # Mock voice service
        mock_voice_instance = Mock()
        mock_voice_instance.clone_voice.return_value = {
            'duration_seconds': 3.0,
            'cloning_time_ms': 800.0,
            'audio_stats': {'rms': 0.4, 'peak': 0.7}
        }
        mock_voice_instance.get_speaker_list.return_value = [
            {'speaker_id': 'speaker1', 'reference_duration': 5.0}
        ]
        mock_voice.return_value = mock_voice_instance
        
        # Mock repository
        mock_repo_instance = Mock()
        mock_repo_instance.create_record.return_value = 123
        mock_repo_instance.get_record.return_value = AudioRecordData(
            id=123,
            prompt="test prompt",
            audio_path="/tmp/test.wav",
            duration_seconds=10.0,
            sample_rate=48000,
            model_name="test-model",
            temperature=0.8,
            generation_time_ms=1000.0,
            metadata={},
            tags=[]
        )
        mock_repo_instance.list_records.return_value = []
        mock_repo.return_value = mock_repo_instance
        
        # Mock database manager
        mock_db_instance = Mock()
        mock_db_instance.get_health_status.return_value = {'status': 'healthy'}
        mock_db_instance.initialize.return_value = None
        mock_db.return_value = mock_db_instance
        
        yield {
            'audio_service': mock_audio_instance,
            'voice_service': mock_voice_instance,
            'repository': mock_repo_instance,
            'db_manager': mock_db_instance
        }


@pytest.fixture
def app_config():
    """Test configuration for the FastAPI app."""
    return {
        'debug': True,
        'cors_origins': ['http://localhost:3000'],
        'enable_rate_limiting': False,  # Disable for testing
        'log_requests': False,
        'enable_docs': True
    }


@pytest.fixture
def client(mock_services, app_config):
    """Test client for the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not available")
    
    app = create_app(app_config)
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, client, mock_services):
        """Test basic health check endpoint."""
        response = client.get("/api/v1/health/")
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] in ['healthy', 'warning']
        assert 'timestamp' in data
        assert 'version' in data
        assert 'services' in data
        assert 'system_info' in data
    
    def test_metrics_endpoint(self, client, mock_services):
        """Test metrics endpoint."""
        response = client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, dict)


class TestGenerationEndpoints:
    """Test audio generation endpoints."""
    
    def test_generate_audio_success(self, client, mock_services):
        """Test successful audio generation."""
        request_data = {
            "prompt": "A cat meowing",
            "duration_seconds": 5.0,
            "temperature": 0.7,
            "save_to_db": True
        }
        
        response = client.post("/api/v1/generate/", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data['prompt'] == "A cat meowing"
        assert data['duration_seconds'] == 10.0  # From mock
        assert data['generation_id'] == "123"
        assert 'generation_time_ms' in data
        assert 'audio_url' in data
        
        # Verify service was called
        mock_services['audio_service'].generate_audio.assert_called_once()
        mock_services['repository'].create_record.assert_called_once()
    
    def test_generate_audio_validation_error(self, client, mock_services):
        """Test audio generation with invalid input."""
        request_data = {
            "prompt": "",  # Empty prompt should fail
            "duration_seconds": 5.0
        }
        
        response = client.post("/api/v1/generate/", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_generate_audio_duration_validation(self, client, mock_services):
        """Test duration validation."""
        request_data = {
            "prompt": "Test prompt",
            "duration_seconds": 100.0  # Too long
        }
        
        response = client.post("/api/v1/generate/", json=request_data)
        assert response.status_code == 422
    
    def test_transform_audio_success(self, client, mock_services):
        """Test successful audio transformation."""
        # Create a mock audio file
        audio_content = b"fake audio data"
        
        response = client.post(
            "/api/v1/generate/transform",
            data={
                "prompt": "Add reverb",
                "strength": 0.5
            },
            files={
                "audio_file": ("test.wav", io.BytesIO(audio_content), "audio/wav")
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['prompt'] == "Add reverb"
        assert data['strength'] == 0.5
        assert 'transformation_time_ms' in data
        
        # Verify service was called
        mock_services['audio_service'].transform_audio.assert_called_once()
    
    def test_transform_audio_invalid_file(self, client, mock_services):
        """Test transformation with non-audio file."""
        text_content = b"not audio data"
        
        response = client.post(
            "/api/v1/generate/transform",
            data={"prompt": "Add reverb"},
            files={
                "audio_file": ("test.txt", io.BytesIO(text_content), "text/plain")
            }
        )
        
        assert response.status_code == 400
        assert "audio file" in response.json()['detail'].lower()
    
    def test_generation_history(self, client, mock_services):
        """Test generation history endpoint."""
        response = client.get("/api/v1/generate/history")
        assert response.status_code == 200
        
        data = response.json()
        assert 'generations' in data
        assert 'total_count' in data
        assert 'page' in data
        assert 'page_size' in data
        assert 'has_next' in data
        
        # Test pagination
        response = client.get("/api/v1/generate/history?page=2&page_size=10")
        assert response.status_code == 200
    
    def test_generation_history_with_filter(self, client, mock_services):
        """Test generation history with model filter."""
        response = client.get("/api/v1/generate/history?model_name=test-model")
        assert response.status_code == 200
        
        # Verify repository was called with filter
        mock_services['repository'].list_records.assert_called()


class TestVoiceEndpoints:
    """Test voice cloning endpoints."""
    
    def test_clone_voice_success(self, client, mock_services):
        """Test successful voice cloning."""
        audio_content = b"fake reference audio"
        
        response = client.post(
            "/api/v1/voice/clone",
            data={
                "text": "Hello, this is a test",
                "speaker_id": "test_speaker",
                "prosody_transfer": True
            },
            files={
                "reference_audio": ("ref.wav", io.BytesIO(audio_content), "audio/wav")
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['text'] == "Hello, this is a test"
        assert data['speaker_id'] == "test_speaker"
        assert 'cloning_time_ms' in data
        
        # Verify service was called
        mock_services['voice_service'].clone_voice.assert_called_once()
    
    def test_clone_voice_invalid_file(self, client, mock_services):
        """Test voice cloning with invalid reference file."""
        text_content = b"not audio"
        
        response = client.post(
            "/api/v1/voice/clone",
            data={"text": "Test text"},
            files={
                "reference_audio": ("test.txt", io.BytesIO(text_content), "text/plain")
            }
        )
        
        assert response.status_code == 400
        assert "audio file" in response.json()['detail'].lower()
    
    def test_list_speakers(self, client, mock_services):
        """Test speaker list endpoint."""
        response = client.get("/api/v1/voice/speakers")
        assert response.status_code == 200
        
        data = response.json()
        assert 'speakers' in data
        assert isinstance(data['speakers'], list)
        
        # Verify service was called
        mock_services['voice_service'].get_speaker_list.assert_called_once()


class TestAudioEndpoints:
    """Test audio file serving endpoints."""
    
    def test_get_audio_file_not_found(self, client, mock_services):
        """Test getting non-existent audio file."""
        # Mock repository to return None
        mock_services['repository'].get_record.return_value = None
        
        response = client.get("/api/v1/audio/999")
        assert response.status_code == 404
    
    def test_get_audio_file_success(self, client, mock_services):
        """Test successful audio file retrieval."""
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b"fake audio data")
            tmp_path = tmp.name
        
        try:
            # Mock repository to return record with existing file path
            record = AudioRecordData(
                id=123,
                prompt="test",
                audio_path=tmp_path,
                duration_seconds=10.0,
                sample_rate=48000,
                model_name="test",
                temperature=0.8,
                generation_time_ms=1000.0,
                metadata={},
                tags=[]
            )
            mock_services['repository'].get_record.return_value = record
            
            response = client.get("/api/v1/audio/123")
            assert response.status_code == 200
            assert response.headers['content-type'] == 'audio/wav'
            
        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint(self, client, mock_services):
        """Test root API information endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data['name'] == "Fugatto Audio Lab API"
        assert 'version' in data
        assert 'endpoints' in data
        assert data['docs_url'] == "/docs"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_internal_server_error(self, client, mock_services):
        """Test internal server error handling."""
        # Make the service raise an exception
        mock_services['audio_service'].generate_audio.side_effect = Exception("Test error")
        
        request_data = {
            "prompt": "Test prompt",
            "duration_seconds": 5.0
        }
        
        response = client.post("/api/v1/generate/", json=request_data)
        assert response.status_code == 500
        
        data = response.json()
        assert 'error' in data
        assert data['type'] == 'internal_error'
    
    def test_not_found_endpoint(self, client, mock_services):
        """Test 404 for non-existent endpoint."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for the API."""
    
    def test_full_generation_workflow(self, client, mock_services):
        """Test complete generation workflow."""
        # 1. Generate audio
        gen_response = client.post("/api/v1/generate/", json={
            "prompt": "A dog barking",
            "duration_seconds": 3.0,
            "save_to_db": True
        })
        assert gen_response.status_code == 200
        generation_id = gen_response.json()['generation_id']
        
        # 2. Check history
        history_response = client.get("/api/v1/generate/history")
        assert history_response.status_code == 200
        
        # 3. Try to get audio file (will fail in test due to mocking)
        audio_response = client.get(f"/api/v1/audio/{generation_id}")
        # This may succeed or fail depending on mocking setup
    
    def test_voice_cloning_workflow(self, client, mock_services):
        """Test complete voice cloning workflow."""
        audio_content = b"fake reference audio"
        
        # 1. Clone voice
        clone_response = client.post(
            "/api/v1/voice/clone",
            data={"text": "Test cloning", "speaker_id": "new_speaker"},
            files={"reference_audio": ("ref.wav", io.BytesIO(audio_content), "audio/wav")}
        )
        assert clone_response.status_code == 200
        
        # 2. List speakers
        speakers_response = client.get("/api/v1/voice/speakers")
        assert speakers_response.status_code == 200


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])