"""Tests for the database layer of Fugatto Audio Lab."""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from fugatto_lab.database import (
    DatabaseManager, get_db_manager, AudioRepository, 
    SessionRepository, ExperimentRepository
)
from fugatto_lab.database.models import AudioRecordData, UserSessionData, ExperimentRunData


class TestDatabaseManager:
    """Test the DatabaseManager class."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        yield f"sqlite:///{db_path}"
        
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    def test_initialization_sqlite(self, temp_db_path):
        """Test database manager initialization with SQLite."""
        db_manager = DatabaseManager(temp_db_path)
        db_manager.initialize()
        
        assert db_manager._initialized == True
        assert db_manager.database_url == temp_db_path
    
    def test_health_status_healthy(self, temp_db_path):
        """Test health status when database is healthy."""
        db_manager = DatabaseManager(temp_db_path)
        db_manager.initialize()
        
        health = db_manager.get_health_status()
        
        assert health['status'] == 'healthy'
        assert 'response_time_ms' in health
        assert health['initialized'] == True
    
    def test_health_status_unhealthy(self):
        """Test health status when database is unhealthy."""
        # Use invalid database URL
        db_manager = DatabaseManager("sqlite:///nonexistent/path/db.sqlite")
        
        health = db_manager.get_health_status()
        
        assert health['status'] == 'unhealthy'
        assert 'error' in health
    
    def test_get_stats(self, temp_db_path):
        """Test database statistics retrieval."""
        db_manager = DatabaseManager(temp_db_path)
        db_manager.initialize()
        
        stats = db_manager.get_stats()
        
        assert isinstance(stats, dict)
        assert 'audio_records_count' in stats
        assert 'user_sessions_count' in stats
        assert 'experiment_runs_count' in stats
        
        # All counts should be 0 for new database
        assert stats['audio_records_count'] == 0
        assert stats['user_sessions_count'] == 0
        assert stats['experiment_runs_count'] == 0
    
    def test_execute_query(self, temp_db_path):
        """Test direct query execution."""
        db_manager = DatabaseManager(temp_db_path)
        db_manager.initialize()
        
        # Test simple query
        result = db_manager.execute_query("SELECT 1 as test_value")
        
        assert len(result) == 1
        # Note: result format depends on SQLAlchemy availability
    
    def test_cleanup_old_records(self, temp_db_path):
        """Test cleanup of old records."""
        db_manager = DatabaseManager(temp_db_path)
        db_manager.initialize()
        
        # Should not fail even with empty tables
        deleted_count = db_manager.cleanup_old_records('audio_records', days_old=30)
        assert deleted_count == 0
    
    def test_session_context_manager(self, temp_db_path):
        """Test database session context manager."""
        db_manager = DatabaseManager(temp_db_path)
        db_manager.initialize()
        
        # Test successful session
        with db_manager.get_session() as session:
            # Session should be valid
            assert session is not None
        
        # Test session with exception
        try:
            with db_manager.get_session() as session:
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected


class TestAudioRepository:
    """Test the AudioRepository class."""
    
    @pytest.fixture
    def temp_db_and_repo(self):
        """Create temporary database and repository."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        db_url = f"sqlite:///{db_path}"
        db_manager = DatabaseManager(db_url)
        db_manager.initialize()
        
        repo = AudioRepository(db_manager)
        
        yield repo
        
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    def test_create_record(self, temp_db_and_repo):
        """Test creating an audio record."""
        repo = temp_db_and_repo
        
        record_data = AudioRecordData(
            prompt="A cat meowing",
            audio_path="/tmp/test.wav",
            duration_seconds=5.0,
            sample_rate=48000,
            model_name="test-model",
            temperature=0.8,
            generation_time_ms=1000.0,
            metadata={"quality": "high"},
            tags=["test", "cat"]
        )
        
        record_id = repo.create_record(record_data)
        
        assert isinstance(record_id, int)
        assert record_id > 0
    
    def test_get_record(self, temp_db_and_repo):
        """Test retrieving an audio record."""
        repo = temp_db_and_repo
        
        # Create a record first
        record_data = AudioRecordData(
            prompt="Test prompt",
            audio_path="/tmp/test.wav",
            duration_seconds=3.0,
            sample_rate=48000,
            model_name="test-model",
            temperature=0.7,
            generation_time_ms=500.0,
            metadata={},
            tags=[]
        )
        
        record_id = repo.create_record(record_data)
        
        # Retrieve the record
        retrieved = repo.get_record(record_id)
        
        assert retrieved is not None
        assert retrieved.id == record_id
        assert retrieved.prompt == "Test prompt"
        assert retrieved.duration_seconds == 3.0
        assert retrieved.temperature == 0.7
    
    def test_get_nonexistent_record(self, temp_db_and_repo):
        """Test retrieving a non-existent record."""
        repo = temp_db_and_repo
        
        result = repo.get_record(999)
        assert result is None
    
    def test_list_records(self, temp_db_and_repo):
        """Test listing audio records."""
        repo = temp_db_and_repo
        
        # Create multiple records
        for i in range(5):
            record_data = AudioRecordData(
                prompt=f"Test prompt {i}",
                audio_path=f"/tmp/test{i}.wav",
                duration_seconds=float(i + 1),
                sample_rate=48000,
                model_name="test-model",
                temperature=0.8,
                generation_time_ms=100.0 * i,
                metadata={},
                tags=[]
            )
            repo.create_record(record_data)
        
        # Test basic listing
        records = repo.list_records(limit=3)
        assert len(records) <= 3
        
        # Test pagination
        page1 = repo.list_records(limit=2, offset=0)
        page2 = repo.list_records(limit=2, offset=2)
        
        assert len(page1) == 2
        assert len(page2) == 2
        
        # Ensure different records
        page1_ids = [r.id for r in page1]
        page2_ids = [r.id for r in page2]
        assert not set(page1_ids).intersection(set(page2_ids))
    
    def test_search_records(self, temp_db_and_repo):
        """Test searching audio records."""
        repo = temp_db_and_repo
        
        # Create records with different prompts
        prompts = ["A cat meowing loudly", "A dog barking", "Cat and dog playing"]
        for prompt in prompts:
            record_data = AudioRecordData(
                prompt=prompt,
                audio_path="/tmp/test.wav",
                duration_seconds=5.0,
                sample_rate=48000,
                model_name="test-model",
                temperature=0.8,
                generation_time_ms=1000.0,
                metadata={},
                tags=[]
            )
            repo.create_record(record_data)
        
        # Search for records containing "cat"
        cat_records = repo.search_records("cat", limit=10)
        
        assert len(cat_records) == 2  # Should find 2 records with "cat"
        for record in cat_records:
            assert "cat" in record.prompt.lower()
    
    def test_get_statistics(self, temp_db_and_repo):
        """Test getting repository statistics."""
        repo = temp_db_and_repo
        
        # Create some test records
        for i in range(3):
            record_data = AudioRecordData(
                prompt=f"Test {i}",
                audio_path="/tmp/test.wav",
                duration_seconds=5.0 + i,
                sample_rate=48000,
                model_name=f"model-{i % 2}",  # Two different models
                temperature=0.8,
                generation_time_ms=1000.0 + i * 100,
                metadata={},
                tags=[]
            )
            repo.create_record(record_data)
        
        stats = repo.get_statistics()
        
        assert stats['total_records'] == 3
        assert stats['avg_duration'] is not None
        assert stats['avg_generation_time'] is not None
        assert stats['unique_models'] == 2
    
    def test_delete_record(self, temp_db_and_repo):
        """Test deleting an audio record."""
        repo = temp_db_and_repo
        
        # Create a record
        record_data = AudioRecordData(
            prompt="To be deleted",
            audio_path="/tmp/test.wav",
            duration_seconds=5.0,
            sample_rate=48000,
            model_name="test-model",
            temperature=0.8,
            generation_time_ms=1000.0,
            metadata={},
            tags=[]
        )
        
        record_id = repo.create_record(record_data)
        
        # Verify it exists
        assert repo.get_record(record_id) is not None
        
        # Delete it
        deleted = repo.delete_record(record_id)
        assert deleted == True
        
        # Verify it's gone
        assert repo.get_record(record_id) is None
        
        # Try to delete non-existent record
        not_deleted = repo.delete_record(999)
        assert not_deleted == False


class TestSessionRepository:
    """Test the SessionRepository class."""
    
    @pytest.fixture
    def temp_db_and_repo(self):
        """Create temporary database and session repository."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        db_url = f"sqlite:///{db_path}"
        db_manager = DatabaseManager(db_url)
        db_manager.initialize()
        
        repo = SessionRepository(db_manager)
        
        yield repo
        
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    def test_create_session(self, temp_db_and_repo):
        """Test creating a user session."""
        repo = temp_db_and_repo
        
        session_data = UserSessionData(
            session_id="test_session_123",
            user_agent="Test Browser 1.0",
            ip_address="192.168.1.1",
            session_data={"theme": "dark", "language": "en"}
        )
        
        session_id = repo.create_session(session_data)
        
        assert isinstance(session_id, int)
        assert session_id > 0
    
    def test_get_session(self, temp_db_and_repo):
        """Test retrieving a user session."""
        repo = temp_db_and_repo
        
        # Create session
        session_data = UserSessionData(
            session_id="retrieve_test",
            user_agent="Test Browser",
            ip_address="127.0.0.1",
            session_data={"key": "value"}
        )
        
        repo.create_session(session_data)
        
        # Retrieve session
        retrieved = repo.get_session("retrieve_test")
        
        assert retrieved is not None
        assert retrieved.session_id == "retrieve_test"
        assert retrieved.user_agent == "Test Browser"
        assert retrieved.session_data == {"key": "value"}
    
    def test_update_session_activity(self, temp_db_and_repo):
        """Test updating session activity timestamp."""
        repo = temp_db_and_repo
        
        session_data = UserSessionData(session_id="activity_test")
        repo.create_session(session_data)
        
        # Update activity
        updated = repo.update_session_activity("activity_test")
        assert updated == True
        
        # Try to update non-existent session
        not_updated = repo.update_session_activity("nonexistent")
        assert not_updated == False
    
    def test_deactivate_session(self, temp_db_and_repo):
        """Test deactivating a session."""
        repo = temp_db_and_repo
        
        session_data = UserSessionData(session_id="deactivate_test")
        repo.create_session(session_data)
        
        # Deactivate session
        deactivated = repo.deactivate_session("deactivate_test")
        assert deactivated == True
        
        # Try to deactivate non-existent session
        not_deactivated = repo.deactivate_session("nonexistent")
        assert not_deactivated == False


class TestExperimentRepository:
    """Test the ExperimentRepository class."""
    
    @pytest.fixture
    def temp_db_and_repo(self):
        """Create temporary database and experiment repository."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        db_url = f"sqlite:///{db_path}"
        db_manager = DatabaseManager(db_url)
        db_manager.initialize()
        
        repo = ExperimentRepository(db_manager)
        
        yield repo
        
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    def test_create_experiment(self, temp_db_and_repo):
        """Test creating an experiment run."""
        repo = temp_db_and_repo
        
        experiment_data = ExperimentRunData(
            experiment_name="test_generation",
            parameters={"temperature": 0.8, "duration": 10.0},
            metrics={"setup_time": 0.5}
        )
        
        experiment_id = repo.create_experiment(experiment_data)
        
        assert isinstance(experiment_id, int)
        assert experiment_id > 0
    
    def test_update_experiment_status(self, temp_db_and_repo):
        """Test updating experiment status."""
        repo = temp_db_and_repo
        
        # Create experiment
        experiment_data = ExperimentRunData(
            experiment_name="status_test",
            parameters={"param": "value"}
        )
        
        experiment_id = repo.create_experiment(experiment_data)
        
        # Update to completed
        results = {"generated_files": 5, "avg_quality": 4.2}
        updated = repo.update_experiment_status(
            experiment_id, 
            status="completed", 
            results=results
        )
        
        assert updated == True
        
        # Verify update
        experiment = repo.get_experiment(experiment_id)
        assert experiment.status == "completed"
        assert experiment.results == results
        assert experiment.completed_at is not None
    
    def test_update_experiment_failed(self, temp_db_and_repo):
        """Test updating experiment to failed status."""
        repo = temp_db_and_repo
        
        experiment_data = ExperimentRunData(
            experiment_name="fail_test",
            parameters={}
        )
        
        experiment_id = repo.create_experiment(experiment_data)
        
        # Update to failed
        updated = repo.update_experiment_status(
            experiment_id,
            status="failed",
            error_message="Test error occurred"
        )
        
        assert updated == True
        
        # Verify update
        experiment = repo.get_experiment(experiment_id)
        assert experiment.status == "failed"
        assert experiment.error_message == "Test error occurred"
    
    def test_get_experiment(self, temp_db_and_repo):
        """Test retrieving an experiment."""
        repo = temp_db_and_repo
        
        experiment_data = ExperimentRunData(
            experiment_name="retrieve_test",
            parameters={"test": True},
            results={"success": True}
        )
        
        experiment_id = repo.create_experiment(experiment_data)
        
        # Retrieve experiment
        retrieved = repo.get_experiment(experiment_id)
        
        assert retrieved is not None
        assert retrieved.id == experiment_id
        assert retrieved.experiment_name == "retrieve_test"
        assert retrieved.parameters == {"test": True}
    
    def test_list_experiments(self, temp_db_and_repo):
        """Test listing experiments with filters."""
        repo = temp_db_and_repo
        
        # Create multiple experiments
        experiments = [
            ("exp1", "running", {"param1": 1}),
            ("exp1", "completed", {"param1": 2}),
            ("exp2", "running", {"param2": 3}),
        ]
        
        for name, status, params in experiments:
            exp_data = ExperimentRunData(
                experiment_name=name,
                parameters=params,
                status=status
            )
            exp_id = repo.create_experiment(exp_data)
            
            # Update status if not running
            if status != "running":
                repo.update_experiment_status(exp_id, status)
        
        # Test listing all experiments
        all_experiments = repo.list_experiments()
        assert len(all_experiments) == 3
        
        # Test filtering by experiment name
        exp1_experiments = repo.list_experiments(experiment_name="exp1")
        assert len(exp1_experiments) == 2
        
        # Test filtering by status
        running_experiments = repo.list_experiments(status="running")
        assert len(running_experiments) == 2  # exp1 and exp2 both have running instances


class TestDatabaseIntegration:
    """Integration tests for the database layer."""
    
    def test_full_workflow(self):
        """Test complete database workflow."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db_url = f"sqlite:///{db_path}"
            db_manager = DatabaseManager(db_url)
            db_manager.initialize()
            
            # Test audio repository
            audio_repo = AudioRepository(db_manager)
            record_data = AudioRecordData(
                prompt="Integration test",
                audio_path="/tmp/integration.wav",
                duration_seconds=5.0,
                sample_rate=48000,
                model_name="integration-model",
                temperature=0.8,
                generation_time_ms=1000.0,
                metadata={"test": True},
                tags=["integration"]
            )
            
            audio_id = audio_repo.create_record(record_data)
            assert audio_id > 0
            
            # Test session repository
            session_repo = SessionRepository(db_manager)
            session_data = UserSessionData(
                session_id="integration_session",
                session_data={"integration": True}
            )
            
            session_id = session_repo.create_session(session_data)
            assert session_id > 0
            
            # Test experiment repository
            exp_repo = ExperimentRepository(db_manager)
            exp_data = ExperimentRunData(
                experiment_name="integration_experiment",
                parameters={"integration": True}
            )
            
            exp_id = exp_repo.create_experiment(exp_data)
            assert exp_id > 0
            
            # Test database stats
            stats = db_manager.get_stats()
            assert stats['audio_records_count'] == 1
            assert stats['user_sessions_count'] == 1
            assert stats['experiment_runs_count'] == 1
            
        finally:
            Path(db_path).unlink(missing_ok=True)
    
    def test_global_db_manager(self):
        """Test global database manager singleton."""
        manager1 = get_db_manager()
        manager2 = get_db_manager()
        
        # Should be the same instance
        assert manager1 is manager2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])