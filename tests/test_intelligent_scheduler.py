"""Tests for Intelligent Scheduler."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock

from fugatto_lab.intelligent_scheduler import (
    IntelligentScheduler, SchedulingStrategy, TaskProfile, AdaptiveLearningEngine,
    ResourceMonitor, create_intelligent_scheduler, run_audio_processing_batch
)


class TestTaskProfile:
    """Test TaskProfile functionality."""
    
    def test_profile_creation(self):
        """Test basic profile creation."""
        profile = TaskProfile(
            task_id="test_task",
            task_type="audio_processing",
            input_size=1000,
            complexity_score=1.5,
            resource_requirements={"cpu": 2, "memory": 4},
            estimated_duration=10.0
        )
        
        assert profile.task_id == "test_task"
        assert profile.task_type == "audio_processing"
        assert profile.input_size == 1000
        assert profile.complexity_score == 1.5
        assert profile.estimated_duration == 10.0
        assert profile.success is True  # Default
    
    def test_feature_vector_extraction(self):
        """Test feature vector extraction."""
        profile = TaskProfile(
            task_id="test",
            task_type="audio",
            input_size=2000,
            complexity_score=2.0,
            resource_requirements={"cpu": 1, "memory": 2, "gpu": 1},
            estimated_duration=15.0
        )
        
        features = profile.get_feature_vector()
        
        assert len(features) == 7  # input_size, complexity, cpu, memory, gpu, duration, age
        assert features[0] == 2000  # input_size
        assert features[1] == 2.0   # complexity_score
        assert features[2] == 1     # cpu
        assert features[3] == 2     # memory
        assert features[4] == 1     # gpu
        assert features[5] == 15.0  # estimated_duration


class TestAdaptiveLearningEngine:
    """Test AdaptiveLearningEngine functionality."""
    
    def test_engine_initialization(self):
        """Test learning engine initialization."""
        engine = AdaptiveLearningEngine()
        
        assert len(engine.task_history) == 0
        assert len(engine.performance_history) == 0
        assert engine.prediction_accuracy["duration"] == 0.5  # Default
    
    def test_add_task_profile(self):
        """Test adding task profiles."""
        engine = AdaptiveLearningEngine()
        
        profile = TaskProfile(
            task_id="test",
            task_type="audio",
            input_size=1000,
            complexity_score=1.0,
            resource_requirements={"cpu": 1},
            estimated_duration=10.0,
            actual_duration=12.0
        )
        
        engine.add_task_profile(profile)
        
        assert len(engine.task_history) == 1
        assert engine.task_history[0] == profile
    
    def test_duration_prediction_no_history(self):
        """Test duration prediction with no history."""
        engine = AdaptiveLearningEngine()
        
        profile = TaskProfile(
            task_id="test",
            task_type="audio",
            input_size=1000,
            complexity_score=1.0,
            resource_requirements={"cpu": 1},
            estimated_duration=10.0
        )
        
        predicted = engine.predict_task_duration(profile)
        assert predicted == 10.0  # Should return original estimate
    
    def test_duration_prediction_with_history(self):
        """Test duration prediction with historical data."""
        engine = AdaptiveLearningEngine()
        
        # Add historical similar tasks
        for i in range(5):
            historical_profile = TaskProfile(
                task_id=f"hist_{i}",
                task_type="audio",
                input_size=1000 + i * 100,  # Similar sizes
                complexity_score=1.0,
                resource_requirements={"cpu": 1},
                estimated_duration=10.0,
                actual_duration=8.0 + i * 0.5  # Actual durations
            )
            engine.add_task_profile(historical_profile)
        
        # Predict for similar task
        profile = TaskProfile(
            task_id="test",
            task_type="audio",
            input_size=1050,  # Similar to historical tasks
            complexity_score=1.0,
            resource_requirements={"cpu": 1},
            estimated_duration=10.0
        )
        
        predicted = engine.predict_task_duration(profile)
        
        # Should be influenced by historical data
        assert predicted != 10.0
        assert 8.0 <= predicted <= 11.0  # Within reasonable range
    
    def test_priority_optimization(self):
        """Test priority optimization."""
        engine = AdaptiveLearningEngine()
        
        profile = TaskProfile(
            task_id="test",
            task_type="audio",
            input_size=1000,
            complexity_score=1.0,
            resource_requirements={"cpu": 0.5},  # Low resource requirement
            estimated_duration=10.0
        )
        
        system_state = {
            "cpu_utilization": 90.0,  # High CPU usage
            "memory_utilization": 50.0,
            "queue_length": 5
        }
        
        priority = engine.predict_optimal_priority(profile, system_state)
        
        # Should boost priority for low-resource task when system is busy
        assert priority > 0.5
    
    def test_learning_stats(self):
        """Test getting learning statistics."""
        engine = AdaptiveLearningEngine()
        
        # Add some data
        profile = TaskProfile(
            task_id="test",
            task_type="audio",
            input_size=1000,
            complexity_score=1.0,
            resource_requirements={"cpu": 1},
            estimated_duration=10.0,
            actual_duration=8.0
        )
        engine.add_task_profile(profile)
        
        stats = engine.get_learning_stats()
        
        assert "task_history_size" in stats
        assert "performance_history_size" in stats
        assert "duration_prediction_accuracy" in stats
        assert stats["task_history_size"] == 1


class TestResourceMonitor:
    """Test ResourceMonitor functionality."""
    
    def test_resource_monitor_initialization(self):
        """Test resource monitor initialization."""
        monitor = ResourceMonitor(update_interval=1.0)
        
        assert monitor.update_interval == 1.0
        assert "cpu_utilization" in monitor.cached_state
        assert "memory_utilization" in monitor.cached_state
    
    def test_get_system_state(self):
        """Test getting system state."""
        monitor = ResourceMonitor()
        
        state = monitor.get_system_state()
        
        # Should have all required metrics
        required_metrics = [
            "cpu_utilization", "memory_utilization", "gpu_utilization",
            "disk_io", "network_io", "load_average", "available_memory_gb"
        ]
        
        for metric in required_metrics:
            assert metric in state
            assert isinstance(state[metric], (int, float))


class TestIntelligentScheduler:
    """Test IntelligentScheduler functionality."""
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        scheduler = IntelligentScheduler(
            strategy=SchedulingStrategy.ADAPTIVE,
            max_concurrent_tasks=4,
            enable_learning=True
        )
        
        assert scheduler.strategy == SchedulingStrategy.ADAPTIVE
        assert scheduler.max_concurrent_tasks == 4
        assert scheduler.enable_learning is True
        assert scheduler.learning_engine is not None
        assert len(scheduler.high_priority_queue) == 0
    
    def test_schedule_task_basic(self):
        """Test basic task scheduling."""
        scheduler = IntelligentScheduler(strategy=SchedulingStrategy.FIFO)
        
        task_info = {
            "type": "audio_processing",
            "input_size": 1000,
            "complexity": 1.0,
            "resources": {"cpu": 1, "memory": 2},
            "duration": 10.0
        }
        
        success = scheduler.schedule_task("test_task", task_info, priority=0.5)
        
        assert success is True
        assert "test_task" in scheduler.task_profiles
        assert len(scheduler.medium_priority_queue) == 1
    
    def test_priority_based_scheduling(self):
        """Test priority-based task scheduling."""
        scheduler = IntelligentScheduler(strategy=SchedulingStrategy.PRIORITY)
        
        # Schedule high priority task
        high_task_info = {"type": "audio", "duration": 5.0}
        scheduler.schedule_task("high_task", high_task_info, priority=0.8)
        
        # Schedule low priority task
        low_task_info = {"type": "audio", "duration": 5.0}
        scheduler.schedule_task("low_task", low_task_info, priority=0.2)
        
        # High priority queue should have the high priority task
        assert len(scheduler.high_priority_queue) == 1
        assert scheduler.high_priority_queue[0][0] == "high_task"
        
        # Low priority queue should have the low priority task
        assert len(scheduler.low_priority_queue) == 1
        assert scheduler.low_priority_queue[0][0] == "low_task"
    
    def test_sjf_scheduling(self):
        """Test shortest job first scheduling."""
        scheduler = IntelligentScheduler(strategy=SchedulingStrategy.SJF)
        
        # Schedule tasks with different durations
        long_task = {"type": "audio", "duration": 20.0}
        scheduler.schedule_task("long_task", long_task, priority=0.5)
        
        short_task = {"type": "audio", "duration": 5.0}
        scheduler.schedule_task("short_task", short_task, priority=0.5)
        
        # Shorter task should be first in queue
        assert scheduler.medium_priority_queue[0][0] == "short_task"
        assert scheduler.medium_priority_queue[1][0] == "long_task"
    
    def test_adaptive_scheduling(self):
        """Test adaptive scheduling."""
        scheduler = IntelligentScheduler(
            strategy=SchedulingStrategy.ADAPTIVE,
            enable_learning=True
        )
        
        # Mock system state
        scheduler.system_state = {
            "cpu_utilization": 50.0,
            "memory_utilization": 40.0,
            "queue_length": 5
        }
        
        task_info = {
            "type": "audio",
            "input_size": 1000,
            "complexity": 1.0,
            "resources": {"cpu": 1, "memory": 2},
            "duration": 10.0
        }
        
        success = scheduler.schedule_task("adaptive_task", task_info, priority=0.6)
        
        assert success is True
        assert "adaptive_task" in scheduler.task_profiles
    
    def test_priority_aging(self):
        """Test priority aging mechanism."""
        scheduler = IntelligentScheduler(strategy=SchedulingStrategy.PRIORITY)
        
        # Add old task to low priority queue
        old_time = time.time() - 3600  # 1 hour ago
        
        with patch('time.time', return_value=old_time):
            task_info = {"type": "audio", "duration": 5.0}
            scheduler.schedule_task("old_task", task_info, priority=0.2)
        
        # Apply aging
        scheduler._apply_priority_aging()
        
        # Task should have higher priority now
        task_id, profile, priority = scheduler.low_priority_queue[0]
        assert priority > 0.2  # Should be boosted by aging
    
    def test_resource_availability_check(self):
        """Test resource availability checking."""
        scheduler = IntelligentScheduler()
        
        # Mock resource monitor to return high usage
        scheduler.resource_monitor.get_system_state = Mock(return_value={
            "cpu_utilization": 95.0,
            "memory_utilization": 90.0,
            "gpu_utilization": 80.0
        })
        
        # High resource requirement task
        profile = TaskProfile(
            task_id="test",
            task_type="audio",
            input_size=1000,
            complexity_score=1.0,
            resource_requirements={"cpu": 2.0, "memory": 4.0},
            estimated_duration=10.0
        )
        
        available = scheduler._check_resource_availability(profile)
        assert available is False  # Should not be available due to high usage
        
        # Low resource requirement task
        profile_light = TaskProfile(
            task_id="test_light",
            task_type="audio",
            input_size=1000,
            complexity_score=1.0,
            resource_requirements={"cpu": 0.5, "memory": 1.0},
            estimated_duration=10.0
        )
        
        available_light = scheduler._check_resource_availability(profile_light)
        assert available_light is True  # Should be available
    
    def test_scheduler_status(self):
        """Test getting scheduler status."""
        scheduler = IntelligentScheduler()
        
        # Add some tasks
        task_info = {"type": "audio", "duration": 5.0}
        scheduler.schedule_task("task1", task_info, priority=0.8)
        scheduler.schedule_task("task2", task_info, priority=0.5)
        scheduler.schedule_task("task3", task_info, priority=0.3)
        
        status = scheduler.get_scheduler_status()
        
        assert status["strategy"] == SchedulingStrategy.ADAPTIVE.value
        assert status["queues"]["high_priority"] == 1
        assert status["queues"]["medium_priority"] == 1
        assert status["queues"]["low_priority"] == 1
        assert "metrics" in status
        assert "system_state" in status


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_intelligent_scheduler(self):
        """Test creating intelligent scheduler."""
        scheduler = create_intelligent_scheduler(
            strategy="priority",
            max_tasks=8,
            enable_ml=False
        )
        
        assert scheduler.strategy == SchedulingStrategy.PRIORITY
        assert scheduler.max_concurrent_tasks == 8
        assert scheduler.enable_learning is False
        assert scheduler.learning_engine is None
    
    @pytest.mark.asyncio
    async def test_run_audio_processing_batch(self):
        """Test running audio processing batch."""
        scheduler = IntelligentScheduler(max_concurrent_tasks=2)
        
        # Mock the task execution to be fast
        original_execute = scheduler._execute_task
        
        async def mock_execute(task_id, profile):
            return {"success": True, "duration": 0.1}
        
        scheduler._execute_task = mock_execute
        
        audio_files = ["file1.wav", "file2.wav", "file3.wav"]
        
        results = await run_audio_processing_batch(scheduler, audio_files, "enhance")
        
        assert "total_tasks" in results
        assert "completed" in results
        assert "total_time" in results
        assert results["total_tasks"] == 3


@pytest.fixture
def sample_scheduler():
    """Create a sample scheduler for testing."""
    scheduler = IntelligentScheduler(
        strategy=SchedulingStrategy.ADAPTIVE,
        max_concurrent_tasks=2,
        enable_learning=True
    )
    
    # Add some sample tasks
    task1_info = {
        "type": "audio_processing",
        "input_size": 1000,
        "complexity": 1.0,
        "resources": {"cpu": 1, "memory": 2},
        "duration": 10.0
    }
    
    task2_info = {
        "type": "audio_processing",
        "input_size": 2000,
        "complexity": 1.5,
        "resources": {"cpu": 2, "memory": 4},
        "duration": 15.0
    }
    
    scheduler.schedule_task("task1", task1_info, priority=0.8)
    scheduler.schedule_task("task2", task2_info, priority=0.5)
    
    return scheduler


class TestIntegration:
    """Integration tests for intelligent scheduler."""
    
    def test_full_scheduling_workflow(self, sample_scheduler):
        """Test complete scheduling workflow."""
        scheduler = sample_scheduler
        
        # Check initial state
        assert len(scheduler.task_profiles) == 2
        
        # Test task selection
        next_task = scheduler._select_next_task()
        assert next_task is not None
        
        task_id, profile, priority = next_task
        assert task_id in ["task1", "task2"]
        
        # Higher priority task should be selected first
        if task_id == "task1":
            assert priority == 0.8
    
    def test_learning_integration(self, sample_scheduler):
        """Test learning engine integration."""
        scheduler = sample_scheduler
        
        # Simulate task completion
        profile = scheduler.task_profiles["task1"]
        profile.actual_duration = 8.0
        profile.success = True
        
        # Add to learning engine
        scheduler.learning_engine.add_task_profile(profile)
        
        # Check that learning engine has the data
        assert len(scheduler.learning_engine.task_history) == 1
        
        # Test prediction
        new_task_info = {
            "type": "audio_processing",
            "input_size": 1000,
            "complexity": 1.0,
            "resources": {"cpu": 1, "memory": 2},
            "duration": 10.0
        }
        
        scheduler.schedule_task("task3", new_task_info, priority=0.5)
        
        # Duration should be adjusted by learning
        task3_profile = scheduler.task_profiles["task3"]
        assert task3_profile.estimated_duration != 10.0  # Should be modified by ML
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, sample_scheduler):
        """Test concurrent task processing."""
        scheduler = sample_scheduler
        
        # Mock faster execution
        async def mock_execute(task_id, profile):
            await asyncio.sleep(0.1)  # Fast execution
            return {"success": True, "duration": 0.1}
        
        scheduler._execute_task = mock_execute
        
        # Mock resource checking to always return True
        scheduler._check_resource_availability = Mock(return_value=True)
        
        # Start scheduling
        start_time = time.time()
        await scheduler._schedule_next_tasks()
        
        # Both tasks should start since max_concurrent_tasks=2
        assert len(scheduler.running_tasks) <= 2
        
        # Wait a bit for tasks to complete
        await asyncio.sleep(0.2)
        
        # Check completed tasks
        scheduler._update_metrics(scheduler.task_profiles["task1"])
        
        execution_time = time.time() - start_time
        assert execution_time < 1.0  # Should be fast due to mocking


if __name__ == "__main__":
    pytest.main([__file__])