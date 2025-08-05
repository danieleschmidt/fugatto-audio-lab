"""Tests for Quantum Task Planner."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch

from fugatto_lab.quantum_planner import (
    QuantumTaskPlanner, QuantumTask, TaskPriority, QuantumResourceManager,
    create_audio_generation_pipeline, run_quantum_audio_pipeline
)


class TestQuantumTask:
    """Test QuantumTask functionality."""
    
    def test_task_creation(self):
        """Test basic task creation."""
        task = QuantumTask(
            id="test_task",
            name="Test Task",
            description="A test task",
            priority=TaskPriority.HIGH,
            estimated_duration=10.0
        )
        
        assert task.id == "test_task"
        assert task.name == "Test Task"
        assert task.priority == TaskPriority.HIGH
        assert task.estimated_duration == 10.0
        assert not task.is_completed
        assert task.is_ready  # Should be ready initially
    
    def test_quantum_state_initialization(self):
        """Test quantum state initialization."""
        task = QuantumTask(
            id="test",
            name="Test",
            description="Test",
            priority=TaskPriority.MEDIUM,
            estimated_duration=5.0
        )
        
        # Check quantum state properties
        assert "ready" in task.quantum_state
        assert "waiting" in task.quantum_state
        assert "blocked" in task.quantum_state
        assert "completed" in task.quantum_state
        
        # State probabilities should sum close to 1
        total_prob = sum(task.quantum_state.values())
        assert 0.95 <= total_prob <= 1.05
    
    def test_state_collapse(self):
        """Test quantum state collapse."""
        task = QuantumTask(
            id="test",
            name="Test",
            description="Test",
            priority=TaskPriority.MEDIUM,
            estimated_duration=5.0
        )
        
        collapsed_state = task.collapse_state()
        assert collapsed_state in ["ready", "waiting", "blocked", "completed"]
    
    def test_dependencies_affect_state(self):
        """Test that dependencies affect quantum state."""
        task_with_deps = QuantumTask(
            id="test",
            name="Test",
            description="Test",
            priority=TaskPriority.MEDIUM,
            estimated_duration=5.0,
            dependencies=["dep1", "dep2"]
        )
        
        task_without_deps = QuantumTask(
            id="test2",
            name="Test2",
            description="Test2",
            priority=TaskPriority.MEDIUM,
            estimated_duration=5.0
        )
        
        # Task with dependencies should have higher waiting probability
        assert task_with_deps.quantum_state["waiting"] > task_without_deps.quantum_state["waiting"]


class TestQuantumResourceManager:
    """Test QuantumResourceManager functionality."""
    
    def test_resource_allocation(self):
        """Test resource allocation and deallocation."""
        manager = QuantumResourceManager()
        
        # Test successful allocation
        requirements = {"cpu_cores": 2, "memory_gb": 4}
        success = manager.allocate_resources("task1", requirements)
        assert success
        
        # Test that resources are tracked
        assert "task1" in manager.allocated_resources
        assert manager.allocated_resources["task1"] == requirements
    
    def test_resource_limits(self):
        """Test resource allocation limits."""
        manager = QuantumResourceManager()
        
        # Allocate all CPU cores
        requirements = {"cpu_cores": 4}
        success1 = manager.allocate_resources("task1", requirements)
        assert success1
        
        # Try to allocate more than available
        requirements2 = {"cpu_cores": 2}
        success2 = manager.allocate_resources("task2", requirements2)
        assert not success2  # Should fail
    
    def test_resource_deallocation(self):
        """Test resource deallocation."""
        manager = QuantumResourceManager()
        
        requirements = {"cpu_cores": 2, "memory_gb": 4}
        manager.allocate_resources("task1", requirements)
        
        # Deallocate resources
        manager.deallocate_resources("task1")
        
        # Resources should be freed
        assert "task1" not in manager.allocated_resources
        
        # Should be able to allocate again
        success = manager.allocate_resources("task2", requirements)
        assert success
    
    def test_resource_utilization(self):
        """Test resource utilization calculation."""
        manager = QuantumResourceManager()
        
        # Initially should be 0% utilized
        utilization = manager.get_resource_utilization()
        assert utilization["cpu_cores"] == 0.0
        assert utilization["memory_gb"] == 0.0
        
        # Allocate 50% of resources
        requirements = {"cpu_cores": 2, "memory_gb": 4}
        manager.allocate_resources("task1", requirements)
        
        utilization = manager.get_resource_utilization()
        assert utilization["cpu_cores"] == 50.0
        assert utilization["memory_gb"] == 50.0


class TestQuantumTaskPlanner:
    """Test QuantumTaskPlanner functionality."""
    
    def test_planner_initialization(self):
        """Test planner initialization."""
        planner = QuantumTaskPlanner(max_concurrent_tasks=4)
        
        assert planner.max_concurrent_tasks == 4
        assert len(planner.tasks) == 0
        assert len(planner.task_queue) == 0
        assert len(planner.running_tasks) == 0
        assert len(planner.completed_tasks) == 0
    
    def test_add_task(self):
        """Test adding tasks to planner."""
        planner = QuantumTaskPlanner()
        
        task = QuantumTask(
            id="test_task",
            name="Test Task",
            description="A test task",
            priority=TaskPriority.HIGH,
            estimated_duration=10.0
        )
        
        task_id = planner.add_task(task)
        
        assert task_id == "test_task"
        assert "test_task" in planner.tasks
        assert len(planner.task_queue) == 1
    
    def test_create_audio_processing_task(self):
        """Test creating audio processing tasks."""
        planner = QuantumTaskPlanner()
        
        task = planner.create_audio_processing_task(
            name="Generate Audio",
            audio_file="test.wav",
            operation="generate",
            parameters={"prompt": "test sound", "duration": 10.0},
            priority=TaskPriority.HIGH
        )
        
        assert task.name == "Generate Audio"
        assert task.context["audio_file"] == "test.wav"
        assert task.context["operation"] == "generate"
        assert task.context["parameters"]["prompt"] == "test sound"
        assert task.priority == TaskPriority.HIGH
    
    def test_batch_processing_task_creation(self):
        """Test creating batch processing tasks."""
        planner = QuantumTaskPlanner()
        
        files = ["file1.wav", "file2.wav", "file3.wav", "file4.wav", "file5.wav"]
        tasks = planner.create_batch_processing_task(
            files=files,
            operation="enhance",
            batch_size=2
        )
        
        # Should create 3 batches (2+2+1)
        assert len(tasks) == 3
        
        # Check first batch
        assert len(tasks[0].context["files"]) == 2
        assert tasks[0].context["files"] == ["file1.wav", "file2.wav"]
        
        # Check last batch
        assert len(tasks[2].context["files"]) == 1
        assert tasks[2].context["files"] == ["file5.wav"]
        
        # Check entanglements
        assert len(tasks[1].entangled_tasks) == 1  # Should reference previous task
    
    def test_task_optimization(self):
        """Test task order optimization."""
        planner = QuantumTaskPlanner()
        
        # Add tasks with different priorities
        high_task = QuantumTask("high", "High", "High priority", TaskPriority.HIGH, 5.0)
        medium_task = QuantumTask("medium", "Medium", "Medium priority", TaskPriority.MEDIUM, 10.0)
        low_task = QuantumTask("low", "Low", "Low priority", TaskPriority.LOW, 3.0)
        
        planner.add_task(low_task)
        planner.add_task(medium_task)
        planner.add_task(high_task)
        
        # Optimize task order
        optimized_order = planner.optimize_task_order()
        
        # High priority task should be first
        assert optimized_order[0].priority == TaskPriority.HIGH
    
    @pytest.mark.asyncio
    async def test_task_execution(self):
        """Test basic task execution."""
        planner = QuantumTaskPlanner(max_concurrent_tasks=2)
        
        # Add a simple task
        task = planner.create_audio_processing_task(
            name="Test Audio",
            audio_file="test.wav",
            operation="analyze",  # Should be quick
            parameters={"duration": 0.1},  # Very short for testing
            priority=TaskPriority.HIGH
        )
        
        planner.add_task(task)
        
        # Execute tasks with short timeout
        with patch('asyncio.sleep') as mock_sleep:
            mock_sleep.return_value = asyncio.sleep(0.01)  # Speed up execution
            
            results = await asyncio.wait_for(planner.execute_tasks(), timeout=5.0)
            
            assert results["completed_tasks"] == 1
            assert results["failed_tasks"] == 0
            assert len(planner.completed_tasks) == 1
    
    def test_get_model_info(self):
        """Test getting planner information."""
        planner = QuantumTaskPlanner()
        
        info = planner.get_system_status()
        
        assert "total_tasks" in info
        assert "pending_tasks" in info
        assert "running_tasks" in info
        assert "completed_tasks" in info
        assert "resource_utilization" in info


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_audio_generation_pipeline(self):
        """Test audio generation pipeline creation."""
        prompts = ["ocean waves", "forest sounds", "city noise"]
        
        planner = create_audio_generation_pipeline(prompts, TaskPriority.HIGH)
        
        assert len(planner.tasks) == 3
        
        # All tasks should be generation tasks
        for task in planner.tasks.values():
            assert task.context["operation"] == "generate"
            assert task.priority == TaskPriority.HIGH
    
    @pytest.mark.asyncio
    async def test_run_quantum_audio_pipeline(self):
        """Test running quantum audio pipeline."""
        prompts = ["test sound"]
        planner = create_audio_generation_pipeline(prompts, TaskPriority.HIGH)
        
        # Mock the execution to be faster
        with patch('asyncio.sleep') as mock_sleep:
            mock_sleep.return_value = asyncio.sleep(0.01)
            
            results = await asyncio.wait_for(
                run_quantum_audio_pipeline(planner), 
                timeout=5.0
            )
            
            assert "execution_time" in results
            assert "completed_tasks" in results
            assert results["completed_tasks"] >= 0


class TestQuantumStateEvolution:
    """Test quantum state evolution and coherence."""
    
    def test_state_update(self):
        """Test quantum state updates."""
        task = QuantumTask(
            id="test",
            name="Test",
            description="Test",
            priority=TaskPriority.MEDIUM,
            estimated_duration=5.0
        )
        
        original_state = task.quantum_state.copy()
        
        # Update state
        new_probabilities = {"ready": 0.8, "waiting": 0.1, "blocked": 0.05, "completed": 0.05}
        task.update_quantum_state(new_probabilities)
        
        # State should be updated and normalized
        assert task.quantum_state != original_state
        total_prob = sum(task.quantum_state.values())
        assert 0.99 <= total_prob <= 1.01  # Allow for floating point precision
    
    def test_completion_detection(self):
        """Test task completion detection."""
        task = QuantumTask(
            id="test",
            name="Test",
            description="Test",
            priority=TaskPriority.MEDIUM,
            estimated_duration=5.0
        )
        
        # Initially not completed
        assert not task.is_completed
        
        # Set completion state
        task.update_quantum_state({"completed": 1.0, "ready": 0.0, "waiting": 0.0, "blocked": 0.0})
        
        # Should be detected as completed
        assert task.is_completed


@pytest.fixture
def sample_planner():
    """Create a sample planner for testing."""
    planner = QuantumTaskPlanner(max_concurrent_tasks=2)
    
    # Add some sample tasks
    task1 = planner.create_audio_processing_task(
        name="Task 1",
        audio_file="file1.wav",
        operation="generate",
        parameters={"prompt": "test1"},
        priority=TaskPriority.HIGH
    )
    
    task2 = planner.create_audio_processing_task(
        name="Task 2",
        audio_file="file2.wav",
        operation="transform",
        parameters={"prompt": "test2"},
        priority=TaskPriority.MEDIUM
    )
    
    planner.add_task(task1)
    planner.add_task(task2)
    
    return planner


class TestIntegration:
    """Integration tests for quantum planner."""
    
    def test_full_workflow(self, sample_planner):
        """Test complete workflow with multiple tasks."""
        planner = sample_planner
        
        # Check initial state
        assert len(planner.tasks) == 2
        assert len(planner.task_queue) == 2
        
        # Tasks should be properly ordered
        optimized_order = planner.optimize_task_order()
        assert len(optimized_order) == 2
        
        # High priority task should be first
        assert optimized_order[0].priority == TaskPriority.HIGH
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self, sample_planner):
        """Test concurrent task execution."""
        planner = sample_planner
        
        # Mock faster execution
        with patch('asyncio.sleep') as mock_sleep:
            mock_sleep.return_value = asyncio.sleep(0.01)
            
            start_time = time.time()
            results = await asyncio.wait_for(planner.execute_tasks(), timeout=10.0)
            execution_time = time.time() - start_time
            
            # Should complete both tasks
            assert results["completed_tasks"] == 2
            assert results["failed_tasks"] == 0
            
            # Should be reasonably fast due to concurrency
            assert execution_time < 5.0  # Much faster than sequential


if __name__ == "__main__":
    pytest.main([__file__])