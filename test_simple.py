#!/usr/bin/env python3
"""Simple test script without external dependencies."""

import sys
import os
import time
import asyncio

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_imports():
    """Test that our modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from fugatto_lab.quantum_planner import QuantumTask, TaskPriority
        print("✅ quantum_planner imported successfully")
        
        from fugatto_lab.intelligent_scheduler import SchedulingStrategy, TaskProfile
        print("✅ intelligent_scheduler imported successfully")
        
        from fugatto_lab.robust_error_handling import ValidationError, InputValidator
        print("✅ robust_error_handling imported successfully")
        
        from fugatto_lab.advanced_monitoring import MetricType, MetricsCollector
        print("✅ advanced_monitoring imported successfully")
        
        from fugatto_lab.security_framework import SecurityLevel, SecurityContext
        print("✅ security_framework imported successfully")
        
        from fugatto_lab.performance_optimization import CachePolicy, HighPerformanceCache
        print("✅ performance_optimization imported successfully")
        
        from fugatto_lab.auto_scaling import ScalingPolicy, WorkerInstance
        print("✅ auto_scaling imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_quantum_task():
    """Test QuantumTask functionality."""
    print("\n🧪 Testing QuantumTask...")
    
    try:
        from fugatto_lab.quantum_planner import QuantumTask, TaskPriority
        
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
        
        # Test quantum state
        assert "ready" in task.quantum_state
        assert "waiting" in task.quantum_state
        assert "blocked" in task.quantum_state
        assert "completed" in task.quantum_state
        
        # Test state collapse
        collapsed = task.collapse_state()
        assert collapsed in ["ready", "waiting", "blocked", "completed"]
        
        print("✅ QuantumTask works correctly")
        return True
        
    except Exception as e:
        print(f"❌ QuantumTask test failed: {e}")
        return False

def test_cache():
    """Test HighPerformanceCache functionality."""
    print("\n🧪 Testing HighPerformanceCache...")
    
    try:
        from fugatto_lab.performance_optimization import HighPerformanceCache, CachePolicy
        
        cache = HighPerformanceCache(
            max_size_mb=10,
            max_entries=100,
            policy=CachePolicy.LRU,
            enable_persistence=False
        )
        
        # Test put/get
        success = cache.put("key1", "value1")
        assert success is True
        
        value = cache.get("key1")
        assert value == "value1"
        
        # Test miss
        miss_value = cache.get("nonexistent")
        assert miss_value is None
        
        # Test stats
        stats = cache.get_stats()
        assert stats["entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        
        cache.shutdown()
        
        print("✅ HighPerformanceCache works correctly")
        return True
        
    except Exception as e:
        print(f"❌ HighPerformanceCache test failed: {e}")
        return False

def test_input_validator():
    """Test InputValidator functionality."""
    print("\n🧪 Testing InputValidator...")
    
    try:
        from fugatto_lab.robust_error_handling import InputValidator, ValidationError
        
        validator = InputValidator()
        
        # Test string sanitization
        clean_string = validator.sanitize_string("Hello <script>alert('xss')</script> World")
        assert "<script>" not in clean_string
        assert "Hello" in clean_string
        assert "World" in clean_string
        
        # Test filename sanitization
        clean_filename = validator.sanitize_filename("../../../etc/passwd")
        assert "../" not in clean_filename
        assert clean_filename != "../../../etc/passwd"
        
        # Test sample rate validation
        valid_sr = validator.validate_sample_rate(44100)
        assert valid_sr == 44100
        
        # Test invalid sample rate
        try:
            validator.validate_sample_rate(-1000)
            assert False, "Should have raised ValidationError"
        except ValidationError:
            pass  # Expected
        
        print("✅ InputValidator works correctly")
        return True
        
    except Exception as e:
        print(f"❌ InputValidator test failed: {e}")
        return False

def test_metrics_collector():
    """Test MetricsCollector functionality."""
    print("\n🧪 Testing MetricsCollector...")
    
    try:
        from fugatto_lab.advanced_monitoring import MetricsCollector, MetricType
        
        collector = MetricsCollector(buffer_size=1000, flush_interval=60.0)
        
        # Record some metrics
        collector.record_metric("test.counter", 1.0, MetricType.COUNTER)
        collector.set_gauge("test.gauge", 50.0)
        collector.record_timer("test.timer", 0.5)
        
        # Get stats
        stats = collector.get_all_metrics()
        assert stats["total_metrics"] == 3
        assert stats["unique_metrics"] == 3
        
        collector.shutdown()
        
        print("✅ MetricsCollector works correctly")
        return True
        
    except Exception as e:
        print(f"❌ MetricsCollector test failed: {e}")
        return False

async def test_auto_scaler():
    """Test AutoScaler functionality."""
    print("\n🧪 Testing AutoScaler...")
    
    try:
        from fugatto_lab.auto_scaling import AutoScaler, WorkerInstance, WorkerState
        
        # Create auto-scaler
        scaler = AutoScaler(
            min_workers=1,
            max_workers=5,
            target_utilization=70.0
        )
        
        # Add a worker
        worker = scaler.add_worker("test_worker", capacity=10)
        assert worker.worker_id == "test_worker"
        assert worker.capacity == 10
        assert worker.state == WorkerState.INITIALIZING
        
        # Wait a moment for initialization
        await asyncio.sleep(0.1)
        
        # Test task processing (mock)
        task_data = {
            "id": "test_task",
            "processing_time": 0.1  # Short processing time
        }
        
        # Process task
        result = await scaler.process_task(task_data)
        assert "status" in result
        assert "task_id" in result
        
        # Get status
        status = scaler.get_scaling_status()
        assert "policy" in status
        assert "current_metrics" in status
        
        scaler.shutdown()
        
        print("✅ AutoScaler works correctly")
        return True
        
    except Exception as e:
        print(f"❌ AutoScaler test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting Fugatto Audio Lab Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_quantum_task, 
        test_cache,
        test_input_validator,
        test_metrics_collector,
        test_auto_scaler
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if asyncio.iscoroutinefunction(test):
                result = await test()
            else:
                result = test()
                
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)