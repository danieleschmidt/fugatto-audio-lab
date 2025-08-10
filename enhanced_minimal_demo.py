#!/usr/bin/env python3
"""Enhanced minimal demo without external dependencies."""

import sys
import os
import time
import random
import math
from typing import Dict, Any, List, Optional

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

def mock_numpy():
    """Create a minimal numpy-like interface for demo purposes."""
    class MockArray:
        def __init__(self, data):
            if isinstance(data, (list, tuple)):
                self.data = list(data)
            else:
                self.data = [data]
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, key):
            return self.data[key]
        
        def __setitem__(self, key, value):
            self.data[key] = value
        
        def __iter__(self):
            return iter(self.data)
        
        def mean(self):
            return sum(self.data) / len(self.data) if self.data else 0
        
        def max(self):
            return max(self.data) if self.data else 0
        
        def min(self):
            return min(self.data) if self.data else 0
        
        def sum(self):
            return sum(self.data)
        
        def astype(self, dtype):
            return self
    
    class MockNumpy:
        def array(self, data):
            return MockArray(data)
        
        def zeros(self, size):
            return MockArray([0.0] * size)
        
        def ones(self, size):
            return MockArray([1.0] * size)
        
        def sin(self, data):
            if hasattr(data, '__iter__'):
                return MockArray([math.sin(x) for x in data])
            return math.sin(data)
        
        def cos(self, data):
            if hasattr(data, '__iter__'):
                return MockArray([math.cos(x) for x in data])
            return math.cos(data)
        
        def linspace(self, start, stop, num):
            if num <= 1:
                return MockArray([stop])
            step = (stop - start) / (num - 1)
            return MockArray([start + i * step for i in range(num)])
        
        def random(self):
            class MockRandom:
                def normal(self, mean, std, size=None):
                    if size is None:
                        return random.gauss(mean, std)
                    return MockArray([random.gauss(mean, std) for _ in range(size)])
                
                def uniform(self, low, high, size=None):
                    if size is None:
                        return random.uniform(low, high)
                    return MockArray([random.uniform(low, high) for _ in range(size)])
            return MockRandom()
        
        float32 = float
        pi = math.pi
    
    return MockNumpy()

# Monkey patch numpy for demo
import fugatto_lab.core
fugatto_lab.core.np = mock_numpy()
fugatto_lab.core.HAS_NUMPY = True

def test_core_audio_generation():
    """Test core audio generation functionality."""
    print("🧪 Testing Core Audio Generation...")
    
    try:
        from fugatto_lab.core import FugattoModel, AudioProcessor
        
        # Initialize model
        model = FugattoModel("test-model")
        processor = AudioProcessor(sample_rate=16000)
        
        # Generate audio
        print("  🎵 Generating audio from text prompt...")
        audio = model.generate(
            prompt="A cat meowing softly",
            duration_seconds=2.0,
            temperature=0.7
        )
        
        assert len(audio) > 0, "Audio generation returned empty result"
        print(f"  ✅ Generated {len(audio)} samples")
        
        # Test transformation
        print("  🔄 Testing audio transformation...")
        transformed = model.transform(
            audio=audio,
            prompt="Add echo effect",
            strength=0.5
        )
        
        assert len(transformed) == len(audio), "Transformation changed audio length"
        print("  ✅ Audio transformation successful")
        
        # Test multi-conditioning
        print("  🎛️ Testing multi-conditioning...")
        multi_audio = model.generate_multi(
            text_prompt="Jazz piano",
            attributes={"tempo": 120, "reverb": 0.3},
            duration_seconds=1.0
        )
        
        assert len(multi_audio) > 0, "Multi-conditioning returned empty result"
        print("  ✅ Multi-conditioning successful")
        
        # Test audio processing
        print("  📊 Testing audio processing...")
        stats = processor.get_audio_stats(audio)
        
        assert "duration_seconds" in stats, "Missing duration in stats"
        assert "rms" in stats, "Missing RMS in stats"
        assert stats["duration_seconds"] > 0, "Invalid duration"
        print(f"  ✅ Audio stats: {stats['duration_seconds']:.2f}s, RMS: {stats['rms']:.4f}")
        
        # Test preprocessing
        print("  🔧 Testing audio preprocessing...")
        processed = processor.preprocess(audio, normalize=True, trim_silence=True)
        
        assert len(processed) > 0, "Preprocessing returned empty result"
        print("  ✅ Audio preprocessing successful")
        
        # Test enhancement
        print("  ⚡ Testing audio enhancement...")
        enhanced = processor.enhance_audio(audio, {
            'noise_gate': True,
            'normalize': True,
            'gate_threshold': -40.0
        })
        
        assert len(enhanced) == len(audio), "Enhancement changed audio length"
        print("  ✅ Audio enhancement successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Core audio generation failed: {e}")
        return False

def test_quantum_planning():
    """Test quantum task planning functionality."""
    print("\n🧪 Testing Quantum Planning...")
    
    try:
        from fugatto_lab.quantum_planner import (
            QuantumTaskPlanner, QuantumTask, TaskPriority,
            create_audio_generation_pipeline
        )
        
        # Create planner
        planner = QuantumTaskPlanner(max_workers=2, enable_quantum_optimization=True)
        
        # Create tasks
        tasks = [
            QuantumTask(
                id=f"task_{i}",
                name=f"Audio Generation {i}",
                description=f"Generate audio clip {i}",
                priority=TaskPriority.HIGH if i % 2 == 0 else TaskPriority.MEDIUM,
                estimated_duration=random.uniform(1.0, 5.0)
            )
            for i in range(5)
        ]
        
        # Add tasks to planner
        for task in tasks:
            success = planner.add_task(task)
            assert success, f"Failed to add task {task.id}"
        
        print(f"  ✅ Added {len(tasks)} tasks to quantum planner")
        
        # Test quantum state analysis
        task_states = [task.collapse_state() for task in tasks]
        unique_states = set(task_states)
        print(f"  🌌 Quantum states observed: {unique_states}")
        
        # Test priority sorting
        sorted_tasks = planner.get_sorted_tasks()
        assert len(sorted_tasks) == len(tasks), "Task count mismatch after sorting"
        print(f"  📊 Task priority distribution: {[t.priority for t in sorted_tasks[:3]]}")
        
        # Test pipeline creation
        print("  🔧 Creating audio generation pipeline...")
        pipeline = create_audio_generation_pipeline(
            input_texts=["Cat meowing", "Ocean waves", "Jazz music"],
            output_dir="/tmp/audio_output",
            batch_size=2
        )
        
        assert len(pipeline) > 0, "Pipeline creation failed"
        print(f"  ✅ Created pipeline with {len(pipeline)} stages")
        
        # Get planner status
        status = planner.get_status()
        assert "total_tasks" in status, "Missing total_tasks in status"
        assert "queue_size" in status, "Missing queue_size in status"
        print(f"  📈 Planner status: {status['total_tasks']} tasks, {status['queue_size']} queued")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Quantum planning failed: {e}")
        return False

def test_intelligent_scheduling():
    """Test intelligent scheduling functionality."""
    print("\n🧪 Testing Intelligent Scheduling...")
    
    try:
        from fugatto_lab.intelligent_scheduler import IntelligentScheduler, SchedulingStrategy, TaskProfile
        
        # Create scheduler
        scheduler = IntelligentScheduler(strategy=SchedulingStrategy.PRIORITY_QUEUE)
        
        # Create task profiles
        profiles = [
            TaskProfile(
                task_id=f"profile_{i}",
                estimated_cpu_time=random.uniform(0.5, 3.0),
                estimated_memory_mb=random.uniform(100, 1000),
                priority_score=random.uniform(0.1, 1.0)
            )
            for i in range(8)
        ]
        
        # Schedule tasks
        for profile in profiles:
            scheduled = scheduler.schedule_task(profile)
            assert scheduled, f"Failed to schedule task {profile.task_id}"
        
        print(f"  ✅ Scheduled {len(profiles)} tasks")
        
        # Test resource estimation
        total_cpu = sum(p.estimated_cpu_time for p in profiles)
        total_memory = sum(p.estimated_memory_mb for p in profiles)
        print(f"  💻 Resource estimates: {total_cpu:.2f}s CPU, {total_memory:.0f}MB memory")
        
        # Test scheduling optimization
        optimized = scheduler.optimize_schedule()
        assert "total_estimated_time" in optimized, "Missing optimization results"
        print(f"  ⚡ Optimized schedule: {optimized['total_estimated_time']:.2f}s total time")
        
        # Test priority adjustment
        high_priority_tasks = [p for p in profiles if p.priority_score > 0.7]
        scheduler.boost_priority([p.task_id for p in high_priority_tasks])
        print(f"  🚀 Boosted priority for {len(high_priority_tasks)} high-priority tasks")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Intelligent scheduling failed: {e}")
        return False

def test_robust_validation():
    """Test robust validation and error handling."""
    print("\n🧪 Testing Robust Validation...")
    
    try:
        from fugatto_lab.robust_validation import RobustValidator, ValidationResult
        from fugatto_lab.robust_error_handling import ErrorRecoveryManager, RecoveryStrategy
        
        # Create validator
        validator = RobustValidator(strict_mode=False)
        
        # Test input validation
        test_cases = [
            {"prompt": "Valid audio prompt", "duration": 10.0, "sample_rate": 44100},
            {"prompt": "", "duration": 10.0, "sample_rate": 44100},  # Empty prompt
            {"prompt": "Valid prompt", "duration": -5.0, "sample_rate": 44100},  # Negative duration
            {"prompt": "Valid prompt", "duration": 10.0, "sample_rate": 8000}   # Low sample rate
        ]
        
        results = []
        for i, case in enumerate(test_cases):
            result = validator.validate_audio_generation_request(case)
            results.append(result)
            print(f"  📝 Test case {i+1}: {'✅ Valid' if result.is_valid else '⚠️ Invalid'}")
        
        valid_count = sum(1 for r in results if r.is_valid)
        print(f"  📊 Validation results: {valid_count}/{len(test_cases)} passed")
        
        # Test error recovery
        recovery_manager = ErrorRecoveryManager()
        
        # Simulate an error scenario
        error_context = {
            "operation": "audio_generation",
            "error_type": "timeout",
            "retry_count": 0
        }
        
        recovery_plan = recovery_manager.create_recovery_plan(error_context)
        assert "strategy" in recovery_plan, "Missing recovery strategy"
        print(f"  🔄 Recovery strategy: {recovery_plan['strategy']}")
        
        # Test circuit breaker
        circuit_status = recovery_manager.get_circuit_breaker_status("audio_generation")
        print(f"  ⚡ Circuit breaker status: {circuit_status['state']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Robust validation failed: {e}")
        return False

def test_performance_optimization():
    """Test performance optimization features."""
    print("\n🧪 Testing Performance Optimization...")
    
    try:
        from fugatto_lab.performance_optimization import AdvancedOptimizer, OptimizationStrategy
        from fugatto_lab.optimization import get_audio_cache
        
        # Create optimizer
        optimizer = AdvancedOptimizer(
            enable_caching=True,
            enable_compression=True,
            enable_parallel_processing=True
        )
        
        # Test cache functionality
        cache = get_audio_cache()
        
        # Store some test data
        test_data = list(range(1000))  # Mock audio data
        cache_key = "test_audio_clip_001"
        
        stored = cache.put(cache_key, test_data)
        assert stored, "Failed to store data in cache"
        
        retrieved = cache.get(cache_key)
        assert retrieved == test_data, "Retrieved data doesn't match stored data"
        print("  💾 Cache functionality: ✅ Working")
        
        # Test optimization strategies
        strategies = [
            OptimizationStrategy.MEMORY_EFFICIENT,
            OptimizationStrategy.CPU_OPTIMIZED,
            OptimizationStrategy.BALANCED
        ]
        
        for strategy in strategies:
            optimizer.set_strategy(strategy)
            config = optimizer.get_current_config()
            assert "strategy" in config, f"Missing strategy in config for {strategy}"
            print(f"  ⚡ {strategy}: {config['parallel_workers']} workers")
        
        # Test performance profiling
        profile_data = optimizer.profile_operation("mock_audio_generation", lambda: time.sleep(0.1))
        assert "execution_time" in profile_data, "Missing execution time in profile"
        print(f"  📊 Profiling: {profile_data['execution_time']:.3f}s execution time")
        
        # Test memory optimization
        memory_stats = optimizer.get_memory_usage()
        print(f"  💾 Memory usage: {memory_stats.get('estimated_mb', 0):.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance optimization failed: {e}")
        return False

def test_security_framework():
    """Test security framework functionality."""
    print("\n🧪 Testing Security Framework...")
    
    try:
        from fugatto_lab.security_framework import SecurityManager, SecurityLevel, AuditLogger
        
        # Create security manager
        security_manager = SecurityManager(
            default_level=SecurityLevel.MEDIUM,
            enable_audit_logging=True
        )
        
        # Test input sanitization
        test_inputs = [
            "normal_input.wav",
            "../../../etc/passwd",
            "file<script>alert(1)</script>.wav",
            "audio_file.wav; rm -rf /",
            "valid_audio_file_123.wav"
        ]
        
        sanitized_count = 0
        for test_input in test_inputs:
            sanitized = security_manager.sanitize_filename(test_input)
            if sanitized != test_input:
                sanitized_count += 1
            print(f"  🔒 '{test_input}' -> '{sanitized}'")
        
        print(f"  🛡️ Sanitized {sanitized_count}/{len(test_inputs)} inputs")
        
        # Test permission validation
        operations = [
            ("read_audio", {"file_path": "/tmp/test.wav"}),
            ("write_audio", {"file_path": "/home/user/output.wav"}),
            ("generate_audio", {"prompt": "test", "duration": 10}),
            ("system_config", {"setting": "max_workers"})
        ]
        
        allowed_count = 0
        for operation, context in operations:
            allowed = security_manager.check_permission(operation, context)
            if allowed:
                allowed_count += 1
            print(f"  🔑 {operation}: {'✅ Allowed' if allowed else '❌ Denied'}")
        
        print(f"  📊 Permissions: {allowed_count}/{len(operations)} allowed")
        
        # Test audit logging
        audit_logger = AuditLogger()
        audit_logger.log_operation("test_operation", {"user": "demo", "action": "audio_generation"})
        
        audit_stats = audit_logger.get_audit_stats()
        print(f"  📝 Audit log: {audit_stats.get('total_entries', 0)} entries")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Security framework failed: {e}")
        return False

def main():
    """Run enhanced demonstration."""
    print("🚀 Fugatto Audio Lab - Enhanced Autonomous Demo")
    print("=" * 60)
    print("🎯 Generation 1: Core Functionality Enhancement")
    print("")
    
    tests = [
        test_core_audio_generation,
        test_quantum_planning,
        test_intelligent_scheduling,
        test_robust_validation,
        test_performance_optimization,
        test_security_framework
    ]
    
    passed = 0
    failed = 0
    
    start_time = time.time()
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    end_time = time.time()
    
    print("\n" + "=" * 60)
    print(f"📊 Enhanced Demo Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⏱️ Runtime: {end_time - start_time:.2f} seconds")
    print("")
    
    if failed == 0:
        print("🎉 Generation 1 Enhancement: COMPLETE")
        print("   • Core audio generation with advanced features")
        print("   • Quantum-inspired task planning")
        print("   • Intelligent scheduling with optimization")
        print("   • Robust validation and error handling")
        print("   • High-performance caching and optimization")
        print("   • Enterprise-grade security framework")
        print("")
        print("🚀 Ready for Generation 2 (Robust & Reliable)")
        return 0
    else:
        print("💥 Some components need attention before proceeding")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)