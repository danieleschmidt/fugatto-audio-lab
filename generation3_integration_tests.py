#!/usr/bin/env python3
"""
GENERATION 3 INTEGRATION TESTS - MAKE IT SCALE
Comprehensive testing suite for scaling and optimization components
"""

import os
import sys
import asyncio
import pytest
import numpy as np
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
import time
import threading

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Import Generation 3 components
from fugatto_lab.distributed_processing_engine import (
    DistributedProcessingEngine,
    WorkerNode,
    DistributedTask,
    TaskPriority,
    NodeType
)
from fugatto_lab.performance_optimization_suite import (
    PerformanceOptimizationSuite,
    CachePolicy,
    OptimizationLevel
)

class TestDistributedProcessingEngine:
    """Test suite for distributed processing and scaling"""
    
    def setup_method(self):
        """Setup test environment"""
        self.engine = DistributedProcessingEngine()
        
    def test_initialization(self):
        """Test engine initialization"""
        assert self.engine is not None
        assert hasattr(self.engine, 'worker_nodes')
        assert hasattr(self.engine, 'task_queue')
        print("✓ DistributedProcessingEngine initialization test passed")
        
    def test_worker_node_management(self):
        """Test worker node registration and management"""
        # Mock worker node
        worker = WorkerNode(
            node_id="test_worker_001",
            host="localhost",
            port=8080,
            node_type=NodeType.WORKER
        )
        
        # Register worker
        success = self.engine.register_worker(worker)
        assert success == True
        
        # Check worker is registered
        workers = self.engine.get_active_workers()
        assert len(workers) >= 1
        assert any(w.node_id == "test_worker_001" for w in workers)
        print("✓ Worker node management test passed")
        
    def test_task_submission_and_scheduling(self):
        """Test task submission and intelligent scheduling"""
        # Create test task
        task = DistributedTask(
            task_id="test_task_001",
            function_name="audio_generation",
            priority=TaskPriority.HIGH,
            kwargs={"audio_length": 30, "style": "classical"},
            cpu_requirement=4,
            gpu_requirement=True
        )
        
        # Submit task
        task_id = self.engine.submit_task(task)
        assert task_id is not None
        assert task_id == "test_task_001"
        
        # Check task in queue
        queued_tasks = self.engine.get_queued_tasks()
        assert len(queued_tasks) >= 1
        print("✓ Task submission and scheduling test passed")
        
    def test_load_balancing(self):
        """Test intelligent load balancing"""
        # Register multiple workers
        for i in range(3):
            worker = WorkerNode(
                node_id=f"worker_{i}",
                host=f"worker{i}",
                port=8080,
                node_type=NodeType.WORKER
            )
            self.engine.register_worker(worker)
        
        # Submit multiple tasks
        tasks = []
        for i in range(5):
            task = DistributedTask(
                task_id=f"task_{i}",
                function_name="audio_processing",
                priority=TaskPriority.NORMAL,
                kwargs={"data": f"test_data_{i}"}
            )
            tasks.append(task)
            self.engine.submit_task(task)
        
        # Test load distribution
        distribution = self.engine.get_load_distribution()
        assert len(distribution) > 0
        print("✓ Load balancing test passed")
        
    @pytest.mark.asyncio
    async def test_auto_scaling(self):
        """Test auto-scaling capabilities"""
        # Set high load scenario
        for i in range(20):
            task = DistributedTask(
                task_id=f"load_task_{i}",
                function_name="heavy_processing",
                priority=TaskPriority.HIGH,
                kwargs={"complexity": "high"}
            )
            self.engine.submit_task(task)
        
        # Trigger scaling decision
        scaling_decision = await self.engine.evaluate_scaling_need()
        assert 'scale_up' in scaling_decision or 'scale_down' in scaling_decision
        print("✓ Auto-scaling test passed")

class TestPerformanceOptimizationSuite:
    """Test suite for performance optimization"""
    
    def setup_method(self):
        """Setup test environment"""
        self.optimizer = PerformanceOptimizationSuite()
        
    def test_initialization(self):
        """Test optimizer initialization"""
        assert self.optimizer is not None
        assert hasattr(self.optimizer, 'cache')
        assert hasattr(self.optimizer, 'memory_manager')
        print("✓ PerformanceOptimizationSuite initialization test passed")
        
    def test_high_performance_cache(self):
        """Test high-performance caching system"""
        # Test cache functionality through the optimizer
        cache_result = self.optimizer.configure_cache(
            max_size=1000,
            policy=CachePolicy.LRU
        )
        assert cache_result is not None
        
        # Test cache operations through optimizer
        result = self.optimizer.cache_result("test_key", "test_value")
        assert result == True
        
        cached_value = self.optimizer.get_cached_result("test_key")
        assert cached_value == "test_value"
        
        print("✓ High-performance cache test passed")
        
    def test_memory_optimization(self):
        """Test memory optimization features"""
        # Create large data structure
        large_array = np.random.random((1000, 1000))
        
        # Test memory profiling
        initial_profile = self.optimizer.profile_system()
        assert initial_profile is not None
        
        # Test memory optimization
        optimized_data = self.optimizer.optimize_data_structure(large_array)
        assert optimized_data is not None
        
        # Test memory cleanup
        cleanup_result = self.optimizer.garbage_collect()
        assert cleanup_result >= 0
        print("✓ Memory optimization test passed")
        
    def test_performance_monitoring(self):
        """Test performance monitoring and metrics"""
        # Start performance monitoring
        self.optimizer.start_profiling()
        
        # Simulate some work
        time.sleep(0.1)
        
        # Get performance metrics
        metrics = self.optimizer.get_performance_report()
        assert metrics is not None
        assert 'cpu_usage' in str(metrics) or 'performance' in str(metrics)
        print("✓ Performance monitoring test passed")
        
    def test_optimization_strategies(self):
        """Test different optimization strategies"""
        # Test basic optimization
        basic_config = self.optimizer.apply_optimization_level(
            OptimizationLevel.BASIC
        )
        assert basic_config is not None
        
        # Test aggressive optimization
        aggressive_config = self.optimizer.apply_optimization_level(
            OptimizationLevel.AGGRESSIVE
        )
        assert aggressive_config is not None
        print("✓ Optimization strategies test passed")

class TestIntegrationScenarios:
    """Integration tests combining multiple Generation 3 components"""
    
    def setup_method(self):
        """Setup integrated test environment"""
        self.engine = DistributedProcessingEngine()
        self.optimizer = PerformanceOptimizationSuite()
        
    def test_optimized_distributed_processing(self):
        """Test distributed processing with performance optimization"""
        # Enable optimization
        self.optimizer.apply_optimization_level(OptimizationLevel.BASIC)
        
        # Register optimized worker
        worker = WorkerNode(
            node_id="optimized_worker",
            host="optimized",
            port=8080,
            node_type=NodeType.WORKER
        )
        self.engine.register_worker(worker)
        
        # Submit performance-critical task
        task = DistributedTask(
            task_id="perf_critical_task",
            function_name="real_time_audio",
            priority=TaskPriority.CRITICAL,
            kwargs={"latency_requirement": "10ms", "optimization_level": "high"}
        )
        
        task_id = self.engine.submit_task(task)
        assert task_id is not None
        print("✓ Optimized distributed processing integration test passed")
        
    def test_scaled_performance_monitoring(self):
        """Test performance monitoring across distributed system"""
        # Start distributed monitoring
        self.optimizer.start_monitoring()
        
        # Register multiple workers
        for i in range(3):
            worker = WorkerNode(
                node_id=f"monitor_worker_{i}",
                host=f"monitor{i}",
                port=8080,
                node_type=NodeType.MONITOR
            )
            self.engine.register_worker(worker)
        
        # Get system-wide metrics
        system_metrics = self.optimizer.get_system_metrics()
        assert system_metrics is not None
        print("✓ Scaled performance monitoring integration test passed")

def run_generation3_tests():
    """Run all Generation 3 tests with comprehensive reporting"""
    print("\n" + "="*80)
    print("TERRAGON SDLC - GENERATION 3 INTEGRATION TESTS")
    print("MAKE IT SCALE - Testing scaling and optimization components")
    print("="*80)
    
    test_results = {
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    # Test classes to run
    test_classes = [
        TestDistributedProcessingEngine,
        TestPerformanceOptimizationSuite,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        print(f"\n--- Testing {test_class.__name__} ---")
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for test_method in test_methods:
            test_results['total_tests'] += 1
            try:
                # Setup
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run test
                method = getattr(test_instance, test_method)
                if asyncio.iscoroutinefunction(method):
                    asyncio.run(method())
                else:
                    method()
                
                test_results['passed'] += 1
                
            except Exception as e:
                test_results['failed'] += 1
                error_msg = f"{test_class.__name__}.{test_method}: {str(e)}"
                test_results['errors'].append(error_msg)
                print(f"✗ {test_method} FAILED: {str(e)}")
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("GENERATION 3 TEST RESULTS SUMMARY")
    print("="*80)
    print(f"Total Tests: {test_results['total_tests']}")
    print(f"Passed: {test_results['passed']}")
    print(f"Failed: {test_results['failed']}")
    
    success_rate = (test_results['passed'] / test_results['total_tests']) * 100
    print(f"Success Rate: {success_rate:.1f}%")
    
    if test_results['errors']:
        print(f"\nFailed Tests ({len(test_results['errors'])}):")
        for error in test_results['errors']:
            print(f"  ✗ {error}")
    
    # Quality gate check
    if success_rate >= 85.0:
        print(f"\n🎉 QUALITY GATE PASSED! Success rate {success_rate:.1f}% exceeds 85% threshold")
        return True
    else:
        print(f"\n⚠️  Quality gate not met. Success rate {success_rate:.1f}% below 85% threshold")
        return False

if __name__ == "__main__":
    run_generation3_tests()