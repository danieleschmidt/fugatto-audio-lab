"""Tests for Performance Optimization."""

import pytest
import time
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from fugatto_lab.performance_optimization import (
    HighPerformanceCache, CachePolicy, CacheEntry,
    ParallelProcessor, MemoryManager,
    ComputationOptimizer, PerformanceOptimizer, OptimizationLevel,
    get_global_optimizer, performance_optimized, cached_result
)


class TestCacheEntry:
    """Test CacheEntry functionality."""
    
    def test_cache_entry_creation(self):
        """Test basic cache entry creation."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            size_bytes=1024,
            ttl_seconds=300
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.size_bytes == 1024
        assert entry.ttl_seconds == 300
        assert entry.access_count == 0
        assert not entry.is_expired()
    
    def test_cache_entry_expiration(self):
        """Test cache entry expiration."""
        # Create entry that expires immediately
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl_seconds=0.001
        )
        
        # Wait for expiration
        time.sleep(0.002)
        
        assert entry.is_expired()
    
    def test_cache_entry_access_tracking(self):
        """Test access tracking."""
        entry = CacheEntry(key="test", value="value")
        
        assert entry.access_count == 0
        
        entry.access()
        assert entry.access_count == 1
        
        entry.access()
        assert entry.access_count == 2


class TestHighPerformanceCache:
    """Test HighPerformanceCache functionality."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = HighPerformanceCache(
            max_size_mb=10,
            max_entries=100,
            policy=CachePolicy.LRU,
            enable_persistence=False
        )
        
        assert cache.max_size_bytes == 10 * 1024 * 1024
        assert cache.max_entries == 100
        assert cache.policy == CachePolicy.LRU
        assert cache.current_size_bytes == 0
        assert len(cache.cache) == 0
    
    def test_basic_put_get(self):
        """Test basic put and get operations."""
        cache = HighPerformanceCache(enable_persistence=False)
        
        # Put value
        success = cache.put("key1", "value1")
        assert success is True
        
        # Get value
        value = cache.get("key1")
        assert value == "value1"
        
        # Get non-existent key
        value = cache.get("non_existent")
        assert value is None
    
    def test_cache_hit_miss_stats(self):
        """Test cache hit/miss statistics."""
        cache = HighPerformanceCache(enable_persistence=False)
        
        # Initial stats
        assert cache.hits == 0
        assert cache.misses == 0
        
        # Miss
        cache.get("non_existent")
        assert cache.misses == 1
        
        # Put and hit
        cache.put("key1", "value1")
        cache.get("key1")
        assert cache.hits == 1
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = HighPerformanceCache(enable_persistence=False)
        
        # Put with short TTL
        cache.put("key1", "value1", ttl_seconds=0.1)
        
        # Should be available immediately
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        assert cache.get("key1") is None
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = HighPerformanceCache(
            max_entries=2,
            policy=CachePolicy.LRU,
            enable_persistence=False
        )
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add another key (should evict key2)
        cache.put("key3", "value3")
        
        # key1 should still be there, key2 should be evicted
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
    
    def test_size_limits(self):
        """Test cache size limits."""
        cache = HighPerformanceCache(
            max_size_mb=1,  # Very small cache
            enable_persistence=False
        )
        
        # Create large value
        large_value = "x" * (512 * 1024)  # 512KB
        
        # Should succeed
        success1 = cache.put("key1", large_value)
        assert success1 is True
        
        # Another large value should trigger eviction
        success2 = cache.put("key2", large_value)
        assert success2 is True
        
        # First key might be evicted due to size
        # (This is implementation dependent)
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = HighPerformanceCache(enable_persistence=False)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert len(cache.cache) == 2
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.current_size_bytes == 0
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = HighPerformanceCache(enable_persistence=False)
        
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate_percent"] == 50.0
        assert stats["policy"] == CachePolicy.ADAPTIVE.value


class TestParallelProcessor:
    """Test ParallelProcessor functionality."""
    
    def test_processor_initialization(self):
        """Test parallel processor initialization."""
        processor = ParallelProcessor(
            max_workers=4,
            use_processes=False,
            chunk_size=100
        )
        
        assert processor.max_workers == 4
        assert processor.use_processes is False
        assert processor.chunk_size == 100
        assert processor.task_count == 0
    
    def test_map_parallel_basic(self):
        """Test basic parallel mapping."""
        processor = ParallelProcessor(max_workers=2, use_processes=False)
        
        def square(x):
            return x * x
        
        items = [1, 2, 3, 4, 5]
        results = processor.map_parallel(square, items)
        
        assert results == [1, 4, 9, 16, 25]
        assert processor.task_count == 5
    
    def test_map_parallel_empty_input(self):
        """Test parallel mapping with empty input."""
        processor = ParallelProcessor(max_workers=2, use_processes=False)
        
        def square(x):
            return x * x
        
        results = processor.map_parallel(square, [])
        assert results == []
    
    def test_map_parallel_with_errors(self):
        """Test parallel mapping with errors."""
        processor = ParallelProcessor(max_workers=2, use_processes=False)
        
        def problematic_func(x):
            if x == 3:
                raise ValueError("Error on 3")
            return x * 2
        
        items = [1, 2, 3, 4, 5]
        results = processor.map_parallel(problematic_func, items)
        
        # Should handle error gracefully
        assert len(results) == 5
        assert results[0] == 2  # 1 * 2
        assert results[1] == 4  # 2 * 2
        assert results[2] is None  # Error case
        assert results[3] == 8  # 4 * 2
        assert results[4] == 10  # 5 * 2
    
    def test_process_chunks(self):
        """Test chunk processing."""
        processor = ParallelProcessor(max_workers=2, chunk_size=3)
        
        def sum_chunk(chunk):
            return [sum(chunk)]
        
        data = list(range(10))  # [0, 1, 2, ..., 9]
        results = processor.process_chunks(sum_chunk, data)
        
        # Should process in chunks and flatten results
        expected = [sum([0, 1, 2]), sum([3, 4, 5]), sum([6, 7, 8]), sum([9])]
        assert results == expected
    
    @pytest.mark.asyncio
    async def test_async_map(self):
        """Test async parallel mapping."""
        processor = ParallelProcessor(max_workers=2, use_processes=False)
        
        async def async_square(x):
            await asyncio.sleep(0.01)  # Simulate async work
            return x * x
        
        items = [1, 2, 3, 4, 5]
        results = await processor.async_map(async_square, items)
        
        assert results == [1, 4, 9, 16, 25]
    
    @pytest.mark.asyncio
    async def test_async_map_with_errors(self):
        """Test async mapping with errors."""
        processor = ParallelProcessor(max_workers=2, use_processes=False)
        
        async def async_problematic(x):
            if x == 3:
                raise ValueError("Error on 3")
            return x * 2
        
        items = [1, 2, 3, 4, 5]
        results = await processor.async_map(async_problematic, items)
        
        # Should handle errors
        assert len(results) == 5
        assert results[2] is None  # Error case
        assert results[0] == 2
        assert results[1] == 4
    
    def test_processor_stats(self):
        """Test processor statistics."""
        processor = ParallelProcessor(max_workers=4)
        
        # Process some tasks
        processor.map_parallel(lambda x: x * 2, [1, 2, 3, 4, 5])
        
        stats = processor.get_stats()
        
        assert stats["max_workers"] == 4
        assert stats["total_tasks"] == 5
        assert stats["total_time"] > 0
        assert stats["avg_time_per_task"] > 0


class TestMemoryManager:
    """Test MemoryManager functionality."""
    
    def test_memory_manager_initialization(self):
        """Test memory manager initialization."""
        manager = MemoryManager()
        
        assert len(manager.memory_pools) == 0
        assert len(manager.allocation_stats) == 0
    
    def test_create_memory_pool(self):
        """Test memory pool creation."""
        manager = MemoryManager()
        
        pool = manager.create_pool("test_pool", chunk_size=1000, max_chunks=10)
        
        assert isinstance(pool, np.ndarray)
        assert len(pool) == 1000 * 10
        assert "test_pool" in manager.memory_pools
        
        pool_info = manager.memory_pools["test_pool"]
        assert pool_info["chunk_size"] == 1000
        assert pool_info["max_chunks"] == 10
        assert pool_info["used_chunks"] == 0
    
    def test_pool_allocation_deallocation(self):
        """Test pool allocation and deallocation."""
        manager = MemoryManager()
        manager.create_pool("test_pool", chunk_size=100, max_chunks=5)
        
        # Allocate chunk
        chunk = manager.allocate_from_pool("test_pool")
        
        assert chunk is not None
        assert len(chunk) == 100
        assert manager.memory_pools["test_pool"]["used_chunks"] == 1
        
        # Pool exhaustion test
        for _ in range(4):  # Allocate remaining chunks
            manager.allocate_from_pool("test_pool")
        
        # Should be exhausted now
        exhausted_chunk = manager.allocate_from_pool("test_pool")
        assert exhausted_chunk is None
    
    def test_array_optimization(self):
        """Test array optimization."""
        manager = MemoryManager()
        
        # Create float64 array
        array = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        original_size = array.nbytes
        
        # Optimize to float32
        optimized = manager.optimize_array(array, target_dtype=np.float32)
        
        assert optimized.dtype == np.float32
        assert optimized.nbytes < original_size
        assert np.allclose(array, optimized)  # Values should be preserved
    
    def test_auto_optimization(self):
        """Test automatic array optimization."""
        manager = MemoryManager()
        
        # Create int64 array with small values
        array = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        
        # Should automatically optimize to smaller int type
        optimized = manager.optimize_array(array)
        
        assert optimized.dtype in [np.int16, np.int32]
        assert optimized.nbytes <= array.nbytes
        assert np.array_equal(array, optimized)


class TestComputationOptimizer:
    """Test ComputationOptimizer functionality."""
    
    def test_optimizer_initialization(self):
        """Test computation optimizer initialization."""
        optimizer = ComputationOptimizer()
        
        assert len(optimizer.optimization_cache) == 0
        assert len(optimizer.compilation_cache) == 0
    
    def test_numpy_optimization_decorator(self):
        """Test numpy operations optimization."""
        optimizer = ComputationOptimizer()
        
        @optimizer.optimize_numpy_operations
        def test_function(data):
            return np.sum(data)
        
        # Test with list (should be converted to numpy array)
        result = test_function([1, 2, 3, 4, 5])
        assert result == 15
        
        # Test with numpy array
        result = test_function(np.array([1, 2, 3, 4, 5]))
        assert result == 15
    
    def test_memoization_decorator(self):
        """Test memoization decorator."""
        optimizer = ComputationOptimizer()
        
        call_count = 0
        
        @optimizer.memoize_expensive(ttl_seconds=60)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call with same args (should use cache)
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Not incremented
        
        # Different args (should call function)
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2
    
    def test_batch_processing_decorator(self):
        """Test batch processing decorator."""
        optimizer = ComputationOptimizer()
        
        @optimizer.batch_process(batch_size=3)
        def process_batch(data_list):
            return [x * 2 for x in data_list]
        
        # Small batch (no splitting)
        result1 = process_batch([1, 2])
        assert result1 == [2, 4]
        
        # Large batch (should be split)
        result2 = process_batch([1, 2, 3, 4, 5, 6, 7])
        assert result2 == [2, 4, 6, 8, 10, 12, 14]


class TestPerformanceOptimizer:
    """Test PerformanceOptimizer functionality."""
    
    def test_optimizer_initialization(self):
        """Test performance optimizer initialization."""
        optimizer = PerformanceOptimizer(
            optimization_level=OptimizationLevel.MODERATE,
            cache_size_mb=512,
            enable_parallel=True
        )
        
        assert optimizer.optimization_level == OptimizationLevel.MODERATE
        assert optimizer.enable_parallel is True
        assert optimizer.cache is not None
        assert optimizer.parallel_processor is not None
    
    def test_audio_processing_optimization(self):
        """Test audio processing function optimization."""
        optimizer = PerformanceOptimizer(optimization_level=OptimizationLevel.BASIC)
        
        def audio_function(data):
            return np.mean(data)
        
        optimized_func = optimizer.optimize_audio_processing(audio_function)
        
        # Test that it still works
        test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = optimized_func(test_data)
        
        assert result == 3.0
    
    def test_cached_audio_operation(self):
        """Test cached audio operation decorator."""
        optimizer = PerformanceOptimizer()
        
        call_count = 0
        
        @optimizer.cached_audio_operation("test_op", ttl_seconds=60)
        def audio_operation(data, param=1.0):
            nonlocal call_count
            call_count += 1
            return np.sum(data) * param
        
        data = np.array([1, 2, 3, 4, 5])
        
        # First call
        result1 = audio_operation(data, param=2.0)
        assert result1 == 30.0  # (1+2+3+4+5) * 2
        assert call_count == 1
        assert optimizer.optimization_stats["cache_misses"] == 1
        
        # Second call (should use cache)
        result2 = audio_operation(data, param=2.0)
        assert result2 == 30.0
        assert call_count == 1  # Not incremented
        assert optimizer.optimization_stats["cache_hits"] == 1
    
    def test_parallel_audio_batch(self):
        """Test parallel audio batch processing."""
        optimizer = PerformanceOptimizer(enable_parallel=True)
        
        def process_item(item):
            return item * 2
        
        items = [1, 2, 3, 4, 5]
        results = optimizer.parallel_audio_batch(process_item, items)
        
        assert results == [2, 4, 6, 8, 10]
        assert optimizer.optimization_stats["parallel_tasks"] == 5
    
    def test_audio_array_optimization(self):
        """Test audio array optimization."""
        optimizer = PerformanceOptimizer()
        
        # Create float64 audio array
        audio = np.array([0.1, 0.2, 0.3, -0.1, -0.2], dtype=np.float64)
        
        optimized = optimizer.optimize_audio_array(audio)
        
        assert optimized.dtype == np.float32
        assert optimizer.optimization_stats["memory_optimizations"] == 1
        assert np.allclose(audio, optimized)
    
    def test_performance_report(self):
        """Test performance report generation."""
        optimizer = PerformanceOptimizer()
        
        # Generate some activity
        optimizer.optimize_audio_array(np.array([1, 2, 3], dtype=np.float64))
        
        report = optimizer.get_performance_report()
        
        assert "optimization_level" in report
        assert "cache_stats" in report
        assert "optimization_stats" in report
        assert "memory_usage" in report


class TestDecorators:
    """Test performance optimization decorators."""
    
    def test_performance_optimized_decorator(self):
        """Test performance_optimized decorator."""
        
        @performance_optimized(cache_ttl=60, optimization_level=OptimizationLevel.BASIC)
        def test_function(data):
            return np.sum(data)
        
        data = np.array([1, 2, 3, 4, 5])
        result = test_function(data)
        
        assert result == 15
    
    def test_cached_result_decorator(self):
        """Test cached_result decorator."""
        
        call_count = 0
        
        @cached_result(ttl_seconds=60)
        def cached_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call
        result1 = cached_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call (should use cache)
        result2 = cached_function(1, 2)
        assert result2 == 3
        assert call_count == 1
    
    def test_parallel_batch_processor_decorator(self):
        """Test parallel_batch_processor decorator."""
        
        @parallel_batch_processor(batch_size=3)
        def process_items(items):
            return [item * 2 for item in items]
        
        # Small batch
        result1 = process_items([1, 2])
        assert result1 == [2, 4]
        
        # Large batch (should be processed in parallel)
        result2 = process_items([1, 2, 3, 4, 5, 6, 7])
        assert result2 == [2, 4, 6, 8, 10, 12, 14]


class TestGlobalOptimizer:
    """Test global optimizer functionality."""
    
    def test_get_global_optimizer(self):
        """Test getting global optimizer instance."""
        optimizer1 = get_global_optimizer()
        optimizer2 = get_global_optimizer()
        
        # Should return same instance
        assert optimizer1 is optimizer2
        assert isinstance(optimizer1, PerformanceOptimizer)


@pytest.fixture
def temp_cache_file():
    """Create temporary cache file for testing."""
    fd, temp_file = tempfile.mkstemp(suffix='.pkl')
    os.close(fd)
    yield temp_file
    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)


class TestCachePersistence:
    """Test cache persistence functionality."""
    
    def test_cache_persistence_disabled(self):
        """Test cache with persistence disabled."""
        cache = HighPerformanceCache(enable_persistence=False)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Should not create any files
        cache.shutdown()
    
    def test_cache_persistence_enabled(self, temp_cache_file):
        """Test cache with persistence enabled."""
        # Create cache with persistence
        cache1 = HighPerformanceCache(
            enable_persistence=True,
            persistence_file=temp_cache_file
        )
        
        cache1.put("key1", "value1")
        cache1.put("key2", "value2")
        
        # Manually save
        cache1._save_cache()
        cache1.shutdown()
        
        # Create new cache instance with same file
        cache2 = HighPerformanceCache(
            enable_persistence=True,
            persistence_file=temp_cache_file
        )
        
        # Should load previous data
        assert cache2.get("key1") == "value1"
        assert cache2.get("key2") == "value2"
        
        cache2.shutdown()


class TestIntegration:
    """Integration tests for performance optimization."""
    
    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        optimizer = PerformanceOptimizer(
            optimization_level=OptimizationLevel.AGGRESSIVE,
            cache_size_mb=100,
            enable_parallel=True
        )
        
        # Define audio processing function
        @optimizer.cached_audio_operation("test_audio_proc")
        def process_audio(audio_data, gain=1.0):
            return audio_data * gain
        
        # Optimize array
        audio_data = np.random.rand(1000).astype(np.float64)
        optimized_data = optimizer.optimize_audio_array(audio_data)
        
        # Process audio (should be cached)
        result1 = process_audio(optimized_data, gain=2.0)
        result2 = process_audio(optimized_data, gain=2.0)  # From cache
        
        assert np.allclose(result1, result2)
        assert optimizer.optimization_stats["cache_hits"] >= 1
        
        # Generate performance report
        report = optimizer.get_performance_report()
        assert report["optimization_stats"]["cache_hits"] >= 1
        assert report["optimization_stats"]["memory_optimizations"] >= 1
    
    def test_parallel_processing_integration(self):
        """Test parallel processing integration."""
        optimizer = PerformanceOptimizer(enable_parallel=True)
        
        def heavy_computation(data):
            # Simulate heavy computation
            return np.sum(data ** 2)
        
        # Create multiple data arrays
        data_arrays = [np.random.rand(100) for _ in range(10)]
        
        # Process in parallel
        start_time = time.time()
        results = optimizer.parallel_audio_batch(heavy_computation, data_arrays)
        parallel_time = time.time() - start_time
        
        # Process sequentially for comparison
        start_time = time.time()
        sequential_results = [heavy_computation(data) for data in data_arrays]
        sequential_time = time.time() - start_time
        
        # Results should be the same
        assert len(results) == len(sequential_results)
        for r1, r2 in zip(results, sequential_results):
            assert abs(r1 - r2) < 1e-10  # Should be identical
        
        # Parallel should generally be faster (though not guaranteed in tests)
        # This is more of a smoke test
        assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__])