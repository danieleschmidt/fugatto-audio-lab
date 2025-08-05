"""Performance Optimization Engine for Fugatto Audio Lab.

Advanced performance optimization including caching, parallel processing,
memory management, and computational efficiency improvements.
"""

import asyncio
import multiprocessing
import threading
import time
import pickle
import json
import hashlib
import logging
import functools
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict, OrderedDict
import numpy as np
import weakref

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    FIFO = "fifo"        # First In, First Out
    TTL = "ttl"          # Time To Live
    ADAPTIVE = "adaptive" # Adaptive based on usage patterns


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    NONE = 0
    BASIC = 1
    MODERATE = 2
    AGGRESSIVE = 3
    EXTREME = 4


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    def access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


class HighPerformanceCache:
    """High-performance multi-level cache system."""
    
    def __init__(self, 
                 max_size_mb: int = 1024,
                 max_entries: int = 10000,
                 policy: CachePolicy = CachePolicy.ADAPTIVE,
                 enable_persistence: bool = True,
                 persistence_file: str = "cache_data.pkl"):
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.policy = policy
        self.enable_persistence = enable_persistence
        self.persistence_file = persistence_file
        
        # Cache storage
        self.cache = OrderedDict()  # key -> CacheEntry
        self.current_size_bytes = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background maintenance
        self.maintenance_interval = 60.0  # 1 minute
        self.running = True
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.maintenance_thread.start()
        
        # Load persistent cache
        if self.enable_persistence:
            self._load_cache()
        
        logger.info(f"HighPerformanceCache initialized: {max_size_mb}MB, {max_entries} entries, {policy.value} policy")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check expiration
                if entry.is_expired():
                    self._remove_entry(key)
                    self.misses += 1
                    return None
                
                # Update access info
                entry.access()
                
                # Move to end for LRU
                if self.policy in [CachePolicy.LRU, CachePolicy.ADAPTIVE]:
                    self.cache.move_to_end(key)
                
                self.hits += 1
                return entry.value
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> bool:
        """Put value in cache."""
        with self.lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            except:
                size_bytes = 1024  # Fallback estimate
            
            # Check if value is too large
            if size_bytes > self.max_size_bytes // 2:
                logger.warning(f"Value too large for cache: {size_bytes} bytes")
                return False
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Make space if needed
            while (len(self.cache) >= self.max_entries or 
                   self.current_size_bytes + size_bytes > self.max_size_bytes):
                if not self._evict_entry():
                    break  # Could not evict any more entries
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds
            )
            
            # Add to cache
            self.cache[key] = entry
            self.current_size_bytes += size_bytes
            
            return True
    
    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache."""
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0
            logger.info("Cache cleared")
    
    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_size_bytes -= entry.size_bytes
            del self.cache[key]
    
    def _evict_entry(self) -> bool:
        """Evict one entry based on policy."""
        if not self.cache:
            return False
        
        if self.policy == CachePolicy.LRU:
            # Remove least recently used (first in OrderedDict)
            key = next(iter(self.cache))
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
        elif self.policy == CachePolicy.FIFO:
            # Remove oldest entry
            key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
        elif self.policy == CachePolicy.TTL:
            # Remove expired entries first, then oldest
            expired_keys = [k for k, entry in self.cache.items() if entry.is_expired()]
            if expired_keys:
                key = expired_keys[0]
            else:
                key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
        elif self.policy == CachePolicy.ADAPTIVE:
            # Adaptive policy based on access patterns
            key = self._adaptive_eviction()
        else:
            key = next(iter(self.cache))
        
        self._remove_entry(key)
        self.evictions += 1
        return True
    
    def _adaptive_eviction(self) -> str:
        """Adaptive eviction based on usage patterns."""
        current_time = time.time()
        
        # Score entries based on multiple factors
        scores = {}
        for key, entry in self.cache.items():
            # Factors: recency, frequency, size, age
            recency_score = 1.0 / (current_time - entry.last_accessed + 1)
            frequency_score = entry.access_count / (current_time - entry.created_at + 1)
            size_penalty = entry.size_bytes / self.max_size_bytes
            
            # Combined score (lower is worse)
            scores[key] = recency_score + frequency_score - size_penalty
        
        # Return key with lowest score
        return min(scores.keys(), key=lambda k: scores[k])
    
    def _maintenance_loop(self):
        """Background maintenance loop."""
        while self.running:
            try:
                time.sleep(self.maintenance_interval)
                
                with self.lock:
                    # Remove expired entries
                    expired_keys = [
                        key for key, entry in self.cache.items()
                        if entry.is_expired()
                    ]
                    
                    for key in expired_keys:
                        self._remove_entry(key)
                    
                    if expired_keys:
                        logger.debug(f"Removed {len(expired_keys)} expired cache entries")
                    
                    # Save cache if persistence enabled
                    if self.enable_persistence:
                        self._save_cache()
                        
            except Exception as e:
                logger.error(f"Cache maintenance error: {e}")
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            cache_data = {
                'entries': dict(self.cache),
                'stats': {
                    'hits': self.hits,
                    'misses': self.misses,
                    'evictions': self.evictions
                }
            }
            
            with open(self.persistence_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        except Exception as e:
            logger.error(f"Cache save error: {e}")
    
    def _load_cache(self):
        """Load cache from disk."""
        try:
            if Path(self.persistence_file).exists():
                with open(self.persistence_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.cache = OrderedDict(cache_data['entries'])
                stats = cache_data.get('stats', {})
                self.hits = stats.get('hits', 0)
                self.misses = stats.get('misses', 0)
                self.evictions = stats.get('evictions', 0)
                
                # Recalculate current size
                self.current_size_bytes = sum(entry.size_bytes for entry in self.cache.values())
                
                logger.info(f"Loaded {len(self.cache)} cache entries from disk")
                
        except Exception as e:
            logger.error(f"Cache load error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "entries": len(self.cache),
                "size_mb": self.current_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization_percent": (self.current_size_bytes / self.max_size_bytes * 100),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate_percent": hit_rate,
                "policy": self.policy.value
            }
    
    def shutdown(self):
        """Shutdown cache system."""
        self.running = False
        if self.maintenance_thread.is_alive():
            self.maintenance_thread.join(timeout=5.0)
        
        if self.enable_persistence:
            self._save_cache()


class ParallelProcessor:
    """High-performance parallel processing engine."""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 use_processes: bool = False,
                 chunk_size: int = 1000):
        
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        
        # Initialize executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Performance tracking
        self.task_count = 0
        self.total_time = 0.0
        
        logger.info(f"ParallelProcessor initialized: {self.max_workers} workers, "
                   f"{'processes' if use_processes else 'threads'}")
    
    def map_parallel(self, func: Callable, items: List[Any], 
                    progress_callback: Optional[Callable] = None) -> List[Any]:
        """Execute function in parallel over list of items."""
        if not items:
            return []
        
        start_time = time.time()
        results = []
        
        try:
            # Submit all tasks
            future_to_index = {}
            for i, item in enumerate(items):
                future = self.executor.submit(func, item)
                future_to_index[future] = i
            
            # Collect results maintaining order
            result_dict = {}
            completed = 0
            
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    index = future_to_index[future]
                    result_dict[index] = result
                    
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(items))
                        
                except Exception as e:
                    index = future_to_index[future]
                    result_dict[index] = None
                    logger.error(f"Parallel task {index} failed: {e}")
            
            # Reconstruct ordered results
            results = [result_dict.get(i) for i in range(len(items))]
            
        except Exception as e:
            logger.error(f"Parallel processing error: {e}")
            # Fallback to sequential processing
            results = [func(item) for item in items]
        
        # Update stats
        duration = time.time() - start_time
        self.task_count += len(items)
        self.total_time += duration
        
        logger.debug(f"Parallel processing: {len(items)} items in {duration:.2f}s")
        
        return results
    
    def process_chunks(self, func: Callable, data: List[Any], 
                      chunk_size: Optional[int] = None) -> List[Any]:
        """Process data in parallel chunks."""
        chunk_size = chunk_size or self.chunk_size
        
        if len(data) <= chunk_size:
            return func(data)
        
        # Split into chunks
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Process chunks in parallel
        chunk_results = self.map_parallel(func, chunks)
        
        # Flatten results
        results = []
        for chunk_result in chunk_results:
            if isinstance(chunk_result, list):
                results.extend(chunk_result)
            else:
                results.append(chunk_result)
        
        return results
    
    async def async_map(self, async_func: Callable, items: List[Any]) -> List[Any]:
        """Execute async function in parallel."""
        if not items:
            return []
        
        # Create tasks
        tasks = [async_func(item) for item in items]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        clean_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Async task {i} failed: {result}")
                clean_results.append(None)
            else:
                clean_results.append(result)
        
        return clean_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        avg_time_per_task = (self.total_time / self.task_count) if self.task_count > 0 else 0
        
        return {
            "max_workers": self.max_workers,
            "executor_type": "ProcessPoolExecutor" if self.use_processes else "ThreadPoolExecutor",
            "total_tasks": self.task_count,
            "total_time": self.total_time,
            "avg_time_per_task": avg_time_per_task,
            "throughput_tasks_per_sec": (self.task_count / self.total_time) if self.total_time > 0 else 0
        }
    
    def shutdown(self):
        """Shutdown parallel processor."""
        self.executor.shutdown(wait=True)


class MemoryManager:
    """Advanced memory management and optimization."""
    
    def __init__(self):
        self.memory_pools = {}
        self.allocation_stats = defaultdict(int)
        self.peak_usage = 0
        self.weak_refs = weakref.WeakSet()
        
        logger.info("MemoryManager initialized")
    
    def create_pool(self, name: str, chunk_size: int, max_chunks: int = 100) -> np.ndarray:
        """Create memory pool for efficient allocation."""
        pool_size = chunk_size * max_chunks
        memory_pool = np.empty(pool_size, dtype=np.float32)
        
        self.memory_pools[name] = {
            'pool': memory_pool,
            'chunk_size': chunk_size,
            'max_chunks': max_chunks,
            'used_chunks': 0,
            'free_list': list(range(max_chunks))
        }
        
        logger.info(f"Created memory pool '{name}': {chunk_size} x {max_chunks} chunks")
        return memory_pool
    
    def allocate_from_pool(self, pool_name: str) -> Optional[np.ndarray]:
        """Allocate chunk from memory pool."""
        if pool_name not in self.memory_pools:
            return None
        
        pool_info = self.memory_pools[pool_name]
        
        if not pool_info['free_list']:
            return None  # Pool exhausted
        
        chunk_index = pool_info['free_list'].pop(0)
        pool_info['used_chunks'] += 1
        
        start_idx = chunk_index * pool_info['chunk_size']
        end_idx = start_idx + pool_info['chunk_size']
        
        chunk = pool_info['pool'][start_idx:end_idx]
        self.allocation_stats[pool_name] += 1
        
        return chunk
    
    def deallocate_to_pool(self, pool_name: str, chunk: np.ndarray):
        """Return chunk to memory pool."""
        if pool_name not in self.memory_pools:
            return
        
        pool_info = self.memory_pools[pool_name]
        
        # Find chunk index (simplified - in practice would need more robust tracking)
        chunk_size = pool_info['chunk_size']
        chunk_index = (chunk.data.obj.ctypes.data - pool_info['pool'].ctypes.data) // (chunk_size * 4)
        
        if 0 <= chunk_index < pool_info['max_chunks']:
            pool_info['free_list'].append(chunk_index)
            pool_info['used_chunks'] -= 1
    
    def optimize_array(self, array: np.ndarray, target_dtype: np.dtype = None) -> np.ndarray:
        """Optimize numpy array for memory efficiency."""
        original_size = array.nbytes
        
        # Convert to optimal dtype if not specified
        if target_dtype is None:
            if array.dtype == np.float64:
                target_dtype = np.float32  # Usually sufficient precision
            elif array.dtype == np.int64:
                max_val = np.max(np.abs(array))
                if max_val < 2**15:
                    target_dtype = np.int16
                elif max_val < 2**31:
                    target_dtype = np.int32
                else:
                    target_dtype = array.dtype
        
        # Convert dtype if beneficial
        if target_dtype != array.dtype:
            optimized = array.astype(target_dtype)
        else:
            optimized = array
        
        # Ensure contiguous memory layout
        if not optimized.flags['C_CONTIGUOUS']:
            optimized = np.ascontiguousarray(optimized)
        
        new_size = optimized.nbytes
        if new_size < original_size:
            logger.debug(f"Memory optimization: {original_size} -> {new_size} bytes "
                        f"({(1 - new_size/original_size)*100:.1f}% reduction)")
        
        return optimized
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            
            return {
                "rss_mb": process.memory_info().rss / 1024 / 1024,
                "vms_mb": process.memory_info().vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "pool_usage": {
                    name: {
                        "used_chunks": info['used_chunks'],
                        "max_chunks": info['max_chunks'],
                        "utilization": info['used_chunks'] / info['max_chunks'] * 100
                    }
                    for name, info in self.memory_pools.items()
                },
                "allocation_stats": dict(self.allocation_stats)
            }
        except ImportError:
            return {"error": "psutil not available"}


class ComputationOptimizer:
    """Advanced computation optimization techniques."""
    
    def __init__(self):
        self.optimization_cache = {}
        self.compilation_cache = {}
        
        logger.info("ComputationOptimizer initialized")
    
    def optimize_numpy_operations(self, func: Callable) -> Callable:
        """Optimize numpy operations with vectorization."""
        def optimized_func(*args, **kwargs):
            # Convert lists to numpy arrays for better performance
            optimized_args = []
            for arg in args:
                if isinstance(arg, list) and len(arg) > 100:
                    optimized_args.append(np.array(arg))
                else:
                    optimized_args.append(arg)
            
            # Set numpy to use optimal number of threads
            original_threads = None
            try:
                # Try to optimize numpy threading
                import numpy as np
                if hasattr(np, '__config__'):
                    original_threads = np.get_num_threads() if hasattr(np, 'get_num_threads') else None
                    if hasattr(np, 'set_num_threads'):
                        np.set_num_threads(min(8, multiprocessing.cpu_count()))
            except:
                pass
            
            try:
                return func(*optimized_args, **kwargs)
            finally:
                # Restore original thread count
                if original_threads is not None:
                    try:
                        np.set_num_threads(original_threads)
                    except:
                        pass
        
        return optimized_func
    
    def memoize_expensive(self, ttl_seconds: int = 300):
        """Memoization decorator for expensive computations."""
        def decorator(func):
            cache = {}
            
            def wrapper(*args, **kwargs):
                # Create cache key
                key = hashlib.md5(
                    pickle.dumps((args, sorted(kwargs.items())), protocol=pickle.HIGHEST_PROTOCOL)
                ).hexdigest()
                
                # Check cache
                if key in cache:
                    result, timestamp = cache[key]
                    if time.time() - timestamp < ttl_seconds:
                        return result
                    else:
                        del cache[key]
                
                # Compute result
                result = func(*args, **kwargs)
                cache[key] = (result, time.time())
                
                # Limit cache size
                if len(cache) > 1000:
                    # Remove oldest entries
                    oldest_keys = sorted(cache.keys(), key=lambda k: cache[k][1])[:100]
                    for old_key in oldest_keys:
                        del cache[old_key]
                
                return result
            
            return wrapper
        return decorator
    
    def batch_process(self, batch_size: int = 32):
        """Decorator for efficient batch processing."""
        def decorator(func):
            def wrapper(data_list, *args, **kwargs):
                if len(data_list) <= batch_size:
                    return func(data_list, *args, **kwargs)
                
                # Process in batches
                results = []
                for i in range(0, len(data_list), batch_size):
                    batch = data_list[i:i + batch_size]
                    batch_result = func(batch, *args, **kwargs)
                    results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
                
                return results
            return wrapper
        return decorator


class PerformanceOptimizer:
    """Main performance optimization orchestrator."""
    
    def __init__(self, 
                 optimization_level: OptimizationLevel = OptimizationLevel.MODERATE,
                 cache_size_mb: int = 1024,
                 enable_parallel: bool = True):
        
        self.optimization_level = optimization_level
        self.enable_parallel = enable_parallel
        
        # Initialize components
        self.cache = HighPerformanceCache(
            max_size_mb=cache_size_mb,
            policy=CachePolicy.ADAPTIVE
        )
        
        if enable_parallel:
            self.parallel_processor = ParallelProcessor(
                use_processes=optimization_level.value >= OptimizationLevel.AGGRESSIVE.value
            )
        else:
            self.parallel_processor = None
        
        self.memory_manager = MemoryManager()
        self.computation_optimizer = ComputationOptimizer()
        
        # Performance metrics
        self.optimization_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "parallel_tasks": 0,
            "memory_optimizations": 0,
            "computation_optimizations": 0
        }
        
        logger.info(f"PerformanceOptimizer initialized: level={optimization_level.name}")
    
    def optimize_audio_processing(self, func: Callable) -> Callable:
        """Comprehensive optimization for audio processing functions."""
        
        # Apply different optimizations based on level
        optimized_func = func
        
        if self.optimization_level.value >= OptimizationLevel.BASIC.value:
            # Basic numpy optimizations
            optimized_func = self.computation_optimizer.optimize_numpy_operations(optimized_func)
        
        if self.optimization_level.value >= OptimizationLevel.MODERATE.value:
            # Add memoization for expensive operations
            optimized_func = self.computation_optimizer.memoize_expensive(ttl_seconds=600)(optimized_func)
        
        if self.optimization_level.value >= OptimizationLevel.AGGRESSIVE.value:
            # Add batch processing optimization
            optimized_func = self.computation_optimizer.batch_process(batch_size=16)(optimized_func)
        
        return optimized_func
    
    def cached_audio_operation(self, operation_name: str, ttl_seconds: int = 3600):
        """Decorator for caching audio operations."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Create cache key
                cache_key = f"{operation_name}:{hashlib.md5(pickle.dumps((args, kwargs))).hexdigest()}"
                
                # Try cache first
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    self.optimization_stats["cache_hits"] += 1
                    return cached_result
                
                # Compute result
                result = func(*args, **kwargs)
                
                # Cache result
                self.cache.put(cache_key, result, ttl_seconds=ttl_seconds)
                self.optimization_stats["cache_misses"] += 1
                
                return result
            return wrapper
        return decorator
    
    def parallel_audio_batch(self, batch_func: Callable, items: List[Any]) -> List[Any]:
        """Process audio batch in parallel."""
        if not self.enable_parallel or not self.parallel_processor:
            return [batch_func(item) for item in items]
        
        results = self.parallel_processor.map_parallel(batch_func, items)
        self.optimization_stats["parallel_tasks"] += len(items)
        
        return results
    
    def optimize_audio_array(self, audio: np.ndarray) -> np.ndarray:
        """Optimize audio array for memory and computation efficiency."""
        optimized = self.memory_manager.optimize_array(audio, target_dtype=np.float32)
        self.optimization_stats["memory_optimizations"] += 1
        return optimized
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "optimization_level": self.optimization_level.name,
            "cache_stats": self.cache.get_stats(),
            "optimization_stats": self.optimization_stats.copy(),
            "memory_usage": self.memory_manager.get_memory_usage()
        }
        
        if self.parallel_processor:
            report["parallel_stats"] = self.parallel_processor.get_stats()
        
        return report
    
    def save_performance_report(self, filepath: str):
        """Save performance report to file."""
        report = self.get_performance_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to: {filepath}")
    
    def shutdown(self):
        """Shutdown performance optimizer."""
        self.cache.shutdown()
        if self.parallel_processor:
            self.parallel_processor.shutdown()


# Global performance optimizer
_global_optimizer = None

def get_global_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


# Performance decorators

def performance_optimized(cache_ttl: int = 3600, 
                         enable_parallel: bool = False,
                         optimization_level: OptimizationLevel = OptimizationLevel.MODERATE):
    """Comprehensive performance optimization decorator."""
    def decorator(func):
        optimizer = get_global_optimizer()
        
        # Apply audio processing optimizations
        optimized_func = optimizer.optimize_audio_processing(func)
        
        # Add caching
        if cache_ttl > 0:
            optimized_func = optimizer.cached_audio_operation(
                func.__name__, ttl_seconds=cache_ttl
            )(optimized_func)
        
        return optimized_func
    return decorator


def cached_result(ttl_seconds: int = 3600, cache_key_func: Callable = None):
    """Simple result caching decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            optimizer = get_global_optimizer()
            
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hashlib.md5(pickle.dumps((args, kwargs))).hexdigest()}"
            
            # Try cache
            result = optimizer.cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            optimizer.cache.put(cache_key, result, ttl_seconds=ttl_seconds)
            
            return result
        return wrapper
    return decorator


def parallel_batch_processor(batch_size: int = 32):
    """Decorator for parallel batch processing."""
    def decorator(func):
        def wrapper(items, *args, **kwargs):
            optimizer = get_global_optimizer()
            
            if len(items) <= batch_size or not optimizer.enable_parallel:
                return func(items, *args, **kwargs)
            
            # Create batch function
            def batch_func(batch_items):
                return func(batch_items, *args, **kwargs)
            
            # Split into batches
            batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
            
            # Process in parallel
            batch_results = optimizer.parallel_audio_batch(batch_func, batches)
            
            # Flatten results
            results = []
            for batch_result in batch_results:
                if isinstance(batch_result, list):
                    results.extend(batch_result)
                else:
                    results.append(batch_result)
            
            return results
        return wrapper
    return decorator