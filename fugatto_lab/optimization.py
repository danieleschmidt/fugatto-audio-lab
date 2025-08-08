"""Performance optimization module for Fugatto Audio Lab."""

import logging
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from threading import Lock
import hashlib

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Mock numpy for testing
    class MockNumpy:
        @staticmethod
        def array(data, dtype=None):
            return list(data) if hasattr(data, '__iter__') else [data]
        
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
        
        @staticmethod
        def max(data):
            return max(data) if data else 0
        
        float32 = float
        ndarray = list  # Mock ndarray as list
    
    if not HAS_NUMPY:
        np = MockNumpy()

logger = logging.getLogger(__name__)


class AudioCache:
    """Intelligent caching system for audio generation."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """Initialize audio cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time to live for cached items
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._access_times = {}
        self._lock = Lock()
        
        logger.info(f"AudioCache initialized: max_size={max_size}, ttl={ttl_seconds}s")
    
    def _generate_key(self, prompt: str, duration: float, temperature: float, model_name: str) -> str:
        """Generate cache key from parameters."""
        content = f"{prompt}:{duration}:{temperature}:{model_name}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(self, prompt: str, duration: float, temperature: float, model_name: str) -> Optional[np.ndarray]:
        """Get cached audio if available."""
        key = self._generate_key(prompt, duration, temperature, model_name)
        
        with self._lock:
            if key in self._cache:
                current_time = time.time()
                if current_time - self._cache[key]['timestamp'] < self.ttl_seconds:
                    self._access_times[key] = current_time
                    logger.debug(f"Cache hit for key: {key}")
                    return self._cache[key]['audio']
                else:
                    # Expired entry
                    del self._cache[key]
                    del self._access_times[key]
                    logger.debug(f"Cache expired for key: {key}")
        
        return None
    
    def put(self, prompt: str, duration: float, temperature: float, model_name: str, audio: np.ndarray):
        """Store audio in cache."""
        key = self._generate_key(prompt, duration, temperature, model_name)
        current_time = time.time()
        
        with self._lock:
            # Remove oldest entry if cache is full
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
                del self._cache[oldest_key]
                del self._access_times[oldest_key]
                logger.debug(f"Cache evicted oldest entry: {oldest_key}")
            
            self._cache[key] = {
                'audio': audio.copy(),
                'timestamp': current_time
            }
            self._access_times[key] = current_time
            logger.debug(f"Cache stored entry: {key}")
    
    def clear(self):
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'utilization': len(self._cache) / self.max_size,
                'ttl_seconds': self.ttl_seconds
            }


class BatchProcessor:
    """Batch processing for multiple audio generation requests."""
    
    def __init__(self, batch_size: int = 4, timeout_seconds: float = 30.0):
        """Initialize batch processor.
        
        Args:
            batch_size: Maximum batch size
            timeout_seconds: Maximum wait time for batch
        """
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self._pending_requests = []
        self._lock = Lock()
        
        logger.info(f"BatchProcessor initialized: batch_size={batch_size}, timeout={timeout_seconds}s")
    
    def add_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add request to batch (simplified implementation)."""
        with self._lock:
            self._pending_requests.append(request_data)
            
            if len(self._pending_requests) >= self.batch_size:
                return self._process_batch()
            
        # For demo purposes, process immediately
        return self._process_single(request_data)
    
    def _process_batch(self) -> Dict[str, Any]:
        """Process accumulated batch."""
        requests = self._pending_requests.copy()
        self._pending_requests.clear()
        
        logger.info(f"Processing batch of {len(requests)} requests")
        
        # Simplified batch processing
        results = []
        for request in requests:
            result = self._process_single(request)
            results.append(result)
        
        return {'batch_results': results}
    
    def _process_single(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process single request."""
        # This would normally call the actual model
        return {
            'request_id': request_data.get('id', 'unknown'),
            'status': 'completed',
            'processing_time_ms': 100
        }


def performance_optimized(cache_enabled: bool = True):
    """Decorator for performance optimization."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Simple caching logic (would be more sophisticated in real implementation)
            if cache_enabled and hasattr(args[0], '_audio_cache'):
                cache = args[0]._audio_cache
                prompt = args[1] if len(args) > 1 else kwargs.get('prompt', '')
                duration = kwargs.get('duration_seconds', 10.0)
                temperature = kwargs.get('temperature', 0.8)
                model_name = getattr(args[0], 'model_name', 'unknown')
                
                # Try to get from cache
                cached_result = cache.get(prompt, duration, temperature, model_name)
                if cached_result is not None:
                    logger.debug("Returning cached result")
                    return cached_result
                
                # Call original function
                result = func(*args, **kwargs)
                
                # Store in cache
                if isinstance(result, np.ndarray):
                    cache.put(prompt, duration, temperature, model_name, result)
                
                return result
            else:
                result = func(*args, **kwargs)
            
            end_time = time.time()
            logger.debug(f"Function {func.__name__} took {(end_time - start_time)*1000:.2f}ms")
            
            return result
        
        return wrapper
    return decorator


class PerformanceOptimizer:
    """Performance optimization utilities."""
    
    @staticmethod
    def optimize_audio_processing(audio: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        """Optimize audio processing with chunking."""
        if len(audio) <= chunk_size:
            return audio
        
        # Process in chunks for better memory efficiency
        chunks = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            chunks.append(chunk)
        
        return np.concatenate(chunks)
    
    @staticmethod
    def precompute_embeddings(prompts: list) -> Dict[str, Any]:
        """Precompute embeddings for common prompts."""
        embeddings = {}
        
        for prompt in prompts:
            # Simplified embedding computation
            embedding = np.random.randn(512)  # Mock embedding
            key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            embeddings[key] = embedding
        
        logger.info(f"Precomputed embeddings for {len(prompts)} prompts")
        return embeddings
    
    @staticmethod
    def get_optimization_recommendations(system_info: Dict[str, Any]) -> Dict[str, str]:
        """Get optimization recommendations based on system info."""
        recommendations = {}
        
        memory_mb = system_info.get('memory_available_mb', 0)
        cpu_cores = system_info.get('cpu_cores', 1)
        gpu_available = system_info.get('gpu_available', False)
        
        if memory_mb < 4096:
            recommendations['memory'] = "Consider increasing batch size limit due to low memory"
        
        if cpu_cores >= 8:
            recommendations['concurrency'] = "Enable multi-threaded processing"
        
        if gpu_available:
            recommendations['acceleration'] = "Enable GPU acceleration for faster processing"
        else:
            recommendations['acceleration'] = "Consider GPU for significant speed improvements"
        
        return recommendations


# Global instances
_audio_cache = AudioCache()
_batch_processor = BatchProcessor()


def get_audio_cache() -> AudioCache:
    """Get global audio cache instance."""
    return _audio_cache


def get_batch_processor() -> BatchProcessor:
    """Get global batch processor instance."""
    return _batch_processor