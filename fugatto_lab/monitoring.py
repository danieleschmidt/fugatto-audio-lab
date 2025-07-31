"""Performance monitoring and metrics collection for Fugatto Audio Lab.

This module provides comprehensive monitoring capabilities including
performance metrics, health checks, and observability features.
"""

import time
import logging
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class AudioGenerationMetrics:
    """Metrics for audio generation operations."""
    
    prompt_length: int
    duration_seconds: float
    generation_time_ms: float
    model_name: str
    output_size_bytes: int
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: str


@dataclass
class SystemHealthMetrics:
    """System health and resource utilization metrics."""
    
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    gpu_memory_used_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    temperature_celsius: Optional[float] = None


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self, enable_detailed_logging: bool = False):
        self.enable_detailed_logging = enable_detailed_logging
        self.metrics_history: List[AudioGenerationMetrics] = []
        self.logger = logging.getLogger(__name__)
        
        if enable_detailed_logging:
            self.logger.setLevel(logging.DEBUG)
    
    def start_generation_timing(self) -> float:
        """Start timing an audio generation operation."""
        return time.time()
    
    def record_generation_metrics(self, start_time: float, prompt: str, 
                                duration_seconds: float, output_audio: Any,
                                model_name: str = "unknown") -> AudioGenerationMetrics:
        """Record metrics for completed audio generation."""
        end_time = time.time()
        generation_time_ms = (end_time - start_time) * 1000
        
        # Get system metrics
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Estimate output size (for numpy array)
        output_size_bytes = getattr(output_audio, 'nbytes', 0)
        
        metrics = AudioGenerationMetrics(
            prompt_length=len(prompt),
            duration_seconds=duration_seconds,
            generation_time_ms=generation_time_ms,
            model_name=model_name,
            output_size_bytes=output_size_bytes,
            memory_usage_mb=memory_info.used / (1024 * 1024),
            cpu_usage_percent=cpu_percent,
            timestamp=datetime.utcnow().isoformat()
        )
        
        self.metrics_history.append(metrics)
        
        if self.enable_detailed_logging:
            self.logger.debug(f"Audio generation metrics: {asdict(metrics)}")
        
        return metrics
    
    def get_system_health(self) -> SystemHealthMetrics:
        """Get current system health metrics."""
        cpu_percent = psutil.cpu_percent(interval=1.0)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health = SystemHealthMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_usage_percent=(disk.used / disk.total) * 100
        )
        
        # Try to get GPU metrics if available
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                health.gpu_memory_used_mb = gpu.memoryUsed
                health.gpu_utilization_percent = gpu.load * 100
                health.temperature_celsius = gpu.temperature
        except (ImportError, Exception):
            # GPU monitoring not available
            pass
        
        return health
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and statistics."""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics recorded yet"}
        
        generation_times = [m.generation_time_ms for m in self.metrics_history]
        memory_usage = [m.memory_usage_mb for m in self.metrics_history]
        cpu_usage = [m.cpu_usage_percent for m in self.metrics_history]
        
        return {
            "total_generations": len(self.metrics_history),
            "avg_generation_time_ms": sum(generation_times) / len(generation_times),
            "max_generation_time_ms": max(generation_times),
            "min_generation_time_ms": min(generation_times),
            "avg_memory_usage_mb": sum(memory_usage) / len(memory_usage),
            "avg_cpu_usage_percent": sum(cpu_usage) / len(cpu_usage),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for the system."""
        health = self.get_system_health()
        summary = self.get_performance_summary()
        
        # Determine overall health status
        status = "healthy"
        warnings = []
        
        if health.cpu_percent > 80:
            status = "warning"
            warnings.append("High CPU usage")
        
        if health.memory_percent > 85:
            status = "warning"
            warnings.append("High memory usage")
        
        if health.disk_usage_percent > 90:
            status = "critical"
            warnings.append("Disk space critically low")
        
        if health.gpu_memory_used_mb and health.gpu_memory_used_mb > 8000:  # 8GB
            status = "warning"
            warnings.append("High GPU memory usage")
        
        return {
            "status": status,
            "warnings": warnings,
            "system_health": asdict(health),
            "performance_summary": summary,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global monitor instance
_monitor_instance: Optional[PerformanceMonitor] = None


def get_monitor(enable_detailed_logging: bool = False) -> PerformanceMonitor:
    """Get or create global performance monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PerformanceMonitor(enable_detailed_logging)
    return _monitor_instance


def monitor_generation(func):
    """Decorator to automatically monitor audio generation functions."""
    def wrapper(*args, **kwargs):
        monitor = get_monitor()
        start_time = monitor.start_generation_timing()
        
        try:
            result = func(*args, **kwargs)
            
            # Extract parameters for metrics
            prompt = args[0] if args else kwargs.get('prompt', 'unknown')
            duration = kwargs.get('duration_seconds', 0.0)
            model_name = getattr(args[0], 'model_name', 'unknown') if args else 'unknown'
            
            monitor.record_generation_metrics(
                start_time, prompt, duration, result, model_name
            )
            
            return result
        except Exception as e:
            logging.error(f"Audio generation failed: {e}")
            raise
    
    return wrapper