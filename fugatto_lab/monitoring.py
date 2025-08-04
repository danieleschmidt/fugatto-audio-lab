"""Performance monitoring and metrics collection for Fugatto Audio Lab.

This module provides comprehensive monitoring capabilities including
performance metrics, health checks, and observability features.
"""

import time
import logging
import psutil
import sys
import subprocess
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


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


class HealthChecker:
    """System health checker for Fugatto Audio Lab components."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def check_python_version(self) -> Dict[str, Any]:
        """Check Python version compatibility."""
        version = sys.version_info
        major, minor = version.major, version.minor
        
        # Require Python 3.10+
        if major >= 3 and minor >= 10:
            return {
                'healthy': True,
                'message': f'Python {major}.{minor}.{version.micro} ✓',
                'version': f'{major}.{minor}.{version.micro}'
            }
        else:
            return {
                'healthy': False,
                'message': f'Python {major}.{minor}.{version.micro} (requires 3.10+)',
                'version': f'{major}.{minor}.{version.micro}'
            }
    
    def check_torch_installation(self) -> Dict[str, Any]:
        """Check PyTorch installation and CUDA availability."""
        try:
            import torch
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                message = f'PyTorch {torch_version} with CUDA ({gpu_count} GPU(s): {gpu_name}) ✓'
            else:
                message = f'PyTorch {torch_version} (CPU only) ⚠️'
            
            return {
                'healthy': True,
                'message': message,
                'version': torch_version,
                'cuda_available': cuda_available,
                'gpu_count': gpu_count if cuda_available else 0
            }
        except ImportError:
            return {
                'healthy': False,
                'message': 'PyTorch not installed ❌',
                'version': None,
                'cuda_available': False,
                'gpu_count': 0
            }
    
    def check_audio_libraries(self) -> Dict[str, Any]:
        """Check audio processing libraries."""
        libraries = {}
        overall_healthy = True
        
        # Check core audio libraries
        for lib_name, import_name in [
            ('librosa', 'librosa'),
            ('soundfile', 'soundfile'),
            ('numpy', 'numpy'),
            ('scipy', 'scipy')
        ]:
            try:
                lib = __import__(import_name)
                version = getattr(lib, '__version__', 'unknown')
                libraries[lib_name] = {
                    'installed': True,
                    'version': version,
                    'status': '✓'
                }
            except ImportError:
                libraries[lib_name] = {
                    'installed': False,
                    'version': None,
                    'status': '❌'
                }
                if lib_name in ['numpy']:  # Critical libraries
                    overall_healthy = False
        
        # Check optional libraries
        for lib_name, import_name in [
            ('transformers', 'transformers'),
            ('gradio', 'gradio'),
            ('matplotlib', 'matplotlib')
        ]:
            try:
                lib = __import__(import_name)
                version = getattr(lib, '__version__', 'unknown')
                libraries[lib_name] = {
                    'installed': True,
                    'version': version,
                    'status': '✓'
                }
            except ImportError:
                libraries[lib_name] = {
                    'installed': False,
                    'version': None,
                    'status': '⚠️ (optional)'
                }
        
        installed_count = sum(1 for lib in libraries.values() if lib['installed'])
        total_count = len(libraries)
        
        return {
            'healthy': overall_healthy,
            'message': f'Audio libraries: {installed_count}/{total_count} available',
            'libraries': libraries
        }
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            used_percent = (disk_usage.used / disk_usage.total) * 100
            
            if used_percent > 90:
                healthy = False
                message = f'Disk space critically low: {free_gb:.1f}GB free ({used_percent:.1f}% used) ❌'
            elif used_percent > 80:
                healthy = True
                message = f'Disk space low: {free_gb:.1f}GB free ({used_percent:.1f}% used) ⚠️'
            else:
                healthy = True
                message = f'Disk space: {free_gb:.1f}GB free ({100-used_percent:.1f}% available) ✓'
            
            return {
                'healthy': healthy,
                'message': message,
                'free_gb': free_gb,
                'total_gb': total_gb,
                'used_percent': used_percent
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f'Could not check disk space: {e}',
                'free_gb': None,
                'total_gb': None,
                'used_percent': None
            }
    
    def check_memory(self) -> Dict[str, Any]:
        """Check system memory."""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            used_percent = memory.percent
            
            if used_percent > 90:
                healthy = False
                message = f'Memory critically low: {available_gb:.1f}GB available ({used_percent:.1f}% used) ❌'
            elif used_percent > 80:
                healthy = True
                message = f'Memory usage high: {available_gb:.1f}GB available ({used_percent:.1f}% used) ⚠️'
            else:
                healthy = True
                message = f'Memory: {available_gb:.1f}GB available ({used_percent:.1f}% used) ✓'
            
            return {
                'healthy': healthy,
                'message': message,
                'available_gb': available_gb,
                'total_gb': total_gb,
                'used_percent': used_percent
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f'Could not check memory: {e}',
                'available_gb': None,
                'total_gb': None,
                'used_percent': None
            }
    
    def check_model_availability(self) -> Dict[str, Any]:
        """Check if models can be loaded."""
        try:
            from .core import FugattoModel
            
            # Try to initialize a model (this doesn't actually load weights)
            model = FugattoModel("nvidia/fugatto-base")
            
            return {
                'healthy': True,
                'message': 'Model initialization successful ✓',
                'model_class_available': True
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f'Model initialization failed: {e} ❌',
                'model_class_available': False
            }
    
    def check_all(self) -> Dict[str, Dict[str, Any]]:
        """Run all health checks."""
        checks = {
            'Python Version': self.check_python_version(),
            'PyTorch': self.check_torch_installation(),
            'Audio Libraries': self.check_audio_libraries(),
            'Disk Space': self.check_disk_space(),
            'Memory': self.check_memory(),
            'Model Loading': self.check_model_availability()
        }
        
        self.logger.info("Health check completed")
        return checks

def get_monitor():
    """Get monitor instance."""
    return HealthChecker()
