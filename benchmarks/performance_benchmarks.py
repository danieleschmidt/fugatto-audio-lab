"""Performance benchmarking suite for Fugatto Audio Lab.

This module provides comprehensive benchmarking capabilities to measure
and track performance improvements across different system configurations.
"""

import time
import json
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

from fugatto_lab.core import FugattoModel, AudioProcessor
from fugatto_lab.monitoring import get_monitor


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks."""
    
    name: str
    description: str
    iterations: int = 10
    warmup_iterations: int = 3
    audio_durations: List[float] = None
    prompt_lengths: List[int] = None
    
    def __post_init__(self):
        if self.audio_durations is None:
            self.audio_durations = [1.0, 5.0, 10.0, 30.0]
        if self.prompt_lengths is None:
            self.prompt_lengths = [10, 50, 100, 200]


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    
    config_name: str
    metric_name: str
    value: float
    unit: str
    timestamp: str
    metadata: Dict[str, Any] = None


class PerformanceBenchmark:
    """Performance benchmarking framework."""
    
    def __init__(self, model: FugattoModel, processor: AudioProcessor):
        self.model = model
        self.processor = processor
        self.monitor = get_monitor()
        self.results: List[BenchmarkResult] = []
    
    def warmup(self, iterations: int = 3) -> None:
        """Warm up the system before benchmarking."""
        print(f"Warming up system with {iterations} iterations...")
        for i in range(iterations):
            _ = self.model.generate("warmup prompt", duration_seconds=1.0)
    
    def benchmark_generation_latency(self, config: BenchmarkConfig) -> Dict[str, float]:
        """Benchmark audio generation latency."""
        print(f"Benchmarking generation latency: {config.name}")
        
        latencies = []
        
        for duration in config.audio_durations:
            duration_latencies = []
            
            for i in range(config.iterations):
                prompt = f"Benchmark prompt for {duration}s audio generation {i}"
                
                start_time = time.time()
                audio = self.model.generate(prompt, duration_seconds=duration)
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                duration_latencies.append(latency_ms)
            
            # Record statistics for this duration
            avg_latency = statistics.mean(duration_latencies)
            self._record_result(config.name, f"latency_{duration}s", avg_latency, "ms")
            
            latencies.extend(duration_latencies)
        
        overall_stats = {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "min": min(latencies),
            "max": max(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99)
        }
        
        return overall_stats
    
    def benchmark_throughput(self, config: BenchmarkConfig) -> Dict[str, float]:
        """Benchmark audio generation throughput."""
        print(f"Benchmarking throughput: {config.name}")
        
        total_audio_duration = 0.0
        total_wall_time = 0.0
        
        start_time = time.time()
        
        for i in range(config.iterations):
            duration = 5.0  # Fixed duration for throughput testing
            prompt = f"Throughput test prompt {i}"
            
            gen_start = time.time()
            audio = self.model.generate(prompt, duration_seconds=duration)
            gen_end = time.time()
            
            total_audio_duration += duration
            total_wall_time += (gen_end - gen_start)
        
        end_time = time.time()
        
        # Calculate throughput metrics
        real_time_factor = total_audio_duration / total_wall_time
        overall_throughput = total_audio_duration / (end_time - start_time)
        
        throughput_stats = {
            "real_time_factor": real_time_factor,
            "audio_seconds_per_wall_second": overall_throughput,
            "generations_per_minute": (config.iterations / (end_time - start_time)) * 60
        }
        
        self._record_result(config.name, "real_time_factor", real_time_factor, "x")
        self._record_result(config.name, "throughput", overall_throughput, "audio_s/wall_s")
        
        return throughput_stats
    
    def benchmark_memory_usage(self, config: BenchmarkConfig) -> Dict[str, float]:
        """Benchmark memory usage during generation."""
        print(f"Benchmarking memory usage: {config.name}")
        
        import psutil
        process = psutil.Process()
        
        memory_usage = []
        peak_memory = 0
        
        for duration in config.audio_durations:
            for i in range(config.iterations):
                # Measure memory before generation
                mem_before = process.memory_info().rss / (1024 * 1024)  # MB
                
                prompt = f"Memory test prompt {i}"
                audio = self.model.generate(prompt, duration_seconds=duration)
                
                # Measure memory after generation
                mem_after = process.memory_info().rss / (1024 * 1024)  # MB
                mem_diff = mem_after - mem_before
                
                memory_usage.append(mem_diff)
                peak_memory = max(peak_memory, mem_after)
        
        memory_stats = {
            "avg_memory_increase_mb": statistics.mean(memory_usage),
            "max_memory_increase_mb": max(memory_usage),
            "peak_memory_usage_mb": peak_memory,
            "min_memory_increase_mb": min(memory_usage)
        }
        
        self._record_result(config.name, "avg_memory_increase", 
                          memory_stats["avg_memory_increase_mb"], "MB")
        
        return memory_stats
    
    def benchmark_audio_quality_metrics(self, config: BenchmarkConfig) -> Dict[str, float]:
        """Benchmark audio quality consistency."""
        print(f"Benchmarking audio quality: {config.name}")
        
        audio_qualities = []
        
        for i in range(config.iterations):
            prompt = "High quality audio generation test"
            audio = self.model.generate(prompt, duration_seconds=5.0)
            
            # Basic quality metrics
            if audio is not None and len(audio) > 0:
                # Signal-to-noise ratio approximation
                signal_power = np.mean(audio ** 2)
                noise_floor = np.percentile(np.abs(audio), 10)  # Bottom 10% as noise
                snr = 10 * np.log10(signal_power / (noise_floor ** 2 + 1e-10))
                
                # Dynamic range
                dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (noise_floor + 1e-10))
                
                audio_qualities.append({
                    "snr_db": snr,
                    "dynamic_range_db": dynamic_range,
                    "peak_amplitude": np.max(np.abs(audio)),
                    "rms_level": np.sqrt(np.mean(audio ** 2))
                })
        
        if audio_qualities:
            quality_stats = {
                "avg_snr_db": statistics.mean([q["snr_db"] for q in audio_qualities]),
                "avg_dynamic_range_db": statistics.mean([q["dynamic_range_db"] for q in audio_qualities]),
                "avg_peak_amplitude": statistics.mean([q["peak_amplitude"] for q in audio_qualities]),
                "avg_rms_level": statistics.mean([q["rms_level"] for q in audio_qualities])
            }
        else:
            quality_stats = {"error": "No valid audio generated"}
        
        return quality_stats
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmark suites."""
        print("Starting comprehensive performance benchmark...")
        
        configs = [
            BenchmarkConfig("latency_test", "Audio generation latency", iterations=5),
            BenchmarkConfig("throughput_test", "Audio generation throughput", iterations=10),
            BenchmarkConfig("memory_test", "Memory usage during generation", iterations=5),
            BenchmarkConfig("quality_test", "Audio quality consistency", iterations=8)
        ]
        
        results = {}
        
        # Warm up system
        self.warmup()
        
        for config in configs:
            if "latency" in config.name:
                results[config.name] = self.benchmark_generation_latency(config)
            elif "throughput" in config.name:
                results[config.name] = self.benchmark_throughput(config)
            elif "memory" in config.name:
                results[config.name] = self.benchmark_memory_usage(config)
            elif "quality" in config.name:
                results[config.name] = self.benchmark_audio_quality_metrics(config)
        
        # Generate summary report
        summary = {
            "benchmark_timestamp": datetime.utcnow().isoformat(),
            "total_benchmarks": len(configs),
            "results": results,
            "system_info": self._get_system_info(),
            "model_info": {
                "name": getattr(self.model, 'model_name', 'unknown'),
                "class": self.model.__class__.__name__
            }
        }
        
        return summary
    
    def _record_result(self, config_name: str, metric_name: str, 
                      value: float, unit: str, metadata: Dict[str, Any] = None) -> None:
        """Record a benchmark result."""
        result = BenchmarkResult(
            config_name=config_name,
            metric_name=metric_name,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow().isoformat(),
            metadata=metadata or {}
        )
        self.results.append(result)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
        }
    
    def save_results(self, filepath: str) -> None:
        """Save benchmark results to file."""
        results_data = {
            "results": [asdict(r) for r in self.results],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Benchmark results saved to {filepath}")


# Benchmark runner function for CLI usage
def run_benchmarks(model_name: str = "nvidia/fugatto-base") -> Dict[str, Any]:
    """Run benchmarks and return results."""
    from fugatto_lab.core import FugattoModel, AudioProcessor
    
    model = FugattoModel.from_pretrained(model_name)
    processor = AudioProcessor()
    
    benchmark = PerformanceBenchmark(model, processor)
    return benchmark.run_comprehensive_benchmark()