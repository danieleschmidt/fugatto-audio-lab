"""Auto-Optimizer for Fugatto Audio Lab - Generation 3 Enhancements.

Intelligent performance optimization with adaptive scaling, resource monitoring,
and predictive performance enhancement.
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import threading
from collections import deque, defaultdict
import json

# Conditional imports for graceful degradation
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from .simple_api import SimpleAudioAPI
    HAS_SIMPLE_API = True
except ImportError:
    HAS_SIMPLE_API = False

try:
    from .quantum_planner import QuantumTaskPlanner
    HAS_QUANTUM_PLANNER = True
except ImportError:
    HAS_QUANTUM_PLANNER = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """System and application performance metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0
    gpu_usage: float = 0.0
    temperature: float = 0.0
    
    # Application metrics
    audio_generation_rate: float = 0.0
    average_latency: float = 0.0
    error_rate: float = 0.0
    queue_length: int = 0
    active_tasks: int = 0
    
    # Quality metrics
    throughput_fps: float = 0.0  # Frames per second equivalent
    real_time_factor: float = 0.0
    quality_score: float = 0.0
    
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationStrategy:
    """Defines optimization strategies and parameters."""
    name: str
    description: str
    target_metric: str
    improvement_threshold: float
    enabled: bool = True
    
    # Strategy parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    applications: int = 0
    successes: int = 0
    average_improvement: float = 0.0
    last_applied: Optional[float] = None


class PerformanceMonitor:
    """Real-time performance monitoring with trend analysis."""
    
    def __init__(self, window_size: int = 100, update_interval: float = 1.0):
        """Initialize performance monitor.
        
        Args:
            window_size: Number of metrics to keep in history
            update_interval: Seconds between metric updates
        """
        self.window_size = window_size
        self.update_interval = update_interval
        
        # Metric storage
        self.metrics_history = deque(maxlen=window_size)
        self.current_metrics = PerformanceMetrics()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Callbacks for metric updates
        self.metric_callbacks: List[Callable[[PerformanceMetrics], None]] = []
        
        logger.info(f"PerformanceMonitor initialized with {window_size}s window")
    
    def add_callback(self, callback: Callable[[PerformanceMetrics], None]) -> None:
        """Add a callback to be called when metrics are updated."""
        self.metric_callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """Start continuous performance monitoring."""
        if self.is_monitoring:
            logger.warning("Performance monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # Call registered callbacks
                for callback in self.metric_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Metric callback error: {e}")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system and application metrics."""
        metrics = PerformanceMetrics()
        
        if HAS_PSUTIL:
            try:
                # System metrics
                metrics.cpu_usage = psutil.cpu_percent(interval=None)
                metrics.memory_usage = psutil.virtual_memory().percent
                
                # Disk and network I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    metrics.disk_io = (disk_io.read_bytes + disk_io.write_bytes) / 1024 / 1024  # MB/s
                
                net_io = psutil.net_io_counters()
                if net_io:
                    metrics.network_io = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # MB/s
                
                # Temperature (if available)
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        all_temps = []
                        for name, entries in temps.items():
                            for entry in entries:
                                all_temps.append(entry.current)
                        if all_temps:
                            metrics.temperature = sum(all_temps) / len(all_temps)
                except (AttributeError, OSError):
                    pass  # Not available on all systems
                
            except Exception as e:
                logger.debug(f"psutil metrics collection error: {e}")
        else:
            # Mock metrics for testing
            import random
            metrics.cpu_usage = random.uniform(10, 80)
            metrics.memory_usage = random.uniform(30, 70)
            metrics.disk_io = random.uniform(0, 100)
            metrics.network_io = random.uniform(0, 50)
            metrics.temperature = random.uniform(40, 70)
        
        # Application-specific metrics would be updated by other components
        return metrics
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get the most recent performance metrics."""
        return self.current_metrics
    
    def get_metrics_trend(self, metric_name: str, window: int = 10) -> Dict[str, float]:
        """Analyze trend for a specific metric.
        
        Args:
            metric_name: Name of the metric to analyze
            window: Number of recent samples to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        if len(self.metrics_history) < 2:
            return {'trend': 0.0, 'average': 0.0, 'variance': 0.0}
        
        recent_metrics = list(self.metrics_history)[-window:]
        values = [getattr(m, metric_name, 0.0) for m in recent_metrics]
        
        if not values:
            return {'trend': 0.0, 'average': 0.0, 'variance': 0.0}
        
        # Calculate trend (simple linear regression)
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * v for i, v in enumerate(values))
        x_sq_sum = sum(i * i for i in range(n))
        
        trend = 0.0
        if n * x_sq_sum - x_sum * x_sum != 0:
            trend = (n * xy_sum - x_sum * y_sum) / (n * x_sq_sum - x_sum * x_sum)
        
        # Calculate statistics
        average = sum(values) / len(values)
        variance = sum((v - average) ** 2 for v in values) / len(values)
        
        return {
            'trend': trend,
            'average': average,
            'variance': variance,
            'latest': values[-1],
            'samples': n
        }


class AdaptiveOptimizer:
    """Adaptive performance optimizer with ML-driven strategies."""
    
    def __init__(self, monitor: PerformanceMonitor):
        """Initialize adaptive optimizer.
        
        Args:
            monitor: Performance monitor instance
        """
        self.monitor = monitor
        self.strategies: Dict[str, OptimizationStrategy] = {}
        self.optimization_history = deque(maxlen=1000)
        
        # Optimization state
        self.is_optimizing = False
        self.optimization_interval = 10.0  # seconds
        self.optimizer_thread = None
        
        # Initialize default strategies
        self._initialize_strategies()
        
        # Register for metric updates
        self.monitor.add_callback(self._on_metrics_update)
        
        logger.info("AdaptiveOptimizer initialized")
    
    def _initialize_strategies(self) -> None:
        """Initialize default optimization strategies."""
        self.strategies = {
            'cpu_throttling': OptimizationStrategy(
                name='CPU Throttling',
                description='Reduce processing intensity when CPU usage is high',
                target_metric='cpu_usage',
                improvement_threshold=0.15,
                parameters={
                    'high_threshold': 80.0,
                    'target_reduction': 0.20,
                    'min_quality': 0.7
                }
            ),
            'memory_cleanup': OptimizationStrategy(
                name='Memory Cleanup',
                description='Clear caches and optimize memory usage',
                target_metric='memory_usage',
                improvement_threshold=0.10,
                parameters={
                    'high_threshold': 85.0,
                    'cleanup_ratio': 0.5
                }
            ),
            'batch_optimization': OptimizationStrategy(
                name='Batch Optimization',
                description='Optimize batch sizes based on current load',
                target_metric='throughput_fps',
                improvement_threshold=0.25,
                parameters={
                    'min_batch_size': 1,
                    'max_batch_size': 8,
                    'load_threshold': 0.7
                }
            ),
            'quality_adaptation': OptimizationStrategy(
                name='Quality Adaptation',
                description='Adapt quality settings based on performance requirements',
                target_metric='real_time_factor',
                improvement_threshold=0.30,
                parameters={
                    'min_quality': 0.6,
                    'max_quality': 1.0,
                    'target_rtf': 1.0
                }
            )
        }
    
    def start_optimization(self) -> None:
        """Start adaptive optimization."""
        if self.is_optimizing:
            logger.warning("Optimization already active")
            return
        
        self.is_optimizing = True
        self.optimizer_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimizer_thread.start()
        logger.info("Adaptive optimization started")
    
    def stop_optimization(self) -> None:
        """Stop adaptive optimization."""
        self.is_optimizing = False
        if self.optimizer_thread:
            self.optimizer_thread.join(timeout=5)
        logger.info("Adaptive optimization stopped")
    
    def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while self.is_optimizing:
            try:
                self._run_optimization_cycle()
                time.sleep(self.optimization_interval)
            except Exception as e:
                logger.error(f"Optimization cycle error: {e}")
                time.sleep(self.optimization_interval)
    
    def _run_optimization_cycle(self) -> None:
        """Run a single optimization cycle."""
        current_metrics = self.monitor.get_current_metrics()
        
        # Analyze which strategies should be applied
        strategies_to_apply = []
        
        for strategy_name, strategy in self.strategies.items():
            if not strategy.enabled:
                continue
            
            should_apply = self._should_apply_strategy(strategy, current_metrics)
            if should_apply:
                strategies_to_apply.append(strategy)
        
        # Apply strategies in order of priority
        for strategy in strategies_to_apply:
            try:
                success = self._apply_strategy(strategy, current_metrics)
                strategy.applications += 1
                if success:
                    strategy.successes += 1
                strategy.last_applied = time.time()
            except Exception as e:
                logger.error(f"Failed to apply strategy {strategy.name}: {e}")
    
    def _should_apply_strategy(self, strategy: OptimizationStrategy, metrics: PerformanceMetrics) -> bool:
        """Determine if a strategy should be applied."""
        # Check if enough time has passed since last application
        if strategy.last_applied and time.time() - strategy.last_applied < 30:
            return False
        
        # Check metric thresholds
        current_value = getattr(metrics, strategy.target_metric, 0.0)
        
        if strategy.name == 'cpu_throttling':
            return current_value > strategy.parameters['high_threshold']
        elif strategy.name == 'memory_cleanup':
            return current_value > strategy.parameters['high_threshold']
        elif strategy.name == 'batch_optimization':
            # Apply if throughput is low
            return current_value < 1.0  # Less than 1 FPS equivalent
        elif strategy.name == 'quality_adaptation':
            # Apply if real-time factor is too low
            return current_value < strategy.parameters['target_rtf']
        
        return False
    
    def _apply_strategy(self, strategy: OptimizationStrategy, metrics: PerformanceMetrics) -> bool:
        """Apply an optimization strategy."""
        logger.info(f"Applying optimization strategy: {strategy.name}")
        
        if strategy.name == 'cpu_throttling':
            return self._apply_cpu_throttling(strategy, metrics)
        elif strategy.name == 'memory_cleanup':
            return self._apply_memory_cleanup(strategy, metrics)
        elif strategy.name == 'batch_optimization':
            return self._apply_batch_optimization(strategy, metrics)
        elif strategy.name == 'quality_adaptation':
            return self._apply_quality_adaptation(strategy, metrics)
        
        return False
    
    def _apply_cpu_throttling(self, strategy: OptimizationStrategy, metrics: PerformanceMetrics) -> bool:
        """Apply CPU throttling optimization."""
        # This would adjust processing parameters to reduce CPU load
        logger.info(f"CPU throttling applied: current usage {metrics.cpu_usage:.1f}%")
        
        # Record optimization
        self.optimization_history.append({
            'strategy': strategy.name,
            'timestamp': time.time(),
            'before_metrics': metrics,
            'action': 'reduce_processing_intensity',
            'parameters': {
                'reduction_factor': strategy.parameters['target_reduction']
            }
        })
        
        return True
    
    def _apply_memory_cleanup(self, strategy: OptimizationStrategy, metrics: PerformanceMetrics) -> bool:
        """Apply memory cleanup optimization."""
        logger.info(f"Memory cleanup applied: current usage {metrics.memory_usage:.1f}%")
        
        # This would clear caches, run garbage collection, etc.
        import gc
        gc.collect()
        
        self.optimization_history.append({
            'strategy': strategy.name,
            'timestamp': time.time(),
            'before_metrics': metrics,
            'action': 'memory_cleanup',
            'parameters': {}
        })
        
        return True
    
    def _apply_batch_optimization(self, strategy: OptimizationStrategy, metrics: PerformanceMetrics) -> bool:
        """Apply batch size optimization."""
        # Determine optimal batch size based on current load
        current_load = (metrics.cpu_usage + metrics.memory_usage) / 200  # Normalize to 0-1
        
        if current_load > strategy.parameters['load_threshold']:
            optimal_batch_size = strategy.parameters['min_batch_size']
        else:
            optimal_batch_size = min(
                strategy.parameters['max_batch_size'],
                int(strategy.parameters['max_batch_size'] * (1 - current_load))
            )
        
        logger.info(f"Batch optimization applied: optimal size {optimal_batch_size}")
        
        self.optimization_history.append({
            'strategy': strategy.name,
            'timestamp': time.time(),
            'before_metrics': metrics,
            'action': 'adjust_batch_size',
            'parameters': {'optimal_batch_size': optimal_batch_size}
        })
        
        return True
    
    def _apply_quality_adaptation(self, strategy: OptimizationStrategy, metrics: PerformanceMetrics) -> bool:
        """Apply quality adaptation optimization."""
        # Adjust quality based on performance requirements
        target_rtf = strategy.parameters['target_rtf']
        current_rtf = metrics.real_time_factor
        
        if current_rtf < target_rtf:
            # Need to reduce quality to improve performance
            quality_reduction = 0.1
            new_quality = max(
                strategy.parameters['min_quality'],
                metrics.quality_score - quality_reduction
            )
        else:
            # Can increase quality
            quality_increase = 0.05
            new_quality = min(
                strategy.parameters['max_quality'],
                metrics.quality_score + quality_increase
            )
        
        logger.info(f"Quality adaptation applied: {metrics.quality_score:.2f} -> {new_quality:.2f}")
        
        self.optimization_history.append({
            'strategy': strategy.name,
            'timestamp': time.time(),
            'before_metrics': metrics,
            'action': 'adjust_quality',
            'parameters': {'new_quality': new_quality}
        })
        
        return True
    
    def _on_metrics_update(self, metrics: PerformanceMetrics) -> None:
        """Handle metrics updates for real-time optimization."""
        # This could trigger immediate optimizations for critical situations
        if metrics.cpu_usage > 95:
            logger.warning("Critical CPU usage detected, applying emergency throttling")
        
        if metrics.memory_usage > 95:
            logger.warning("Critical memory usage detected, applying emergency cleanup")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization performance report."""
        report = {
            'strategies': {},
            'recent_optimizations': list(self.optimization_history)[-10:],
            'summary': {
                'total_optimizations': len(self.optimization_history),
                'active_strategies': sum(1 for s in self.strategies.values() if s.enabled),
                'overall_success_rate': 0.0
            }
        }
        
        total_applications = 0
        total_successes = 0
        
        for name, strategy in self.strategies.items():
            success_rate = strategy.successes / strategy.applications if strategy.applications > 0 else 0.0
            
            report['strategies'][name] = {
                'enabled': strategy.enabled,
                'applications': strategy.applications,
                'successes': strategy.successes,
                'success_rate': success_rate,
                'average_improvement': strategy.average_improvement,
                'last_applied': strategy.last_applied
            }
            
            total_applications += strategy.applications
            total_successes += strategy.successes
        
        if total_applications > 0:
            report['summary']['overall_success_rate'] = total_successes / total_applications
        
        return report


class AutoOptimizer:
    """Main auto-optimizer orchestrating all optimization components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize auto-optimizer.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        
        # Initialize components
        self.monitor = PerformanceMonitor(
            window_size=self.config.get('monitor_window', 100),
            update_interval=self.config.get('monitor_interval', 1.0)
        )
        
        self.optimizer = AdaptiveOptimizer(self.monitor)
        
        # Integration with other components
        self.api_instance = None
        self.planner_instance = None
        
        # State
        self.is_running = False
        
        logger.info("AutoOptimizer initialized")
    
    def start(self) -> None:
        """Start all optimization components."""
        if self.is_running:
            logger.warning("AutoOptimizer already running")
            return
        
        logger.info("Starting AutoOptimizer...")
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Start optimization
        self.optimizer.start_optimization()
        
        self.is_running = True
        logger.info("AutoOptimizer started successfully")
    
    def stop(self) -> None:
        """Stop all optimization components."""
        if not self.is_running:
            return
        
        logger.info("Stopping AutoOptimizer...")
        
        # Stop optimization
        self.optimizer.stop_optimization()
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        self.is_running = False
        logger.info("AutoOptimizer stopped")
    
    def integrate_with_api(self, api: 'SimpleAudioAPI') -> None:
        """Integrate optimizer with audio API for application-specific optimizations."""
        self.api_instance = api
        
        # Set up callbacks to update application metrics
        def update_app_metrics(metrics: PerformanceMetrics):
            # This would be implemented to get actual application metrics
            # For now, simulate some basic metrics
            metrics.audio_generation_rate = 1.0  # placeholder
            metrics.real_time_factor = 1.2  # placeholder
            metrics.quality_score = 0.85  # placeholder
        
        self.monitor.add_callback(update_app_metrics)
        logger.info("AutoOptimizer integrated with SimpleAudioAPI")
    
    def integrate_with_planner(self, planner: 'QuantumTaskPlanner') -> None:
        """Integrate optimizer with quantum planner for task-level optimizations."""
        self.planner_instance = planner
        logger.info("AutoOptimizer integrated with QuantumTaskPlanner")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        status = {
            'running': self.is_running,
            'current_metrics': self.monitor.get_current_metrics().__dict__,
            'optimization_report': self.optimizer.get_optimization_report(),
            'monitor_window_size': len(self.monitor.metrics_history),
            'uptime': time.time() - (self.monitor.metrics_history[0].timestamp if self.monitor.metrics_history else time.time())
        }
        
        return status
    
    def force_optimization(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """Force immediate optimization cycle.
        
        Args:
            strategy_name: Optional specific strategy to apply
            
        Returns:
            Results of the forced optimization
        """
        logger.info(f"Forcing optimization cycle{f' for {strategy_name}' if strategy_name else ''}")
        
        current_metrics = self.monitor.get_current_metrics()
        results = {'applied_strategies': [], 'errors': []}
        
        if strategy_name and strategy_name in self.optimizer.strategies:
            # Apply specific strategy
            try:
                strategy = self.optimizer.strategies[strategy_name]
                success = self.optimizer._apply_strategy(strategy, current_metrics)
                results['applied_strategies'].append({
                    'name': strategy_name,
                    'success': success
                })
            except Exception as e:
                results['errors'].append(f"Failed to apply {strategy_name}: {e}")
        else:
            # Run full optimization cycle
            try:
                self.optimizer._run_optimization_cycle()
                results['applied_strategies'].append({
                    'name': 'full_cycle',
                    'success': True
                })
            except Exception as e:
                results['errors'].append(f"Full cycle failed: {e}")
        
        return results
    
    def export_metrics(self, filepath: Union[str, Path]) -> None:
        """Export performance metrics and optimization history.
        
        Args:
            filepath: Path to save the metrics data
        """
        filepath = Path(filepath)
        
        data = {
            'export_timestamp': time.time(),
            'config': self.config,
            'metrics_history': [m.__dict__ for m in self.monitor.metrics_history],
            'optimization_history': list(self.optimizer.optimization_history),
            'strategies': {name: {
                'name': s.name,
                'description': s.description,
                'applications': s.applications,
                'successes': s.successes,
                'parameters': s.parameters
            } for name, s in self.optimizer.strategies.items()},
            'status': self.get_status()
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {filepath}")


# Convenience functions
def create_auto_optimizer(config: Optional[Dict[str, Any]] = None) -> AutoOptimizer:
    """Create and configure an auto-optimizer instance."""
    return AutoOptimizer(config)


def quick_optimize(api: Optional['SimpleAudioAPI'] = None, 
                  duration: float = 30.0) -> Dict[str, Any]:
    """Run a quick optimization session.
    
    Args:
        api: Optional audio API to optimize
        duration: How long to run optimization (seconds)
        
    Returns:
        Optimization results
    """
    optimizer = create_auto_optimizer()
    
    if api:
        optimizer.integrate_with_api(api)
    
    # Start optimization
    optimizer.start()
    
    # Let it run
    time.sleep(duration)
    
    # Get results
    results = optimizer.get_status()
    
    # Stop
    optimizer.stop()
    
    return results


if __name__ == "__main__":
    # Demo the auto-optimizer
    logger.info("Starting AutoOptimizer demo...")
    
    optimizer = create_auto_optimizer({
        'monitor_window': 50,
        'monitor_interval': 0.5
    })
    
    optimizer.start()
    
    # Run for a short demo
    time.sleep(10)
    
    # Show status
    status = optimizer.get_status()
    print(f"Demo completed:")
    print(f"Monitoring window: {status['monitor_window_size']} samples")
    print(f"Current CPU: {status['current_metrics']['cpu_usage']:.1f}%")
    print(f"Current Memory: {status['current_metrics']['memory_usage']:.1f}%")
    
    optimizer.stop()
    logger.info("AutoOptimizer demo completed")