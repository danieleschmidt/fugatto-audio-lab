"""Advanced Monitoring and Observability System for Fugatto Audio Lab.

Comprehensive monitoring, metrics collection, alerting, and observability
for production audio processing workflows.
"""

import time
import json
import asyncio
import threading
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics for monitoring."""
    COUNTER = "counter"           # Monotonically increasing values
    GAUGE = "gauge"              # Point-in-time values
    HISTOGRAM = "histogram"       # Distribution of values
    TIMER = "timer"              # Time-based measurements
    RATE = "rate"                # Rate of events per time unit


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"        # Immediate action required
    WARNING = "warning"          # Attention needed
    INFO = "info"               # Informational


@dataclass
class MetricValue:
    """Individual metric measurement."""
    
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "type": self.metric_type.value
        }


@dataclass
class Alert:
    """Alert definition and state."""
    
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str
    threshold: float
    metric_name: str
    triggered: bool = False
    trigger_time: Optional[float] = None
    resolve_time: Optional[float] = None
    trigger_count: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "condition": self.condition,
            "threshold": self.threshold,
            "metric_name": self.metric_name,
            "triggered": self.triggered,
            "trigger_time": self.trigger_time,
            "resolve_time": self.resolve_time,
            "trigger_count": self.trigger_count,
            "tags": self.tags
        }


class MetricsCollector:
    """High-performance metrics collection and aggregation."""
    
    def __init__(self, buffer_size: int = 10000, flush_interval: float = 30.0):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Metric storage
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.aggregated_metrics = defaultdict(list)
        self.metric_metadata = {}
        
        # Statistics tracking
        self.metric_counts = defaultdict(int)
        self.last_flush_time = time.time()
        
        # Background processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.running = True
        self.flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self.flush_thread.start()
        
        logger.info("MetricsCollector initialized")
    
    def record_metric(self, name: str, value: float, 
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Dict[str, str] = None):
        """Record a metric measurement."""
        metric = MetricValue(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {}
        )
        
        self.metrics_buffer.append(metric)
        self.metric_counts[name] += 1
        
        # Store metadata
        if name not in self.metric_metadata:
            self.metric_metadata[name] = {
                "type": metric_type.value,
                "first_seen": time.time(),
                "tags_seen": set()
            }
        
        # Track unique tag combinations
        if tags:
            tag_signature = tuple(sorted(tags.items()))
            self.metric_metadata[name]["tags_seen"].add(tag_signature)
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, tags)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric value."""
        self.record_metric(name, value, MetricType.GAUGE, tags)
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timer measurement."""
        self.record_metric(name, duration, MetricType.TIMER, tags)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value."""
        self.record_metric(name, value, MetricType.HISTOGRAM, tags)
    
    def get_metric_summary(self, name: str, window_seconds: float = 300) -> Dict[str, Any]:
        """Get statistical summary of a metric over time window."""
        cutoff_time = time.time() - window_seconds
        
        # Filter metrics within time window
        recent_values = [
            metric.value for metric in self.metrics_buffer
            if metric.name == name and metric.timestamp >= cutoff_time
        ]
        
        if not recent_values:
            return {"name": name, "count": 0, "window_seconds": window_seconds}
        
        values = np.array(recent_values)
        
        return {
            "name": name,
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "sum": float(np.sum(values)),
            "window_seconds": window_seconds
        }
    
    def _flush_loop(self):
        """Background thread for periodic metric flushing."""
        while self.running:
            try:
                time.sleep(self.flush_interval)
                self._flush_metrics()
            except Exception as e:
                logger.error(f"Metrics flush error: {e}")
    
    def _flush_metrics(self):
        """Flush metrics to aggregated storage."""
        current_time = time.time()
        
        # Convert buffer to list for processing
        metrics_to_process = list(self.metrics_buffer)
        
        # Aggregate metrics by name and type
        for metric in metrics_to_process:
            key = f"{metric.name}:{metric.metric_type.value}"
            self.aggregated_metrics[key].append(metric)
        
        # Trim old metrics from aggregated storage
        cutoff_time = current_time - 3600  # Keep 1 hour of data
        for key in self.aggregated_metrics:
            self.aggregated_metrics[key] = [
                m for m in self.aggregated_metrics[key]
                if m.timestamp >= cutoff_time
            ]
        
        self.last_flush_time = current_time
        logger.debug(f"Flushed metrics: {len(metrics_to_process)} metrics processed")
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics and metadata."""
        return {
            "buffer_size": len(self.metrics_buffer),
            "total_metrics": sum(self.metric_counts.values()),
            "unique_metrics": len(self.metric_counts),
            "metric_counts": dict(self.metric_counts),
            "metadata": {
                name: {
                    "type": meta["type"],
                    "first_seen": meta["first_seen"],
                    "unique_tag_combinations": len(meta["tags_seen"])
                }
                for name, meta in self.metric_metadata.items()
            },
            "last_flush": self.last_flush_time
        }
    
    def shutdown(self):
        """Shutdown metrics collector."""
        self.running = False
        if self.flush_thread.is_alive():
            self.flush_thread.join(timeout=5.0)
        self.executor.shutdown(wait=True)


class AlertManager:
    """Alert management and notification system."""
    
    def __init__(self, check_interval: float = 10.0):
        self.check_interval = check_interval
        self.alerts = {}  # alert_id -> Alert
        self.alert_handlers = []  # List of alert handler functions
        self.alert_history = deque(maxlen=1000)
        
        # Alert evaluation state
        self.running = True
        self.check_thread = threading.Thread(target=self._check_loop, daemon=True)
        self.check_thread.start()
        
        logger.info("AlertManager initialized")
    
    def add_alert(self, alert: Alert):
        """Add an alert definition."""
        self.alerts[alert.alert_id] = alert
        logger.info(f"Added alert: {alert.name}")
    
    def create_threshold_alert(self, alert_id: str, name: str, description: str,
                             metric_name: str, condition: str, threshold: float,
                             severity: AlertSeverity = AlertSeverity.WARNING,
                             tags: Dict[str, str] = None) -> Alert:
        """Create a threshold-based alert."""
        alert = Alert(
            alert_id=alert_id,
            name=name,
            description=description,
            severity=severity,
            condition=condition,
            threshold=threshold,
            metric_name=metric_name,
            tags=tags or {}
        )
        
        self.add_alert(alert)
        return alert
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__name__}")
    
    def check_alerts(self, metrics_collector: MetricsCollector):
        """Check all alerts against current metrics."""
        for alert in self.alerts.values():
            try:
                self._evaluate_alert(alert, metrics_collector)
            except Exception as e:
                logger.error(f"Alert evaluation error for {alert.name}: {e}")
    
    def _evaluate_alert(self, alert: Alert, metrics_collector: MetricsCollector):
        """Evaluate a single alert."""
        # Get recent metric summary
        metric_summary = metrics_collector.get_metric_summary(alert.metric_name, 60)
        
        if metric_summary["count"] == 0:
            return  # No data to evaluate
        
        # Get value based on condition
        if alert.condition == "mean_above":
            value = metric_summary["mean"]
            triggered = value > alert.threshold
        elif alert.condition == "mean_below":
            value = metric_summary["mean"]
            triggered = value < alert.threshold
        elif alert.condition == "max_above":
            value = metric_summary["max"]
            triggered = value > alert.threshold
        elif alert.condition == "min_below":
            value = metric_summary["min"]
            triggered = value < alert.threshold
        elif alert.condition == "p95_above":
            value = metric_summary["p95"]
            triggered = value > alert.threshold
        elif alert.condition == "count_above":
            value = metric_summary["count"]
            triggered = value > alert.threshold
        else:
            logger.warning(f"Unknown alert condition: {alert.condition}")
            return
        
        # Check for state change
        was_triggered = alert.triggered
        
        if triggered and not was_triggered:
            # Alert triggered
            alert.triggered = True
            alert.trigger_time = time.time()
            alert.trigger_count += 1
            
            # Notify handlers
            self._notify_alert_triggered(alert, value)
            
        elif not triggered and was_triggered:
            # Alert resolved
            alert.triggered = False
            alert.resolve_time = time.time()
            
            # Notify handlers
            self._notify_alert_resolved(alert, value)
    
    def _notify_alert_triggered(self, alert: Alert, value: float):
        """Notify handlers that alert was triggered."""
        logger.warning(f"ALERT TRIGGERED: {alert.name} - {alert.description} (value: {value})")
        
        # Add to history
        self.alert_history.append({
            "alert_id": alert.alert_id,
            "name": alert.name,
            "action": "triggered",
            "value": value,
            "timestamp": time.time()
        })
        
        # Call handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def _notify_alert_resolved(self, alert: Alert, value: float):
        """Notify handlers that alert was resolved."""
        logger.info(f"ALERT RESOLVED: {alert.name} (value: {value})")
        
        # Add to history
        self.alert_history.append({
            "alert_id": alert.alert_id,
            "name": alert.name,
            "action": "resolved",
            "value": value,
            "timestamp": time.time()
        })
        
        # Call handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def _check_loop(self):
        """Background thread for periodic alert checking."""
        while self.running:
            try:
                time.sleep(self.check_interval)
                # Alert checking happens when metrics collector is passed
            except Exception as e:
                logger.error(f"Alert check loop error: {e}")
    
    def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert status."""
        active_alerts = [alert for alert in self.alerts.values() if alert.triggered]
        
        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active_alerts),
            "alert_history_size": len(self.alert_history),
            "alerts": [alert.to_dict() for alert in self.alerts.values()],
            "recent_history": list(self.alert_history)[-10:]
        }
    
    def shutdown(self):
        """Shutdown alert manager."""
        self.running = False
        if self.check_thread.is_alive():
            self.check_thread.join(timeout=5.0)


class PerformanceProfiler:
    """Performance profiling and bottleneck detection."""
    
    def __init__(self):
        self.profiles = {}
        self.call_counts = defaultdict(int)
        self.total_times = defaultdict(float)
        self.min_times = defaultdict(lambda: float('inf'))
        self.max_times = defaultdict(float)
        
        logger.info("PerformanceProfiler initialized")
    
    def profile_function(self, func_name: str):
        """Decorator for profiling function performance."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self._record_profile(func_name, duration)
            return wrapper
        return decorator
    
    def profile_async_function(self, func_name: str):
        """Decorator for profiling async function performance."""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self._record_profile(func_name, duration)
            return wrapper
        return decorator
    
    def _record_profile(self, func_name: str, duration: float):
        """Record profiling data for a function call."""
        self.call_counts[func_name] += 1
        self.total_times[func_name] += duration
        self.min_times[func_name] = min(self.min_times[func_name], duration)
        self.max_times[func_name] = max(self.max_times[func_name], duration)
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get performance profile summary."""
        summary = {}
        
        for func_name in self.call_counts:
            count = self.call_counts[func_name]
            total_time = self.total_times[func_name]
            avg_time = total_time / count if count > 0 else 0
            
            summary[func_name] = {
                "call_count": count,
                "total_time": total_time,
                "average_time": avg_time,
                "min_time": self.min_times[func_name],
                "max_time": self.max_times[func_name],
                "time_percentage": 0  # Will be calculated below
            }
        
        # Calculate time percentages
        total_system_time = sum(self.total_times.values())
        if total_system_time > 0:
            for func_name in summary:
                summary[func_name]["time_percentage"] = (
                    summary[func_name]["total_time"] / total_system_time * 100
                )
        
        return summary
    
    def get_bottlenecks(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        summary = self.get_profile_summary()
        
        # Sort by total time descending
        sorted_functions = sorted(
            summary.items(),
            key=lambda x: x[1]["total_time"],
            reverse=True
        )
        
        return [
            {"function": name, **stats}
            for name, stats in sorted_functions[:top_n]
        ]


class SystemMonitor:
    """System resource monitoring and health checks."""
    
    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self.system_metrics = deque(maxlen=1000)
        self.health_checks = {}
        
        # Background monitoring
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("SystemMonitor initialized")
    
    def add_health_check(self, name: str, check_func: Callable[[], bool], 
                        description: str = ""):
        """Add a health check function."""
        self.health_checks[name] = {
            "function": check_func,
            "description": description,
            "last_check": 0,
            "last_result": None,
            "failure_count": 0
        }
        logger.info(f"Added health check: {name}")
    
    def _monitor_loop(self):
        """Background system monitoring loop."""
        while self.running:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)
                
                # Run health checks
                self._run_health_checks()
                
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"System monitor error: {e}")
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                network_stats = {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            except:
                network_stats = {}
            
            return {
                "timestamp": time.time(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "load_avg_1m": load_avg[0],
                    "load_avg_5m": load_avg[1],
                    "load_avg_15m": load_avg[2]
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "swap": {
                    "total": swap.total,
                    "used": swap.used,
                    "free": swap.free,
                    "percent": swap.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent
                },
                "network": network_stats
            }
            
        except ImportError:
            # Fallback without psutil
            return {
                "timestamp": time.time(),
                "cpu": {"percent": 0, "count": 1},
                "memory": {"percent": 0, "total": 0},
                "disk": {"percent": 0, "total": 0},
                "network": {}
            }
    
    def _run_health_checks(self):
        """Run all registered health checks."""
        current_time = time.time()
        
        for name, check_info in self.health_checks.items():
            try:
                result = check_info["function"]()
                check_info["last_check"] = current_time
                check_info["last_result"] = result
                
                if not result:
                    check_info["failure_count"] += 1
                    logger.warning(f"Health check failed: {name}")
                else:
                    check_info["failure_count"] = 0
                    
            except Exception as e:
                check_info["last_check"] = current_time
                check_info["last_result"] = False
                check_info["failure_count"] += 1
                logger.error(f"Health check error for {name}: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        if not self.system_metrics:
            return {"status": "no_data"}
        
        latest = self.system_metrics[-1]
        
        # Calculate health score
        health_score = self._calculate_health_score()
        
        return {
            "timestamp": latest["timestamp"],
            "health_score": health_score,
            "cpu_percent": latest["cpu"]["percent"],
            "memory_percent": latest["memory"]["percent"],
            "disk_percent": latest["disk"]["percent"],
            "load_average": latest["cpu"].get("load_avg_1m", 0),
            "health_checks": {
                name: {
                    "status": info["last_result"],
                    "failure_count": info["failure_count"],
                    "last_check": info["last_check"]
                }
                for name, info in self.health_checks.items()
            }
        }
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        if not self.system_metrics:
            return 0.0
        
        latest = self.system_metrics[-1]
        score = 100.0
        
        # CPU score
        cpu_percent = latest["cpu"]["percent"]
        if cpu_percent > 90:
            score -= 30
        elif cpu_percent > 70:
            score -= 15
        elif cpu_percent > 50:
            score -= 5
        
        # Memory score
        memory_percent = latest["memory"]["percent"]
        if memory_percent > 95:
            score -= 25
        elif memory_percent > 85:
            score -= 15
        elif memory_percent > 70:
            score -= 5
        
        # Disk score
        disk_percent = latest["disk"]["percent"]
        if disk_percent > 95:
            score -= 20
        elif disk_percent > 85:
            score -= 10
        elif disk_percent > 75:
            score -= 5
        
        # Health checks score
        failed_checks = sum(1 for info in self.health_checks.values()
                          if info["last_result"] is False)
        if failed_checks > 0:
            score -= failed_checks * 10
        
        return max(0.0, score)
    
    def shutdown(self):
        """Shutdown system monitor."""
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)


class AdvancedMonitoringSystem:
    """Comprehensive monitoring system orchestrator."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.metrics_collector = MetricsCollector(
            buffer_size=self.config.get("metrics_buffer_size", 10000),
            flush_interval=self.config.get("metrics_flush_interval", 30.0)
        )
        
        self.alert_manager = AlertManager(
            check_interval=self.config.get("alert_check_interval", 10.0)
        )
        
        self.profiler = PerformanceProfiler()
        
        self.system_monitor = SystemMonitor(
            check_interval=self.config.get("system_check_interval", 5.0)
        )
        
        # Setup default alerts
        self._setup_default_alerts()
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Background alert checking
        self.alert_check_thread = threading.Thread(target=self._alert_check_loop, daemon=True)
        self.alert_check_thread.start()
        
        logger.info("AdvancedMonitoringSystem initialized")
    
    def _setup_default_alerts(self):
        """Setup default system alerts."""
        # CPU usage alert
        self.alert_manager.create_threshold_alert(
            alert_id="cpu_high",
            name="High CPU Usage",
            description="CPU usage is above 85%",
            metric_name="system.cpu.percent",
            condition="mean_above",
            threshold=85.0,
            severity=AlertSeverity.WARNING
        )
        
        # Memory usage alert
        self.alert_manager.create_threshold_alert(
            alert_id="memory_high",
            name="High Memory Usage",
            description="Memory usage is above 90%",
            metric_name="system.memory.percent",
            condition="mean_above",
            threshold=90.0,
            severity=AlertSeverity.CRITICAL
        )
        
        # Disk usage alert
        self.alert_manager.create_threshold_alert(
            alert_id="disk_high",
            name="High Disk Usage",
            description="Disk usage is above 90%",
            metric_name="system.disk.percent",
            condition="mean_above",
            threshold=90.0,
            severity=AlertSeverity.WARNING
        )
        
        # Error rate alert
        self.alert_manager.create_threshold_alert(
            alert_id="error_rate_high",
            name="High Error Rate",
            description="Error rate is above 5%",
            metric_name="errors.rate",
            condition="mean_above",
            threshold=5.0,
            severity=AlertSeverity.WARNING
        )
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        
        def check_memory_available():
            """Check if sufficient memory is available."""
            try:
                import psutil
                memory = psutil.virtual_memory()
                return memory.available > 1024 * 1024 * 1024  # 1GB minimum
            except:
                return True  # Default to healthy if can't check
        
        def check_disk_space():
            """Check if sufficient disk space is available."""
            try:
                import psutil
                disk = psutil.disk_usage('/')
                return disk.free > 5 * 1024 * 1024 * 1024  # 5GB minimum
            except:
                return True
        
        self.system_monitor.add_health_check(
            "memory_available",
            check_memory_available,
            "Check if at least 1GB memory is available"
        )
        
        self.system_monitor.add_health_check(
            "disk_space",
            check_disk_space,
            "Check if at least 5GB disk space is available"
        )
    
    def _alert_check_loop(self):
        """Background thread for checking alerts."""
        while True:
            try:
                time.sleep(self.alert_manager.check_interval)
                self.alert_manager.check_alerts(self.metrics_collector)
                
                # Collect system metrics
                system_status = self.system_monitor.get_system_status()
                if system_status.get("status") != "no_data":
                    # Record system metrics
                    self.metrics_collector.set_gauge("system.cpu.percent", 
                                                   system_status["cpu_percent"])
                    self.metrics_collector.set_gauge("system.memory.percent", 
                                                   system_status["memory_percent"])
                    self.metrics_collector.set_gauge("system.disk.percent", 
                                                   system_status["disk_percent"])
                    self.metrics_collector.set_gauge("system.health.score", 
                                                   system_status["health_score"])
            except Exception as e:
                logger.error(f"Alert check loop error: {e}")
    
    def record_audio_processing_metrics(self, operation: str, duration: float, 
                                      success: bool, file_size: int = 0,
                                      input_duration: float = 0):
        """Record metrics for audio processing operations."""
        tags = {"operation": operation}
        
        # Record processing time
        self.metrics_collector.record_timer(f"audio.processing.duration", duration, tags)
        
        # Record success/failure
        if success:
            self.metrics_collector.increment_counter("audio.processing.success", 1.0, tags)
        else:
            self.metrics_collector.increment_counter("audio.processing.failure", 1.0, tags)
        
        # Record throughput metrics
        if file_size > 0:
            throughput = file_size / duration if duration > 0 else 0
            self.metrics_collector.set_gauge("audio.processing.throughput_bytes_per_sec", 
                                           throughput, tags)
        
        if input_duration > 0:
            realtime_factor = input_duration / duration if duration > 0 else 0
            self.metrics_collector.set_gauge("audio.processing.realtime_factor", 
                                           realtime_factor, tags)
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        return {
            "timestamp": time.time(),
            "metrics": self.metrics_collector.get_all_metrics(),
            "alerts": self.alert_manager.get_alert_status(),
            "system": self.system_monitor.get_system_status(),
            "performance": {
                "profile_summary": self.profiler.get_profile_summary(),
                "bottlenecks": self.profiler.get_bottlenecks(5)
            }
        }
    
    def save_monitoring_report(self, filepath: str):
        """Save comprehensive monitoring report."""
        report = self.get_comprehensive_status()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Monitoring report saved to: {filepath}")
    
    def shutdown(self):
        """Shutdown all monitoring components."""
        logger.info("Shutting down monitoring system")
        
        self.metrics_collector.shutdown()
        self.alert_manager.shutdown()
        self.system_monitor.shutdown()


# Global monitoring instance
_global_monitoring = None

def get_global_monitoring() -> AdvancedMonitoringSystem:
    """Get global monitoring system instance."""
    global _global_monitoring
    if _global_monitoring is None:
        _global_monitoring = AdvancedMonitoringSystem()
    return _global_monitoring


# Convenient decorator for monitoring function performance
def monitor_performance(operation_name: str = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        func_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitoring = get_global_monitoring()
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                monitoring.metrics_collector.record_timer(f"function.{func_name}.duration", duration)
                
                if success:
                    monitoring.metrics_collector.increment_counter(f"function.{func_name}.success")
                else:
                    monitoring.metrics_collector.increment_counter(f"function.{func_name}.failure")
        
        return wrapper
    return decorator


# Async version
def monitor_async_performance(operation_name: str = None):
    """Decorator to monitor async function performance."""
    def decorator(func):
        func_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            monitoring = get_global_monitoring()
            start_time = time.time()
            success = True
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                monitoring.metrics_collector.record_timer(f"function.{func_name}.duration", duration)
                
                if success:
                    monitoring.metrics_collector.increment_counter(f"function.{func_name}.success")
                else:
                    monitoring.metrics_collector.increment_counter(f"function.{func_name}.failure")
        
        return wrapper
    return decorator