"""Fugatto Audio Lab: Toolkit for Controllable Audio Generation.

A plug-and-play generative audio playground with live "prompt → sound" preview
for NVIDIA's Fugatto transformer with text+audio multi-conditioning.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

from fugatto_lab.core import FugattoModel, AudioProcessor
from fugatto_lab.quantum_planner import QuantumTaskPlanner, QuantumTask, TaskPriority
from fugatto_lab.intelligent_scheduler import IntelligentScheduler, SchedulingStrategy
from fugatto_lab.robust_error_handling import RobustErrorHandler, ValidationError, InputValidator
from fugatto_lab.advanced_monitoring import AdvancedMonitoringSystem, MetricsCollector
from fugatto_lab.security_framework import SecurityManager, SecurityContext
from fugatto_lab.performance_optimization import PerformanceOptimizer, HighPerformanceCache
from fugatto_lab.auto_scaling import AutoScaler, LoadBalancer

__all__ = [
    "FugattoModel", 
    "AudioProcessor", 
    "QuantumTaskPlanner", 
    "QuantumTask", 
    "TaskPriority",
    "IntelligentScheduler",
    "SchedulingStrategy",
    "RobustErrorHandler",
    "ValidationError",
    "InputValidator",
    "AdvancedMonitoringSystem",
    "MetricsCollector",
    "SecurityManager",
    "SecurityContext",
    "PerformanceOptimizer",
    "HighPerformanceCache",
    "AutoScaler",
    "LoadBalancer",
    "__version__"
]