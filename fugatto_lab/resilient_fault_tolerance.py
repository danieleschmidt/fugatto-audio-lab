"""
Resilient Fault Tolerance System with Advanced Recovery Mechanisms
Generation 2: Comprehensive Fault Tolerance, Circuit Breakers, and Self-Healing
"""

import time
import math
import threading
import asyncio
import random
import traceback
import functools
import inspect
import numpy as np
from typing import Dict, Any, Optional, List, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import logging
import json
from pathlib import Path
import uuid

# Fault tolerance components
class FaultType(Enum):
    """Types of faults the system can handle."""
    TRANSIENT = "transient"  # Temporary network issues, timeouts
    INTERMITTENT = "intermittent"  # Occasional failures
    PERMANENT = "permanent"  # Persistent failures requiring intervention
    CASCADING = "cascading"  # Failures that propagate
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # Memory, CPU, disk issues
    EXTERNAL_DEPENDENCY = "external_dependency"  # Third-party service failures
    DATA_CORRUPTION = "data_corruption"  # Data integrity issues
    CONFIGURATION = "configuration"  # Configuration errors

class RecoveryStrategy(Enum):
    """Recovery strategies for different fault types."""
    RETRY = "retry"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK = "fallback"
    BULKHEAD = "bulkhead"
    TIMEOUT = "timeout"
    RATE_LIMITING = "rate_limiting"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    SELF_HEALING = "self_healing"
    ROLLBACK = "rollback"
    FAILOVER = "failover"

class SystemState(Enum):
    """System operational states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILING = "failing"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"

@dataclass
class FaultEvent:
    """Fault event information."""
    fault_id: str
    fault_type: FaultType
    severity: str  # low, medium, high, critical
    component: str
    description: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_attempts: int = 0
    resolved: bool = False
    resolution_time: Optional[float] = None
    
    def __post_init__(self):
        if not self.fault_id:
            self.fault_id = str(uuid.uuid4())
        if self.timestamp == 0:
            self.timestamp = time.time()

@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking."""
    state: str = "closed"  # closed, open, half_open
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    open_timeout: float = 60.0  # Seconds to wait before half-open
    failure_threshold: int = 5
    success_threshold: int = 3
    
    def should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        return (self.state == "open" and 
                time.time() - self.last_failure_time > self.open_timeout)

@dataclass
class HealthCheck:
    """Health check configuration and state."""
    name: str
    check_function: Callable[[], bool]
    interval: float = 30.0  # Seconds between checks
    timeout: float = 10.0  # Timeout for check
    last_check: float = 0.0
    last_result: bool = True
    consecutive_failures: int = 0
    failure_threshold: int = 3

class ResilientFaultTolerance:
    """
    Advanced fault tolerance system with self-healing capabilities.
    
    Generation 2 Features:
    - Circuit breakers for external dependencies
    - Intelligent retry mechanisms with backoff
    - Bulkhead isolation patterns
    - Self-healing and auto-recovery
    - Health monitoring and alerting
    - Graceful degradation strategies
    - Resource protection and rate limiting
    - Fault injection for testing
    - Comprehensive metrics and observability
    """
    
    def __init__(self,
                 enable_circuit_breakers: bool = True,
                 enable_health_monitoring: bool = True,
                 enable_self_healing: bool = True,
                 max_retry_attempts: int = 3,
                 base_retry_delay: float = 1.0):
        """
        Initialize resilient fault tolerance system.
        
        Args:
            enable_circuit_breakers: Enable circuit breaker patterns
            enable_health_monitoring: Enable health check monitoring
            enable_self_healing: Enable self-healing mechanisms
            max_retry_attempts: Default maximum retry attempts
            base_retry_delay: Base delay for exponential backoff
        """
        self.enable_circuit_breakers = enable_circuit_breakers
        self.enable_health_monitoring = enable_health_monitoring
        self.enable_self_healing = enable_self_healing
        self.max_retry_attempts = max_retry_attempts
        self.base_retry_delay = base_retry_delay
        
        # System state management
        self.system_state = SystemState.HEALTHY
        self.state_change_history: List[Tuple[SystemState, float]] = []
        self.component_states: Dict[str, SystemState] = {}
        
        # Fault tracking
        self.active_faults: Dict[str, FaultEvent] = {}
        self.fault_history: deque = deque(maxlen=10000)  # Keep last 10k faults
        self.fault_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.circuit_breaker_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Health monitoring
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_monitor_running = False
        self.health_monitor_thread: Optional[threading.Thread] = None
        
        # Retry mechanisms
        self.retry_policies: Dict[str, Dict[str, Any]] = {}
        self.active_retries: Dict[str, Dict[str, Any]] = {}
        
        # Bulkhead isolation
        self.resource_pools: Dict[str, ThreadPoolExecutor] = {}
        self.resource_limits: Dict[str, int] = {}
        
        # Self-healing mechanisms
        self.healing_strategies: Dict[str, Callable] = {}
        self.healing_history: List[Dict[str, Any]] = []
        
        # Metrics and monitoring
        self.metrics = {
            'total_faults': 0,
            'resolved_faults': 0,
            'active_faults': 0,
            'circuit_breaker_trips': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'health_check_failures': 0,
            'self_healing_events': 0,
            'retry_attempts': 0,
            'timeout_events': 0
        }
        
        # Configuration
        self.fault_tolerance_config = {
            'circuit_breaker_failure_threshold': 5,
            'circuit_breaker_recovery_timeout': 60.0,
            'circuit_breaker_success_threshold': 3,
            'health_check_interval': 30.0,
            'health_check_timeout': 10.0,
            'max_concurrent_recoveries': 5,
            'fault_correlation_window': 300.0,  # 5 minutes
            'auto_healing_enabled': True,
            'degradation_thresholds': {
                'cpu_usage': 80.0,
                'memory_usage': 85.0,
                'error_rate': 5.0,
                'response_time': 5000.0  # ms
            }
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize default recovery strategies
        self._initialize_recovery_strategies()
        
        # Initialize default health checks
        self._initialize_default_health_checks()
        
        # Start health monitoring if enabled
        if self.enable_health_monitoring:
            self.start_health_monitoring()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("ResilientFaultTolerance system initialized")

    def resilient_call(self, func: Callable, *args, 
                      component_name: str = None,
                      retry_policy: Optional[Dict[str, Any]] = None,
                      timeout: Optional[float] = None,
                      circuit_breaker: bool = True,
                      fallback: Optional[Callable] = None,
                      **kwargs) -> Any:
        """
        Execute function call with comprehensive fault tolerance.
        
        Args:
            func: Function to execute
            *args: Function arguments
            component_name: Component name for tracking
            retry_policy: Custom retry policy
            timeout: Call timeout in seconds
            circuit_breaker: Enable circuit breaker protection
            fallback: Fallback function if main function fails
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
        """
        component_name = component_name or func.__name__
        
        # Check circuit breaker
        if circuit_breaker and self.enable_circuit_breakers:
            cb_check = self._check_circuit_breaker(component_name)
            if not cb_check['allow_call']:
                if fallback:
                    return self._execute_fallback(fallback, *args, **kwargs)
                raise Exception(f"Circuit breaker open for {component_name}")
        
        # Execute with retry and fault tolerance
        return self._execute_with_resilience(
            func, args, kwargs, component_name, retry_policy, timeout, fallback
        )

    def register_circuit_breaker(self, component_name: str,
                                failure_threshold: int = 5,
                                recovery_timeout: float = 60.0,
                                success_threshold: int = 3) -> None:
        """
        Register circuit breaker for component.
        
        Args:
            component_name: Name of component to protect
            failure_threshold: Failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            success_threshold: Successes needed to close circuit
        """
        with self._lock:
            self.circuit_breakers[component_name] = CircuitBreakerState(
                failure_threshold=failure_threshold,
                open_timeout=recovery_timeout,
                success_threshold=success_threshold
            )
        
        self.logger.info(f"Circuit breaker registered for {component_name}")

    def register_health_check(self, name: str, check_function: Callable[[], bool],
                            interval: float = 30.0, timeout: float = 10.0,
                            failure_threshold: int = 3) -> None:
        """
        Register health check for monitoring.
        
        Args:
            name: Health check name
            check_function: Function that returns True if healthy
            interval: Check interval in seconds
            timeout: Check timeout in seconds
            failure_threshold: Consecutive failures before unhealthy
        """
        health_check = HealthCheck(
            name=name,
            check_function=check_function,
            interval=interval,
            timeout=timeout,
            failure_threshold=failure_threshold
        )
        
        with self._lock:
            self.health_checks[name] = health_check
        
        self.logger.info(f"Health check registered: {name}")

    def register_healing_strategy(self, fault_pattern: str, 
                                 healing_function: Callable[[FaultEvent], bool]) -> None:
        """
        Register self-healing strategy for fault pattern.
        
        Args:
            fault_pattern: Pattern to match for healing
            healing_function: Function to execute for healing
        """
        with self._lock:
            self.healing_strategies[fault_pattern] = healing_function
        
        self.logger.info(f"Healing strategy registered for pattern: {fault_pattern}")

    def report_fault(self, fault_type: FaultType, component: str, 
                    description: str, severity: str = "medium",
                    context: Optional[Dict[str, Any]] = None,
                    exception: Optional[Exception] = None) -> str:
        """
        Report a fault to the tolerance system.
        
        Args:
            fault_type: Type of fault
            component: Component where fault occurred
            description: Fault description
            severity: Fault severity (low, medium, high, critical)
            context: Additional context information
            exception: Exception that caused the fault
            
        Returns:
            Fault ID for tracking
        """
        fault_event = FaultEvent(
            fault_id="",  # Will be generated
            fault_type=fault_type,
            severity=severity,
            component=component,
            description=description,
            context=context or {},
            stack_trace=traceback.format_exc() if exception else None
        )
        
        with self._lock:
            self.active_faults[fault_event.fault_id] = fault_event
            self.fault_history.append(fault_event)
            self.fault_patterns[component].append(fault_event.timestamp)
            
            # Update metrics
            self.metrics['total_faults'] += 1
            self.metrics['active_faults'] = len(self.active_faults)
            
            # Update circuit breaker
            if self.enable_circuit_breakers and component in self.circuit_breakers:
                self._record_circuit_breaker_failure(component)
            
            # Update component state
            self._update_component_state(component, fault_event)
            
            # Update system state
            self._update_system_state()
        
        # Trigger recovery if enabled
        if self.enable_self_healing:
            self._trigger_self_healing(fault_event)
        
        self.logger.warning(f"Fault reported: {fault_event.fault_id} in {component}")
        return fault_event.fault_id

    def resolve_fault(self, fault_id: str, resolution_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Mark fault as resolved.
        
        Args:
            fault_id: ID of fault to resolve
            resolution_info: Information about resolution
            
        Returns:
            True if fault was resolved successfully
        """
        with self._lock:
            if fault_id not in self.active_faults:
                return False
            
            fault = self.active_faults[fault_id]
            fault.resolved = True
            fault.resolution_time = time.time()
            
            if resolution_info:
                fault.context.update(resolution_info)
            
            # Remove from active faults
            del self.active_faults[fault_id]
            
            # Update metrics
            self.metrics['resolved_faults'] += 1
            self.metrics['active_faults'] = len(self.active_faults)
            
            # Update component state if needed
            self._check_component_recovery(fault.component)
        
        self.logger.info(f"Fault resolved: {fault_id}")
        return True

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status.
        
        Returns:
            System health information
        """
        with self._lock:
            # Component health status
            component_health = {}
            for name, state in self.component_states.items():
                component_health[name] = {
                    'state': state.value,
                    'circuit_breaker': self.circuit_breakers.get(name, {}).state if name in self.circuit_breakers else None,
                    'active_faults': len([f for f in self.active_faults.values() if f.component == name])
                }
            
            # Health check status
            health_check_status = {}
            for name, check in self.health_checks.items():
                health_check_status[name] = {
                    'healthy': check.last_result,
                    'last_check': check.last_check,
                    'consecutive_failures': check.consecutive_failures
                }
            
            # Recent fault summary
            recent_faults = [
                f for f in self.fault_history 
                if time.time() - f.timestamp < 3600  # Last hour
            ]
            
            fault_summary = {
                'total_recent': len(recent_faults),
                'by_severity': defaultdict(int),
                'by_component': defaultdict(int),
                'by_type': defaultdict(int)
            }
            
            for fault in recent_faults:
                fault_summary['by_severity'][fault.severity] += 1
                fault_summary['by_component'][fault.component] += 1
                fault_summary['by_type'][fault.fault_type.value] += 1
        
        return {
            'system_state': self.system_state.value,
            'component_health': component_health,
            'health_checks': health_check_status,
            'active_faults': len(self.active_faults),
            'recent_fault_summary': dict(fault_summary['by_severity']),
            'circuit_breakers': {
                name: cb.state for name, cb in self.circuit_breakers.items()
            },
            'metrics': self.metrics.copy(),
            'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }

    def inject_fault(self, component: str, fault_type: FaultType, 
                    duration: float = 10.0, severity: str = "medium") -> str:
        """
        Inject fault for testing purposes.
        
        Args:
            component: Component to inject fault into
            fault_type: Type of fault to inject
            duration: Duration of fault in seconds
            severity: Severity of injected fault
            
        Returns:
            Fault ID for tracking
        """
        fault_id = self.report_fault(
            fault_type, component,
            f"Injected fault for testing - {fault_type.value}",
            severity,
            {'injected': True, 'duration': duration}
        )
        
        # Schedule automatic resolution
        def resolve_injected_fault():
            time.sleep(duration)
            self.resolve_fault(fault_id, {'auto_resolved': True})
        
        threading.Thread(target=resolve_injected_fault, daemon=True).start()
        
        self.logger.info(f"Fault injected: {fault_id} in {component} for {duration}s")
        return fault_id

    # Internal methods for fault tolerance mechanisms
    
    def _execute_with_resilience(self, func: Callable, args: tuple, kwargs: dict,
                               component_name: str, retry_policy: Optional[Dict[str, Any]],
                               timeout: Optional[float], fallback: Optional[Callable]) -> Any:
        """Execute function with comprehensive resilience mechanisms."""
        # Get retry policy
        retry_config = retry_policy or self._get_default_retry_policy(component_name)
        
        last_exception = None
        
        for attempt in range(retry_config.get('max_attempts', self.max_retry_attempts)):
            try:
                # Execute with timeout if specified
                if timeout:
                    result = self._execute_with_timeout(func, args, kwargs, timeout)
                else:
                    result = func(*args, **kwargs)
                
                # Record success for circuit breaker
                if self.enable_circuit_breakers and component_name in self.circuit_breakers:
                    self._record_circuit_breaker_success(component_name)
                
                return result
                
            except Exception as e:
                last_exception = e
                self.metrics['retry_attempts'] += 1
                
                # Check if exception is retryable
                if not self._is_retryable_exception(e, retry_config):
                    break
                
                # Apply backoff before retry
                if attempt < retry_config.get('max_attempts', self.max_retry_attempts) - 1:
                    backoff_delay = self._calculate_backoff_delay(attempt, retry_config)
                    time.sleep(backoff_delay)
                
                # Report fault
                self.report_fault(
                    self._classify_exception(e),
                    component_name,
                    f"Function call failed: {str(e)}",
                    "medium",
                    {'attempt': attempt + 1, 'exception_type': type(e).__name__},
                    e
                )
        
        # All retries failed - try fallback
        if fallback:
            try:
                return self._execute_fallback(fallback, *args, **kwargs)
            except Exception as fallback_exception:
                self.logger.error(f"Fallback also failed for {component_name}: {fallback_exception}")
        
        # Update circuit breaker failure
        if self.enable_circuit_breakers and component_name in self.circuit_breakers:
            self._record_circuit_breaker_failure(component_name)
        
        # Final failure
        raise last_exception

    def _execute_with_timeout(self, func: Callable, args: tuple, kwargs: dict, timeout: float) -> Any:
        """Execute function with timeout protection."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout)
                return result
            except TimeoutError:
                self.metrics['timeout_events'] += 1
                raise Exception(f"Function call timed out after {timeout} seconds")

    def _execute_fallback(self, fallback: Callable, *args, **kwargs) -> Any:
        """Execute fallback function."""
        try:
            # Fallback functions might have different signatures
            sig = inspect.signature(fallback)
            if len(sig.parameters) == 0:
                return fallback()
            else:
                return fallback(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Fallback execution failed: {e}")
            raise

    def _check_circuit_breaker(self, component_name: str) -> Dict[str, Any]:
        """Check circuit breaker state for component."""
        if component_name not in self.circuit_breakers:
            return {'allow_call': True, 'state': 'not_configured'}
        
        cb = self.circuit_breakers[component_name]
        
        if cb.state == "closed":
            return {'allow_call': True, 'state': 'closed'}
        elif cb.state == "open":
            if cb.should_attempt_reset():
                cb.state = "half_open"
                return {'allow_call': True, 'state': 'half_open'}
            else:
                return {'allow_call': False, 'state': 'open'}
        elif cb.state == "half_open":
            return {'allow_call': True, 'state': 'half_open'}
        
        return {'allow_call': False, 'state': 'unknown'}

    def _record_circuit_breaker_failure(self, component_name: str) -> None:
        """Record failure for circuit breaker."""
        if component_name not in self.circuit_breakers:
            return
        
        cb = self.circuit_breakers[component_name]
        cb.failure_count += 1
        cb.last_failure_time = time.time()
        
        if cb.state == "closed" and cb.failure_count >= cb.failure_threshold:
            cb.state = "open"
            self.metrics['circuit_breaker_trips'] += 1
            self.logger.warning(f"Circuit breaker opened for {component_name}")
        elif cb.state == "half_open":
            cb.state = "open"
            self.logger.warning(f"Circuit breaker returned to open state for {component_name}")

    def _record_circuit_breaker_success(self, component_name: str) -> None:
        """Record success for circuit breaker."""
        if component_name not in self.circuit_breakers:
            return
        
        cb = self.circuit_breakers[component_name]
        cb.success_count += 1
        cb.last_success_time = time.time()
        
        if cb.state == "half_open" and cb.success_count >= cb.success_threshold:
            cb.state = "closed"
            cb.failure_count = 0
            cb.success_count = 0
            self.logger.info(f"Circuit breaker closed for {component_name}")

    def _get_default_retry_policy(self, component_name: str) -> Dict[str, Any]:
        """Get default retry policy for component."""
        if component_name in self.retry_policies:
            return self.retry_policies[component_name]
        
        return {
            'max_attempts': self.max_retry_attempts,
            'base_delay': self.base_retry_delay,
            'max_delay': 30.0,
            'exponential_backoff': True,
            'jitter': True,
            'retryable_exceptions': [
                'ConnectionError', 'TimeoutError', 'TemporaryFailure'
            ]
        }

    def _is_retryable_exception(self, exception: Exception, retry_config: Dict[str, Any]) -> bool:
        """Check if exception is retryable based on policy."""
        retryable_types = retry_config.get('retryable_exceptions', [])
        exception_type = type(exception).__name__
        
        # Check specific exception types
        if exception_type in retryable_types:
            return True
        
        # Check for transient failures
        transient_indicators = ['timeout', 'connection', 'network', 'temporary', 'unavailable']
        error_msg = str(exception).lower()
        
        return any(indicator in error_msg for indicator in transient_indicators)

    def _classify_exception(self, exception: Exception) -> FaultType:
        """Classify exception into fault type."""
        error_msg = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        
        if 'timeout' in error_msg or 'timeout' in exception_type:
            return FaultType.TRANSIENT
        elif 'connection' in error_msg or 'network' in error_msg:
            return FaultType.EXTERNAL_DEPENDENCY
        elif 'memory' in error_msg or 'resource' in error_msg:
            return FaultType.RESOURCE_EXHAUSTION
        elif 'permission' in error_msg or 'access' in error_msg:
            return FaultType.CONFIGURATION
        elif 'corrupt' in error_msg or 'invalid' in error_msg:
            return FaultType.DATA_CORRUPTION
        else:
            return FaultType.INTERMITTENT

    def _calculate_backoff_delay(self, attempt: int, retry_config: Dict[str, Any]) -> float:
        """Calculate backoff delay for retry attempt."""
        base_delay = retry_config.get('base_delay', self.base_retry_delay)
        max_delay = retry_config.get('max_delay', 30.0)
        
        if retry_config.get('exponential_backoff', True):
            delay = base_delay * (2 ** attempt)
        else:
            delay = base_delay
        
        # Apply jitter
        if retry_config.get('jitter', True):
            delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
        
        return min(delay, max_delay)

    def _update_component_state(self, component: str, fault_event: FaultEvent) -> None:
        """Update component state based on fault."""
        current_state = self.component_states.get(component, SystemState.HEALTHY)
        
        # Determine new state based on fault severity and frequency
        recent_faults = [
            f for f in self.fault_history
            if f.component == component and time.time() - f.timestamp < 300  # Last 5 minutes
        ]
        
        critical_faults = [f for f in recent_faults if f.severity == 'critical']
        high_faults = [f for f in recent_faults if f.severity == 'high']
        
        if len(critical_faults) > 0:
            new_state = SystemState.FAILING
        elif len(high_faults) >= 3:
            new_state = SystemState.CRITICAL
        elif len(recent_faults) >= 5:
            new_state = SystemState.DEGRADED
        else:
            new_state = current_state
        
        if new_state != current_state:
            self.component_states[component] = new_state
            self.logger.warning(f"Component {component} state changed: {current_state.value} -> {new_state.value}")

    def _update_system_state(self) -> None:
        """Update overall system state based on component states."""
        if not self.component_states:
            return
        
        component_states = list(self.component_states.values())
        
        # Determine system state based on worst component state
        if SystemState.FAILING in component_states:
            new_state = SystemState.FAILING
        elif SystemState.CRITICAL in component_states:
            new_state = SystemState.CRITICAL
        elif SystemState.DEGRADED in component_states:
            new_state = SystemState.DEGRADED
        else:
            new_state = SystemState.HEALTHY
        
        if new_state != self.system_state:
            self.state_change_history.append((self.system_state, time.time()))
            self.system_state = new_state
            self.logger.warning(f"System state changed to: {new_state.value}")

    def _check_component_recovery(self, component: str) -> None:
        """Check if component has recovered from faults."""
        recent_faults = [
            f for f in self.active_faults.values()
            if f.component == component
        ]
        
        if len(recent_faults) == 0:
            # No active faults - component might have recovered
            if component in self.component_states:
                old_state = self.component_states[component]
                if old_state != SystemState.HEALTHY:
                    self.component_states[component] = SystemState.HEALTHY
                    self.logger.info(f"Component {component} recovered: {old_state.value} -> healthy")
                    self._update_system_state()

    def _initialize_recovery_strategies(self) -> None:
        """Initialize default recovery strategies."""
        
        def memory_pressure_recovery(fault: FaultEvent) -> bool:
            """Recovery strategy for memory pressure."""
            try:
                # Trigger garbage collection
                import gc
                gc.collect()
                
                # Clear any internal caches (simplified)
                if hasattr(self, '_clear_caches'):
                    self._clear_caches()
                
                return True
            except Exception as e:
                self.logger.error(f"Memory pressure recovery failed: {e}")
                return False
        
        def connection_recovery(fault: FaultEvent) -> bool:
            """Recovery strategy for connection issues."""
            try:
                component = fault.component
                
                # Reset circuit breaker if applicable
                if component in self.circuit_breakers:
                    cb = self.circuit_breakers[component]
                    if cb.state == "open":
                        cb.state = "half_open"
                        self.logger.info(f"Attempting connection recovery for {component}")
                
                return True
            except Exception as e:
                self.logger.error(f"Connection recovery failed: {e}")
                return False
        
        def configuration_recovery(fault: FaultEvent) -> bool:
            """Recovery strategy for configuration issues."""
            try:
                # Reload configuration (simplified)
                self.logger.info(f"Attempting configuration recovery for {fault.component}")
                
                # In production, this would reload config files, environment variables, etc.
                return True
            except Exception as e:
                self.logger.error(f"Configuration recovery failed: {e}")
                return False
        
        # Register recovery strategies
        self.healing_strategies.update({
            'memory': memory_pressure_recovery,
            'connection': connection_recovery,
            'configuration': configuration_recovery
        })

    def _initialize_default_health_checks(self) -> None:
        """Initialize default health checks."""
        
        def system_health_check() -> bool:
            """Basic system health check."""
            try:
                # Check if system state is acceptable
                acceptable_states = [SystemState.HEALTHY, SystemState.DEGRADED]
                return self.system_state in acceptable_states
            except Exception:
                return False
        
        def memory_health_check() -> bool:
            """Memory usage health check."""
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                return memory_percent < self.fault_tolerance_config['degradation_thresholds']['memory_usage']
            except ImportError:
                # psutil not available, assume healthy
                return True
            except Exception:
                return False
        
        def circuit_breaker_health_check() -> bool:
            """Circuit breaker health check."""
            try:
                open_breakers = [
                    name for name, cb in self.circuit_breakers.items()
                    if cb.state == "open"
                ]
                # Health check fails if more than 50% of circuit breakers are open
                return len(open_breakers) <= len(self.circuit_breakers) * 0.5
            except Exception:
                return False
        
        # Register default health checks
        self.register_health_check("system_health", system_health_check, 30.0)
        self.register_health_check("memory_health", memory_health_check, 60.0)
        self.register_health_check("circuit_breakers", circuit_breaker_health_check, 45.0)

    def _trigger_self_healing(self, fault: FaultEvent) -> None:
        """Trigger self-healing mechanisms for fault."""
        if not self.enable_self_healing:
            return
        
        # Find matching healing strategies
        matching_strategies = []
        
        for pattern, strategy in self.healing_strategies.items():
            if (pattern in fault.component.lower() or 
                pattern in fault.description.lower() or
                pattern in fault.fault_type.value):
                matching_strategies.append((pattern, strategy))
        
        # Execute healing strategies
        for pattern, strategy in matching_strategies:
            try:
                self.logger.info(f"Attempting self-healing with strategy: {pattern}")
                
                success = strategy(fault)
                
                healing_event = {
                    'fault_id': fault.fault_id,
                    'strategy': pattern,
                    'success': success,
                    'timestamp': time.time()
                }
                
                self.healing_history.append(healing_event)
                
                if success:
                    self.metrics['successful_recoveries'] += 1
                    self.logger.info(f"Self-healing successful for {fault.component}")
                    
                    # Attempt to resolve the fault
                    self.resolve_fault(fault.fault_id, {
                        'resolution_method': 'self_healing',
                        'strategy': pattern
                    })
                else:
                    self.metrics['failed_recoveries'] += 1
                    self.logger.warning(f"Self-healing failed for {fault.component}")
                
                self.metrics['self_healing_events'] += 1
                
            except Exception as e:
                self.logger.error(f"Self-healing strategy {pattern} failed: {e}")
                self.metrics['failed_recoveries'] += 1

    def start_health_monitoring(self) -> None:
        """Start health monitoring background thread."""
        if self.health_monitor_running:
            return
        
        self.health_monitor_running = True
        self.start_time = time.time()
        
        def health_monitor_loop():
            while self.health_monitor_running:
                try:
                    current_time = time.time()
                    
                    for name, health_check in self.health_checks.items():
                        # Check if it's time for this health check
                        if current_time - health_check.last_check >= health_check.interval:
                            try:
                                # Execute health check with timeout
                                with ThreadPoolExecutor(max_workers=1) as executor:
                                    future = executor.submit(health_check.check_function)
                                    result = future.result(timeout=health_check.timeout)
                                
                                health_check.last_check = current_time
                                health_check.last_result = result
                                
                                if result:
                                    health_check.consecutive_failures = 0
                                else:
                                    health_check.consecutive_failures += 1
                                    self.metrics['health_check_failures'] += 1
                                    
                                    # Report fault if threshold exceeded
                                    if health_check.consecutive_failures >= health_check.failure_threshold:
                                        self.report_fault(
                                            FaultType.INTERMITTENT,
                                            f"health_check_{name}",
                                            f"Health check {name} failed {health_check.consecutive_failures} times",
                                            "high",
                                            {'health_check': name}
                                        )
                                
                            except Exception as e:
                                health_check.consecutive_failures += 1
                                self.metrics['health_check_failures'] += 1
                                self.logger.error(f"Health check {name} failed: {e}")
                    
                    # Sleep for a short interval
                    time.sleep(5.0)
                    
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    time.sleep(10.0)  # Longer sleep on error
        
        self.health_monitor_thread = threading.Thread(
            target=health_monitor_loop,
            daemon=True,
            name="health_monitor"
        )
        self.health_monitor_thread.start()
        
        self.logger.info("Health monitoring started")

    def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        self.health_monitor_running = False
        
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            self.health_monitor_thread.join(timeout=5.0)
        
        self.logger.info("Health monitoring stopped")

    def get_fault_analysis(self) -> Dict[str, Any]:
        """Get comprehensive fault analysis and recommendations."""
        current_time = time.time()
        
        # Analyze fault patterns
        pattern_analysis = {}
        for component, timestamps in self.fault_patterns.items():
            recent_faults = [t for t in timestamps if current_time - t < 3600]  # Last hour
            
            if recent_faults:
                fault_rate = len(recent_faults) / 3600  # Faults per second
                pattern_analysis[component] = {
                    'recent_fault_count': len(recent_faults),
                    'fault_rate': fault_rate,
                    'trend': 'increasing' if fault_rate > 0.001 else 'stable'
                }
        
        # Recovery effectiveness
        recovery_stats = {
            'total_healing_attempts': len(self.healing_history),
            'successful_healings': len([h for h in self.healing_history if h['success']]),
            'healing_success_rate': 0.0
        }
        
        if recovery_stats['total_healing_attempts'] > 0:
            recovery_stats['healing_success_rate'] = (
                recovery_stats['successful_healings'] / recovery_stats['total_healing_attempts']
            )
        
        # Recommendations
        recommendations = self._generate_fault_recommendations(pattern_analysis)
        
        return {
            'fault_patterns': pattern_analysis,
            'recovery_effectiveness': recovery_stats,
            'circuit_breaker_status': {
                name: {
                    'state': cb.state,
                    'failure_count': cb.failure_count,
                    'success_count': cb.success_count
                }
                for name, cb in self.circuit_breakers.items()
            },
            'system_resilience_score': self._calculate_resilience_score(),
            'recommendations': recommendations
        }

    def _generate_fault_recommendations(self, pattern_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations based on fault patterns."""
        recommendations = []
        
        # High fault rate components
        for component, stats in pattern_analysis.items():
            if stats['fault_rate'] > 0.01:  # More than 1 fault per 100 seconds
                recommendations.append({
                    'component': component,
                    'issue': 'high_fault_rate',
                    'recommendation': f'Investigate {component} - fault rate of {stats["fault_rate"]:.4f}/sec is concerning',
                    'priority': 'high'
                })
        
        # Circuit breaker recommendations
        for name, cb in self.circuit_breakers.items():
            if cb.state == 'open' and cb.failure_count > 10:
                recommendations.append({
                    'component': name,
                    'issue': 'persistent_circuit_breaker_open',
                    'recommendation': f'Circuit breaker for {name} has been open with {cb.failure_count} failures',
                    'priority': 'critical'
                })
        
        # Health check recommendations
        failing_checks = [
            name for name, check in self.health_checks.items()
            if not check.last_result and check.consecutive_failures > 0
        ]
        
        for check_name in failing_checks:
            recommendations.append({
                'component': check_name,
                'issue': 'health_check_failure',
                'recommendation': f'Health check {check_name} is failing - investigate underlying issues',
                'priority': 'medium'
            })
        
        return recommendations

    def _calculate_resilience_score(self) -> float:
        """Calculate overall system resilience score."""
        scores = []
        
        # Health score based on system state
        state_scores = {
            SystemState.HEALTHY: 1.0,
            SystemState.DEGRADED: 0.7,
            SystemState.CRITICAL: 0.4,
            SystemState.FAILING: 0.1,
            SystemState.RECOVERING: 0.6,
            SystemState.MAINTENANCE: 0.8
        }
        scores.append(state_scores.get(self.system_state, 0.5))
        
        # Recovery effectiveness score
        if self.healing_history:
            successful_healings = len([h for h in self.healing_history if h['success']])
            healing_rate = successful_healings / len(self.healing_history)
            scores.append(healing_rate)
        else:
            scores.append(0.5)  # Neutral score if no healing history
        
        # Circuit breaker score
        if self.circuit_breakers:
            closed_breakers = len([cb for cb in self.circuit_breakers.values() if cb.state == 'closed'])
            cb_score = closed_breakers / len(self.circuit_breakers)
            scores.append(cb_score)
        else:
            scores.append(1.0)  # Perfect score if no circuit breakers configured
        
        # Health check score
        if self.health_checks:
            healthy_checks = len([hc for hc in self.health_checks.values() if hc.last_result])
            hc_score = healthy_checks / len(self.health_checks)
            scores.append(hc_score)
        else:
            scores.append(1.0)  # Perfect score if no health checks
        
        # Calculate weighted average
        return sum(scores) / len(scores) if scores else 0.0