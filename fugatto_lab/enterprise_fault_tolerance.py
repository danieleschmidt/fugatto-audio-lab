"""Enterprise Fault Tolerance System - Generation 2 Enhancement.

Advanced fault tolerance with circuit breakers, bulkheads, health checks,
and self-healing capabilities for production-grade reliability.
"""

import asyncio
import logging
import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set

try:
    import aiohttp
except ImportError:
    aiohttp = None

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests due to failures
    HALF_OPEN = "half_open" # Testing if service has recovered


class HealthStatus(Enum):
    """Component health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Self-healing recovery strategies."""
    RESTART = "restart"
    RECONNECT = "reconnect"
    FAILOVER = "failover"
    DEGRADE = "degrade"
    CIRCUIT_BREAK = "circuit_break"
    BACKOFF = "backoff"


@dataclass
class FailureRecord:
    """Record of a system failure."""
    timestamp: float
    component: str
    error_type: str
    error_message: str
    severity: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 30.0      # Seconds before trying half-open
    success_threshold: int = 3          # Successes in half-open to close
    timeout: float = 10.0               # Request timeout
    exception_whitelist: Set[str] = field(default_factory=set)
    metrics_window: int = 60            # Seconds for metrics window


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    interval: float = 30.0              # Check interval in seconds
    timeout: float = 5.0                # Check timeout
    max_retries: int = 3                # Retries before marking unhealthy
    degraded_threshold: float = 0.8     # Response time threshold for degraded
    unhealthy_threshold: float = 2.0    # Response time threshold for unhealthy


class CircuitBreaker:
    """Production-grade circuit breaker implementation."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0
        self.failure_count = 0
        self.success_count = 0
        
        # Metrics
        self.request_count = 0
        self.success_count_total = 0
        self.failure_count_total = 0
        self.last_error: Optional[Exception] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metrics window
        self.metrics_history = deque(maxlen=self.config.metrics_window)
        
        logger.info(f"Circuit breaker '{name}' initialized: {self.config}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for protecting functions with circuit breaker."""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            self.request_count += 1
            
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                    self._transition_to_half_open()
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
            
            # Record request attempt
            request_start = time.time()
            
            try:
                # Execute function with timeout
                if asyncio.iscoroutinefunction(func):
                    result = asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
                else:
                    # For sync functions, we can't easily apply timeout without threading
                    result = func(*args, **kwargs)
                
                # Record success
                self._record_success(time.time() - request_start)
                return result
                
            except Exception as e:
                # Check if exception should be ignored
                if type(e).__name__ in self.config.exception_whitelist:
                    self._record_success(time.time() - request_start)
                    raise
                
                # Record failure
                self._record_failure(e, time.time() - request_start)
                raise
    
    def _record_success(self, duration: float) -> None:
        """Record successful execution."""
        self.success_count_total += 1
        
        # Record metrics
        self.metrics_history.append({
            'timestamp': time.time(),
            'success': True,
            'duration': duration
        })
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        
        logger.debug(f"Circuit breaker '{self.name}' recorded success (duration: {duration:.3f}s)")
    
    def _record_failure(self, error: Exception, duration: float) -> None:
        """Record failed execution."""
        self.failure_count += 1
        self.failure_count_total += 1
        self.last_failure_time = time.time()
        self.last_error = error
        
        # Record metrics
        self.metrics_history.append({
            'timestamp': time.time(),
            'success': False,
            'duration': duration,
            'error': str(error)
        })
        
        # Check if should open circuit
        if (self.state == CircuitState.CLOSED and 
            self.failure_count >= self.config.failure_threshold):
            self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        
        logger.warning(f"Circuit breaker '{self.name}' recorded failure: {error}")
    
    def _transition_to_open(self) -> None:
        """Transition circuit to OPEN state."""
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()
        logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN")
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")
    
    def _transition_to_closed(self) -> None:
        """Transition circuit to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED")
    
    def force_open(self) -> None:
        """Force circuit to OPEN state."""
        with self._lock:
            self._transition_to_open()
    
    def force_close(self) -> None:
        """Force circuit to CLOSED state."""
        with self._lock:
            self._transition_to_closed()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 requests
            
            success_rate = 0.0
            avg_duration = 0.0
            
            if recent_metrics:
                successes = sum(1 for m in recent_metrics if m['success'])
                success_rate = successes / len(recent_metrics)
                avg_duration = sum(m['duration'] for m in recent_metrics) / len(recent_metrics)
            
            return {
                'name': self.name,
                'state': self.state.value,
                'request_count': self.request_count,
                'success_count_total': self.success_count_total,
                'failure_count_total': self.failure_count_total,
                'current_failure_count': self.failure_count,
                'success_rate': success_rate,
                'average_duration': avg_duration,
                'last_error': str(self.last_error) if self.last_error else None,
                'last_failure_time': self.last_failure_time,
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'recovery_timeout': self.config.recovery_timeout,
                    'timeout': self.config.timeout
                }
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self, component_name: str, config: Optional[HealthCheckConfig] = None):
        self.component_name = component_name
        self.config = config or HealthCheckConfig()
        
        self.current_status = HealthStatus.UNKNOWN
        self.last_check_time = 0.0
        self.consecutive_failures = 0
        self.health_history = deque(maxlen=50)
        
        # Callbacks
        self.health_check_func: Optional[Callable] = None
        self.status_change_callbacks: List[Callable] = []
        
        # Control
        self.checking_active = False
        self.check_thread: Optional[threading.Thread] = None
        
        logger.info(f"Health checker for '{component_name}' initialized")
    
    def set_health_check_function(self, func: Callable[[], bool]) -> None:
        """Set the function to call for health checks."""
        self.health_check_func = func
    
    def add_status_change_callback(self, callback: Callable[[HealthStatus, HealthStatus], None]) -> None:
        """Add callback for status changes."""
        self.status_change_callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.checking_active or not self.health_check_func:
            return
        
        self.checking_active = True
        self.check_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.check_thread.start()
        
        logger.info(f"Health monitoring started for '{self.component_name}'")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.checking_active = False
        if self.check_thread and self.check_thread.is_alive():
            self.check_thread.join(timeout=2.0)
        
        logger.info(f"Health monitoring stopped for '{self.component_name}'")
    
    def _monitoring_loop(self) -> None:
        """Main health monitoring loop."""
        while self.checking_active:
            try:
                self.perform_health_check()
                time.sleep(self.config.interval)
            except Exception as e:
                logger.error(f"Error in health monitoring loop for '{self.component_name}': {e}")
                time.sleep(self.config.interval)
    
    def perform_health_check(self) -> HealthStatus:
        """Perform single health check."""
        if not self.health_check_func:
            logger.warning(f"No health check function set for '{self.component_name}'")
            return self.current_status
        
        check_start = time.time()
        
        try:
            # Perform health check with timeout
            check_result = self._call_with_timeout(self.health_check_func, self.config.timeout)
            check_duration = time.time() - check_start
            
            # Determine status based on result and timing
            if check_result:
                if check_duration <= self.config.degraded_threshold:
                    new_status = HealthStatus.HEALTHY
                elif check_duration <= self.config.unhealthy_threshold:
                    new_status = HealthStatus.DEGRADED
                else:
                    new_status = HealthStatus.UNHEALTHY
                
                self.consecutive_failures = 0
            else:
                new_status = HealthStatus.UNHEALTHY
                self.consecutive_failures += 1
            
            # Record check result
            self.health_history.append({
                'timestamp': check_start,
                'status': new_status,
                'duration': check_duration,
                'success': check_result,
                'consecutive_failures': self.consecutive_failures
            })
            
            # Update status if changed
            if new_status != self.current_status:
                old_status = self.current_status
                self.current_status = new_status
                self.last_check_time = check_start
                
                # Notify callbacks
                for callback in self.status_change_callbacks:
                    try:
                        callback(old_status, new_status)
                    except Exception as e:
                        logger.error(f"Error in status change callback: {e}")
                
                logger.info(f"Health status changed for '{self.component_name}': {old_status.value} -> {new_status.value}")
            
            return new_status
            
        except Exception as e:
            logger.error(f"Health check failed for '{self.component_name}': {e}")
            self.consecutive_failures += 1
            
            # Record failure
            self.health_history.append({
                'timestamp': check_start,
                'status': HealthStatus.UNHEALTHY,
                'duration': time.time() - check_start,
                'success': False,
                'error': str(e),
                'consecutive_failures': self.consecutive_failures
            })
            
            # Update to unhealthy if too many consecutive failures
            if self.consecutive_failures >= self.config.max_retries:
                if self.current_status != HealthStatus.UNHEALTHY:
                    old_status = self.current_status
                    self.current_status = HealthStatus.UNHEALTHY
                    
                    for callback in self.status_change_callbacks:
                        try:
                            callback(old_status, HealthStatus.UNHEALTHY)
                        except Exception as e:
                            logger.error(f"Error in status change callback: {e}")
            
            return self.current_status
    
    def _call_with_timeout(self, func: Callable, timeout: float) -> Any:
        """Call function with timeout (simplified implementation)."""
        # Note: This is a simplified timeout implementation
        # In production, you'd want to use proper threading or async timeouts
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Health check timed out")
        
        # Set timeout (Unix only)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            result = func()
            signal.alarm(0)  # Cancel alarm
            return result
        except AttributeError:
            # Windows or no signal support - just call function
            return func()
        except TimeoutError:
            logger.warning(f"Health check timeout for '{self.component_name}'")
            return False
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        recent_checks = list(self.health_history)[-10:]
        
        success_rate = 0.0
        avg_duration = 0.0
        
        if recent_checks:
            successes = sum(1 for check in recent_checks if check['success'])
            success_rate = successes / len(recent_checks)
            avg_duration = sum(check['duration'] for check in recent_checks) / len(recent_checks)
        
        return {
            'component_name': self.component_name,
            'current_status': self.current_status.value,
            'last_check_time': self.last_check_time,
            'consecutive_failures': self.consecutive_failures,
            'success_rate': success_rate,
            'average_duration': avg_duration,
            'monitoring_active': self.checking_active,
            'recent_checks': recent_checks,
            'config': {
                'interval': self.config.interval,
                'timeout': self.config.timeout,
                'max_retries': self.config.max_retries
            }
        }


class SelfHealingManager:
    """Self-healing system that automatically recovers from failures."""
    
    def __init__(self):
        self.recovery_strategies: Dict[str, List[Callable]] = defaultdict(list)
        self.failure_history = deque(maxlen=100)
        self.recovery_history = deque(maxlen=100)
        self.healing_active = True
        
        # Strategy effectiveness tracking
        self.strategy_effectiveness: Dict[str, float] = defaultdict(lambda: 0.5)
        
        logger.info("Self-healing manager initialized")
    
    def register_recovery_strategy(self, component: str, strategy: RecoveryStrategy,
                                 recovery_func: Callable[[], bool]) -> None:
        """Register a recovery strategy for a component."""
        strategy_key = f"{component}:{strategy.value}"
        self.recovery_strategies[strategy_key].append(recovery_func)
        
        logger.info(f"Registered recovery strategy '{strategy.value}' for component '{component}'")
    
    def handle_failure(self, component: str, error: Exception, 
                      context: Optional[Dict[str, Any]] = None) -> bool:
        """Handle component failure with appropriate recovery strategy."""
        if not self.healing_active:
            logger.info("Self-healing disabled, not attempting recovery")
            return False
        
        failure_record = FailureRecord(
            timestamp=time.time(),
            component=component,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=self._classify_error_severity(error),
            context=context or {}
        )
        
        self.failure_history.append(failure_record)
        
        logger.warning(f"Handling failure for component '{component}': {error}")
        
        # Determine appropriate recovery strategy
        strategy = self._select_recovery_strategy(component, error, failure_record)
        
        if strategy:
            return self._execute_recovery_strategy(component, strategy, failure_record)
        else:
            logger.error(f"No recovery strategy available for component '{component}'")
            return False
    
    def _classify_error_severity(self, error: Exception) -> str:
        """Classify error severity for recovery prioritization."""
        error_type = type(error).__name__
        
        critical_errors = {
            'OutOfMemoryError', 'SystemError', 'OSError'
        }
        
        high_errors = {
            'ConnectionError', 'TimeoutError', 'CircuitBreakerOpenError'
        }
        
        medium_errors = {
            'ValueError', 'KeyError', 'AttributeError'
        }
        
        if error_type in critical_errors:
            return 'critical'
        elif error_type in high_errors:
            return 'high'
        elif error_type in medium_errors:
            return 'medium'
        else:
            return 'low'
    
    def _select_recovery_strategy(self, component: str, error: Exception,
                                failure_record: FailureRecord) -> Optional[RecoveryStrategy]:
        """Select the most appropriate recovery strategy."""
        error_type = type(error).__name__
        severity = failure_record.severity
        
        # Strategy selection based on error type and severity
        if severity == 'critical':
            if 'Memory' in error_type:
                return RecoveryStrategy.RESTART
            else:
                return RecoveryStrategy.FAILOVER
        
        elif severity == 'high':
            if 'Connection' in error_type or 'Timeout' in error_type:
                return RecoveryStrategy.RECONNECT
            elif 'CircuitBreaker' in error_type:
                return RecoveryStrategy.BACKOFF
            else:
                return RecoveryStrategy.RESTART
        
        elif severity == 'medium':
            return RecoveryStrategy.DEGRADE
        
        else:  # low severity
            return RecoveryStrategy.BACKOFF
    
    def _execute_recovery_strategy(self, component: str, strategy: RecoveryStrategy,
                                 failure_record: FailureRecord) -> bool:
        """Execute the selected recovery strategy."""
        strategy_key = f"{component}:{strategy.value}"
        recovery_functions = self.recovery_strategies.get(strategy_key, [])
        
        if not recovery_functions:
            logger.warning(f"No recovery functions registered for strategy '{strategy_key}'")
            return False
        
        recovery_start = time.time()
        failure_record.recovery_attempted = True
        
        logger.info(f"Executing recovery strategy '{strategy.value}' for component '{component}'")
        
        # Try each recovery function until one succeeds
        for i, recovery_func in enumerate(recovery_functions):
            try:
                logger.debug(f"Attempting recovery function {i+1}/{len(recovery_functions)}")
                
                success = recovery_func()
                
                if success:
                    recovery_duration = time.time() - recovery_start
                    failure_record.recovery_successful = True
                    
                    # Record successful recovery
                    self.recovery_history.append({
                        'timestamp': recovery_start,
                        'component': component,
                        'strategy': strategy.value,
                        'duration': recovery_duration,
                        'success': True,
                        'function_index': i
                    })
                    
                    # Update strategy effectiveness
                    current_effectiveness = self.strategy_effectiveness[strategy_key]
                    self.strategy_effectiveness[strategy_key] = min(1.0, current_effectiveness * 1.1)
                    
                    logger.info(f"Recovery successful for component '{component}' using strategy '{strategy.value}' (duration: {recovery_duration:.2f}s)")
                    return True
                    
            except Exception as recovery_error:
                logger.error(f"Recovery function {i+1} failed for component '{component}': {recovery_error}")
                continue
        
        # All recovery attempts failed
        recovery_duration = time.time() - recovery_start
        
        self.recovery_history.append({
            'timestamp': recovery_start,
            'component': component,
            'strategy': strategy.value,
            'duration': recovery_duration,
            'success': False,
            'attempts': len(recovery_functions)
        })
        
        # Reduce strategy effectiveness
        current_effectiveness = self.strategy_effectiveness[strategy_key]
        self.strategy_effectiveness[strategy_key] = max(0.1, current_effectiveness * 0.9)
        
        logger.error(f"All recovery attempts failed for component '{component}' using strategy '{strategy.value}'")
        return False
    
    def get_healing_report(self) -> Dict[str, Any]:
        """Get comprehensive self-healing report."""
        recent_failures = list(self.failure_history)[-10:]
        recent_recoveries = list(self.recovery_history)[-10:]
        
        # Calculate recovery statistics
        total_failures = len(self.failure_history)
        successful_recoveries = sum(1 for failure in self.failure_history if failure.recovery_successful)
        recovery_rate = successful_recoveries / total_failures if total_failures > 0 else 0.0
        
        # Strategy effectiveness summary
        strategy_summary = {}
        for strategy_key, effectiveness in self.strategy_effectiveness.items():
            component, strategy = strategy_key.split(':', 1)
            if component not in strategy_summary:
                strategy_summary[component] = {}
            strategy_summary[component][strategy] = effectiveness
        
        return {
            'healing_active': self.healing_active,
            'total_failures': total_failures,
            'successful_recoveries': successful_recoveries,
            'recovery_rate': recovery_rate,
            'recent_failures': [{
                'timestamp': f.timestamp,
                'component': f.component,
                'error_type': f.error_type,
                'severity': f.severity,
                'recovery_attempted': f.recovery_attempted,
                'recovery_successful': f.recovery_successful
            } for f in recent_failures],
            'recent_recoveries': recent_recoveries,
            'strategy_effectiveness': strategy_summary,
            'registered_strategies': list(self.recovery_strategies.keys())
        }
    
    def enable_healing(self) -> None:
        """Enable self-healing."""
        self.healing_active = True
        logger.info("Self-healing enabled")
    
    def disable_healing(self) -> None:
        """Disable self-healing."""
        self.healing_active = False
        logger.info("Self-healing disabled")


class EnterpriseFaultToleranceSystem:
    """Main enterprise fault tolerance orchestrator."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.health_checkers: Dict[str, HealthChecker] = {}
        self.self_healer = SelfHealingManager()
        
        # System monitoring
        self.system_health = HealthStatus.UNKNOWN
        self.monitoring_active = False
        
        logger.info("Enterprise fault tolerance system initialized")
    
    def create_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Create and register a circuit breaker."""
        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def create_health_checker(self, component_name: str, 
                            health_check_func: Callable[[], bool],
                            config: Optional[HealthCheckConfig] = None) -> HealthChecker:
        """Create and register a health checker."""
        health_checker = HealthChecker(component_name, config)
        health_checker.set_health_check_function(health_check_func)
        
        # Add status change callback for self-healing integration
        health_checker.add_status_change_callback(self._on_health_status_change)
        
        self.health_checkers[component_name] = health_checker
        return health_checker
    
    def register_recovery_strategy(self, component: str, strategy: RecoveryStrategy,
                                 recovery_func: Callable[[], bool]) -> None:
        """Register a recovery strategy with the self-healing manager."""
        self.self_healer.register_recovery_strategy(component, strategy, recovery_func)
    
    def _on_health_status_change(self, old_status: HealthStatus, new_status: HealthStatus) -> None:
        """Handle health status changes for self-healing integration."""
        if new_status == HealthStatus.UNHEALTHY:
            # Trigger self-healing for unhealthy components
            for component_name, health_checker in self.health_checkers.items():
                if health_checker.current_status == HealthStatus.UNHEALTHY:
                    # Create a synthetic error for self-healing
                    error = Exception(f"Component '{component_name}' became unhealthy")
                    self.self_healer.handle_failure(component_name, error)
        
        # Update overall system health
        self._update_system_health()
    
    def _update_system_health(self) -> None:
        """Update overall system health based on component health."""
        if not self.health_checkers:
            self.system_health = HealthStatus.UNKNOWN
            return
        
        health_statuses = [checker.current_status for checker in self.health_checkers.values()]
        
        # Determine overall health
        if all(status == HealthStatus.HEALTHY for status in health_statuses):
            self.system_health = HealthStatus.HEALTHY
        elif any(status == HealthStatus.UNHEALTHY for status in health_statuses):
            self.system_health = HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in health_statuses):
            self.system_health = HealthStatus.DEGRADED
        else:
            self.system_health = HealthStatus.UNKNOWN
    
    def start_monitoring(self) -> None:
        """Start monitoring all health checkers."""
        for health_checker in self.health_checkers.values():
            health_checker.start_monitoring()
        
        self.monitoring_active = True
        logger.info("Enterprise fault tolerance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring all health checkers."""
        for health_checker in self.health_checkers.values():
            health_checker.stop_monitoring()
        
        self.monitoring_active = False
        logger.info("Enterprise fault tolerance monitoring stopped")
    
    def get_system_report(self) -> Dict[str, Any]:
        """Get comprehensive system fault tolerance report."""
        circuit_breaker_reports = {name: cb.get_metrics() for name, cb in self.circuit_breakers.items()}
        health_checker_reports = {name: hc.get_health_report() for name, hc in self.health_checkers.items()}
        
        return {
            'system_health': self.system_health.value,
            'monitoring_active': self.monitoring_active,
            'circuit_breakers': circuit_breaker_reports,
            'health_checkers': health_checker_reports,
            'self_healing': self.self_healer.get_healing_report(),
            'summary': {
                'total_circuit_breakers': len(self.circuit_breakers),
                'total_health_checkers': len(self.health_checkers),
                'open_circuit_breakers': sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN),
                'unhealthy_components': sum(1 for hc in self.health_checkers.values() if hc.current_status == HealthStatus.UNHEALTHY)
            }
        }
    
    def force_recovery(self, component: str) -> bool:
        """Force recovery attempt for a specific component."""
        error = Exception(f"Forced recovery for component '{component}'")
        return self.self_healer.handle_failure(component, error, {'forced': True})


# Factory functions and decorators

def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> Callable:
    """Decorator for protecting functions with circuit breaker."""
    cb = CircuitBreaker(name, config)
    return cb


def with_health_check(component_name: str, health_func: Callable[[], bool],
                     config: Optional[HealthCheckConfig] = None) -> Callable:
    """Decorator for adding health checking to functions."""
    def decorator(func: Callable) -> Callable:
        health_checker = HealthChecker(component_name, config)
        health_checker.set_health_check_function(health_func)
        health_checker.start_monitoring()
        
        def wrapper(*args, **kwargs):
            if health_checker.current_status == HealthStatus.UNHEALTHY:
                raise Exception(f"Component '{component_name}' is unhealthy")
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def create_enterprise_fault_tolerance() -> EnterpriseFaultToleranceSystem:
    """Create enterprise fault tolerance system with standard configuration."""
    return EnterpriseFaultToleranceSystem()


if __name__ == "__main__":
    # Demonstration
    import random
    
    # Create fault tolerance system
    ft_system = create_enterprise_fault_tolerance()
    
    # Mock functions for demonstration
    def mock_database_call():
        """Mock database call that sometimes fails."""
        if random.random() < 0.3:  # 30% failure rate
            raise ConnectionError("Database connection failed")
        time.sleep(random.uniform(0.1, 0.5))  # Simulate work
        return "Database result"
    
    def mock_database_health_check():
        """Mock database health check."""
        return random.random() > 0.2  # 80% healthy
    
    def mock_database_recovery():
        """Mock database recovery function."""
        logger.info("Attempting database recovery...")
        time.sleep(1.0)  # Simulate recovery time
        return random.random() > 0.3  # 70% success rate
    
    print("Starting enterprise fault tolerance demonstration...")
    
    # Create circuit breaker for database
    db_circuit_breaker = ft_system.create_circuit_breaker(
        "database", 
        CircuitBreakerConfig(failure_threshold=3, recovery_timeout=5.0)
    )
    
    # Create health checker for database
    db_health_checker = ft_system.create_health_checker(
        "database",
        mock_database_health_check,
        HealthCheckConfig(interval=2.0)
    )
    
    # Register recovery strategy
    ft_system.register_recovery_strategy(
        "database",
        RecoveryStrategy.RECONNECT,
        mock_database_recovery
    )
    
    # Start monitoring
    ft_system.start_monitoring()
    
    try:
        # Simulate some operations
        for i in range(20):
            try:
                result = db_circuit_breaker.call(mock_database_call)
                print(f"Operation {i+1}: Success - {result}")
            except Exception as e:
                print(f"Operation {i+1}: Failed - {e}")
            
            time.sleep(1.0)
        
        # Generate report
        report = ft_system.get_system_report()
        
        print("\n=== ENTERPRISE FAULT TOLERANCE REPORT ===")
        print(f"System Health: {report['system_health']}")
        print(f"\nSummary:")
        for key, value in report['summary'].items():
            print(f"  {key}: {value}")
        
        print(f"\nCircuit Breakers:")
        for name, metrics in report['circuit_breakers'].items():
            print(f"  {name}: {metrics['state']} (success rate: {metrics['success_rate']:.2%})")
        
        print(f"\nHealth Checkers:")
        for name, health in report['health_checkers'].items():
            print(f"  {name}: {health['current_status']} (failures: {health['consecutive_failures']})")
        
        print(f"\nSelf-Healing:")
        healing_report = report['self_healing']
        print(f"  Recovery Rate: {healing_report['recovery_rate']:.2%}")
        print(f"  Total Failures: {healing_report['total_failures']}")
        print(f"  Successful Recoveries: {healing_report['successful_recoveries']}")
    
    finally:
        # Stop monitoring
        ft_system.stop_monitoring()
        print("\nEnterprise fault tolerance demonstration completed.")
