"""Robust validation and error handling for enhanced quantum task planning.

Advanced validation, error recovery, and resilience features for production-grade
quantum-inspired task planning with comprehensive monitoring and alerting.
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import hashlib
import uuid
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for robust handling."""
    CRITICAL = "critical"       # System-threatening errors
    HIGH = "high"              # Major functionality impacted
    MEDIUM = "medium"          # Moderate impact, degraded performance
    LOW = "low"               # Minor issues, logging only
    INFO = "info"             # Informational, no action needed


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"                    # Retry operation
    FALLBACK = "fallback"              # Use alternative approach
    DEGRADE = "degrade"                # Reduce functionality gracefully
    ESCALATE = "escalate"              # Hand off to higher level
    TERMINATE = "terminate"            # Stop operation
    CIRCUIT_BREAK = "circuit_break"    # Temporarily disable feature


@dataclass
class ValidationError:
    """Structured validation error with recovery information."""
    
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    error_type: str = "validation_error"
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    timestamp: float = field(default_factory=time.time)
    stack_trace: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/reporting."""
        return {
            "error_id": self.error_id,
            "error_type": self.error_type,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "context": self.context,
            "recovery_strategy": self.recovery_strategy.value,
            "timestamp": self.timestamp,
            "stack_trace": self.stack_trace
        }


class RobustValidator:
    """Comprehensive validation with error recovery."""
    
    def __init__(self):
        self.validation_cache = {}
        self.error_history = deque(maxlen=1000)
        self.validation_rules = {}
        self.custom_validators = {}
        
        # Circuit breaker state
        self.circuit_breakers = {}
        self.failure_thresholds = {
            "task_validation": 10,  # failures per 5 minutes
            "resource_validation": 5,
            "dependency_validation": 8
        }
        
        logger.info("RobustValidator initialized with comprehensive error handling")
    
    def register_validator(self, name: str, validator_func: Callable) -> None:
        """Register custom validation function."""
        self.custom_validators[name] = validator_func
        logger.debug(f"Registered custom validator: {name}")
    
    def validate_task(self, task_data: Dict[str, Any]) -> Tuple[bool, Optional[ValidationError]]:
        """Comprehensive task validation with error recovery."""
        try:
            # Check circuit breaker
            if self._is_circuit_open("task_validation"):
                return False, ValidationError(
                    error_type="circuit_breaker",
                    severity=ErrorSeverity.HIGH,
                    message="Task validation circuit breaker is open",
                    recovery_strategy=RecoveryStrategy.DEGRADE
                )
            
            # Required field validation
            required_fields = ["id", "name", "operation", "estimated_duration"]
            missing_fields = [field for field in required_fields if field not in task_data]
            
            if missing_fields:
                error = ValidationError(
                    error_type="missing_required_fields",
                    severity=ErrorSeverity.HIGH,
                    message=f"Missing required fields: {missing_fields}",
                    details={"missing_fields": missing_fields},
                    context={"task_data": task_data},
                    recovery_strategy=RecoveryStrategy.FALLBACK
                )
                self._record_error(error)
                return False, error
            
            # Data type validation
            type_errors = []
            
            if not isinstance(task_data.get("estimated_duration"), (int, float)) or task_data["estimated_duration"] <= 0:
                type_errors.append("estimated_duration must be positive number")
            
            if not isinstance(task_data.get("name"), str) or len(task_data["name"].strip()) == 0:
                type_errors.append("name must be non-empty string")
            
            if type_errors:
                error = ValidationError(
                    error_type="invalid_data_types",
                    severity=ErrorSeverity.MEDIUM,
                    message="Invalid data types in task",
                    details={"type_errors": type_errors},
                    recovery_strategy=RecoveryStrategy.FALLBACK
                )
                self._record_error(error)
                return False, error
            
            # Business logic validation
            if task_data.get("estimated_duration", 0) > 7200:  # 2 hours max
                error = ValidationError(
                    error_type="duration_limit_exceeded",
                    severity=ErrorSeverity.MEDIUM,
                    message="Task duration exceeds 2 hour limit",
                    details={"duration": task_data["estimated_duration"]},
                    recovery_strategy=RecoveryStrategy.DEGRADE
                )
                self._record_error(error)
                return False, error
            
            # Resource validation
            resources = task_data.get("resources", {})
            if isinstance(resources, dict):
                resource_validation = self._validate_resources(resources)
                if not resource_validation[0]:
                    return False, resource_validation[1]
            
            # Custom validator execution
            for validator_name, validator_func in self.custom_validators.items():
                try:
                    is_valid, custom_error = validator_func(task_data)
                    if not is_valid:
                        error = ValidationError(
                            error_type=f"custom_validation_{validator_name}",
                            severity=ErrorSeverity.MEDIUM,
                            message=f"Custom validation failed: {validator_name}",
                            details={"validator": validator_name, "error": str(custom_error)},
                            recovery_strategy=RecoveryStrategy.ESCALATE
                        )
                        self._record_error(error)
                        return False, error
                except Exception as e:
                    logger.warning(f"Custom validator {validator_name} failed: {e}")
            
            # Success - record positive validation
            self._record_success("task_validation")
            return True, None
            
        except Exception as e:
            error = ValidationError(
                error_type="validation_exception",
                severity=ErrorSeverity.CRITICAL,
                message=f"Unexpected error during task validation: {str(e)}",
                details={"exception": str(e)},
                stack_trace=traceback.format_exc(),
                recovery_strategy=RecoveryStrategy.ESCALATE
            )
            self._record_error(error)
            return False, error
    
    def _validate_resources(self, resources: Dict[str, Any]) -> Tuple[bool, Optional[ValidationError]]:
        """Validate resource requirements."""
        try:
            for resource_type, amount in resources.items():
                if not isinstance(amount, (int, float)) or amount < 0:
                    return False, ValidationError(
                        error_type="invalid_resource_amount",
                        severity=ErrorSeverity.MEDIUM,
                        message=f"Invalid resource amount for {resource_type}: {amount}",
                        details={"resource_type": resource_type, "amount": amount},
                        recovery_strategy=RecoveryStrategy.FALLBACK
                    )
                
                # Resource-specific validation
                if resource_type == "memory_gb" and amount > 32:
                    return False, ValidationError(
                        error_type="excessive_memory_request",
                        severity=ErrorSeverity.HIGH,
                        message=f"Memory request {amount}GB exceeds 32GB limit",
                        recovery_strategy=RecoveryStrategy.DEGRADE
                    )
                
                if resource_type == "cpu_cores" and amount > 16:
                    return False, ValidationError(
                        error_type="excessive_cpu_request", 
                        severity=ErrorSeverity.HIGH,
                        message=f"CPU request {amount} cores exceeds 16 core limit",
                        recovery_strategy=RecoveryStrategy.DEGRADE
                    )
            
            return True, None
            
        except Exception as e:
            return False, ValidationError(
                error_type="resource_validation_exception",
                severity=ErrorSeverity.CRITICAL,
                message=f"Error validating resources: {str(e)}",
                stack_trace=traceback.format_exc()
            )
    
    def validate_dependencies(self, task_id: str, dependencies: List[str], 
                            all_tasks: Dict[str, Any]) -> Tuple[bool, Optional[ValidationError]]:
        """Validate task dependencies with cycle detection."""
        try:
            if self._is_circuit_open("dependency_validation"):
                return False, ValidationError(
                    error_type="circuit_breaker",
                    severity=ErrorSeverity.HIGH,
                    message="Dependency validation circuit breaker is open",
                    recovery_strategy=RecoveryStrategy.DEGRADE
                )
            
            # Check for missing dependencies
            missing_deps = [dep_id for dep_id in dependencies if dep_id not in all_tasks]
            if missing_deps:
                error = ValidationError(
                    error_type="missing_dependencies",
                    severity=ErrorSeverity.HIGH,
                    message=f"Dependencies not found: {missing_deps}",
                    details={"missing_dependencies": missing_deps, "task_id": task_id},
                    recovery_strategy=RecoveryStrategy.FALLBACK
                )
                self._record_error(error)
                return False, error
            
            # Cycle detection using DFS
            if self._has_circular_dependency(task_id, dependencies, all_tasks):
                error = ValidationError(
                    error_type="circular_dependency",
                    severity=ErrorSeverity.CRITICAL,
                    message=f"Circular dependency detected for task {task_id}",
                    details={"task_id": task_id, "dependencies": dependencies},
                    recovery_strategy=RecoveryStrategy.TERMINATE
                )
                self._record_error(error)
                return False, error
            
            self._record_success("dependency_validation")
            return True, None
            
        except Exception as e:
            error = ValidationError(
                error_type="dependency_validation_exception",
                severity=ErrorSeverity.CRITICAL,
                message=f"Error validating dependencies: {str(e)}",
                stack_trace=traceback.format_exc()
            )
            self._record_error(error)
            return False, error
    
    def _has_circular_dependency(self, task_id: str, dependencies: List[str], 
                                all_tasks: Dict[str, Any], visited: Optional[set] = None) -> bool:
        """Detect circular dependencies using DFS."""
        if visited is None:
            visited = set()
        
        if task_id in visited:
            return True  # Cycle detected
        
        visited.add(task_id)
        
        # Check each dependency recursively
        for dep_id in dependencies:
            if dep_id in all_tasks:
                dep_dependencies = all_tasks[dep_id].get("dependencies", [])
                if self._has_circular_dependency(dep_id, dep_dependencies, all_tasks, visited.copy()):
                    return True
        
        return False
    
    def _is_circuit_open(self, circuit_name: str) -> bool:
        """Check if circuit breaker is open."""
        if circuit_name not in self.circuit_breakers:
            self.circuit_breakers[circuit_name] = {
                "failures": 0,
                "last_failure": 0,
                "state": "closed",  # closed, open, half_open
                "next_attempt": 0
            }
            return False
        
        breaker = self.circuit_breakers[circuit_name]
        current_time = time.time()
        
        if breaker["state"] == "open":
            # Check if we should try half-open
            if current_time > breaker["next_attempt"]:
                breaker["state"] = "half_open"
                logger.info(f"Circuit breaker {circuit_name} moving to half-open")
                return False
            return True
        
        return False
    
    def _record_error(self, error: ValidationError):
        """Record validation error and update circuit breakers."""
        self.error_history.append(error)
        
        # Update circuit breaker
        error_category = error.error_type.split('_')[0] + "_validation"
        if error_category in self.circuit_breakers:
            breaker = self.circuit_breakers[error_category]
            breaker["failures"] += 1
            breaker["last_failure"] = time.time()
            
            # Check if we should open the circuit
            threshold = self.failure_thresholds.get(error_category, 10)
            if breaker["failures"] >= threshold and breaker["state"] == "closed":
                breaker["state"] = "open"
                breaker["next_attempt"] = time.time() + 300  # 5 minute cooldown
                logger.warning(f"Circuit breaker {error_category} opened due to {breaker['failures']} failures")
        
        logger.error(f"Validation error recorded: {error.message}")
    
    def _record_success(self, category: str):
        """Record successful validation."""
        if category in self.circuit_breakers:
            breaker = self.circuit_breakers[category]
            if breaker["state"] == "half_open":
                # Reset circuit breaker on successful operation
                breaker["state"] = "closed"
                breaker["failures"] = 0
                logger.info(f"Circuit breaker {category} reset to closed")
            elif breaker["state"] == "closed":
                # Gradually reduce failure count on successes
                breaker["failures"] = max(0, breaker["failures"] - 1)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary."""
        if not self.error_history:
            return {"total_errors": 0, "summary": "No errors recorded"}
        
        errors_by_type = defaultdict(int)
        errors_by_severity = defaultdict(int)
        recent_errors = []
        
        for error in self.error_history:
            errors_by_type[error.error_type] += 1
            errors_by_severity[error.severity.value] += 1
            
            # Include recent errors (last hour)
            if time.time() - error.timestamp < 3600:
                recent_errors.append(error.to_dict())
        
        return {
            "total_errors": len(self.error_history),
            "errors_by_type": dict(errors_by_type),
            "errors_by_severity": dict(errors_by_severity),
            "recent_errors_count": len(recent_errors),
            "recent_errors": recent_errors[-10:],  # Last 10 recent errors
            "circuit_breakers": self.circuit_breakers.copy()
        }


class EnhancedErrorHandler:
    """Enhanced error handling with automatic recovery."""
    
    def __init__(self):
        self.error_handlers = {}
        self.recovery_strategies = {}
        self.error_metrics = defaultdict(int)
        self.recovery_success_rate = {}
        
        # Default recovery strategies
        self._setup_default_strategies()
        
        logger.info("EnhancedErrorHandler initialized")
    
    def _setup_default_strategies(self):
        """Setup default recovery strategies."""
        self.recovery_strategies[RecoveryStrategy.RETRY] = self._retry_strategy
        self.recovery_strategies[RecoveryStrategy.FALLBACK] = self._fallback_strategy
        self.recovery_strategies[RecoveryStrategy.DEGRADE] = self._degrade_strategy
        self.recovery_strategies[RecoveryStrategy.ESCALATE] = self._escalate_strategy
        self.recovery_strategies[RecoveryStrategy.TERMINATE] = self._terminate_strategy
        self.recovery_strategies[RecoveryStrategy.CIRCUIT_BREAK] = self._circuit_break_strategy
    
    async def handle_error(self, error: ValidationError, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle error with appropriate recovery strategy."""
        try:
            self.error_metrics[error.error_type] += 1
            
            logger.error(f"Handling error {error.error_id}: {error.message}")
            
            # Apply recovery strategy
            recovery_func = self.recovery_strategies.get(error.recovery_strategy)
            if recovery_func:
                success, result = await recovery_func(error, context)
                
                # Track recovery success rate
                strategy_key = error.recovery_strategy.value
                if strategy_key not in self.recovery_success_rate:
                    self.recovery_success_rate[strategy_key] = {"attempts": 0, "successes": 0}
                
                self.recovery_success_rate[strategy_key]["attempts"] += 1
                if success:
                    self.recovery_success_rate[strategy_key]["successes"] += 1
                
                return success, result
            else:
                logger.error(f"No recovery strategy found for {error.recovery_strategy}")
                return False, None
                
        except Exception as e:
            logger.critical(f"Error handler failed: {e}")
            traceback.print_exc()
            return False, None
    
    async def _retry_strategy(self, error: ValidationError, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Retry strategy with exponential backoff."""
        max_retries = context.get("max_retries", 3)
        retry_count = context.get("retry_count", 0)
        original_operation = context.get("operation")
        
        if retry_count >= max_retries:
            logger.warning(f"Max retries ({max_retries}) exceeded for error {error.error_id}")
            return False, "max_retries_exceeded"
        
        # Exponential backoff
        backoff_time = min(2 ** retry_count, 30)  # Max 30 seconds
        logger.info(f"Retrying operation after {backoff_time}s (attempt {retry_count + 1}/{max_retries})")
        
        await asyncio.sleep(backoff_time)
        
        # Update context for next retry
        context["retry_count"] = retry_count + 1
        
        if original_operation and callable(original_operation):
            try:
                result = await original_operation(context)
                logger.info(f"Retry successful after {retry_count + 1} attempts")
                return True, result
            except Exception as e:
                logger.warning(f"Retry attempt {retry_count + 1} failed: {e}")
                if retry_count + 1 < max_retries:
                    return await self._retry_strategy(error, context)
                return False, str(e)
        
        return False, "no_operation_to_retry"
    
    async def _fallback_strategy(self, error: ValidationError, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Fallback to alternative implementation."""
        fallback_operation = context.get("fallback_operation")
        
        if fallback_operation and callable(fallback_operation):
            try:
                logger.info(f"Executing fallback for error {error.error_id}")
                result = await fallback_operation(context)
                return True, result
            except Exception as e:
                logger.error(f"Fallback operation failed: {e}")
                return False, str(e)
        
        # Default fallback - simplified operation
        if error.error_type in ["missing_required_fields", "invalid_data_types"]:
            logger.info("Applying default data sanitization fallback")
            sanitized_data = self._sanitize_task_data(context.get("task_data", {}))
            return True, {"action": "data_sanitized", "data": sanitized_data}
        
        return False, "no_fallback_available"
    
    async def _degrade_strategy(self, error: ValidationError, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Graceful degradation of functionality."""
        degraded_mode = context.get("degraded_mode", {})
        
        logger.info(f"Applying graceful degradation for error {error.error_id}")
        
        # Task-specific degradation
        if error.error_type == "excessive_memory_request":
            # Reduce memory allocation
            original_resources = context.get("task_data", {}).get("resources", {})
            degraded_resources = original_resources.copy()
            degraded_resources["memory_gb"] = min(original_resources.get("memory_gb", 8), 8)
            
            return True, {
                "action": "memory_degraded",
                "original_memory": original_resources.get("memory_gb"),
                "degraded_memory": degraded_resources["memory_gb"],
                "resources": degraded_resources
            }
        
        elif error.error_type == "duration_limit_exceeded":
            # Split into smaller tasks
            original_duration = context.get("task_data", {}).get("estimated_duration", 3600)
            chunk_duration = 1800  # 30 minutes max per chunk
            num_chunks = int((original_duration + chunk_duration - 1) // chunk_duration)
            
            return True, {
                "action": "task_chunked",
                "original_duration": original_duration,
                "num_chunks": num_chunks,
                "chunk_duration": chunk_duration
            }
        
        # Generic degradation
        return True, {
            "action": "generic_degradation",
            "reduced_functionality": True,
            "performance_impact": "moderate"
        }
    
    async def _escalate_strategy(self, error: ValidationError, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Escalate to higher-level error handling."""
        escalation_handler = context.get("escalation_handler")
        
        logger.warning(f"Escalating error {error.error_id} to higher level")
        
        if escalation_handler and callable(escalation_handler):
            try:
                result = await escalation_handler(error, context)
                return True, result
            except Exception as e:
                logger.error(f"Escalation handler failed: {e}")
        
        # Default escalation - alert administrators
        alert_data = {
            "action": "admin_alert",
            "error_id": error.error_id,
            "severity": error.severity.value,
            "message": error.message,
            "requires_manual_intervention": True
        }
        
        logger.critical(f"ADMIN ALERT: {error.message}")
        return True, alert_data
    
    async def _terminate_strategy(self, error: ValidationError, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Safely terminate operation."""
        logger.error(f"Terminating operation due to critical error {error.error_id}")
        
        # Cleanup resources if cleanup function provided
        cleanup_func = context.get("cleanup_operation")
        if cleanup_func and callable(cleanup_func):
            try:
                await cleanup_func(context)
                logger.info("Cleanup operation completed successfully")
            except Exception as e:
                logger.error(f"Cleanup operation failed: {e}")
        
        return True, {
            "action": "operation_terminated",
            "reason": error.message,
            "cleanup_performed": cleanup_func is not None
        }
    
    async def _circuit_break_strategy(self, error: ValidationError, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Implement circuit breaker pattern."""
        circuit_name = context.get("circuit_name", "default")
        
        logger.warning(f"Circuit breaker activated for {circuit_name} due to error {error.error_id}")
        
        # Temporarily disable functionality
        return True, {
            "action": "circuit_breaker_activated",
            "circuit_name": circuit_name,
            "disabled_until": time.time() + 300,  # 5 minutes
            "alternative_available": context.get("alternative_available", False)
        }
    
    def _sanitize_task_data(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize task data for fallback processing."""
        sanitized = {
            "id": task_data.get("id", f"task_{int(time.time())}"),
            "name": task_data.get("name", "Unknown Task").strip() or "Unnamed Task",
            "operation": task_data.get("operation", "generic"),
            "estimated_duration": max(1.0, min(float(task_data.get("estimated_duration", 60)), 3600)),
            "resources": {}
        }
        
        # Sanitize resources
        original_resources = task_data.get("resources", {})
        if isinstance(original_resources, dict):
            for resource_type, amount in original_resources.items():
                try:
                    sanitized_amount = max(0.0, float(amount))
                    # Apply reasonable limits
                    if resource_type == "memory_gb":
                        sanitized_amount = min(sanitized_amount, 16)
                    elif resource_type == "cpu_cores":
                        sanitized_amount = min(sanitized_amount, 8)
                    
                    sanitized["resources"][resource_type] = sanitized_amount
                except (ValueError, TypeError):
                    # Skip invalid resource entries
                    continue
        
        return sanitized
    
    def get_handler_metrics(self) -> Dict[str, Any]:
        """Get error handler performance metrics."""
        total_errors = sum(self.error_metrics.values())
        
        recovery_stats = {}
        for strategy, stats in self.recovery_success_rate.items():
            if stats["attempts"] > 0:
                success_rate = stats["successes"] / stats["attempts"]
                recovery_stats[strategy] = {
                    "attempts": stats["attempts"],
                    "successes": stats["successes"],
                    "success_rate": success_rate
                }
        
        return {
            "total_errors_handled": total_errors,
            "errors_by_type": dict(self.error_metrics),
            "recovery_statistics": recovery_stats,
            "average_recovery_rate": (
                sum(stats.get("success_rate", 0) for stats in recovery_stats.values()) / 
                len(recovery_stats) if recovery_stats else 0
            )
        }


class MonitoringEnhancer:
    """Enhanced monitoring for quantum task planning."""
    
    def __init__(self):
        self.health_checks = {}
        self.performance_metrics = defaultdict(list)
        self.alert_thresholds = {}
        self.alert_handlers = []
        
        # Default health checks
        self._setup_default_health_checks()
        
        logger.info("MonitoringEnhancer initialized")
    
    def _setup_default_health_checks(self):
        """Setup default health check functions."""
        self.health_checks["memory_usage"] = self._check_memory_usage
        self.health_checks["cpu_usage"] = self._check_cpu_usage
        self.health_checks["task_queue_health"] = self._check_task_queue_health
        self.health_checks["error_rate"] = self._check_error_rate
        
        # Default alert thresholds
        self.alert_thresholds = {
            "memory_usage": 85.0,        # Percent
            "cpu_usage": 90.0,           # Percent
            "task_queue_length": 100,    # Number of tasks
            "error_rate": 0.1,           # 10% error rate
            "response_time": 5.0         # Seconds
        }
    
    async def run_health_checks(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run all health checks and return status."""
        health_status = {
            "overall_status": "healthy",
            "timestamp": time.time(),
            "checks": {}
        }
        
        for check_name, check_func in self.health_checks.items():
            try:
                result = await check_func(context)
                health_status["checks"][check_name] = result
                
                # Update overall status
                if result["status"] == "critical":
                    health_status["overall_status"] = "critical"
                elif result["status"] == "warning" and health_status["overall_status"] == "healthy":
                    health_status["overall_status"] = "warning"
                    
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                health_status["checks"][check_name] = {
                    "status": "error",
                    "message": f"Check failed: {str(e)}",
                    "timestamp": time.time()
                }
                if health_status["overall_status"] != "critical":
                    health_status["overall_status"] = "degraded"
        
        # Trigger alerts if necessary
        await self._process_alerts(health_status)
        
        return health_status
    
    async def _check_memory_usage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check system memory usage."""
        # Simulate memory check - in production use psutil
        resource_manager = context.get("resource_manager")
        if resource_manager:
            utilization = resource_manager.get_resource_utilization()
            memory_usage = utilization.get("memory_utilization", 0)
        else:
            memory_usage = 45.0  # Mock value
        
        status = "healthy"
        if memory_usage > self.alert_thresholds.get("memory_usage", 85):
            status = "critical"
        elif memory_usage > 70:
            status = "warning"
        
        return {
            "status": status,
            "value": memory_usage,
            "threshold": self.alert_thresholds.get("memory_usage", 85),
            "unit": "percent",
            "message": f"Memory usage: {memory_usage:.1f}%"
        }
    
    async def _check_cpu_usage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check system CPU usage."""
        resource_manager = context.get("resource_manager")
        if resource_manager:
            utilization = resource_manager.get_resource_utilization()
            cpu_usage = utilization.get("cpu_utilization", 0)
        else:
            cpu_usage = 35.0  # Mock value
        
        status = "healthy"
        if cpu_usage > self.alert_thresholds.get("cpu_usage", 90):
            status = "critical"
        elif cpu_usage > 75:
            status = "warning"
        
        return {
            "status": status,
            "value": cpu_usage,
            "threshold": self.alert_thresholds.get("cpu_usage", 90),
            "unit": "percent",
            "message": f"CPU usage: {cpu_usage:.1f}%"
        }
    
    async def _check_task_queue_health(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check task queue health."""
        task_planner = context.get("task_planner")
        if task_planner:
            queue_length = len(task_planner.task_queue)
            running_tasks = len(task_planner.running_tasks)
        else:
            queue_length = 0
            running_tasks = 0
        
        status = "healthy"
        threshold = self.alert_thresholds.get("task_queue_length", 100)
        
        if queue_length > threshold:
            status = "critical"
        elif queue_length > threshold * 0.7:
            status = "warning"
        
        return {
            "status": status,
            "queue_length": queue_length,
            "running_tasks": running_tasks,
            "threshold": threshold,
            "message": f"Queue: {queue_length} pending, {running_tasks} running"
        }
    
    async def _check_error_rate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check system error rate."""
        validator = context.get("validator")
        if validator and hasattr(validator, 'error_history'):
            recent_errors = [
                error for error in validator.error_history
                if time.time() - error.timestamp < 3600  # Last hour
            ]
            error_rate = len(recent_errors) / max(100, 100)  # Assume 100 operations in last hour
        else:
            error_rate = 0.02  # Mock 2% error rate
        
        status = "healthy"
        threshold = self.alert_thresholds.get("error_rate", 0.1)
        
        if error_rate > threshold:
            status = "critical"
        elif error_rate > threshold * 0.5:
            status = "warning"
        
        return {
            "status": status,
            "value": error_rate,
            "threshold": threshold,
            "unit": "ratio",
            "message": f"Error rate: {error_rate:.2%}"
        }
    
    async def _process_alerts(self, health_status: Dict[str, Any]):
        """Process health status and trigger alerts if needed."""
        if health_status["overall_status"] in ["critical", "warning"]:
            alert_data = {
                "severity": health_status["overall_status"],
                "timestamp": health_status["timestamp"],
                "summary": f"System health is {health_status['overall_status']}",
                "details": health_status["checks"]
            }
            
            for alert_handler in self.alert_handlers:
                try:
                    await alert_handler(alert_data)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
    
    def register_alert_handler(self, handler: Callable):
        """Register alert handler function."""
        self.alert_handlers.append(handler)
        logger.info("Registered new alert handler")
    
    def record_performance_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record performance metric for monitoring."""
        metric_data = {
            "value": value,
            "timestamp": time.time(),
            "tags": tags or {}
        }
        
        self.performance_metrics[metric_name].append(metric_data)
        
        # Keep only recent data (last 1000 points)
        if len(self.performance_metrics[metric_name]) > 1000:
            self.performance_metrics[metric_name] = self.performance_metrics[metric_name][-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        summary = {}
        
        for metric_name, data_points in self.performance_metrics.items():
            if not data_points:
                continue
            
            values = [point["value"] for point in data_points[-100:]]  # Last 100 points
            
            summary[metric_name] = {
                "current": values[-1] if values else None,
                "average": sum(values) / len(values) if values else 0,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "count": len(data_points)
            }
        
        return summary


# Integration functions for quantum planner

async def create_robust_quantum_planner(max_concurrent_tasks: int = 4) -> 'QuantumTaskPlanner':
    """Create quantum task planner with robust error handling."""
    from .quantum_planner import QuantumTaskPlanner
    
    # Create enhanced components
    validator = RobustValidator()
    error_handler = EnhancedErrorHandler()
    monitoring = MonitoringEnhancer()
    
    # Create planner
    planner = QuantumTaskPlanner(max_concurrent_tasks)
    
    # Inject robust components
    planner.validator = validator
    planner.error_handler = error_handler
    planner.monitoring = monitoring
    
    # Setup alert handlers
    async def log_alert(alert_data: Dict[str, Any]):
        logger.warning(f"SYSTEM ALERT: {alert_data['summary']}")
    
    monitoring.register_alert_handler(log_alert)
    
    logger.info("Created robust quantum task planner with enhanced error handling")
    return planner


def setup_custom_validation_rules(planner: 'QuantumTaskPlanner', rules: Dict[str, Callable]):
    """Setup custom validation rules for task planner."""
    if hasattr(planner, 'validator'):
        for rule_name, rule_func in rules.items():
            planner.validator.register_validator(rule_name, rule_func)
        logger.info(f"Registered {len(rules)} custom validation rules")
    else:
        logger.warning("Planner does not have validator component")