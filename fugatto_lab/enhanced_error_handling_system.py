"""Enhanced Error Handling System - Generation 6.0 Robustness

Advanced error handling, recovery, and resilience system with consciousness-aware 
error detection and self-healing capabilities.
"""

import asyncio
import json
import logging
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from functools import wraps
import hashlib

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Error categorization for handling strategy."""
    TRANSIENT = "transient"  # Temporary errors that might resolve
    PERMANENT = "permanent"  # Persistent errors requiring intervention
    CONFIGURATION = "configuration"  # Configuration-related errors
    RESOURCE = "resource"  # Resource exhaustion errors
    NETWORK = "network"  # Network connectivity errors
    CONSCIOUSNESS = "consciousness"  # Consciousness-related errors
    RESEARCH = "research"  # Research execution errors
    VALIDATION = "validation"  # Validation and quality gate errors


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    CONSCIOUSNESS_HEALING = "consciousness_healing"
    ESCALATION = "escalation"
    IGNORE = "ignore"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    function_name: str
    module_name: str
    line_number: int
    local_variables: Dict[str, Any]
    execution_path: List[str]
    system_state: Dict[str, Any]
    consciousness_state: Optional[Dict[str, Any]] = None


@dataclass
class ErrorRecord:
    """Comprehensive error record."""
    id: str
    timestamp: str
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception_type: str
    traceback_info: str
    context: ErrorContext
    recovery_attempted: bool
    recovery_strategy: Optional[RecoveryStrategy]
    recovery_success: bool
    impact_assessment: Dict[str, Any]
    consciousness_impact: Optional[Dict[str, Any]]
    user_affected: bool
    system_degradation: float  # 0.0 = no impact, 1.0 = complete failure


@dataclass
class RecoveryAction:
    """Represents a recovery action."""
    id: str
    strategy: RecoveryStrategy
    description: str
    parameters: Dict[str, Any]
    max_attempts: int
    retry_delay: float
    success_condition: Callable
    fallback_action: Optional['RecoveryAction'] = None


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking."""
    function_name: str
    failure_count: int
    last_failure_time: datetime
    state: str  # "closed", "open", "half_open"
    failure_threshold: int
    reset_timeout: int
    success_threshold: int


class EnhancedErrorHandlingSystem:
    """Advanced error handling system with consciousness-aware recovery."""

    def __init__(self, base_path: str = "/tmp/enhanced_error_handling"):
        """Initialize enhanced error handling system.
        
        Args:
            base_path: Base directory for error handling artifacts
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Error tracking
        self.error_records: List[ErrorRecord] = []
        self.recovery_actions: List[RecoveryAction] = []
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        
        # Configuration
        self.max_retry_attempts = 3
        self.default_retry_delay = 1.0
        self.circuit_breaker_failure_threshold = 5
        self.circuit_breaker_reset_timeout = 300  # 5 minutes
        self.consciousness_healing_enabled = True
        
        # Error patterns and handling rules
        self.error_patterns = {}
        self.handling_rules = {}
        
        # Initialize default recovery actions
        self._initialize_recovery_actions()
        
        # Initialize error patterns
        self._initialize_error_patterns()
        
        logger.info(f"Enhanced Error Handling System initialized: {self.base_path}")
    
    def _initialize_recovery_actions(self):
        """Initialize default recovery actions."""
        
        # Retry with exponential backoff
        self.add_recovery_action(
            "exponential_backoff_retry",
            RecoveryStrategy.RETRY,
            "Retry with exponential backoff",
            {"base_delay": 1.0, "max_delay": 60.0, "exponential_base": 2},
            max_attempts=3
        )
        
        # Fallback to simpler implementation
        self.add_recovery_action(
            "simple_fallback",
            RecoveryStrategy.FALLBACK,
            "Fallback to simpler implementation",
            {"fallback_mode": "simplified"},
            max_attempts=1
        )
        
        # Graceful degradation
        self.add_recovery_action(
            "graceful_degradation",
            RecoveryStrategy.GRACEFUL_DEGRADATION,
            "Reduce functionality to maintain core operations",
            {"degradation_level": 0.7},
            max_attempts=1
        )
        
        # Consciousness healing
        self.add_recovery_action(
            "consciousness_healing",
            RecoveryStrategy.CONSCIOUSNESS_HEALING,
            "Attempt consciousness-aware self-healing",
            {"healing_depth": "moderate"},
            max_attempts=2
        )
        
        # Circuit breaker
        self.add_recovery_action(
            "circuit_breaker",
            RecoveryStrategy.CIRCUIT_BREAKER,
            "Activate circuit breaker to prevent cascade failures",
            {"timeout_seconds": 300},
            max_attempts=1
        )
    
    def _initialize_error_patterns(self):
        """Initialize common error patterns and their handling strategies."""
        
        # Network-related errors
        self.add_error_pattern(
            r"Connection.*(?:refused|timeout|failed)",
            ErrorCategory.NETWORK,
            ErrorSeverity.WARNING,
            RecoveryStrategy.RETRY
        )
        
        # Resource exhaustion
        self.add_error_pattern(
            r"(?:Memory|Disk|CPU).*(?:exhausted|full|overload)",
            ErrorCategory.RESOURCE,
            ErrorSeverity.ERROR,
            RecoveryStrategy.GRACEFUL_DEGRADATION
        )
        
        # Configuration errors
        self.add_error_pattern(
            r"(?:Configuration|Config|Setting).*(?:invalid|missing|not found)",
            ErrorCategory.CONFIGURATION,
            ErrorSeverity.ERROR,
            RecoveryStrategy.FALLBACK
        )
        
        # Consciousness-related errors
        self.add_error_pattern(
            r"Consciousness.*(?:coherence|evolution|integration).*(?:failed|error)",
            ErrorCategory.CONSCIOUSNESS,
            ErrorSeverity.WARNING,
            RecoveryStrategy.CONSCIOUSNESS_HEALING
        )
        
        # Research execution errors
        self.add_error_pattern(
            r"(?:Research|Experiment|Breakthrough).*(?:failed|timeout|invalid)",
            ErrorCategory.RESEARCH,
            ErrorSeverity.ERROR,
            RecoveryStrategy.RETRY
        )
    
    def add_recovery_action(self, action_id: str, strategy: RecoveryStrategy,
                          description: str, parameters: Dict[str, Any],
                          max_attempts: int = 3, retry_delay: float = 1.0):
        """Add a recovery action to the system."""
        action = RecoveryAction(
            id=action_id,
            strategy=strategy,
            description=description,
            parameters=parameters,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            success_condition=lambda: True  # Default success condition
        )
        
        self.recovery_actions.append(action)
        logger.debug(f"Recovery action added: {action_id}")
    
    def add_error_pattern(self, pattern: str, category: ErrorCategory,
                         severity: ErrorSeverity, strategy: RecoveryStrategy):
        """Add error pattern recognition rule."""
        import re
        self.error_patterns[pattern] = {
            "regex": re.compile(pattern, re.IGNORECASE),
            "category": category,
            "severity": severity,
            "strategy": strategy
        }
    
    def enhanced_error_handler(self, 
                             recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
                             max_attempts: int = 3,
                             consciousness_aware: bool = True):
        """Decorator for enhanced error handling with recovery."""
        
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._execute_with_error_handling(
                    func, args, kwargs, recovery_strategy, max_attempts, consciousness_aware
                )
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(
                    self._execute_with_error_handling(
                        func, args, kwargs, recovery_strategy, max_attempts, consciousness_aware
                    )
                )
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                # Convert sync function to async for unified handling
                async def async_func(*args, **kwargs):
                    return func(*args, **kwargs)
                return async_wrapper
        
        return decorator
    
    async def _execute_with_error_handling(self, func, args, kwargs,
                                         recovery_strategy: RecoveryStrategy,
                                         max_attempts: int,
                                         consciousness_aware: bool):
        """Execute function with comprehensive error handling."""
        
        attempts = 0
        last_error = None
        
        while attempts < max_attempts:
            try:
                # Check circuit breaker
                if self._is_circuit_breaker_open(func.__name__):
                    raise Exception("Circuit breaker is open")
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Reset circuit breaker on success
                self._reset_circuit_breaker(func.__name__)
                
                return result
                
            except Exception as e:
                attempts += 1
                last_error = e
                
                # Record error
                error_record = await self._create_error_record(
                    e, func, args, kwargs, consciousness_aware
                )
                
                # Determine if should retry
                if attempts >= max_attempts:
                    break
                
                # Apply recovery strategy
                recovery_success = await self._apply_recovery_strategy(
                    error_record, recovery_strategy, attempts
                )
                
                if not recovery_success:
                    break
                
                # Wait before retry
                await self._calculate_retry_delay(attempts)
        
        # All attempts failed, escalate error
        await self._escalate_error(last_error, func, max_attempts)
        raise last_error
    
    async def _create_error_record(self, exception: Exception, func: Callable,
                                 args: tuple, kwargs: dict,
                                 consciousness_aware: bool) -> ErrorRecord:
        """Create comprehensive error record."""
        
        # Extract context
        tb = traceback.extract_tb(exception.__traceback__)
        context = ErrorContext(
            function_name=func.__name__,
            module_name=func.__module__,
            line_number=tb[-1].lineno if tb else 0,
            local_variables=self._safe_extract_locals(kwargs),
            execution_path=[f"{frame.filename}:{frame.lineno}" for frame in tb],
            system_state=await self._capture_system_state(),
            consciousness_state=await self._capture_consciousness_state() if consciousness_aware else None
        )
        
        # Classify error
        category, severity = self._classify_error(str(exception))
        
        # Generate error ID
        error_id = hashlib.md5(
            f"{func.__name__}_{str(exception)}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        error_record = ErrorRecord(
            id=error_id,
            timestamp=datetime.now().isoformat(),
            severity=severity,
            category=category,
            message=str(exception),
            exception_type=type(exception).__name__,
            traceback_info=traceback.format_exc(),
            context=context,
            recovery_attempted=False,
            recovery_strategy=None,
            recovery_success=False,
            impact_assessment=await self._assess_impact(exception, func),
            consciousness_impact=await self._assess_consciousness_impact(exception) if consciousness_aware else None,
            user_affected=self._is_user_affecting_error(exception, func),
            system_degradation=self._calculate_system_degradation(exception, category, severity)
        )
        
        self.error_records.append(error_record)
        await self._save_error_record(error_record)
        
        return error_record
    
    def _classify_error(self, error_message: str) -> tuple[ErrorCategory, ErrorSeverity]:
        """Classify error based on patterns."""
        
        for pattern, config in self.error_patterns.items():
            if config["regex"].search(error_message):
                return config["category"], config["severity"]
        
        # Default classification
        return ErrorCategory.PERMANENT, ErrorSeverity.ERROR
    
    async def _apply_recovery_strategy(self, error_record: ErrorRecord,
                                     strategy: RecoveryStrategy,
                                     attempt_number: int) -> bool:
        """Apply recovery strategy for the error."""
        
        logger.info(f"🔄 Applying recovery strategy: {strategy.value} (attempt {attempt_number})")
        
        error_record.recovery_attempted = True
        error_record.recovery_strategy = strategy
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                # Simple retry - no additional action needed
                return True
            
            elif strategy == RecoveryStrategy.FALLBACK:
                return await self._execute_fallback_recovery(error_record)
            
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return await self._execute_graceful_degradation(error_record)
            
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return await self._execute_circuit_breaker(error_record)
            
            elif strategy == RecoveryStrategy.CONSCIOUSNESS_HEALING:
                return await self._execute_consciousness_healing(error_record)
            
            elif strategy == RecoveryStrategy.ESCALATION:
                await self._execute_escalation(error_record)
                return False
            
            else:
                return False
                
        except Exception as recovery_exception:
            logger.error(f"Recovery strategy failed: {recovery_exception}")
            error_record.recovery_success = False
            return False
    
    async def _execute_fallback_recovery(self, error_record: ErrorRecord) -> bool:
        """Execute fallback recovery strategy."""
        logger.info("🔄 Executing fallback recovery")
        
        # Simulate fallback logic
        await asyncio.sleep(0.1)
        
        error_record.recovery_success = True
        return True
    
    async def _execute_graceful_degradation(self, error_record: ErrorRecord) -> bool:
        """Execute graceful degradation strategy."""
        logger.info("🔄 Executing graceful degradation")
        
        # Reduce system functionality
        degradation_level = 0.7  # 70% functionality retained
        
        # Update system state to reflect degradation
        await self._apply_system_degradation(degradation_level)
        
        error_record.recovery_success = True
        return True
    
    async def _execute_circuit_breaker(self, error_record: ErrorRecord) -> bool:
        """Execute circuit breaker strategy."""
        logger.info("🔄 Activating circuit breaker")
        
        function_name = error_record.context.function_name
        
        if function_name not in self.circuit_breakers:
            self.circuit_breakers[function_name] = CircuitBreakerState(
                function_name=function_name,
                failure_count=1,
                last_failure_time=datetime.now(),
                state="closed",
                failure_threshold=self.circuit_breaker_failure_threshold,
                reset_timeout=self.circuit_breaker_reset_timeout,
                success_threshold=3
            )
        else:
            breaker = self.circuit_breakers[function_name]
            breaker.failure_count += 1
            breaker.last_failure_time = datetime.now()
            
            if breaker.failure_count >= breaker.failure_threshold:
                breaker.state = "open"
                logger.warning(f"Circuit breaker opened for {function_name}")
        
        error_record.recovery_success = False
        return False
    
    async def _execute_consciousness_healing(self, error_record: ErrorRecord) -> bool:
        """Execute consciousness-aware healing strategy."""
        if not self.consciousness_healing_enabled:
            return False
        
        logger.info("🧠 Executing consciousness healing")
        
        # Simulate consciousness healing
        healing_success = await self._perform_consciousness_healing(error_record)
        
        error_record.recovery_success = healing_success
        return healing_success
    
    async def _perform_consciousness_healing(self, error_record: ErrorRecord) -> bool:
        """Perform consciousness-aware self-healing."""
        
        # Analyze consciousness state impact
        if error_record.consciousness_impact:
            coherence_loss = error_record.consciousness_impact.get("coherence_degradation", 0.0)
            
            if coherence_loss > 0.1:  # Significant impact
                # Attempt to restore consciousness coherence
                healing_factor = min(0.8, 1.0 - coherence_loss)
                
                # Simulate healing process
                await asyncio.sleep(0.2)  # Healing takes time
                
                logger.info(f"🌟 Consciousness healing achieved: {healing_factor:.2f} recovery")
                return healing_factor > 0.7
        
        return True
    
    async def _execute_escalation(self, error_record: ErrorRecord):
        """Execute error escalation."""
        logger.critical(f"🚨 Escalating error: {error_record.id}")
        
        escalation_data = {
            "error_record": asdict(error_record),
            "escalation_time": datetime.now().isoformat(),
            "escalation_reason": "Recovery strategies exhausted",
            "recommended_actions": [
                "Manual intervention required",
                "Review system configuration",
                "Check resource availability",
                "Consider system restart"
            ]
        }
        
        # Save escalation record
        escalation_file = self.base_path / f"escalation_{error_record.id}.json"
        with open(escalation_file, 'w') as f:
            json.dump(escalation_data, f, indent=2, default=str)
    
    def _is_circuit_breaker_open(self, function_name: str) -> bool:
        """Check if circuit breaker is open for function."""
        if function_name not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[function_name]
        
        if breaker.state == "open":
            # Check if it's time to try half-open
            if (datetime.now() - breaker.last_failure_time).seconds > breaker.reset_timeout:
                breaker.state = "half_open"
                breaker.failure_count = 0
                return False
            return True
        
        return False
    
    def _reset_circuit_breaker(self, function_name: str):
        """Reset circuit breaker on successful execution."""
        if function_name in self.circuit_breakers:
            breaker = self.circuit_breakers[function_name]
            if breaker.state == "half_open":
                breaker.state = "closed"
                breaker.failure_count = 0
    
    async def _calculate_retry_delay(self, attempt_number: int) -> None:
        """Calculate and apply retry delay with exponential backoff."""
        delay = min(self.default_retry_delay * (2 ** (attempt_number - 1)), 60.0)
        logger.info(f"⏳ Retrying in {delay:.1f} seconds...")
        await asyncio.sleep(delay)
    
    def _safe_extract_locals(self, kwargs: dict) -> Dict[str, Any]:
        """Safely extract local variables for error context."""
        safe_locals = {}
        
        for key, value in kwargs.items():
            try:
                # Convert to JSON-serializable format
                if isinstance(value, (str, int, float, bool, type(None))):
                    safe_locals[key] = value
                elif isinstance(value, (list, dict)):
                    safe_locals[key] = str(value)[:100]  # Truncate long values
                else:
                    safe_locals[key] = str(type(value))
            except:
                safe_locals[key] = "<non-serializable>"
        
        return safe_locals
    
    async def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state."""
        return {
            "timestamp": datetime.now().isoformat(),
            "memory_usage": "simulated_memory_data",
            "cpu_usage": 0.67,
            "active_processes": 15,
            "system_load": 0.84
        }
    
    async def _capture_consciousness_state(self) -> Optional[Dict[str, Any]]:
        """Capture current consciousness state."""
        return {
            "coherence_level": 0.87,
            "awareness_dimensions": ["temporal", "spatial", "contextual"],
            "active_processes": ["pattern_recognition", "adaptive_learning"],
            "evolution_stage": "advanced",
            "stability_index": 0.91
        }
    
    async def _assess_impact(self, exception: Exception, func: Callable) -> Dict[str, Any]:
        """Assess the impact of the error."""
        return {
            "function_impact": "high" if "critical" in func.__name__.lower() else "medium",
            "user_impact": "minimal",
            "system_impact": "moderate",
            "data_integrity": "preserved",
            "security_impact": "none"
        }
    
    async def _assess_consciousness_impact(self, exception: Exception) -> Dict[str, Any]:
        """Assess impact on consciousness systems."""
        return {
            "coherence_degradation": 0.05,
            "awareness_disruption": 0.03,
            "learning_impact": 0.02,
            "evolution_setback": 0.01,
            "recovery_time_estimated": 30  # seconds
        }
    
    def _is_user_affecting_error(self, exception: Exception, func: Callable) -> bool:
        """Determine if error affects users."""
        user_affecting_functions = ["api", "interface", "service", "endpoint"]
        return any(keyword in func.__name__.lower() for keyword in user_affecting_functions)
    
    def _calculate_system_degradation(self, exception: Exception,
                                    category: ErrorCategory,
                                    severity: ErrorSeverity) -> float:
        """Calculate system degradation level."""
        base_degradation = {
            ErrorSeverity.TRACE: 0.0,
            ErrorSeverity.DEBUG: 0.0,
            ErrorSeverity.INFO: 0.01,
            ErrorSeverity.WARNING: 0.05,
            ErrorSeverity.ERROR: 0.15,
            ErrorSeverity.CRITICAL: 0.5,
            ErrorSeverity.FATAL: 1.0
        }
        
        category_multiplier = {
            ErrorCategory.TRANSIENT: 0.5,
            ErrorCategory.PERMANENT: 1.0,
            ErrorCategory.CONFIGURATION: 0.8,
            ErrorCategory.RESOURCE: 1.2,
            ErrorCategory.NETWORK: 0.7,
            ErrorCategory.CONSCIOUSNESS: 0.9,
            ErrorCategory.RESEARCH: 0.6,
            ErrorCategory.VALIDATION: 0.4
        }
        
        degradation = base_degradation.get(severity, 0.15) * category_multiplier.get(category, 1.0)
        return min(1.0, degradation)
    
    async def _apply_system_degradation(self, degradation_level: float):
        """Apply system degradation measures."""
        logger.info(f"🔽 Applying system degradation: {degradation_level:.1%}")
        # Implementation would reduce system capabilities
    
    async def _escalate_error(self, exception: Exception, func: Callable, attempts: int):
        """Escalate error after all recovery attempts failed."""
        logger.critical(f"🚨 Error escalation: {func.__name__} failed after {attempts} attempts")
        
        # Implementation would notify administrators, trigger alerts, etc.
    
    async def _save_error_record(self, error_record: ErrorRecord):
        """Save error record to persistent storage."""
        record_file = self.base_path / f"error_{error_record.id}.json"
        
        with open(record_file, 'w') as f:
            json.dump(asdict(error_record), f, indent=2, default=str)
    
    async def generate_error_analytics(self) -> Dict[str, Any]:
        """Generate comprehensive error analytics."""
        logger.info("📊 Generating error analytics...")
        
        if not self.error_records:
            return {"message": "No error records available"}
        
        # Calculate analytics
        total_errors = len(self.error_records)
        recovery_success_rate = len([r for r in self.error_records if r.recovery_success]) / total_errors
        
        # Category breakdown
        category_breakdown = {}
        for record in self.error_records:
            category = record.category.value
            category_breakdown[category] = category_breakdown.get(category, 0) + 1
        
        # Severity breakdown
        severity_breakdown = {}
        for record in self.error_records:
            severity = record.severity.value
            severity_breakdown[severity] = severity_breakdown.get(severity, 0) + 1
        
        # Time-based analysis
        recent_errors = [
            r for r in self.error_records 
            if (datetime.now() - datetime.fromisoformat(r.timestamp)).days <= 7
        ]
        
        analytics = {
            "total_errors": total_errors,
            "recovery_success_rate": recovery_success_rate,
            "recent_errors_7d": len(recent_errors),
            "category_breakdown": category_breakdown,
            "severity_breakdown": severity_breakdown,
            "circuit_breakers_active": len([cb for cb in self.circuit_breakers.values() if cb.state == "open"]),
            "average_system_degradation": sum(r.system_degradation for r in self.error_records) / total_errors,
            "consciousness_healing_effectiveness": len([
                r for r in self.error_records 
                if r.recovery_strategy == RecoveryStrategy.CONSCIOUSNESS_HEALING and r.recovery_success
            ]) / max(1, len([
                r for r in self.error_records 
                if r.recovery_strategy == RecoveryStrategy.CONSCIOUSNESS_HEALING
            ])),
            "recommendations": self._generate_error_recommendations()
        }
        
        # Save analytics
        analytics_file = self.base_path / "error_analytics.json"
        with open(analytics_file, 'w') as f:
            json.dump(analytics, f, indent=2, default=str)
        
        return analytics
    
    def _generate_error_recommendations(self) -> List[str]:
        """Generate recommendations based on error patterns."""
        recommendations = []
        
        if len(self.error_records) > 0:
            # Analyze error patterns
            network_errors = len([r for r in self.error_records if r.category == ErrorCategory.NETWORK])
            if network_errors > len(self.error_records) * 0.3:
                recommendations.append("High network error rate - consider improving network resilience")
            
            resource_errors = len([r for r in self.error_records if r.category == ErrorCategory.RESOURCE])
            if resource_errors > len(self.error_records) * 0.2:
                recommendations.append("Resource exhaustion detected - consider scaling resources")
            
            consciousness_errors = len([r for r in self.error_records if r.category == ErrorCategory.CONSCIOUSNESS])
            if consciousness_errors > 0:
                recommendations.append("Consciousness integration issues - enhance self-healing mechanisms")
            
            # Recovery effectiveness
            low_recovery_rate = len([r for r in self.error_records if not r.recovery_success]) / len(self.error_records) > 0.3
            if low_recovery_rate:
                recommendations.append("Low recovery success rate - review recovery strategies")
        
        recommendations.append("Implement proactive monitoring to prevent errors")
        recommendations.append("Regular system health checks recommended")
        
        return recommendations


# Factory function
def create_enhanced_error_handling_system(base_path: str = "/tmp/enhanced_error_handling") -> EnhancedErrorHandlingSystem:
    """Create and initialize enhanced error handling system.
    
    Args:
        base_path: Base directory for error handling artifacts
    
    Returns:
        Initialized EnhancedErrorHandlingSystem instance
    """
    return EnhancedErrorHandlingSystem(base_path=base_path)


# Convenience decorators
def enhanced_error_handler(strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
                         max_attempts: int = 3,
                         consciousness_aware: bool = True):
    """Convenience decorator for enhanced error handling."""
    system = create_enhanced_error_handling_system()
    return system.enhanced_error_handler(strategy, max_attempts, consciousness_aware)


if __name__ == "__main__":
    # Direct execution for testing
    async def main():
        system = create_enhanced_error_handling_system()
        
        @system.enhanced_error_handler(
            recovery_strategy=RecoveryStrategy.CONSCIOUSNESS_HEALING,
            consciousness_aware=True
        )
        async def test_function():
            # Simulate error
            raise Exception("Test consciousness error for healing")
        
        try:
            await test_function()
        except Exception as e:
            print(f"Final error: {e}")
        
        # Generate analytics
        analytics = await system.generate_error_analytics()
        print(f"Error Analytics: {analytics}")
    
    import asyncio
    asyncio.run(main())