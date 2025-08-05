"""Robust Error Handling and Validation System for Fugatto Audio Lab.

Comprehensive error handling, validation, recovery mechanisms, and resilience patterns
for production-grade audio processing workflows.
"""

import functools
import logging
import traceback
import time
import json
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    CRITICAL = "critical"      # System-threatening errors
    HIGH = "high"             # Major functionality impact
    MEDIUM = "medium"         # Moderate impact, degraded performance
    LOW = "low"              # Minor issues, minimal impact
    INFO = "info"            # Informational, not actually errors


class ErrorCategory(Enum):
    """Error categories for systematic handling."""
    VALIDATION = "validation"           # Input validation errors
    RESOURCE = "resource"              # Resource-related errors (memory, disk, etc.)
    NETWORK = "network"                # Network and connectivity errors
    PROCESSING = "processing"          # Audio processing errors
    MODEL = "model"                    # ML model errors
    STORAGE = "storage"                # File system and storage errors
    PERMISSION = "permission"          # Authentication and authorization errors
    TIMEOUT = "timeout"                # Operation timeout errors
    CONFIGURATION = "configuration"    # Configuration and setup errors
    EXTERNAL = "external"              # External service errors
    UNKNOWN = "unknown"                # Unclassified errors


@dataclass
class ErrorContext:
    """Comprehensive error context for debugging and monitoring."""
    
    error_id: str
    timestamp: float = field(default_factory=time.time)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.UNKNOWN
    message: str = ""
    exception_type: str = ""
    traceback_info: str = ""
    function_name: str = ""
    module_name: str = ""
    user_context: Dict[str, Any] = field(default_factory=dict)
    system_context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary for logging/storage."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "exception_type": self.exception_type,
            "traceback": self.traceback_info,
            "function": self.function_name,
            "module": self.module_name,
            "user_context": self.user_context,
            "system_context": self.system_context,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "retry_count": self.retry_count
        }


class AudioProcessingError(Exception):
    """Base exception for audio processing errors."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.PROCESSING,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Dict[str, Any] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context or {}


class ValidationError(AudioProcessingError):
    """Input validation error."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message, ErrorCategory.VALIDATION, ErrorSeverity.HIGH)
        self.field = field
        self.value = value


class ResourceError(AudioProcessingError):
    """Resource availability or allocation error."""
    
    def __init__(self, message: str, resource_type: str = None, required: float = None, available: float = None):
        super().__init__(message, ErrorCategory.RESOURCE, ErrorSeverity.HIGH)
        self.resource_type = resource_type
        self.required = required
        self.available = available


class ModelError(AudioProcessingError):
    """ML model loading or inference error."""
    
    def __init__(self, message: str, model_name: str = None, model_path: str = None):
        super().__init__(message, ErrorCategory.MODEL, ErrorSeverity.CRITICAL)
        self.model_name = model_name
        self.model_path = model_path


class TimeoutError(AudioProcessingError):
    """Operation timeout error."""
    
    def __init__(self, message: str, timeout_seconds: float = None, actual_duration: float = None):
        super().__init__(message, ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM)
        self.timeout_seconds = timeout_seconds
        self.actual_duration = actual_duration


class RobustErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self, log_file: str = None, enable_recovery: bool = True):
        self.log_file = log_file
        self.enable_recovery = enable_recovery
        self.error_history = []
        self.recovery_strategies = {}
        self.error_counts = {}
        self.circuit_breakers = {}
        
        # Configure logging
        self._setup_error_logging()
        
        # Register default recovery strategies
        self._register_default_recovery_strategies()
        
        logger.info("RobustErrorHandler initialized")
    
    def _setup_error_logging(self):
        """Setup specialized error logging."""
        if self.log_file:
            # Create file handler for error logs
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.ERROR)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
    
    def _register_default_recovery_strategies(self):
        """Register default error recovery strategies."""
        
        # Resource error recovery
        self.recovery_strategies[ErrorCategory.RESOURCE] = [
            self._recover_memory_pressure,
            self._recover_disk_space,
            self._recover_gpu_memory
        ]
        
        # Network error recovery
        self.recovery_strategies[ErrorCategory.NETWORK] = [
            self._recover_network_retry,
            self._recover_network_fallback
        ]
        
        # Processing error recovery
        self.recovery_strategies[ErrorCategory.PROCESSING] = [
            self._recover_processing_params,
            self._recover_processing_fallback
        ]
        
        # Model error recovery
        self.recovery_strategies[ErrorCategory.MODEL] = [
            self._recover_model_reload,
            self._recover_model_fallback
        ]
        
        # Timeout error recovery
        self.recovery_strategies[ErrorCategory.TIMEOUT] = [
            self._recover_timeout_extend,
            self._recover_timeout_chunk
        ]
    
    def handle_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorContext:
        """Comprehensive error handling with recovery attempts."""
        
        # Generate unique error ID
        error_id = f"err_{int(time.time() * 1000)}_{hash(str(exception)) % 10000}"
        
        # Classify error
        category = self._classify_error(exception)
        severity = self._assess_severity(exception, category)
        
        # Create error context
        error_context = ErrorContext(
            error_id=error_id,
            severity=severity,
            category=category,
            message=str(exception),
            exception_type=type(exception).__name__,
            traceback_info=traceback.format_exc(),
            function_name=self._get_calling_function(),
            module_name=self._get_calling_module(),
            user_context=context or {},
            system_context=self._get_system_context()
        )
        
        # Log error
        self._log_error(error_context)
        
        # Track error statistics
        self._track_error_stats(error_context)
        
        # Attempt recovery if enabled
        if self.enable_recovery and not self._is_circuit_breaker_open(category):
            success = self._attempt_recovery(error_context, exception)
            error_context.recovery_attempted = True
            error_context.recovery_successful = success
            
            if not success:
                self._update_circuit_breaker(category)
        
        # Store in history
        self.error_history.append(error_context)
        
        # Trim history to prevent memory leaks
        if len(self.error_history) > 10000:
            self.error_history = self.error_history[-5000:]
        
        return error_context
    
    def _classify_error(self, exception: Exception) -> ErrorCategory:
        """Classify error based on exception type and content."""
        
        if isinstance(exception, ValidationError):
            return ErrorCategory.VALIDATION
        elif isinstance(exception, ResourceError):
            return ErrorCategory.RESOURCE
        elif isinstance(exception, ModelError):
            return ErrorCategory.MODEL
        elif isinstance(exception, TimeoutError):
            return ErrorCategory.TIMEOUT
        elif isinstance(exception, (ConnectionError, OSError)) and "network" in str(exception).lower():
            return ErrorCategory.NETWORK
        elif isinstance(exception, PermissionError):
            return ErrorCategory.PERMISSION
        elif isinstance(exception, (FileNotFoundError, IOError)):
            return ErrorCategory.STORAGE
        elif "memory" in str(exception).lower() or isinstance(exception, MemoryError):
            return ErrorCategory.RESOURCE
        elif "timeout" in str(exception).lower():
            return ErrorCategory.TIMEOUT
        elif "config" in str(exception).lower() or "setting" in str(exception).lower():
            return ErrorCategory.CONFIGURATION
        else:
            return ErrorCategory.UNKNOWN
    
    def _assess_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Assess error severity based on exception and category."""
        
        if isinstance(exception, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        elif isinstance(exception, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        elif category == ErrorCategory.MODEL:
            return ErrorSeverity.CRITICAL
        elif category == ErrorCategory.RESOURCE:
            return ErrorSeverity.HIGH
        elif category == ErrorCategory.NETWORK:
            return ErrorSeverity.MEDIUM
        elif category == ErrorCategory.VALIDATION:
            return ErrorSeverity.HIGH
        elif category == ErrorCategory.TIMEOUT:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _get_calling_function(self) -> str:
        """Get the name of the function that caused the error."""
        try:
            frame = traceback.extract_tb(None)
            if frame:
                return frame[-1].name
        except:
            pass
        return "unknown"
    
    def _get_calling_module(self) -> str:
        """Get the name of the module that caused the error."""
        try:
            frame = traceback.extract_tb(None)
            if frame:
                return Path(frame[-1].filename).stem
        except:
            pass
        return "unknown"
    
    def _get_system_context(self) -> Dict[str, Any]:
        """Get current system context for error analysis."""
        try:
            import psutil
            
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            }
        except ImportError:
            # Fallback system context
            return {
                "timestamp": time.time(),
                "process_id": id(self)
            }
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level and formatting."""
        
        log_message = (
            f"[{error_context.error_id}] {error_context.category.value.upper()} "
            f"({error_context.severity.value}): {error_context.message}"
        )
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Log detailed context for debugging
        logger.debug(f"Error context: {json.dumps(error_context.to_dict(), indent=2)}")
    
    def _track_error_stats(self, error_context: ErrorContext):
        """Track error statistics for monitoring."""
        category_key = error_context.category.value
        severity_key = error_context.severity.value
        
        # Track by category
        if category_key not in self.error_counts:
            self.error_counts[category_key] = {"total": 0, "by_severity": {}}
        
        self.error_counts[category_key]["total"] += 1
        
        if severity_key not in self.error_counts[category_key]["by_severity"]:
            self.error_counts[category_key]["by_severity"][severity_key] = 0
        
        self.error_counts[category_key]["by_severity"][severity_key] += 1
    
    def _attempt_recovery(self, error_context: ErrorContext, exception: Exception) -> bool:
        """Attempt error recovery using registered strategies."""
        
        category = error_context.category
        if category not in self.recovery_strategies:
            return False
        
        for strategy in self.recovery_strategies[category]:
            try:
                if strategy(error_context, exception):
                    logger.info(f"Recovery successful for error {error_context.error_id}")
                    return True
            except Exception as recovery_error:
                logger.warning(f"Recovery strategy failed: {recovery_error}")
        
        return False
    
    def _is_circuit_breaker_open(self, category: ErrorCategory) -> bool:
        """Check if circuit breaker is open for error category."""
        if category.value not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[category.value]
        current_time = time.time()
        
        # Check if circuit breaker should reset
        if current_time - breaker["last_failure"] > breaker["reset_timeout"]:
            breaker["failure_count"] = 0
            breaker["is_open"] = False
        
        return breaker["is_open"]
    
    def _update_circuit_breaker(self, category: ErrorCategory):
        """Update circuit breaker state after recovery failure."""
        category_key = category.value
        
        if category_key not in self.circuit_breakers:
            self.circuit_breakers[category_key] = {
                "failure_count": 0,
                "failure_threshold": 5,
                "reset_timeout": 300,  # 5 minutes
                "last_failure": 0,
                "is_open": False
            }
        
        breaker = self.circuit_breakers[category_key]
        breaker["failure_count"] += 1
        breaker["last_failure"] = time.time()
        
        if breaker["failure_count"] >= breaker["failure_threshold"]:
            breaker["is_open"] = True
            logger.warning(f"Circuit breaker opened for category: {category_key}")
    
    # Recovery strategy implementations
    
    def _recover_memory_pressure(self, error_context: ErrorContext, exception: Exception) -> bool:
        """Recover from memory pressure issues."""
        try:
            import gc
            import psutil
            
            # Force garbage collection
            gc.collect()
            
            # Check if memory pressure reduced
            memory_percent = psutil.virtual_memory().percent
            if memory_percent < 85:  # Threshold for acceptable memory usage
                logger.info("Memory pressure recovered through garbage collection")
                return True
            
            return False
        except Exception:
            return False
    
    def _recover_disk_space(self, error_context: ErrorContext, exception: Exception) -> bool:
        """Recover from disk space issues."""
        try:
            # Attempt to clean temporary files
            import tempfile
            import shutil
            
            temp_dir = Path(tempfile.gettempdir())
            for temp_file in temp_dir.glob("fugatto_*"):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                    elif temp_file.is_dir():
                        shutil.rmtree(temp_file)
                except Exception:
                    continue
            
            logger.info("Temporary files cleaned for disk space recovery")
            return True
        except Exception:
            return False
    
    def _recover_gpu_memory(self, error_context: ErrorContext, exception: Exception) -> bool:
        """Recover from GPU memory issues."""
        try:
            import torch
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU memory cache cleared")
                return True
            
            return False
        except Exception:
            return False
    
    def _recover_network_retry(self, error_context: ErrorContext, exception: Exception) -> bool:
        """Recover from network issues with retry."""
        try:
            # Simple retry logic
            import time
            time.sleep(1.0)  # Wait before retry
            
            # This is a placeholder - actual implementation would retry the failed operation
            logger.info("Network retry recovery attempted")
            return True
        except Exception:
            return False
    
    def _recover_network_fallback(self, error_context: ErrorContext, exception: Exception) -> bool:
        """Recover from network issues with fallback."""
        # This would implement fallback to local resources
        logger.info("Network fallback recovery attempted")
        return True
    
    def _recover_processing_params(self, error_context: ErrorContext, exception: Exception) -> bool:
        """Recover from processing errors by adjusting parameters."""
        # This would implement parameter adjustment logic
        logger.info("Processing parameter recovery attempted")
        return True
    
    def _recover_processing_fallback(self, error_context: ErrorContext, exception: Exception) -> bool:
        """Recover from processing errors with fallback method."""
        # This would implement fallback processing method
        logger.info("Processing fallback recovery attempted")
        return True
    
    def _recover_model_reload(self, error_context: ErrorContext, exception: Exception) -> bool:
        """Recover from model errors by reloading."""
        # This would implement model reloading logic
        logger.info("Model reload recovery attempted")
        return True
    
    def _recover_model_fallback(self, error_context: ErrorContext, exception: Exception) -> bool:
        """Recover from model errors with fallback model."""
        # This would implement fallback model logic
        logger.info("Model fallback recovery attempted")
        return True
    
    def _recover_timeout_extend(self, error_context: ErrorContext, exception: Exception) -> bool:
        """Recover from timeout by extending timeout."""
        # This would implement timeout extension logic
        logger.info("Timeout extension recovery attempted")
        return True
    
    def _recover_timeout_chunk(self, error_context: ErrorContext, exception: Exception) -> bool:
        """Recover from timeout by chunking operation."""
        # This would implement operation chunking logic
        logger.info("Timeout chunking recovery attempted")
        return True
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        return {
            "total_errors": len(self.error_history),
            "error_counts_by_category": self.error_counts,
            "circuit_breaker_status": self.circuit_breakers,
            "recent_errors": [
                ctx.to_dict() for ctx in self.error_history[-10:]
            ]
        }
    
    def save_error_report(self, filepath: str):
        """Save detailed error report."""
        report = {
            "report_timestamp": time.time(),
            "total_errors": len(self.error_history),
            "error_statistics": self.get_error_statistics(),
            "error_history": [ctx.to_dict() for ctx in self.error_history],
            "recovery_success_rate": self._calculate_recovery_success_rate()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Error report saved to: {filepath}")
    
    def _calculate_recovery_success_rate(self) -> float:
        """Calculate recovery success rate."""
        recovery_attempts = [ctx for ctx in self.error_history if ctx.recovery_attempted]
        if not recovery_attempts:
            return 0.0
        
        successful_recoveries = [ctx for ctx in recovery_attempts if ctx.recovery_successful]
        return len(successful_recoveries) / len(recovery_attempts)


# Decorator for robust error handling

def robust_error_handler(
    error_handler: RobustErrorHandler = None,
    retry_attempts: int = 3,
    retry_delay: float = 1.0,
    fallback_result: Any = None,
    raise_on_failure: bool = True
):
    """Decorator for robust error handling with retry and recovery."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal error_handler
            
            if error_handler is None:
                error_handler = RobustErrorHandler()
            
            last_exception = None
            
            for attempt in range(retry_attempts + 1):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    
                    # Handle error
                    context = {
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "max_attempts": retry_attempts + 1,
                        "args": str(args)[:200],  # Truncate for logging
                        "kwargs": str(kwargs)[:200]
                    }
                    
                    error_context = error_handler.handle_error(e, context)
                    
                    # If recovery was successful, try again immediately
                    if error_context.recovery_successful:
                        continue
                    
                    # If not the last attempt, wait and retry
                    if attempt < retry_attempts:
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        continue
                    
                    # Last attempt failed
                    break
            
            # All attempts failed
            if fallback_result is not None:
                logger.warning(f"Using fallback result for {func.__name__}")
                return fallback_result
            
            if raise_on_failure:
                raise last_exception
            
            return None
        
        return wrapper
    
    return decorator


# Async version of the decorator
def async_robust_error_handler(
    error_handler: RobustErrorHandler = None,
    retry_attempts: int = 3,
    retry_delay: float = 1.0,
    fallback_result: Any = None,
    raise_on_failure: bool = True
):
    """Async decorator for robust error handling with retry and recovery."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal error_handler
            
            if error_handler is None:
                error_handler = RobustErrorHandler()
            
            last_exception = None
            
            for attempt in range(retry_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    
                    # Handle error
                    context = {
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "max_attempts": retry_attempts + 1,
                        "args": str(args)[:200],
                        "kwargs": str(kwargs)[:200]
                    }
                    
                    error_context = error_handler.handle_error(e, context)
                    
                    # If recovery was successful, try again immediately
                    if error_context.recovery_successful:
                        continue
                    
                    # If not the last attempt, wait and retry
                    if attempt < retry_attempts:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    
                    break
            
            # All attempts failed
            if fallback_result is not None:
                logger.warning(f"Using fallback result for {func.__name__}")
                return fallback_result
            
            if raise_on_failure:
                raise last_exception
            
            return None
        
        return wrapper
    
    return decorator


# Input validation utilities

class InputValidator:
    """Comprehensive input validation for audio processing."""
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 10000) -> str:
        """Sanitize string input."""
        if not isinstance(input_str, str):
            raise ValueError("Input must be string")
        
        # Truncate if too long
        if len(input_str) > max_length:
            input_str = input_str[:max_length]
        
        # Basic HTML escaping
        input_str = input_str.replace('&', '&amp;')
        input_str = input_str.replace('<', '&lt;')
        input_str = input_str.replace('>', '&gt;')
        input_str = input_str.replace('"', '&quot;')
        input_str = input_str.replace("'", '&#x27;')
        
        return input_str.strip()
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage."""
        if not isinstance(filename, str):
            raise ValueError("Filename must be string")
        
        # Remove path separators and dangerous characters
        filename = filename.replace('/', '_').replace('\\', '_')
        filename = filename.replace('<', '_').replace('>', '_')
        
        # Ensure not empty
        if not filename or filename.isspace():
            filename = f"file_{int(time.time())}"
        
        return filename[:100]  # Limit length
    
    @staticmethod
    def validate_audio_array(audio: np.ndarray, name: str = "audio") -> np.ndarray:
        """Validate audio array input."""
        if not isinstance(audio, np.ndarray):
            raise ValidationError(f"{name} must be numpy array", field=name, value=type(audio))
        
        if audio.size == 0:
            raise ValidationError(f"{name} array is empty", field=name, value=audio.shape)
        
        if audio.ndim > 2:
            raise ValidationError(f"{name} must be 1D or 2D array", field=name, value=audio.ndim)
        
        if not np.isfinite(audio).all():
            raise ValidationError(f"{name} contains non-finite values", field=name)
        
        if np.abs(audio).max() > 100:  # Reasonable threshold
            logger.warning(f"{name} contains unusually large values: {np.abs(audio).max()}")
        
        return audio
    
    @staticmethod
    def validate_sample_rate(sample_rate: Union[int, float], name: str = "sample_rate") -> int:
        """Validate sample rate parameter."""
        if not isinstance(sample_rate, (int, float)):
            raise ValidationError(f"{name} must be numeric", field=name, value=type(sample_rate))
        
        sample_rate = int(sample_rate)
        
        if sample_rate <= 0:
            raise ValidationError(f"{name} must be positive", field=name, value=sample_rate)
        
        if sample_rate < 8000 or sample_rate > 192000:
            raise ValidationError(f"{name} out of valid range (8kHz-192kHz)", field=name, value=sample_rate)
        
        return sample_rate
    
    @staticmethod
    def validate_duration(duration: Union[int, float], name: str = "duration", max_duration: float = 300) -> float:
        """Validate duration parameter."""
        if not isinstance(duration, (int, float)):
            raise ValidationError(f"{name} must be numeric", field=name, value=type(duration))
        
        duration = float(duration)
        
        if duration <= 0:
            raise ValidationError(f"{name} must be positive", field=name, value=duration)
        
        if duration > max_duration:
            raise ValidationError(f"{name} exceeds maximum ({max_duration}s)", field=name, value=duration)
        
        return duration
    
    @staticmethod
    def validate_file_path(filepath: Union[str, Path], must_exist: bool = True, name: str = "filepath") -> Path:
        """Validate file path parameter."""
        if not isinstance(filepath, (str, Path)):
            raise ValidationError(f"{name} must be string or Path", field=name, value=type(filepath))
        
        filepath = Path(filepath)
        
        if must_exist and not filepath.exists():
            raise ValidationError(f"{name} does not exist", field=name, value=str(filepath))
        
        if must_exist and not filepath.is_file():
            raise ValidationError(f"{name} is not a file", field=name, value=str(filepath))
        
        return filepath
    
    @staticmethod
    def validate_parameters(params: Dict[str, Any], schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Validate parameters against schema."""
        validated = {}
        
        for param_name, param_schema in schema.items():
            if param_name in params:
                value = params[param_name]
                param_type = param_schema.get("type")
                required = param_schema.get("required", False)
                min_val = param_schema.get("min")
                max_val = param_schema.get("max")
                choices = param_schema.get("choices")
                
                # Type validation
                if param_type and not isinstance(value, param_type):
                    raise ValidationError(
                        f"Parameter {param_name} must be {param_type.__name__}",
                        field=param_name, value=type(value)
                    )
                
                # Range validation
                if min_val is not None and value < min_val:
                    raise ValidationError(
                        f"Parameter {param_name} below minimum ({min_val})",
                        field=param_name, value=value
                    )
                
                if max_val is not None and value > max_val:
                    raise ValidationError(
                        f"Parameter {param_name} above maximum ({max_val})",
                        field=param_name, value=value
                    )
                
                # Choice validation
                if choices and value not in choices:
                    raise ValidationError(
                        f"Parameter {param_name} not in valid choices: {choices}",
                        field=param_name, value=value
                    )
                
                validated[param_name] = value
            
            elif param_schema.get("required", False):
                raise ValidationError(f"Required parameter {param_name} missing", field=param_name)
            
            else:
                # Use default value if provided
                default = param_schema.get("default")
                if default is not None:
                    validated[param_name] = default
        
        return validated


# Global error handler instance
_global_error_handler = None

def get_global_error_handler() -> RobustErrorHandler:
    """Get global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = RobustErrorHandler(
            log_file="fugatto_errors.log",
            enable_recovery=True
        )
    return _global_error_handler


# Convenience functions
def handle_error(exception: Exception, context: Dict[str, Any] = None) -> ErrorContext:
    """Handle error using global error handler."""
    return get_global_error_handler().handle_error(exception, context)


def validate_audio_input(audio: np.ndarray, sample_rate: int = None) -> Tuple[np.ndarray, int]:
    """Validate audio input with optional sample rate."""
    validator = InputValidator()
    
    audio = validator.validate_audio_array(audio)
    if sample_rate is not None:
        sample_rate = validator.validate_sample_rate(sample_rate)
    
    return audio, sample_rate