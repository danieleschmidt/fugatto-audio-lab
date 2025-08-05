"""Security Framework for Fugatto Audio Lab.

Comprehensive security implementation including input sanitization,
access control, audit logging, and security monitoring.
"""

import hashlib
import hmac
import secrets
import time
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import base64
import re
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"           # No restrictions
    INTERNAL = "internal"       # Internal use only
    RESTRICTED = "restricted"   # Restricted access
    CONFIDENTIAL = "confidential"  # Confidential data
    SECRET = "secret"          # Highly sensitive


class AuditEventType(Enum):
    """Types of audit events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_EVENT = "system_event"
    SECURITY_VIOLATION = "security_violation"
    CONFIGURATION_CHANGE = "configuration_change"


@dataclass
class SecurityContext:
    """Security context for operations."""
    
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    permissions: List[str] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    mfa_verified: bool = False
    
    def is_expired(self, timeout_seconds: int = 3600) -> bool:
        """Check if security context has expired."""
        return time.time() - self.last_activity > timeout_seconds
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission."""
        return permission in self.permissions
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()


@dataclass
class AuditEvent:
    """Security audit event."""
    
    event_id: str
    event_type: AuditEventType
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: str = "success"  # success, failure, denied
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "details": self.details,
            "risk_score": self.risk_score
        }


class InputSanitizer:
    """Comprehensive input sanitization and validation."""
    
    # Dangerous patterns to detect
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript URLs
        r'on\w+\s*=',                 # Event handlers
        r'expression\s*\(',            # CSS expressions
        r'import\s+',                  # Python imports
        r'exec\s*\(',                  # Exec calls
        r'eval\s*\(',                  # Eval calls
        r'__.*__',                     # Python dunder methods
        r'\.\./.*',                    # Path traversal
        r'[;\|&`$]',                   # Shell metacharacters
    ]
    
    # Safe filename pattern
    SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9._\-\s]{1,255}$')
    
    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                                for pattern in self.DANGEROUS_PATTERNS]
        logger.info("InputSanitizer initialized")
    
    def sanitize_string(self, input_str: str, max_length: int = 10000) -> str:
        """Sanitize string input."""
        if not isinstance(input_str, str):
            raise ValueError("Input must be string")
        
        # Truncate if too long
        if len(input_str) > max_length:
            input_str = input_str[:max_length]
            logger.warning(f"Input truncated to {max_length} characters")
        
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(input_str):
                logger.warning(f"Dangerous pattern detected: {pattern.pattern}")
                raise ValueError("Input contains potentially dangerous content")
        
        # Basic HTML escaping
        input_str = input_str.replace('&', '&amp;')
        input_str = input_str.replace('<', '&lt;')
        input_str = input_str.replace('>', '&gt;')
        input_str = input_str.replace('"', '&quot;')
        input_str = input_str.replace("'", '&#x27;')
        
        return input_str.strip()
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        if not isinstance(filename, str):
            raise ValueError("Filename must be string")
        
        # Remove path separators
        filename = filename.replace('/', '_').replace('\\', '_')
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"|?*]', '_', filename)
        
        # Check against safe pattern
        if not self.SAFE_FILENAME_PATTERN.match(filename):
            # Generate safe filename
            safe_chars = re.sub(r'[^a-zA-Z0-9._\-]', '_', filename)
            filename = safe_chars[:100]  # Limit length
        
        # Ensure not empty
        if not filename or filename.isspace():
            filename = f"file_{int(time.time())}"
        
        return filename
    
    def sanitize_path(self, path: str, base_path: str = None) -> Path:
        """Sanitize file path to prevent directory traversal."""
        if not isinstance(path, str):
            raise ValueError("Path must be string")
        
        # Convert to Path object
        path_obj = Path(path).resolve()
        
        # If base path provided, ensure path is within it
        if base_path:
            base_path_obj = Path(base_path).resolve()
            try:
                path_obj.relative_to(base_path_obj)
            except ValueError:
                raise ValueError("Path traversal attempt detected")
        
        return path_obj
    
    def validate_audio_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate audio processing parameters."""
        safe_params = {}
        
        # Sample rate validation
        if 'sample_rate' in params:
            sr = params['sample_rate']
            if not isinstance(sr, (int, float)) or sr <= 0 or sr > 192000:
                raise ValueError("Invalid sample rate")
            safe_params['sample_rate'] = int(sr)
        
        # Duration validation
        if 'duration' in params:
            duration = params['duration']
            if not isinstance(duration, (int, float)) or duration <= 0 or duration > 3600:
                raise ValueError("Invalid duration")
            safe_params['duration'] = float(duration)
        
        # Temperature validation
        if 'temperature' in params:
            temp = params['temperature']
            if not isinstance(temp, (int, float)) or temp < 0.1 or temp > 2.0:
                raise ValueError("Invalid temperature")
            safe_params['temperature'] = float(temp)
        
        # Prompt validation
        if 'prompt' in params:
            prompt = self.sanitize_string(params['prompt'], max_length=1000)
            safe_params['prompt'] = prompt
        
        return safe_params


class AccessController:
    """Role-based access control system."""
    
    def __init__(self):
        self.roles = {}  # role_name -> permissions
        self.user_roles = {}  # user_id -> roles
        self.resource_permissions = {}  # resource -> required_permissions
        
        # Setup default roles
        self._setup_default_roles()
        
        logger.info("AccessController initialized")
    
    def _setup_default_roles(self):
        """Setup default security roles."""
        self.roles = {
            "admin": [
                "system.admin",
                "audio.generate", "audio.transform", "audio.analyze",
                "user.manage", "config.modify", "logs.view"
            ],
            "user": [
                "audio.generate", "audio.transform", "audio.analyze",
                "audio.upload", "audio.download"
            ],
            "viewer": [
                "audio.view", "audio.download"
            ],
            "service": [
                "audio.process", "audio.cache", "system.monitor"
            ]
        }
        
        self.resource_permissions = {
            "/admin": ["system.admin"],
            "/api/generate": ["audio.generate"],
            "/api/transform": ["audio.transform"],
            "/api/analyze": ["audio.analyze"],
            "/api/upload": ["audio.upload"],
            "/api/download": ["audio.download"],
            "/config": ["config.modify"],
            "/logs": ["logs.view"]
        }
    
    def assign_role(self, user_id: str, role: str):
        """Assign role to user."""
        if role not in self.roles:
            raise ValueError(f"Unknown role: {role}")
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = []
        
        if role not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role)
            logger.info(f"Assigned role '{role}' to user '{user_id}'")
    
    def check_permission(self, security_context: SecurityContext, 
                        resource: str, action: str = None) -> bool:
        """Check if user has permission for resource/action."""
        user_id = security_context.user_id
        
        # Get user permissions from roles
        user_permissions = set()
        if user_id in self.user_roles:
            for role in self.user_roles[user_id]:
                if role in self.roles:
                    user_permissions.update(self.roles[role])
        
        # Add context permissions
        user_permissions.update(security_context.permissions)
        
        # Check resource-specific permissions
        if resource in self.resource_permissions:
            required_perms = self.resource_permissions[resource]
            if not any(perm in user_permissions for perm in required_perms):
                return False
        
        # Check action-specific permissions
        if action:
            action_perm = f"{resource}.{action}".lstrip('.')
            if action_perm not in user_permissions:
                return False
        
        return True
    
    def get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for user."""
        permissions = set()
        
        if user_id in self.user_roles:
            for role in self.user_roles[user_id]:
                if role in self.roles:
                    permissions.update(self.roles[role])
        
        return list(permissions)


class AuditLogger:
    """Security audit logging system."""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = log_file
        self.audit_events = deque(maxlen=10000)  # Keep recent events in memory
        self.event_counts = defaultdict(int)
        self.risk_scores = deque(maxlen=1000)
        
        # Setup file logging
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        self.file_handler.setFormatter(formatter)
        
        # Create audit logger
        self.audit_logger = logging.getLogger('security_audit')
        self.audit_logger.addHandler(self.file_handler)
        self.audit_logger.setLevel(logging.INFO)
        
        logger.info("AuditLogger initialized")
    
    def log_event(self, event: AuditEvent):
        """Log security audit event."""
        # Add to memory storage
        self.audit_events.append(event)
        self.event_counts[event.event_type.value] += 1
        self.risk_scores.append(event.risk_score)
        
        # Log to file
        log_message = json.dumps(event.to_dict())
        
        if event.result == "failure" or event.risk_score > 7.0:
            self.audit_logger.error(log_message)
        elif event.result == "denied" or event.risk_score > 5.0:
            self.audit_logger.warning(log_message)
        else:
            self.audit_logger.info(log_message)
    
    def log_authentication(self, user_id: str, result: str, 
                          ip_address: str = None, details: Dict[str, Any] = None):
        """Log authentication event."""
        event = AuditEvent(
            event_id=f"auth_{int(time.time() * 1000)}_{secrets.randbelow(10000)}",
            event_type=AuditEventType.AUTHENTICATION,
            user_id=user_id,
            ip_address=ip_address,
            action="login",
            result=result,
            details=details or {},
            risk_score=8.0 if result == "failure" else 2.0
        )
        self.log_event(event)
    
    def log_authorization(self, security_context: SecurityContext, 
                         resource: str, action: str, result: str):
        """Log authorization event."""
        event = AuditEvent(
            event_id=f"authz_{int(time.time() * 1000)}_{secrets.randbelow(10000)}",
            event_type=AuditEventType.AUTHORIZATION,
            user_id=security_context.user_id,
            session_id=security_context.session_id,
            ip_address=security_context.ip_address,
            resource=resource,
            action=action,
            result=result,
            risk_score=6.0 if result == "denied" else 1.0
        )
        self.log_event(event)
    
    def log_data_access(self, security_context: SecurityContext, 
                       resource: str, details: Dict[str, Any] = None):
        """Log data access event."""
        event = AuditEvent(
            event_id=f"access_{int(time.time() * 1000)}_{secrets.randbelow(10000)}",
            event_type=AuditEventType.DATA_ACCESS,
            user_id=security_context.user_id,
            session_id=security_context.session_id,
            ip_address=security_context.ip_address,
            resource=resource,
            action="read",
            result="success",
            details=details or {},
            risk_score=1.0
        )
        self.log_event(event)
    
    def log_security_violation(self, user_id: str, violation_type: str, 
                              details: Dict[str, Any] = None, ip_address: str = None):
        """Log security violation."""
        event = AuditEvent(
            event_id=f"violation_{int(time.time() * 1000)}_{secrets.randbelow(10000)}",
            event_type=AuditEventType.SECURITY_VIOLATION,
            user_id=user_id,
            ip_address=ip_address,
            action=violation_type,
            result="detected",
            details=details or {},
            risk_score=9.0
        )
        self.log_event(event)
    
    def get_audit_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get audit summary for time window."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        recent_events = [
            event for event in self.audit_events
            if event.timestamp >= cutoff_time
        ]
        
        # Count by type
        type_counts = defaultdict(int)
        result_counts = defaultdict(int)
        high_risk_events = []
        
        for event in recent_events:
            type_counts[event.event_type.value] += 1
            result_counts[event.result] += 1
            
            if event.risk_score >= 7.0:
                high_risk_events.append(event.to_dict())
        
        return {
            "time_window_hours": time_window_hours,
            "total_events": len(recent_events),
            "events_by_type": dict(type_counts),
            "events_by_result": dict(result_counts),
            "high_risk_events": high_risk_events,
            "average_risk_score": sum(self.risk_scores) / len(self.risk_scores) if self.risk_scores else 0
        }


class RateLimiter:
    """Rate limiting for API protection."""
    
    def __init__(self):
        self.requests = defaultdict(deque)  # ip -> timestamps
        self.limits = {
            "default": {"requests": 100, "window": 3600},  # 100 requests per hour
            "auth": {"requests": 5, "window": 300},        # 5 auth attempts per 5 minutes
            "upload": {"requests": 10, "window": 3600},    # 10 uploads per hour
            "generate": {"requests": 20, "window": 3600}   # 20 generations per hour
        }
        
        # Cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("RateLimiter initialized")
    
    def check_rate_limit(self, identifier: str, limit_type: str = "default") -> bool:
        """Check if request is within rate limit."""
        current_time = time.time()
        limit_config = self.limits.get(limit_type, self.limits["default"])
        
        window_start = current_time - limit_config["window"]
        
        # Clean old requests
        request_queue = self.requests[identifier]
        while request_queue and request_queue[0] < window_start:
            request_queue.popleft()
        
        # Check limit
        if len(request_queue) >= limit_config["requests"]:
            return False
        
        # Record request
        request_queue.append(current_time)
        return True
    
    def _cleanup_loop(self):
        """Background cleanup of old rate limit data."""
        while True:
            try:
                time.sleep(300)  # Clean every 5 minutes
                current_time = time.time()
                
                # Remove old entries
                for identifier in list(self.requests.keys()):
                    request_queue = self.requests[identifier]
                    # Remove requests older than 2 hours
                    cutoff_time = current_time - 7200
                    
                    while request_queue and request_queue[0] < cutoff_time:
                        request_queue.popleft()
                    
                    # Remove empty queues
                    if not request_queue:
                        del self.requests[identifier]
                        
            except Exception as e:
                logger.error(f"Rate limiter cleanup error: {e}")


class SecurityManager:
    """Main security management system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.sanitizer = InputSanitizer()
        self.access_controller = AccessController()
        self.audit_logger = AuditLogger(
            log_file=self.config.get("audit_log_file", "security_audit.log")
        )
        self.rate_limiter = RateLimiter()
        
        # Security contexts
        self.active_sessions = {}  # session_id -> SecurityContext
        self.failed_attempts = defaultdict(list)  # ip_address -> timestamps
        
        # Security settings
        self.session_timeout = self.config.get("session_timeout", 3600)  # 1 hour
        self.max_failed_attempts = self.config.get("max_failed_attempts", 5)
        self.lockout_duration = self.config.get("lockout_duration", 900)  # 15 minutes
        
        # Background cleanup
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("SecurityManager initialized")
    
    def create_session(self, user_id: str, ip_address: str, 
                      user_agent: str, permissions: List[str] = None) -> SecurityContext:
        """Create new security session."""
        session_id = secrets.token_urlsafe(32)
        
        context = SecurityContext(
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            permissions=permissions or []
        )
        
        self.active_sessions[session_id] = context
        
        # Log session creation
        self.audit_logger.log_authentication(
            user_id=user_id,
            result="success",
            ip_address=ip_address,
            details={"session_id": session_id}
        )
        
        logger.info(f"Created session for user {user_id}")
        return context
    
    def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """Validate and return security context."""
        if session_id not in self.active_sessions:
            return None
        
        context = self.active_sessions[session_id]
        
        # Check expiration
        if context.is_expired(self.session_timeout):
            self.invalidate_session(session_id)
            return None
        
        # Update activity
        context.update_activity()
        return context
    
    def invalidate_session(self, session_id: str):
        """Invalidate security session."""
        if session_id in self.active_sessions:
            context = self.active_sessions[session_id]
            del self.active_sessions[session_id]
            
            # Log session invalidation
            self.audit_logger.log_event(AuditEvent(
                event_id=f"session_{int(time.time() * 1000)}",
                event_type=AuditEventType.SYSTEM_EVENT,
                user_id=context.user_id,
                session_id=session_id,
                action="session_invalidated",
                result="success"
            ))
    
    def check_access(self, security_context: SecurityContext, 
                    resource: str, action: str = None) -> bool:
        """Check access permission and log result."""
        has_access = self.access_controller.check_permission(
            security_context, resource, action
        )
        
        result = "success" if has_access else "denied"
        
        # Log authorization attempt
        self.audit_logger.log_authorization(
            security_context, resource, action or "access", result
        )
        
        return has_access
    
    def is_ip_locked(self, ip_address: str) -> bool:
        """Check if IP address is locked due to failed attempts."""
        if ip_address not in self.failed_attempts:
            return False
        
        current_time = time.time()
        cutoff_time = current_time - self.lockout_duration
        
        # Clean old failed attempts
        self.failed_attempts[ip_address] = [
            timestamp for timestamp in self.failed_attempts[ip_address]
            if timestamp > cutoff_time
        ]
        
        # Check if locked
        return len(self.failed_attempts[ip_address]) >= self.max_failed_attempts
    
    def record_failed_attempt(self, ip_address: str, user_id: str = None):
        """Record failed authentication attempt."""
        current_time = time.time()
        self.failed_attempts[ip_address].append(current_time)
        
        # Log security violation if approaching lockout
        if len(self.failed_attempts[ip_address]) >= self.max_failed_attempts - 1:
            self.audit_logger.log_security_violation(
                user_id=user_id or "unknown",
                violation_type="excessive_failed_attempts",
                ip_address=ip_address,
                details={"attempt_count": len(self.failed_attempts[ip_address])}
            )
    
    def sanitize_input(self, input_data: Any, input_type: str = "string") -> Any:
        """Sanitize input data based on type."""
        if input_type == "string":
            return self.sanitizer.sanitize_string(input_data)
        elif input_type == "filename":
            return self.sanitizer.sanitize_filename(input_data)
        elif input_type == "path":
            return self.sanitizer.sanitize_path(input_data)
        elif input_type == "audio_params":
            return self.sanitizer.validate_audio_parameters(input_data)
        else:
            return input_data
    
    def _cleanup_loop(self):
        """Background cleanup of expired sessions and old data."""
        while True:
            try:
                time.sleep(300)  # Clean every 5 minutes
                current_time = time.time()
                
                # Clean expired sessions
                expired_sessions = [
                    session_id for session_id, context in self.active_sessions.items()
                    if context.is_expired(self.session_timeout)
                ]
                
                for session_id in expired_sessions:
                    self.invalidate_session(session_id)
                
                # Clean old failed attempts
                cutoff_time = current_time - self.lockout_duration * 2
                for ip_address in list(self.failed_attempts.keys()):
                    self.failed_attempts[ip_address] = [
                        timestamp for timestamp in self.failed_attempts[ip_address]
                        if timestamp > cutoff_time
                    ]
                    
                    if not self.failed_attempts[ip_address]:
                        del self.failed_attempts[ip_address]
                        
                logger.debug(f"Security cleanup: {len(expired_sessions)} sessions expired")
                
            except Exception as e:
                logger.error(f"Security cleanup error: {e}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "active_sessions": len(self.active_sessions),
            "locked_ips": sum(1 for ip in self.failed_attempts.keys() 
                            if self.is_ip_locked(ip)),
            "recent_violations": len([
                event for event in self.audit_logger.audit_events
                if (event.event_type == AuditEventType.SECURITY_VIOLATION and
                    time.time() - event.timestamp < 3600)
            ]),
            "audit_summary": self.audit_logger.get_audit_summary(24),
            "rate_limit_active": len(self.rate_limiter.requests)
        }
    
    def save_security_report(self, filepath: str):
        """Save comprehensive security report."""
        report = {
            "timestamp": time.time(),
            "security_status": self.get_security_status(),
            "audit_summary": self.audit_logger.get_audit_summary(168),  # 1 week
            "configuration": {
                "session_timeout": self.session_timeout,
                "max_failed_attempts": self.max_failed_attempts,
                "lockout_duration": self.lockout_duration
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Security report saved to: {filepath}")


# Security decorators

def require_permission(permission: str):
    """Decorator to require specific permission."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract security context from kwargs
            security_context = kwargs.get('security_context')
            if not security_context:
                raise PermissionError("No security context provided")
            
            # Check permission
            security_manager = get_global_security_manager()
            if not security_manager.check_access(security_context, permission):
                raise PermissionError(f"Permission denied: {permission}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_rate_limit(limit_type: str = "default"):
    """Decorator to enforce rate limiting."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract identifier (IP address) from kwargs
            identifier = kwargs.get('ip_address', 'unknown')
            
            security_manager = get_global_security_manager()
            if not security_manager.rate_limiter.check_rate_limit(identifier, limit_type):
                raise PermissionError("Rate limit exceeded")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def sanitize_inputs(**input_types):
    """Decorator to sanitize function inputs."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            security_manager = get_global_security_manager()
            
            # Sanitize specified inputs
            for param_name, input_type in input_types.items():
                if param_name in kwargs:
                    kwargs[param_name] = security_manager.sanitize_input(
                        kwargs[param_name], input_type
                    )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Global security manager
_global_security_manager = None

def get_global_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _global_security_manager
    if _global_security_manager is None:
        _global_security_manager = SecurityManager()
    return _global_security_manager


# Utility functions

def generate_secure_token() -> str:
    """Generate cryptographically secure token."""
    return secrets.token_urlsafe(32)


def hash_password(password: str, salt: str = None) -> Tuple[str, str]:
    """Hash password with salt."""
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Use PBKDF2 with SHA256
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # iterations
    )
    
    return base64.b64encode(password_hash).decode('utf-8'), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """Verify password against hash."""
    computed_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, password_hash)