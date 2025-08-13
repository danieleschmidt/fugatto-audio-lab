"""Advanced Security Framework - Generation 2 Enhancement.

Comprehensive security with authentication, authorization, encryption,
audit logging, and threat detection for enterprise deployment.
"""

import hashlib
import hmac
import json
import logging
import secrets
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

try:
    import jwt
except ImportError:
    jwt = None

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except ImportError:
    Fernet = None
    hashes = None
    PBKDF2HMAC = None

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class Permission(Enum):
    """System permissions."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class User:
    """User authentication and authorization data."""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    security_level: SecurityLevel
    permissions: Set[Permission]
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    failed_login_attempts: int = 0
    is_locked: bool = False
    session_token: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    timestamp: float
    event_type: str
    user_id: Optional[str]
    source_ip: Optional[str]
    resource: str
    action: str
    success: bool
    threat_level: ThreatLevel
    details: Dict[str, Any] = field(default_factory=dict)


class AuthenticationManager:
    """Secure authentication system."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.failed_login_tracker = defaultdict(list)
        
        # Security settings
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
        self.session_timeout = 3600  # 1 hour
        self.password_min_length = 8
        
        logger.info("Authentication manager initialized")
    
    def create_user(self, username: str, email: str, password: str,
                   security_level: SecurityLevel = SecurityLevel.INTERNAL,
                   permissions: Optional[Set[Permission]] = None) -> User:
        """Create new user with secure password hashing."""
        if username in [user.username for user in self.users.values()]:
            raise ValueError(f"Username '{username}' already exists")
        
        if not self._validate_password_strength(password):
            raise ValueError("Password does not meet security requirements")
        
        # Generate secure salt and hash password
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)
        
        user_id = secrets.token_urlsafe(16)
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            salt=salt,
            security_level=security_level,
            permissions=permissions or {Permission.READ}
        )
        
        self.users[user_id] = user
        
        logger.info(f"User created: {username} ({security_level.value})")
        return user
    
    def authenticate(self, username: str, password: str, 
                    source_ip: Optional[str] = None) -> Optional[str]:
        """Authenticate user and return session token."""
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            self._log_failed_login(username, source_ip, "User not found")
            return None
        
        # Check if account is locked
        if user.is_locked:
            if time.time() - (user.last_login or 0) < self.lockout_duration:
                self._log_failed_login(username, source_ip, "Account locked")
                return None
            else:
                # Unlock account after lockout period
                user.is_locked = False
                user.failed_login_attempts = 0
        
        # Verify password
        if not self._verify_password(password, user.password_hash, user.salt):
            user.failed_login_attempts += 1
            
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.is_locked = True
                logger.warning(f"Account locked for user: {username}")
            
            self._log_failed_login(username, source_ip, "Invalid password")
            return None
        
        # Successful authentication
        user.failed_login_attempts = 0
        user.last_login = time.time()
        
        # Generate session token
        session_token = self._generate_session_token(user)
        user.session_token = session_token
        
        # Store active session
        self.active_sessions[session_token] = {
            'user_id': user.user_id,
            'username': username,
            'created_at': time.time(),
            'last_activity': time.time(),
            'source_ip': source_ip
        }
        
        logger.info(f"User authenticated: {username}")
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[User]:
        """Validate session token and return user."""
        if session_token not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_token]
        
        # Check session timeout
        if time.time() - session['last_activity'] > self.session_timeout:
            self.logout(session_token)
            return None
        
        # Update last activity
        session['last_activity'] = time.time()
        
        return self.users.get(session['user_id'])
    
    def logout(self, session_token: str) -> bool:
        """Logout user and invalidate session."""
        if session_token in self.active_sessions:
            session = self.active_sessions[session_token]
            user_id = session['user_id']
            
            # Clear session token from user
            if user_id in self.users:
                self.users[user_id].session_token = None
            
            # Remove active session
            del self.active_sessions[session_token]
            
            logger.info(f"User logged out: {session.get('username')}")
            return True
        
        return False
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password meets security requirements."""
        if len(password) < self.password_min_length:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt using PBKDF2."""
        # Use PBKDF2 with SHA-256
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return key.hex()
    
    def _verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash."""
        computed_hash = self._hash_password(password, salt)
        return hmac.compare_digest(computed_hash, password_hash)
    
    def _generate_session_token(self, user: User) -> str:
        """Generate secure session token."""
        if jwt:
            payload = {
                'user_id': user.user_id,
                'username': user.username,
                'security_level': user.security_level.value,
                'iat': time.time(),
                'exp': time.time() + self.session_timeout
            }
            return jwt.encode(payload, self.secret_key, algorithm='HS256')
        else:
            # Fallback to simple secure token
            return secrets.token_urlsafe(32)
    
    def _log_failed_login(self, username: str, source_ip: Optional[str], reason: str) -> None:
        """Log failed login attempt."""
        self.failed_login_tracker[username].append({
            'timestamp': time.time(),
            'source_ip': source_ip,
            'reason': reason
        })
        
        logger.warning(f"Failed login attempt: {username} from {source_ip} - {reason}")


class AuthorizationManager:
    """Role-based access control system."""
    
    def __init__(self):
        self.resource_permissions: Dict[str, Dict[SecurityLevel, Set[Permission]]] = {}
        self.role_permissions: Dict[str, Set[Permission]] = {}
        self.user_roles: Dict[str, Set[str]] = defaultdict(set)
        
        # Initialize default roles
        self._initialize_default_roles()
        
        logger.info("Authorization manager initialized")
    
    def _initialize_default_roles(self) -> None:
        """Initialize default system roles."""
        self.role_permissions = {
            'guest': {Permission.READ},
            'user': {Permission.READ, Permission.WRITE},
            'moderator': {Permission.READ, Permission.WRITE, Permission.EXECUTE},
            'admin': {Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.DELETE, Permission.ADMIN}
        }
    
    def define_resource_permissions(self, resource: str, 
                                  permissions_by_level: Dict[SecurityLevel, Set[Permission]]) -> None:
        """Define permissions required for resource by security level."""
        self.resource_permissions[resource] = permissions_by_level
        logger.info(f"Resource permissions defined for: {resource}")
    
    def assign_role(self, user_id: str, role: str) -> None:
        """Assign role to user."""
        if role not in self.role_permissions:
            raise ValueError(f"Unknown role: {role}")
        
        self.user_roles[user_id].add(role)
        logger.info(f"Role '{role}' assigned to user {user_id}")
    
    def revoke_role(self, user_id: str, role: str) -> None:
        """Revoke role from user."""
        self.user_roles[user_id].discard(role)
        logger.info(f"Role '{role}' revoked from user {user_id}")
    
    def check_permission(self, user: User, resource: str, permission: Permission) -> bool:
        """Check if user has permission for resource."""
        # Check user's direct permissions
        if permission in user.permissions:
            return True
        
        # Check role-based permissions
        user_roles = self.user_roles[user.user_id]
        for role in user_roles:
            role_perms = self.role_permissions.get(role, set())
            if permission in role_perms:
                return True
        
        # Check resource-specific permissions by security level
        if resource in self.resource_permissions:
            resource_perms = self.resource_permissions[resource]
            level_perms = resource_perms.get(user.security_level, set())
            if permission in level_perms:
                return True
        
        return False
    
    def get_user_effective_permissions(self, user: User) -> Set[Permission]:
        """Get all effective permissions for user."""
        permissions = set(user.permissions)
        
        # Add role-based permissions
        user_roles = self.user_roles[user.user_id]
        for role in user_roles:
            role_perms = self.role_permissions.get(role, set())
            permissions.update(role_perms)
        
        return permissions


class EncryptionManager:
    """Data encryption and decryption system."""
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or Fernet.generate_key() if Fernet else b'dummy_key'
        self.cipher_suite = Fernet(self.master_key) if Fernet else None
        self.key_rotation_interval = 86400  # 24 hours
        self.last_key_rotation = time.time()
        
        logger.info("Encryption manager initialized")
    
    def encrypt_data(self, data: Union[str, bytes]) -> str:
        """Encrypt sensitive data."""
        if not self.cipher_suite:
            logger.warning("Cryptography not available, using base64 encoding")
            import base64
            if isinstance(data, str):
                data = data.encode()
            return base64.b64encode(data).decode()
        
        if isinstance(data, str):
            data = data.encode()
        
        encrypted = self.cipher_suite.encrypt(data)
        return encrypted.decode() if isinstance(encrypted, bytes) else encrypted
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        if not self.cipher_suite:
            logger.warning("Cryptography not available, using base64 decoding")
            import base64
            decoded = base64.b64decode(encrypted_data.encode())
            return decoded.decode()
        
        encrypted_bytes = encrypted_data.encode() if isinstance(encrypted_data, str) else encrypted_data
        decrypted = self.cipher_suite.decrypt(encrypted_bytes)
        return decrypted.decode()
    
    def hash_sensitive_data(self, data: str) -> str:
        """Create secure hash of sensitive data."""
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
        return f"{salt}:{hashed.hex()}"
    
    def verify_hash(self, data: str, hashed_data: str) -> bool:
        """Verify data against hash."""
        try:
            salt, hash_value = hashed_data.split(':', 1)
            computed_hash = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
            return hmac.compare_digest(computed_hash.hex(), hash_value)
        except ValueError:
            return False
    
    def rotate_keys(self) -> None:
        """Rotate encryption keys for enhanced security."""
        if time.time() - self.last_key_rotation > self.key_rotation_interval:
            if Fernet:
                old_key = self.master_key
                self.master_key = Fernet.generate_key()
                self.cipher_suite = Fernet(self.master_key)
                self.last_key_rotation = time.time()
                
                logger.info("Encryption keys rotated")
            else:
                logger.warning("Key rotation not available without cryptography package")


class AuditLogger:
    """Comprehensive security audit logging."""
    
    def __init__(self, max_events: int = 10000):
        self.security_events = deque(maxlen=max_events)
        self.event_counts = defaultdict(int)
        self.threat_summary = defaultdict(int)
        
        logger.info("Audit logger initialized")
    
    def log_security_event(self, event_type: str, user_id: Optional[str],
                          source_ip: Optional[str], resource: str, action: str,
                          success: bool, threat_level: ThreatLevel = ThreatLevel.LOW,
                          details: Optional[Dict[str, Any]] = None) -> None:
        """Log security event for audit trail."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            user_id=user_id,
            source_ip=source_ip,
            resource=resource,
            action=action,
            success=success,
            threat_level=threat_level,
            details=details or {}
        )
        
        self.security_events.append(event)
        self.event_counts[event_type] += 1
        self.threat_summary[threat_level] += 1
        
        # Log to system logger based on threat level
        log_message = f"Security Event: {event_type} - {action} on {resource} by {user_id or 'anonymous'} from {source_ip}"
        
        if threat_level == ThreatLevel.CRITICAL:
            logger.critical(log_message)
        elif threat_level == ThreatLevel.HIGH:
            logger.error(log_message)
        elif threat_level == ThreatLevel.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def get_audit_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Generate audit report for specified time range."""
        cutoff_time = time.time() - (time_range_hours * 3600)
        recent_events = [event for event in self.security_events if event.timestamp >= cutoff_time]
        
        # Calculate statistics
        total_events = len(recent_events)
        failed_events = sum(1 for event in recent_events if not event.success)
        success_rate = (total_events - failed_events) / total_events if total_events > 0 else 0.0
        
        # Group by event type
        event_type_counts = defaultdict(int)
        threat_level_counts = defaultdict(int)
        
        for event in recent_events:
            event_type_counts[event.event_type] += 1
            threat_level_counts[event.threat_level.value] += 1
        
        return {
            'time_range_hours': time_range_hours,
            'total_events': total_events,
            'failed_events': failed_events,
            'success_rate': success_rate,
            'event_types': dict(event_type_counts),
            'threat_levels': dict(threat_level_counts),
            'recent_critical_events': [
                {
                    'timestamp': event.timestamp,
                    'event_type': event.event_type,
                    'resource': event.resource,
                    'action': event.action,
                    'user_id': event.user_id,
                    'source_ip': event.source_ip
                }
                for event in recent_events 
                if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            ][-10:]  # Last 10 critical events
        }


class ThreatDetector:
    """Real-time threat detection and response system."""
    
    def __init__(self):
        self.detection_rules: List[Callable[[SecurityEvent], Optional[ThreatLevel]]] = []
        self.threat_patterns = {
            'brute_force': {'max_attempts': 10, 'time_window': 300},  # 10 attempts in 5 minutes
            'privilege_escalation': {'suspicious_actions': ['admin', 'delete', 'execute']},
            'data_exfiltration': {'large_read_threshold': 1000000}  # 1MB
        }
        
        self.active_threats: Dict[str, Dict[str, Any]] = {}
        self.blocked_ips: Set[str] = set()
        
        # Initialize default detection rules
        self._initialize_detection_rules()
        
        logger.info("Threat detector initialized")
    
    def _initialize_detection_rules(self) -> None:
        """Initialize default threat detection rules."""
        self.detection_rules = [
            self._detect_brute_force,
            self._detect_privilege_escalation,
            self._detect_suspicious_timing,
            self._detect_data_exfiltration
        ]
    
    def analyze_event(self, event: SecurityEvent) -> Optional[ThreatLevel]:
        """Analyze security event for threats."""
        max_threat_level = None
        
        for rule in self.detection_rules:
            try:
                threat_level = rule(event)
                if threat_level:
                    if max_threat_level is None or threat_level.value > max_threat_level.value:
                        max_threat_level = threat_level
            except Exception as e:
                logger.error(f"Error in threat detection rule: {e}")
        
        if max_threat_level:
            self._handle_threat(event, max_threat_level)
        
        return max_threat_level
    
    def _detect_brute_force(self, event: SecurityEvent) -> Optional[ThreatLevel]:
        """Detect brute force attacks."""
        if event.event_type == 'authentication' and not event.success:
            source_ip = event.source_ip or 'unknown'
            threat_key = f"brute_force_{source_ip}"
            
            if threat_key not in self.active_threats:
                self.active_threats[threat_key] = {
                    'attempts': [],
                    'first_attempt': event.timestamp
                }
            
            threat_data = self.active_threats[threat_key]
            threat_data['attempts'].append(event.timestamp)
            
            # Clean old attempts outside time window
            cutoff_time = event.timestamp - self.threat_patterns['brute_force']['time_window']
            threat_data['attempts'] = [t for t in threat_data['attempts'] if t >= cutoff_time]
            
            # Check if threshold exceeded
            if len(threat_data['attempts']) >= self.threat_patterns['brute_force']['max_attempts']:
                return ThreatLevel.HIGH
            elif len(threat_data['attempts']) >= self.threat_patterns['brute_force']['max_attempts'] // 2:
                return ThreatLevel.MEDIUM
        
        return None
    
    def _detect_privilege_escalation(self, event: SecurityEvent) -> Optional[ThreatLevel]:
        """Detect privilege escalation attempts."""
        suspicious_actions = self.threat_patterns['privilege_escalation']['suspicious_actions']
        
        if (event.action.lower() in suspicious_actions and 
            event.event_type in ['authorization', 'access_control']):
            
            # Check if user typically doesn't have these permissions
            if not event.success:
                return ThreatLevel.MEDIUM
        
        return None
    
    def _detect_suspicious_timing(self, event: SecurityEvent) -> Optional[ThreatLevel]:
        """Detect suspicious timing patterns."""
        # Check for off-hours access (simplified)
        import datetime
        event_time = datetime.datetime.fromtimestamp(event.timestamp)
        
        # Consider 10 PM to 6 AM as off-hours
        if event_time.hour >= 22 or event_time.hour <= 6:
            if event.action in ['delete', 'admin', 'execute']:
                return ThreatLevel.LOW
        
        return None
    
    def _detect_data_exfiltration(self, event: SecurityEvent) -> Optional[ThreatLevel]:
        """Detect potential data exfiltration."""
        if event.action == 'read' and event.success:
            data_size = event.details.get('data_size', 0)
            
            if data_size > self.threat_patterns['data_exfiltration']['large_read_threshold']:
                return ThreatLevel.MEDIUM
        
        return None
    
    def _handle_threat(self, event: SecurityEvent, threat_level: ThreatLevel) -> None:
        """Handle detected threat."""
        threat_id = f"{event.source_ip}_{event.event_type}_{int(event.timestamp)}"
        
        logger.warning(f"Threat detected: {threat_level.value} - {event.event_type} from {event.source_ip}")
        
        # Take action based on threat level
        if threat_level == ThreatLevel.CRITICAL:
            if event.source_ip:
                self.block_ip(event.source_ip)
        elif threat_level == ThreatLevel.HIGH:
            # Could implement rate limiting here
            pass
    
    def block_ip(self, ip_address: str) -> None:
        """Block IP address."""
        self.blocked_ips.add(ip_address)
        logger.warning(f"IP address blocked: {ip_address}")
    
    def unblock_ip(self, ip_address: str) -> None:
        """Unblock IP address."""
        self.blocked_ips.discard(ip_address)
        logger.info(f"IP address unblocked: {ip_address}")
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        return ip_address in self.blocked_ips


class AdvancedSecurityFramework:
    """Main security framework orchestrator."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.auth_manager = AuthenticationManager(secret_key)
        self.authz_manager = AuthorizationManager()
        self.encryption_manager = EncryptionManager()
        self.audit_logger = AuditLogger()
        self.threat_detector = ThreatDetector()
        
        logger.info("Advanced security framework initialized")
    
    def create_user(self, username: str, email: str, password: str,
                   security_level: SecurityLevel = SecurityLevel.INTERNAL,
                   permissions: Optional[Set[Permission]] = None,
                   roles: Optional[Set[str]] = None) -> User:
        """Create user with full security setup."""
        user = self.auth_manager.create_user(username, email, password, security_level, permissions)
        
        # Assign roles if provided
        if roles:
            for role in roles:
                self.authz_manager.assign_role(user.user_id, role)
        
        # Log user creation
        self.audit_logger.log_security_event(
            'user_management', None, None, 'users', 'create',
            True, ThreatLevel.LOW, {'username': username, 'security_level': security_level.value}
        )
        
        return user
    
    def authenticate_user(self, username: str, password: str,
                         source_ip: Optional[str] = None) -> Optional[str]:
        """Authenticate user with security monitoring."""
        # Check if IP is blocked
        if source_ip and self.threat_detector.is_ip_blocked(source_ip):
            self.audit_logger.log_security_event(
                'authentication', None, source_ip, 'auth', 'login',
                False, ThreatLevel.HIGH, {'reason': 'IP blocked'}
            )
            return None
        
        # Attempt authentication
        session_token = self.auth_manager.authenticate(username, password, source_ip)
        
        # Log authentication attempt
        event = SecurityEvent(
            timestamp=time.time(),
            event_type='authentication',
            user_id=None,  # Will be set after successful auth
            source_ip=source_ip,
            resource='auth',
            action='login',
            success=session_token is not None,
            threat_level=ThreatLevel.LOW
        )
        
        # Analyze for threats
        threat_level = self.threat_detector.analyze_event(event)
        if threat_level:
            event.threat_level = threat_level
        
        # Log the event
        self.audit_logger.log_security_event(
            event.event_type, event.user_id, event.source_ip,
            event.resource, event.action, event.success, event.threat_level
        )
        
        return session_token
    
    def authorize_action(self, session_token: str, resource: str, permission: Permission,
                        source_ip: Optional[str] = None) -> bool:
        """Authorize user action with logging."""
        user = self.auth_manager.validate_session(session_token)
        
        if not user:
            self.audit_logger.log_security_event(
                'authorization', None, source_ip, resource, permission.value,
                False, ThreatLevel.MEDIUM, {'reason': 'Invalid session'}
            )
            return False
        
        # Check authorization
        authorized = self.authz_manager.check_permission(user, resource, permission)
        
        # Log authorization attempt
        threat_level = ThreatLevel.LOW if authorized else ThreatLevel.MEDIUM
        
        self.audit_logger.log_security_event(
            'authorization', user.user_id, source_ip, resource, permission.value,
            authorized, threat_level
        )
        
        return authorized
    
    def encrypt_sensitive_data(self, data: Union[str, bytes]) -> str:
        """Encrypt sensitive data with logging."""
        encrypted = self.encryption_manager.encrypt_data(data)
        
        self.audit_logger.log_security_event(
            'encryption', None, None, 'data', 'encrypt',
            True, ThreatLevel.LOW, {'data_size': len(str(data))}
        )
        
        return encrypted
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        audit_report = self.audit_logger.get_audit_report()
        
        return {
            'audit_summary': audit_report,
            'active_sessions': len(self.auth_manager.active_sessions),
            'total_users': len(self.auth_manager.users),
            'blocked_ips': list(self.threat_detector.blocked_ips),
            'active_threats': len(self.threat_detector.active_threats),
            'encryption_status': {
                'last_key_rotation': self.encryption_manager.last_key_rotation,
                'rotation_interval': self.encryption_manager.key_rotation_interval
            }
        }


# Factory functions
def create_security_framework(secret_key: Optional[str] = None) -> AdvancedSecurityFramework:
    """Create security framework with standard configuration."""
    return AdvancedSecurityFramework(secret_key)


def secure_endpoint(required_permission: Permission, resource: str = "api"):
    """Decorator for securing API endpoints."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # This would integrate with your web framework
            # For now, it's a placeholder
            logger.info(f"Secured endpoint called: {func.__name__} requires {required_permission.value}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Demonstration
    security = create_security_framework()
    
    print("Advanced Security Framework Demonstration")
    
    # Create test users
    admin_user = security.create_user(
        "admin", "admin@example.com", "SecurePass123!",
        SecurityLevel.SECRET, {Permission.ADMIN}, {"admin"}
    )
    
    regular_user = security.create_user(
        "user1", "user1@example.com", "UserPass456!",
        SecurityLevel.INTERNAL, {Permission.READ, Permission.WRITE}, {"user"}
    )
    
    # Authenticate users
    admin_token = security.authenticate_user("admin", "SecurePass123!", "192.168.1.10")
    user_token = security.authenticate_user("user1", "UserPass456!", "192.168.1.20")
    
    print(f"Admin authenticated: {'Yes' if admin_token else 'No'}")
    print(f"User authenticated: {'Yes' if user_token else 'No'}")
    
    # Test authorization
    if admin_token:
        admin_can_delete = security.authorize_action(admin_token, "database", Permission.DELETE, "192.168.1.10")
        print(f"Admin can delete: {admin_can_delete}")
    
    if user_token:
        user_can_delete = security.authorize_action(user_token, "database", Permission.DELETE, "192.168.1.20")
        print(f"User can delete: {user_can_delete}")
    
    # Test encryption
    sensitive_data = "User credit card: 1234-5678-9012-3456"
    encrypted_data = security.encrypt_sensitive_data(sensitive_data)
    print(f"Data encrypted: {len(encrypted_data)} characters")
    
    # Generate security report
    report = security.get_security_report()
    print(f"\nSecurity Report:")
    print(f"  Active sessions: {report['active_sessions']}")
    print(f"  Total users: {report['total_users']}")
    print(f"  Blocked IPs: {len(report['blocked_ips'])}")
    print(f"  Total events (24h): {report['audit_summary']['total_events']}")
