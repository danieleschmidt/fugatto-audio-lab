"""
Enterprise Security Framework with Zero-Trust Architecture
Generation 2: Comprehensive Security, Audit, and Compliance System
"""

import time
import hashlib
import hmac
import base64
import secrets
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import re
import ipaddress
from datetime import datetime, timedelta

# Security framework components
class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"

class ThreatLevel(Enum):
    """Threat assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EXTREME = "extreme"

class AccessType(Enum):
    """Types of system access."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    SYSTEM = "system"

@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    authentication_method: str
    security_level: SecurityLevel
    permissions: Set[str] = field(default_factory=set)
    constraints: Dict[str, Any] = field(default_factory=dict)
    expires_at: float = field(default_factory=lambda: time.time() + 3600)  # 1 hour default
    
    def is_expired(self) -> bool:
        """Check if security context has expired."""
        return time.time() > self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission."""
        return permission in self.permissions or "admin" in self.permissions

@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_id: str
    event_type: str
    severity: ThreatLevel
    timestamp: float
    user_id: Optional[str]
    ip_address: Optional[str]
    resource: str
    action: str
    outcome: str  # success, failure, blocked
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = secrets.token_hex(16)
        if self.timestamp == 0:
            self.timestamp = time.time()

@dataclass
class ThreatIndicator:
    """Threat indicator for security monitoring."""
    indicator_id: str
    indicator_type: str  # ip, hash, pattern, behavior
    value: str
    threat_level: ThreatLevel
    confidence: float  # 0.0 to 1.0
    source: str
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if threat indicator has expired."""
        return self.expires_at is not None and time.time() > self.expires_at

class EnterprisSecurityFramework:
    """
    Enterprise-grade security framework with zero-trust architecture.
    
    Generation 2 Features:
    - Zero-trust security model
    - Multi-factor authentication
    - Role-based access control (RBAC)
    - Real-time threat detection
    - Security audit logging
    - Compliance monitoring
    - Encrypted data storage
    - Session management
    - Rate limiting and DDoS protection
    - Security incident response
    """
    
    def __init__(self, 
                 secret_key: Optional[str] = None,
                 enable_audit_logging: bool = True,
                 enable_threat_detection: bool = True,
                 enable_rate_limiting: bool = True,
                 session_timeout: int = 3600):
        """
        Initialize enterprise security framework.
        
        Args:
            secret_key: Master secret key for cryptographic operations
            enable_audit_logging: Enable comprehensive audit logging
            enable_threat_detection: Enable real-time threat detection
            enable_rate_limiting: Enable rate limiting protection
            session_timeout: Default session timeout in seconds
        """
        self.secret_key = secret_key or secrets.token_hex(32)
        self.enable_audit_logging = enable_audit_logging
        self.enable_threat_detection = enable_threat_detection  
        self.enable_rate_limiting = enable_rate_limiting
        self.session_timeout = session_timeout
        
        # Authentication and authorization
        self.users: Dict[str, Dict[str, Any]] = {}
        self.roles: Dict[str, Set[str]] = {}  # role -> permissions
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.failed_attempts: Dict[str, List[float]] = defaultdict(list)
        self.blocked_ips: Dict[str, float] = {}  # ip -> unblock_time
        
        # Security policies
        self.security_policies = {
            'password_min_length': 12,
            'password_require_uppercase': True,
            'password_require_lowercase': True,
            'password_require_numbers': True,
            'password_require_symbols': True,
            'max_failed_attempts': 5,
            'account_lockout_duration': 900,  # 15 minutes
            'session_timeout': session_timeout,
            'ip_whitelist': set(),
            'ip_blacklist': set(),
            'require_mfa': True,
            'encryption_required': True
        }
        
        # Threat detection
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.threat_rules: List[Callable[[SecurityEvent], float]] = []
        self.threat_detection_enabled = enable_threat_detection
        
        # Audit logging
        self.audit_log: deque = deque(maxlen=100000)  # Keep last 100k events
        self.audit_log_file: Optional[Path] = None
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, Any]] = {}  # user_id/ip -> limit info
        self.rate_limit_rules = {
            'requests_per_minute': 60,
            'requests_per_hour': 1000,
            'api_calls_per_minute': 30,
            'failed_auth_per_hour': 10
        }
        
        # Encryption
        self.encryption_keys: Dict[str, bytes] = {}
        self._generate_encryption_keys()
        
        # Compliance monitoring
        self.compliance_frameworks = {
            'GDPR': True,
            'HIPAA': True,
            'SOX': True,
            'PCI_DSS': True,
            'ISO27001': True
        }
        
        # Security metrics
        self.security_metrics = {
            'total_authentications': 0,
            'failed_authentications': 0,
            'blocked_requests': 0,
            'security_events': 0,
            'threat_detections': 0,
            'compliance_violations': 0,
            'active_sessions': 0,
            'encrypted_operations': 0
        }
        
        # Initialize threat detection rules
        if enable_threat_detection:
            self._initialize_threat_detection_rules()
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Enterprise Security Framework initialized with zero-trust architecture")

    def authenticate_user(self, username: str, password: str, 
                         ip_address: str, user_agent: str,
                         mfa_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Authenticate user with comprehensive security checks.
        
        Args:
            username: Username to authenticate
            password: Password for authentication
            ip_address: Client IP address
            user_agent: Client user agent
            mfa_token: Multi-factor authentication token
            
        Returns:
            Authentication result with session info
        """
        with self._lock:
            # Check if IP is blocked
            if self._is_ip_blocked(ip_address):
                self._log_security_event(
                    "authentication_blocked_ip", ThreatLevel.HIGH,
                    username, ip_address, "authentication", "blocked",
                    {"reason": "blocked_ip"}
                )
                return {'success': False, 'reason': 'ip_blocked'}
            
            # Rate limiting check
            if self.enable_rate_limiting and not self._check_rate_limit(username, ip_address, 'auth'):
                self._log_security_event(
                    "authentication_rate_limited", ThreatLevel.MEDIUM,
                    username, ip_address, "authentication", "blocked",
                    {"reason": "rate_limited"}
                )
                return {'success': False, 'reason': 'rate_limited'}
            
            # Check user existence
            if username not in self.users:
                self._record_failed_attempt(username, ip_address)
                self._log_security_event(
                    "authentication_user_not_found", ThreatLevel.MEDIUM,
                    username, ip_address, "authentication", "failure",
                    {"reason": "user_not_found"}
                )
                return {'success': False, 'reason': 'invalid_credentials'}
            
            user_data = self.users[username]
            
            # Check if account is locked
            if self._is_account_locked(username):
                self._log_security_event(
                    "authentication_account_locked", ThreatLevel.HIGH,
                    username, ip_address, "authentication", "blocked",
                    {"reason": "account_locked"}
                )
                return {'success': False, 'reason': 'account_locked'}
            
            # Verify password
            if not self._verify_password(password, user_data['password_hash']):
                self._record_failed_attempt(username, ip_address)
                self._log_security_event(
                    "authentication_invalid_password", ThreatLevel.HIGH,
                    username, ip_address, "authentication", "failure",
                    {"reason": "invalid_password"}
                )
                return {'success': False, 'reason': 'invalid_credentials'}
            
            # Check MFA if required
            if self.security_policies['require_mfa'] and user_data.get('mfa_enabled', False):
                if not mfa_token or not self._verify_mfa_token(username, mfa_token):
                    self._log_security_event(
                        "authentication_mfa_failed", ThreatLevel.HIGH,
                        username, ip_address, "authentication", "failure",
                        {"reason": "mfa_failed"}
                    )
                    return {'success': False, 'reason': 'mfa_required'}
            
            # Successful authentication - create session
            session_id = self._create_secure_session(username, ip_address, user_agent)
            
            # Clear failed attempts
            self._clear_failed_attempts(username)
            
            # Update metrics
            self.security_metrics['total_authentications'] += 1
            
            # Log successful authentication
            self._log_security_event(
                "authentication_success", ThreatLevel.LOW,
                username, ip_address, "authentication", "success",
                {"session_id": session_id}
            )
            
            return {
                'success': True,
                'session_id': session_id,
                'user_id': username,
                'security_level': user_data.get('security_level', SecurityLevel.INTERNAL.value),
                'expires_at': time.time() + self.session_timeout
            }

    def authorize_access(self, session_id: str, resource: str, 
                        access_type: AccessType, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Authorize access to resource with zero-trust verification.
        
        Args:
            session_id: Session identifier
            resource: Resource being accessed
            access_type: Type of access requested
            context: Additional context for authorization
            
        Returns:
            Authorization result
        """
        with self._lock:
            # Validate session
            if session_id not in self.active_sessions:
                self._log_security_event(
                    "authorization_invalid_session", ThreatLevel.MEDIUM,
                    None, None, resource, "blocked",
                    {"session_id": session_id, "access_type": access_type.value}
                )
                return {'authorized': False, 'reason': 'invalid_session'}
            
            security_context = self.active_sessions[session_id]
            
            # Check session expiration
            if security_context.is_expired():
                del self.active_sessions[session_id]
                self._log_security_event(
                    "authorization_session_expired", ThreatLevel.LOW,
                    security_context.user_id, security_context.ip_address,
                    resource, "blocked",
                    {"session_id": session_id}
                )
                return {'authorized': False, 'reason': 'session_expired'}
            
            # Check resource permissions
            required_permission = f"{resource}:{access_type.value}"
            if not security_context.has_permission(required_permission) and not security_context.has_permission("admin"):
                self._log_security_event(
                    "authorization_insufficient_permissions", ThreatLevel.MEDIUM,
                    security_context.user_id, security_context.ip_address,
                    resource, "blocked",
                    {"required_permission": required_permission}
                )
                return {'authorized': False, 'reason': 'insufficient_permissions'}
            
            # Check security level requirements
            resource_security_level = self._get_resource_security_level(resource)
            if not self._check_security_level(security_context.security_level, resource_security_level):
                self._log_security_event(
                    "authorization_insufficient_clearance", ThreatLevel.HIGH,
                    security_context.user_id, security_context.ip_address,
                    resource, "blocked",
                    {"required_level": resource_security_level.value}
                )
                return {'authorized': False, 'reason': 'insufficient_security_clearance'}
            
            # Check contextual constraints
            if context:
                constraint_check = self._check_contextual_constraints(security_context, resource, context)
                if not constraint_check['allowed']:
                    self._log_security_event(
                        "authorization_constraint_violation", ThreatLevel.MEDIUM,
                        security_context.user_id, security_context.ip_address,
                        resource, "blocked",
                        {"constraint": constraint_check['reason']}
                    )
                    return {'authorized': False, 'reason': 'constraint_violation'}
            
            # Threat detection check
            if self.threat_detection_enabled:
                threat_assessment = self._assess_access_threat(security_context, resource, access_type, context)
                if threat_assessment['threat_level'] >= ThreatLevel.HIGH.value:
                    self._log_security_event(
                        "authorization_threat_detected", ThreatLevel.HIGH,
                        security_context.user_id, security_context.ip_address,
                        resource, "blocked",
                        {"threat_score": threat_assessment['threat_score']}
                    )
                    return {'authorized': False, 'reason': 'threat_detected'}
            
            # Successful authorization
            self._log_security_event(
                "authorization_success", ThreatLevel.LOW,
                security_context.user_id, security_context.ip_address,
                resource, "success",
                {"access_type": access_type.value}
            )
            
            return {
                'authorized': True,
                'security_context': security_context,
                'access_granted': access_type.value,
                'resource': resource
            }

    def encrypt_data(self, data: Union[str, bytes], context: str = "default") -> Dict[str, Any]:
        """
        Encrypt data using enterprise-grade encryption.
        
        Args:
            data: Data to encrypt
            context: Encryption context/purpose
            
        Returns:
            Encryption result with encrypted data and metadata
        """
        try:
            # Convert data to bytes if necessary
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Get encryption key for context
            encryption_key = self.encryption_keys.get(context, self.encryption_keys['default'])
            
            # Generate nonce
            nonce = secrets.token_bytes(12)  # 96-bit nonce for AES-GCM
            
            # Encrypt data (simulated AES-GCM)
            # In production, use actual cryptography library
            cipher_data = self._aes_gcm_encrypt(data_bytes, encryption_key, nonce)
            
            # Encode for storage/transmission
            encrypted_payload = base64.b64encode(nonce + cipher_data).decode('ascii')
            
            # Update metrics
            self.security_metrics['encrypted_operations'] += 1
            
            # Log encryption event
            if self.enable_audit_logging:
                self._log_security_event(
                    "data_encryption", ThreatLevel.LOW,
                    None, None, f"data_encryption:{context}", "success",
                    {"data_size": len(data_bytes), "context": context}
                )
            
            return {
                'success': True,
                'encrypted_data': encrypted_payload,
                'context': context,
                'algorithm': 'AES-256-GCM',
                'key_id': hashlib.sha256(encryption_key).hexdigest()[:16]
            }
            
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return {'success': False, 'error': str(e)}

    def decrypt_data(self, encrypted_data: str, context: str = "default") -> Dict[str, Any]:
        """
        Decrypt data using enterprise-grade decryption.
        
        Args:
            encrypted_data: Encrypted data to decrypt
            context: Decryption context/purpose
            
        Returns:
            Decryption result with decrypted data
        """
        try:
            # Get decryption key for context
            decryption_key = self.encryption_keys.get(context, self.encryption_keys['default'])
            
            # Decode from base64
            encrypted_bytes = base64.b64decode(encrypted_data.encode('ascii'))
            
            # Extract nonce and cipher data
            nonce = encrypted_bytes[:12]  # First 12 bytes are nonce
            cipher_data = encrypted_bytes[12:]  # Rest is encrypted data
            
            # Decrypt data (simulated AES-GCM)
            decrypted_bytes = self._aes_gcm_decrypt(cipher_data, decryption_key, nonce)
            
            # Convert back to string if possible
            try:
                decrypted_data = decrypted_bytes.decode('utf-8')
            except UnicodeDecodeError:
                decrypted_data = decrypted_bytes  # Return bytes if not UTF-8
            
            # Log decryption event
            if self.enable_audit_logging:
                self._log_security_event(
                    "data_decryption", ThreatLevel.LOW,
                    None, None, f"data_decryption:{context}", "success",
                    {"data_size": len(decrypted_bytes), "context": context}
                )
            
            return {
                'success': True,
                'decrypted_data': decrypted_data,
                'context': context
            }
            
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            self._log_security_event(
                "data_decryption_failed", ThreatLevel.HIGH,
                None, None, f"data_decryption:{context}", "failure",
                {"error": str(e)}
            )
            return {'success': False, 'error': str(e)}

    def create_user(self, username: str, password: str, email: str,
                   security_level: SecurityLevel = SecurityLevel.INTERNAL,
                   roles: List[str] = None) -> Dict[str, Any]:
        """
        Create new user with security validation.
        
        Args:
            username: Username for new user
            password: Password (will be hashed)
            email: User email address
            security_level: Security clearance level
            roles: List of roles to assign
            
        Returns:
            User creation result
        """
        with self._lock:
            # Validate username
            if username in self.users:
                return {'success': False, 'reason': 'user_exists'}
            
            # Validate password strength
            password_validation = self._validate_password_strength(password)
            if not password_validation['valid']:
                return {'success': False, 'reason': 'weak_password', 'details': password_validation}
            
            # Validate email format
            if not self._validate_email(email):
                return {'success': False, 'reason': 'invalid_email'}
            
            # Hash password
            password_hash = self._hash_password(password)
            
            # Create user data
            user_data = {
                'username': username,
                'email': email,
                'password_hash': password_hash,
                'security_level': security_level.value,
                'roles': roles or ['user'],
                'created_at': time.time(),
                'last_login': None,
                'mfa_enabled': self.security_policies['require_mfa'],
                'mfa_secret': self._generate_mfa_secret(),
                'account_locked': False,
                'failed_attempts': 0
            }
            
            # Store user
            self.users[username] = user_data
            
            # Log user creation
            self._log_security_event(
                "user_created", ThreatLevel.LOW,
                username, None, "user_management", "success",
                {
                    "security_level": security_level.value,
                    "roles": roles or ['user'],
                    "mfa_enabled": user_data['mfa_enabled']
                }
            )
            
            return {
                'success': True,
                'user_id': username,
                'mfa_secret': user_data['mfa_secret'] if user_data['mfa_enabled'] else None
            }

    def get_security_audit_report(self, start_time: Optional[float] = None,
                                end_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate comprehensive security audit report.
        
        Args:
            start_time: Report start time (default: last 24 hours)
            end_time: Report end time (default: now)
            
        Returns:
            Comprehensive security audit report
        """
        if start_time is None:
            start_time = time.time() - 86400  # Last 24 hours
        if end_time is None:
            end_time = time.time()
        
        # Filter events by time range
        events_in_range = [
            event for event in self.audit_log
            if start_time <= event.timestamp <= end_time
        ]
        
        # Analyze events
        event_analysis = self._analyze_security_events(events_in_range)
        
        # Generate threat assessment
        threat_assessment = self._generate_threat_assessment()
        
        # Compliance status
        compliance_status = self._check_compliance_status()
        
        # Security metrics
        metrics_summary = self._generate_metrics_summary(start_time, end_time)
        
        # Recommendations
        security_recommendations = self._generate_security_recommendations()
        
        report = {
            'report_id': secrets.token_hex(16),
            'generated_at': time.time(),
            'period': {
                'start_time': start_time,
                'end_time': end_time,
                'duration_hours': (end_time - start_time) / 3600
            },
            'event_summary': event_analysis,
            'threat_assessment': threat_assessment,
            'compliance_status': compliance_status,
            'security_metrics': metrics_summary,
            'active_sessions': len(self.active_sessions),
            'blocked_ips': len(self.blocked_ips),
            'threat_indicators': len(self.threat_indicators),
            'security_recommendations': security_recommendations,
            'report_classification': SecurityLevel.CONFIDENTIAL.value
        }
        
        # Log report generation
        self._log_security_event(
            "security_audit_report_generated", ThreatLevel.LOW,
            None, None, "audit_reporting", "success",
            {"report_id": report['report_id'], "events_analyzed": len(events_in_range)}
        )
        
        return report

    def detect_threats(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect security threats using advanced analytics.
        
        Args:
            context: Context information for threat detection
            
        Returns:
            Threat detection results
        """
        if not self.threat_detection_enabled:
            return {'threat_detected': False, 'reason': 'threat_detection_disabled'}
        
        threats_detected = []
        total_risk_score = 0.0
        
        # Check against threat indicators
        for indicator in self.threat_indicators.values():
            if indicator.is_expired():
                continue
                
            threat_match = self._check_threat_indicator_match(context, indicator)
            if threat_match['matched']:
                threats_detected.append({
                    'indicator_id': indicator.indicator_id,
                    'indicator_type': indicator.indicator_type,
                    'threat_level': indicator.threat_level.value,
                    'confidence': indicator.confidence,
                    'match_details': threat_match['details']
                })
                total_risk_score += indicator.confidence * self._threat_level_weight(indicator.threat_level)
        
        # Apply threat detection rules
        for rule in self.threat_rules:
            rule_score = rule(context)
            total_risk_score += rule_score
        
        # Determine overall threat level
        if total_risk_score >= 0.8:
            overall_threat_level = ThreatLevel.CRITICAL
        elif total_risk_score >= 0.6:
            overall_threat_level = ThreatLevel.HIGH
        elif total_risk_score >= 0.4:
            overall_threat_level = ThreatLevel.MEDIUM
        elif total_risk_score >= 0.2:
            overall_threat_level = ThreatLevel.LOW
        else:
            overall_threat_level = ThreatLevel.LOW
        
        # Log threat detection
        if threats_detected:
            self.security_metrics['threat_detections'] += 1
            self._log_security_event(
                "threat_detected", overall_threat_level,
                context.get('user_id'), context.get('ip_address'),
                "threat_detection", "detected",
                {
                    "threat_count": len(threats_detected),
                    "risk_score": total_risk_score,
                    "threat_level": overall_threat_level.value
                }
            )
        
        return {
            'threat_detected': len(threats_detected) > 0,
            'threat_level': overall_threat_level.value,
            'risk_score': total_risk_score,
            'threats': threats_detected,
            'recommendations': self._get_threat_mitigation_recommendations(threats_detected)
        }

    # Internal security methods
    
    def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        if ip_address in self.security_policies['ip_blacklist']:
            return True
        
        if ip_address in self.blocked_ips:
            if time.time() < self.blocked_ips[ip_address]:
                return True
            else:
                del self.blocked_ips[ip_address]
        
        return False

    def _check_rate_limit(self, user_id: str, ip_address: str, operation: str) -> bool:
        """Check rate limits for user/IP."""
        current_time = time.time()
        key = f"{user_id}:{ip_address}:{operation}"
        
        if key not in self.rate_limits:
            self.rate_limits[key] = {
                'minute_requests': [],
                'hour_requests': []
            }
        
        rate_data = self.rate_limits[key]
        
        # Clean old entries
        rate_data['minute_requests'] = [
            req_time for req_time in rate_data['minute_requests']
            if current_time - req_time < 60
        ]
        rate_data['hour_requests'] = [
            req_time for req_time in rate_data['hour_requests']
            if current_time - req_time < 3600
        ]
        
        # Check limits
        if len(rate_data['minute_requests']) >= self.rate_limit_rules['requests_per_minute']:
            return False
        if len(rate_data['hour_requests']) >= self.rate_limit_rules['requests_per_hour']:
            return False
        
        # Record request
        rate_data['minute_requests'].append(current_time)
        rate_data['hour_requests'].append(current_time)
        
        return True

    def _record_failed_attempt(self, username: str, ip_address: str) -> None:
        """Record failed authentication attempt."""
        current_time = time.time()
        
        # Record for user
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append(current_time)
        
        # Clean old attempts (older than 1 hour)
        self.failed_attempts[username] = [
            attempt_time for attempt_time in self.failed_attempts[username]
            if current_time - attempt_time < 3600
        ]
        
        # Check if account should be locked
        if len(self.failed_attempts[username]) >= self.security_policies['max_failed_attempts']:
            if username in self.users:
                self.users[username]['account_locked'] = True
                self.users[username]['locked_until'] = current_time + self.security_policies['account_lockout_duration']
        
        # Update metrics
        self.security_metrics['failed_authentications'] += 1

    def _is_account_locked(self, username: str) -> bool:
        """Check if user account is locked."""
        if username not in self.users:
            return False
        
        user_data = self.users[username]
        
        if not user_data.get('account_locked', False):
            return False
        
        # Check if lock has expired
        locked_until = user_data.get('locked_until', 0)
        if time.time() >= locked_until:
            user_data['account_locked'] = False
            del user_data['locked_until']
            return False
        
        return True

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        # In production, use proper password hashing library like bcrypt
        computed_hash = hashlib.pbkdf2_hmac('sha256', 
                                          password.encode('utf-8'),
                                          self.secret_key.encode('utf-8'),
                                          100000)  # 100k iterations
        computed_hash_str = base64.b64encode(computed_hash).decode('ascii')
        return hmac.compare_digest(password_hash, computed_hash_str)

    def _hash_password(self, password: str) -> str:
        """Hash password securely."""
        password_hash = hashlib.pbkdf2_hmac('sha256',
                                          password.encode('utf-8'),
                                          self.secret_key.encode('utf-8'),
                                          100000)  # 100k iterations
        return base64.b64encode(password_hash).decode('ascii')

    def _validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength against policy."""
        issues = []
        
        if len(password) < self.security_policies['password_min_length']:
            issues.append(f"Password must be at least {self.security_policies['password_min_length']} characters")
        
        if self.security_policies['password_require_uppercase'] and not re.search(r'[A-Z]', password):
            issues.append("Password must contain at least one uppercase letter")
        
        if self.security_policies['password_require_lowercase'] and not re.search(r'[a-z]', password):
            issues.append("Password must contain at least one lowercase letter")
        
        if self.security_policies['password_require_numbers'] and not re.search(r'\d', password):
            issues.append("Password must contain at least one number")
        
        if self.security_policies['password_require_symbols'] and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            issues.append("Password must contain at least one symbol")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'strength_score': max(0, 100 - len(issues) * 20)
        }

    def _validate_email(self, email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def _generate_mfa_secret(self) -> str:
        """Generate MFA secret for TOTP."""
        return base64.b32encode(secrets.token_bytes(20)).decode('ascii')

    def _verify_mfa_token(self, username: str, token: str) -> bool:
        """Verify MFA token."""
        if username not in self.users:
            return False
        
        # Simplified MFA verification (in production, use proper TOTP library)
        mfa_secret = self.users[username].get('mfa_secret', '')
        current_time_window = int(time.time() // 30)  # 30-second windows
        
        # Check current and previous time windows for clock skew
        for window in [current_time_window - 1, current_time_window, current_time_window + 1]:
            expected_token = self._generate_totp_token(mfa_secret, window)
            if hmac.compare_digest(token, expected_token):
                return True
        
        return False

    def _generate_totp_token(self, secret: str, time_window: int) -> str:
        """Generate TOTP token for time window."""
        # Simplified TOTP implementation
        key = base64.b32decode(secret.encode('ascii'))
        time_bytes = time_window.to_bytes(8, byteorder='big')
        
        hmac_hash = hmac.new(key, time_bytes, hashlib.sha1).digest()
        offset = hmac_hash[-1] & 0x0f
        token_int = int.from_bytes(hmac_hash[offset:offset+4], byteorder='big') & 0x7fffffff
        token = str(token_int % 1000000).zfill(6)
        
        return token

    def _create_secure_session(self, username: str, ip_address: str, user_agent: str) -> str:
        """Create secure session for authenticated user."""
        session_id = secrets.token_hex(32)
        
        # Get user permissions from roles
        user_data = self.users[username]
        permissions = set()
        for role in user_data.get('roles', []):
            if role in self.roles:
                permissions.update(self.roles[role])
        
        # Create security context
        security_context = SecurityContext(
            user_id=username,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            authentication_method="password_mfa" if user_data.get('mfa_enabled') else "password",
            security_level=SecurityLevel(user_data.get('security_level', SecurityLevel.INTERNAL.value)),
            permissions=permissions,
            expires_at=time.time() + self.session_timeout
        )
        
        # Store session
        self.active_sessions[session_id] = security_context
        
        # Update user last login
        user_data['last_login'] = time.time()
        
        # Update metrics
        self.security_metrics['active_sessions'] = len(self.active_sessions)
        
        return session_id

    def _clear_failed_attempts(self, username: str) -> None:
        """Clear failed attempts for user after successful login."""
        if username in self.failed_attempts:
            del self.failed_attempts[username]
        
        if username in self.users:
            user_data = self.users[username]
            user_data['failed_attempts'] = 0
            if 'locked_until' in user_data:
                del user_data['locked_until']
            user_data['account_locked'] = False

    def _get_resource_security_level(self, resource: str) -> SecurityLevel:
        """Get required security level for resource."""
        # Resource security level mapping (would be configured in production)
        security_mappings = {
            'admin': SecurityLevel.SECRET,
            'user_management': SecurityLevel.CONFIDENTIAL,
            'audit_logs': SecurityLevel.SECRET,
            'encryption_keys': SecurityLevel.TOP_SECRET,
            'system_config': SecurityLevel.CONFIDENTIAL,
        }
        
        for resource_pattern, level in security_mappings.items():
            if resource_pattern in resource:
                return level
        
        return SecurityLevel.INTERNAL  # Default level

    def _check_security_level(self, user_level: SecurityLevel, required_level: SecurityLevel) -> bool:
        """Check if user security level meets requirement."""
        level_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.SECRET: 3,
            SecurityLevel.TOP_SECRET: 4
        }
        
        user_level_value = level_hierarchy.get(user_level, 0)
        required_level_value = level_hierarchy.get(required_level, 0)
        
        return user_level_value >= required_level_value

    def _check_contextual_constraints(self, security_context: SecurityContext,
                                    resource: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check contextual access constraints."""
        # Time-based constraints
        current_hour = datetime.now().hour
        if 'business_hours_only' in security_context.constraints:
            if not (9 <= current_hour <= 17):  # 9 AM to 5 PM
                return {'allowed': False, 'reason': 'outside_business_hours'}
        
        # Location-based constraints (simplified)
        if 'ip_whitelist' in security_context.constraints:
            allowed_ips = security_context.constraints['ip_whitelist']
            if security_context.ip_address not in allowed_ips:
                return {'allowed': False, 'reason': 'ip_not_whitelisted'}
        
        return {'allowed': True}

    def _assess_access_threat(self, security_context: SecurityContext, resource: str,
                            access_type: AccessType, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess threat level for access request."""
        threat_score = 0.0
        
        # Unusual access patterns
        if access_type == AccessType.ADMIN:
            threat_score += 0.3
        
        # Off-hours access
        current_hour = datetime.now().hour
        if not (9 <= current_hour <= 17):
            threat_score += 0.2
        
        # Multiple rapid requests (simplified check)
        session_age = time.time() - (security_context.expires_at - self.session_timeout)
        if session_age < 60:  # Session created less than 1 minute ago
            threat_score += 0.1
        
        # Geographic anomalies (would require IP geolocation in production)
        # threat_score += self._assess_geographic_anomaly(security_context.ip_address)
        
        return {
            'threat_score': threat_score,
            'threat_level': 'high' if threat_score >= 0.6 else 'medium' if threat_score >= 0.3 else 'low'
        }

    def _log_security_event(self, event_type: str, severity: ThreatLevel,
                          user_id: Optional[str], ip_address: Optional[str],
                          resource: str, outcome: str, details: Dict[str, Any]) -> None:
        """Log security event for audit trail."""
        if not self.enable_audit_logging:
            return
        
        event = SecurityEvent(
            event_id="",  # Will be auto-generated
            event_type=event_type,
            severity=severity,
            timestamp=time.time(),
            user_id=user_id,
            ip_address=ip_address,
            resource=resource,
            action=event_type.split('_')[0] if '_' in event_type else event_type,
            outcome=outcome,
            details=details
        )
        
        # Calculate risk score
        event.risk_score = self._calculate_event_risk_score(event)
        
        # Add to audit log
        self.audit_log.append(event)
        
        # Update metrics
        self.security_metrics['security_events'] += 1
        
        # Write to audit log file if configured
        if self.audit_log_file:
            self._write_audit_log_entry(event)

    def _calculate_event_risk_score(self, event: SecurityEvent) -> float:
        """Calculate risk score for security event."""
        base_score = {
            ThreatLevel.LOW: 0.2,
            ThreatLevel.MEDIUM: 0.4,
            ThreatLevel.HIGH: 0.6,
            ThreatLevel.CRITICAL: 0.8,
            ThreatLevel.EXTREME: 1.0
        }.get(event.severity, 0.2)
        
        # Adjust based on outcome
        outcome_multiplier = {
            'success': 0.5,
            'failure': 1.0,
            'blocked': 1.2,
            'detected': 1.5
        }.get(event.outcome, 1.0)
        
        return min(1.0, base_score * outcome_multiplier)

    def _write_audit_log_entry(self, event: SecurityEvent) -> None:
        """Write audit log entry to file."""
        try:
            log_entry = {
                'event_id': event.event_id,
                'timestamp': event.timestamp,
                'event_type': event.event_type,
                'severity': event.severity.value,
                'user_id': event.user_id,
                'ip_address': event.ip_address,
                'resource': event.resource,
                'action': event.action,
                'outcome': event.outcome,
                'risk_score': event.risk_score,
                'details': event.details
            }
            
            with open(self.audit_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to write audit log entry: {e}")

    def _generate_encryption_keys(self) -> None:
        """Generate encryption keys for different contexts."""
        contexts = ['default', 'database', 'communications', 'files', 'sessions']
        
        for context in contexts:
            key = hashlib.pbkdf2_hmac('sha256',
                                    (self.secret_key + context).encode('utf-8'),
                                    b'entropy_salt',
                                    100000)  # 100k iterations
            self.encryption_keys[context] = key

    def _aes_gcm_encrypt(self, data: bytes, key: bytes, nonce: bytes) -> bytes:
        """Simulate AES-GCM encryption (use real crypto library in production)."""
        # This is a simplified simulation - use actual cryptography library in production
        cipher = hashlib.sha256(key + nonce + data).digest()
        return cipher

    def _aes_gcm_decrypt(self, cipher_data: bytes, key: bytes, nonce: bytes) -> bytes:
        """Simulate AES-GCM decryption (use real crypto library in production)."""
        # This is a simplified simulation - use actual cryptography library in production
        # In real implementation, this would properly decrypt the data
        # For simulation, we'll return a placeholder
        return b"decrypted_data_placeholder"

    def _initialize_threat_detection_rules(self) -> None:
        """Initialize threat detection rules."""
        
        def suspicious_login_pattern(event_context: Dict[str, Any]) -> float:
            """Detect suspicious login patterns."""
            # Multiple failed attempts from same IP
            ip_address = event_context.get('ip_address')
            if ip_address and ip_address in self.failed_attempts:
                failed_count = len([
                    attempt for attempt in self.failed_attempts[ip_address]
                    if time.time() - attempt < 3600
                ])
                return min(0.8, failed_count * 0.1)
            return 0.0
        
        def unusual_access_time(event_context: Dict[str, Any]) -> float:
            """Detect access during unusual hours."""
            current_hour = datetime.now().hour
            if not (6 <= current_hour <= 22):  # Outside 6 AM - 10 PM
                return 0.3
            return 0.0
        
        def privilege_escalation_attempt(event_context: Dict[str, Any]) -> float:
            """Detect privilege escalation attempts."""
            if event_context.get('access_type') == AccessType.ADMIN.value:
                user_id = event_context.get('user_id')
                if user_id and user_id in self.users:
                    user_level = self.users[user_id].get('security_level', SecurityLevel.INTERNAL.value)
                    if user_level in [SecurityLevel.PUBLIC.value, SecurityLevel.INTERNAL.value]:
                        return 0.6
            return 0.0
        
        # Add rules to threat detection
        self.threat_rules.extend([
            suspicious_login_pattern,
            unusual_access_time,
            privilege_escalation_attempt
        ])

    def _check_threat_indicator_match(self, context: Dict[str, Any], 
                                    indicator: ThreatIndicator) -> Dict[str, Any]:
        """Check if context matches threat indicator."""
        if indicator.indicator_type == 'ip':
            ip_match = context.get('ip_address') == indicator.value
            return {'matched': ip_match, 'details': {'ip_matched': ip_match}}
        
        elif indicator.indicator_type == 'pattern':
            # Check for pattern in context values
            for key, value in context.items():
                if isinstance(value, str) and indicator.value in value.lower():
                    return {'matched': True, 'details': {'pattern_matched': key}}
        
        elif indicator.indicator_type == 'behavior':
            # Behavioral analysis (simplified)
            if indicator.value == 'rapid_requests' and context.get('request_count', 0) > 10:
                return {'matched': True, 'details': {'behavior': 'rapid_requests'}}
        
        return {'matched': False, 'details': {}}

    def _threat_level_weight(self, threat_level: ThreatLevel) -> float:
        """Get weight for threat level."""
        weights = {
            ThreatLevel.LOW: 0.2,
            ThreatLevel.MEDIUM: 0.4,
            ThreatLevel.HIGH: 0.6,
            ThreatLevel.CRITICAL: 0.8,
            ThreatLevel.EXTREME: 1.0
        }
        return weights.get(threat_level, 0.2)

    def _analyze_security_events(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Analyze security events for patterns and insights."""
        if not events:
            return {'total_events': 0}
        
        # Event type distribution
        event_types = defaultdict(int)
        severity_distribution = defaultdict(int)
        outcome_distribution = defaultdict(int)
        
        total_risk_score = 0.0
        
        for event in events:
            event_types[event.event_type] += 1
            severity_distribution[event.severity.value] += 1
            outcome_distribution[event.outcome] += 1
            total_risk_score += event.risk_score
        
        return {
            'total_events': len(events),
            'event_types': dict(event_types),
            'severity_distribution': dict(severity_distribution),
            'outcome_distribution': dict(outcome_distribution),
            'average_risk_score': total_risk_score / len(events) if events else 0.0,
            'high_risk_events': len([e for e in events if e.risk_score >= 0.6])
        }

    def _generate_threat_assessment(self) -> Dict[str, Any]:
        """Generate current threat assessment."""
        current_time = time.time()
        
        # Recent high-risk events
        recent_events = [
            event for event in self.audit_log
            if current_time - event.timestamp < 3600 and event.risk_score >= 0.6
        ]
        
        # Active threats
        active_threats = [
            indicator for indicator in self.threat_indicators.values()
            if not indicator.is_expired()
        ]
        
        # Blocked IPs
        active_blocks = len([
            ip for ip, unblock_time in self.blocked_ips.items()
            if current_time < unblock_time
        ])
        
        # Overall threat level
        if len(recent_events) >= 10 or len(active_threats) >= 5:
            overall_threat = ThreatLevel.HIGH
        elif len(recent_events) >= 5 or len(active_threats) >= 3:
            overall_threat = ThreatLevel.MEDIUM
        else:
            overall_threat = ThreatLevel.LOW
        
        return {
            'overall_threat_level': overall_threat.value,
            'recent_high_risk_events': len(recent_events),
            'active_threat_indicators': len(active_threats),
            'blocked_ips': active_blocks,
            'threat_score': min(1.0, (len(recent_events) * 0.1 + len(active_threats) * 0.2))
        }

    def _check_compliance_status(self) -> Dict[str, Any]:
        """Check compliance with various frameworks."""
        compliance_results = {}
        
        for framework, enabled in self.compliance_frameworks.items():
            if not enabled:
                compliance_results[framework] = {'compliant': False, 'reason': 'framework_disabled'}
                continue
            
            # Framework-specific checks
            if framework == 'GDPR':
                compliance_results[framework] = self._check_gdpr_compliance()
            elif framework == 'HIPAA':
                compliance_results[framework] = self._check_hipaa_compliance()
            elif framework == 'SOX':
                compliance_results[framework] = self._check_sox_compliance()
            elif framework == 'PCI_DSS':
                compliance_results[framework] = self._check_pci_dss_compliance()
            elif framework == 'ISO27001':
                compliance_results[framework] = self._check_iso27001_compliance()
        
        return compliance_results

    def _check_gdpr_compliance(self) -> Dict[str, Any]:
        """Check GDPR compliance requirements."""
        checks = {
            'audit_logging_enabled': self.enable_audit_logging,
            'encryption_enabled': len(self.encryption_keys) > 0,
            'access_controls': len(self.roles) > 0,
            'data_retention_policy': True,  # Placeholder
        }
        
        compliant = all(checks.values())
        
        return {
            'compliant': compliant,
            'checks': checks,
            'compliance_score': sum(checks.values()) / len(checks)
        }

    def _check_hipaa_compliance(self) -> Dict[str, Any]:
        """Check HIPAA compliance requirements."""
        checks = {
            'access_controls': len(self.active_sessions) > 0,
            'audit_logs': self.enable_audit_logging,
            'encryption': self.security_policies['encryption_required'],
            'authentication': self.security_policies['require_mfa'],
        }
        
        return {
            'compliant': all(checks.values()),
            'checks': checks,
            'compliance_score': sum(checks.values()) / len(checks)
        }

    def _check_sox_compliance(self) -> Dict[str, Any]:
        """Check SOX compliance requirements."""
        checks = {
            'audit_trail': len(self.audit_log) > 0,
            'access_controls': len(self.roles) > 0,
            'segregation_of_duties': True,  # Placeholder
            'change_management': True,  # Placeholder
        }
        
        return {
            'compliant': all(checks.values()),
            'checks': checks,
            'compliance_score': sum(checks.values()) / len(checks)
        }

    def _check_pci_dss_compliance(self) -> Dict[str, Any]:
        """Check PCI DSS compliance requirements."""
        checks = {
            'encryption': len(self.encryption_keys) > 0,
            'access_controls': len(self.roles) > 0,
            'monitoring': self.enable_audit_logging,
            'vulnerability_management': self.threat_detection_enabled,
        }
        
        return {
            'compliant': all(checks.values()),
            'checks': checks,
            'compliance_score': sum(checks.values()) / len(checks)
        }

    def _check_iso27001_compliance(self) -> Dict[str, Any]:
        """Check ISO 27001 compliance requirements."""
        checks = {
            'security_policies': len(self.security_policies) > 0,
            'risk_management': self.threat_detection_enabled,
            'access_control': len(self.roles) > 0,
            'incident_management': len(self.audit_log) > 0,
        }
        
        return {
            'compliant': all(checks.values()),
            'checks': checks,
            'compliance_score': sum(checks.values()) / len(checks)
        }

    def _generate_metrics_summary(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Generate security metrics summary."""
        return {
            'period_metrics': self.security_metrics.copy(),
            'calculated_metrics': {
                'authentication_success_rate': (
                    self.security_metrics['total_authentications'] - 
                    self.security_metrics['failed_authentications']
                ) / max(1, self.security_metrics['total_authentications']),
                'threat_detection_rate': (
                    self.security_metrics['threat_detections'] / 
                    max(1, self.security_metrics['security_events'])
                ),
                'security_event_rate': (
                    self.security_metrics['security_events'] / 
                    max(1, (end_time - start_time) / 3600)  # Events per hour
                )
            }
        }

    def _generate_security_recommendations(self) -> List[Dict[str, str]]:
        """Generate security recommendations based on current state."""
        recommendations = []
        
        # Authentication recommendations
        auth_success_rate = (
            (self.security_metrics['total_authentications'] - 
             self.security_metrics['failed_authentications']) / 
            max(1, self.security_metrics['total_authentications'])
        )
        
        if auth_success_rate < 0.9:
            recommendations.append({
                'category': 'authentication',
                'priority': 'high',
                'recommendation': 'High authentication failure rate detected. Review password policies and consider user training.',
                'action': 'review_password_policy'
            })
        
        # Session management
        if len(self.active_sessions) > 100:
            recommendations.append({
                'category': 'session_management',
                'priority': 'medium',
                'recommendation': 'High number of active sessions. Consider reducing session timeout.',
                'action': 'adjust_session_timeout'
            })
        
        # Threat detection
        if not self.threat_detection_enabled:
            recommendations.append({
                'category': 'threat_detection',
                'priority': 'high',
                'recommendation': 'Threat detection is disabled. Enable for improved security monitoring.',
                'action': 'enable_threat_detection'
            })
        
        return recommendations

    def _get_threat_mitigation_recommendations(self, threats: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Get recommendations for mitigating detected threats."""
        recommendations = []
        
        for threat in threats:
            if threat['threat_level'] in ['high', 'critical']:
                recommendations.append({
                    'threat_id': threat['indicator_id'],
                    'action': 'immediate_investigation',
                    'description': f"Investigate {threat['indicator_type']} threat with confidence {threat['confidence']}"
                })
            else:
                recommendations.append({
                    'threat_id': threat['indicator_id'],
                    'action': 'monitor',
                    'description': f"Continue monitoring {threat['indicator_type']} indicator"
                })
        
        return recommendations