# Security Guidelines

## Overview

This document outlines security practices, guidelines, and procedures for Fugatto Audio Lab development and deployment.

## Security Architecture

### Threat Model

**Assets Protected:**
- User audio data and prompts
- Model weights and configurations
- API keys and credentials
- User authentication data
- Generated audio content

**Potential Threats:**
- Unauthorized access to audio data
- Model extraction attacks
- Prompt injection attacks
- Data exfiltration
- Malicious audio generation
- Resource exhaustion (DoS)

**Trust Boundaries:**
- Client ↔ API Gateway
- API Gateway ↔ Core Services
- Core Services ↔ Model Runtime
- Model Runtime ↔ Storage

## Input Validation & Sanitization

### Audio Input Validation

```python
# Example secure audio validation
import librosa
import soundfile as sf
from pathlib import Path

def validate_audio_input(audio_path: str) -> bool:
    """Securely validate audio input."""
    
    # Path traversal protection
    path = Path(audio_path).resolve()
    if not str(path).startswith(ALLOWED_UPLOAD_DIR):
        raise SecurityError("Path traversal attempt detected")
    
    # File size limits
    if path.stat().st_size > MAX_AUDIO_SIZE:
        raise SecurityError("Audio file too large")
    
    # Format validation
    try:
        audio, sr = librosa.load(str(path), duration=MAX_DURATION)
        if sr not in ALLOWED_SAMPLE_RATES:
            raise SecurityError("Invalid sample rate")
    except Exception as e:
        raise SecurityError(f"Invalid audio file: {e}")
    
    return True
```

### Text Prompt Sanitization

```python
import re
from typing import str

def sanitize_prompt(prompt: str) -> str:
    """Sanitize user prompts for safe processing."""
    
    # Length limits
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError("Prompt too long")
    
    # Character whitelist
    allowed_chars = re.compile(r'^[a-zA-Z0-9\s\.,!?\-\'\"]*$')
    if not allowed_chars.match(prompt):
        raise ValueError("Prompt contains invalid characters")
    
    # Remove potential injection patterns
    dangerous_patterns = [
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'data:text/html',
        r'vbscript:',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            raise SecurityError("Potentially malicious prompt detected")
    
    return prompt.strip()
```

## Authentication & Authorization

### API Key Management

```python
import secrets
import hashlib
from datetime import datetime, timedelta

class APIKeyManager:
    """Secure API key management."""
    
    def generate_api_key(self, user_id: str) -> dict:
        """Generate a new API key with proper entropy."""
        key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        return {
            'key': key,
            'hash': key_hash,
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(days=365),
            'permissions': ['audio_generate', 'audio_transform']
        }
    
    def validate_api_key(self, provided_key: str) -> dict:
        """Validate API key against stored hash."""
        key_hash = hashlib.sha256(provided_key.encode()).hexdigest()
        
        # Retrieve from secure storage
        stored_key = self.get_key_from_storage(key_hash)
        
        if not stored_key:
            raise AuthenticationError("Invalid API key")
        
        if stored_key['expires_at'] < datetime.utcnow():
            raise AuthenticationError("API key expired")
        
        return stored_key
```

### Rate Limiting

```python
import time
from collections import defaultdict, deque

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if request is within rate limits."""
        now = time.time()
        user_requests = self.requests[user_id]
        
        # Remove old requests outside window
        while user_requests and user_requests[0] < now - self.window_seconds:
            user_requests.popleft()
        
        # Check if within limits
        if len(user_requests) >= self.max_requests:
            return False
        
        # Record this request
        user_requests.append(now)
        return True
```

## Data Protection

### Encryption at Rest

```python
from cryptography.fernet import Fernet
import os

class DataEncryption:
    """Handle encryption of sensitive data."""
    
    def __init__(self):
        # Load key from secure environment
        key = os.environ.get('ENCRYPTION_KEY')
        if not key:
            raise SecurityError("Encryption key not configured")
        self.cipher = Fernet(key.encode())
    
    def encrypt_audio_data(self, audio_bytes: bytes) -> bytes:
        """Encrypt audio data before storage."""
        return self.cipher.encrypt(audio_bytes)
    
    def decrypt_audio_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt audio data after retrieval."""
        return self.cipher.decrypt(encrypted_data)
    
    def encrypt_user_prompts(self, prompt: str) -> str:
        """Encrypt user prompts for secure storage."""
        encrypted = self.cipher.encrypt(prompt.encode())
        return encrypted.decode()
```

### Data Retention

```python
from datetime import datetime, timedelta
import logging

class DataRetentionManager:
    """Manage data lifecycle and retention policies."""
    
    def __init__(self):
        self.retention_policies = {
            'user_prompts': timedelta(days=30),
            'generated_audio': timedelta(days=7),
            'model_cache': timedelta(days=90),
            'logs': timedelta(days=365)
        }
    
    def cleanup_expired_data(self):
        """Remove data past retention period."""
        for data_type, retention_period in self.retention_policies.items():
            cutoff_date = datetime.utcnow() - retention_period
            
            deleted_count = self.delete_data_before_date(data_type, cutoff_date)
            logging.info(f"Deleted {deleted_count} {data_type} records older than {cutoff_date}")
    
    def delete_user_data(self, user_id: str):
        """Complete user data deletion for GDPR compliance."""
        data_types = ['prompts', 'audio_files', 'api_keys', 'usage_logs']
        
        for data_type in data_types:
            self.delete_user_data_by_type(user_id, data_type)
            logging.info(f"Deleted {data_type} for user {user_id}")
```

## Model Security

### Model Integrity Verification

```python
import hashlib
import requests
from pathlib import Path

class ModelVerifier:
    """Verify model integrity and authenticity."""
    
    def __init__(self):
        # Official model checksums from secure source
        self.known_checksums = {
            'nvidia/fugatto-base': 'sha256:a1b2c3d4e5f6...',
            'nvidia/fugatto-large': 'sha256:f6e5d4c3b2a1...'
        }
    
    def verify_model_checksum(self, model_path: Path, expected_checksum: str) -> bool:
        """Verify model file integrity."""
        sha256_hash = hashlib.sha256()
        
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        actual_checksum = sha256_hash.hexdigest()
        return actual_checksum == expected_checksum.split(':')[1]
    
    def download_and_verify_model(self, model_name: str, download_url: str):
        """Securely download and verify model."""
        if model_name not in self.known_checksums:
            raise SecurityError(f"Unknown model: {model_name}")
        
        # Download with TLS verification
        response = requests.get(download_url, verify=True, stream=True)
        response.raise_for_status()
        
        model_path = Path(f"models/{model_name}")
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify integrity
        expected_checksum = self.known_checksums[model_name]
        if not self.verify_model_checksum(model_path, expected_checksum):
            model_path.unlink()  # Delete corrupted file
            raise SecurityError(f"Model checksum verification failed for {model_name}")
```

### Prompt Injection Prevention

```python
import re
from typing import List

class PromptInjectionDetector:
    """Detect and prevent prompt injection attacks."""
    
    def __init__(self):
        self.injection_patterns = [
            # System prompt override attempts
            r'ignore.*previous.*instructions',
            r'system:.*role:.*admin',
            r'</prompt>.*<prompt>',
            
            # Command injection
            r'exec\s*\(',
            r'eval\s*\(',
            r'import\s+os',
            r'subprocess\.',
            
            # Data exfiltration attempts
            r'print.*config',
            r'return.*secret',
            r'reveal.*key',
        ]
    
    def detect_injection(self, prompt: str) -> List[str]:
        """Detect potential injection attempts."""
        detected_patterns = []
        
        for pattern in self.injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE | re.MULTILINE):
                detected_patterns.append(pattern)
        
        return detected_patterns
    
    def is_safe_prompt(self, prompt: str) -> bool:
        """Check if prompt is safe for processing."""
        detected = self.detect_injection(prompt)
        
        if detected:
            logging.warning(f"Prompt injection detected: {detected}")
            return False
        
        return True
```

## Network Security

### TLS Configuration

```python
import ssl
import certifi
from urllib3.util.ssl_ import create_urllib3_context

def create_secure_ssl_context():
    """Create secure SSL context for API connections."""
    context = create_urllib3_context()
    
    # Use only secure protocols
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.maximum_version = ssl.TLSVersion.TLSv1_3
    
    # Use secure cipher suites
    context.set_ciphers('ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384')
    
    # Verify certificates
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(certifi.where())
    
    return context
```

### CORS Configuration

```python
from flask_cors import CORS

def configure_cors(app):
    """Configure secure CORS policies."""
    CORS(app, 
         origins=['https://fugatto-lab.com', 'https://api.fugatto-lab.com'],
         methods=['GET', 'POST'],
         allow_headers=['Content-Type', 'Authorization'],
         expose_headers=['X-Request-ID'],
         supports_credentials=False,  # Avoid credential exposure
         max_age=3600)
```

## Logging & Monitoring

### Security Event Logging

```python
import logging
import json
from datetime import datetime

class SecurityLogger:
    """Centralized security event logging."""
    
    def __init__(self):
        self.logger = logging.getLogger('security')
        handler = logging.FileHandler('security.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_authentication_failure(self, user_id: str, ip_address: str, reason: str):
        """Log authentication failures."""
        event = {
            'event_type': 'authentication_failure',
            'user_id': user_id,
            'ip_address': ip_address,
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.warning(json.dumps(event))
    
    def log_suspicious_activity(self, user_id: str, activity: str, details: dict):
        """Log suspicious user activity."""
        event = {
            'event_type': 'suspicious_activity',
            'user_id': user_id,
            'activity': activity,
            'details': details,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.warning(json.dumps(event))
    
    def log_data_access(self, user_id: str, resource: str, action: str):
        """Log data access for audit trails."""
        event = {
            'event_type': 'data_access',
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(event))
```

### Intrusion Detection

```python
from collections import defaultdict, deque
import time

class IntrusionDetector:
    """Real-time intrusion detection system."""
    
    def __init__(self):
        self.failed_attempts = defaultdict(deque)
        self.blocked_ips = set()
        self.max_failures = 5
        self.time_window = 300  # 5 minutes
        self.block_duration = 3600  # 1 hour
    
    def record_failed_attempt(self, ip_address: str):
        """Record failed authentication attempt."""
        now = time.time()
        attempts = self.failed_attempts[ip_address]
        
        # Remove old attempts
        while attempts and attempts[0] < now - self.time_window:
            attempts.popleft()
        
        # Add current attempt
        attempts.append(now)
        
        # Check if should block
        if len(attempts) >= self.max_failures:
            self.block_ip(ip_address)
    
    def block_ip(self, ip_address: str):
        """Block suspicious IP address."""
        self.blocked_ips.add(ip_address)
        logging.warning(f"Blocked IP address: {ip_address}")
        
        # Schedule automatic unblock
        threading.Timer(self.block_duration, self.unblock_ip, args=[ip_address]).start()
    
    def is_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        return ip_address in self.blocked_ips
```

## Incident Response

### Security Incident Classification

**Severity Levels:**
- **Critical**: Data breach, system compromise, credential theft
- **High**: Unauthorized access attempt, DDoS attack, malware detection
- **Medium**: Failed authentication patterns, suspicious user behavior
- **Low**: Policy violations, configuration issues

### Incident Response Procedures

1. **Detection & Analysis**
   - Automated monitoring alerts
   - Log analysis and correlation
   - User reports and feedback

2. **Containment**
   - Isolate affected systems
   - Block malicious IP addresses
   - Revoke compromised credentials

3. **Eradication**
   - Remove malware/backdoors
   - Patch vulnerabilities
   - Update security controls

4. **Recovery**
   - Restore services from clean backups
   - Implement additional monitoring
   - Conduct post-incident testing

5. **Lessons Learned**
   - Document incident timeline
   - Update response procedures
   - Improve detection capabilities

## Compliance & Standards

### GDPR Compliance

- **Data Minimization**: Collect only necessary data
- **Purpose Limitation**: Use data only for stated purposes
- **Consent Management**: Clear opt-in/opt-out mechanisms
- **Right to Deletion**: Complete data removal on request
- **Data Portability**: Export user data in standard formats
- **Breach Notification**: Report incidents within 72 hours

### SOC 2 Type II Controls

- **Security**: Data protection and access controls
- **Availability**: System uptime and disaster recovery
- **Processing Integrity**: Complete and accurate processing
- **Confidentiality**: Protection of confidential information
- **Privacy**: Collection and use of personal information

## Security Testing

### Automated Security Scanning

```bash
# Static Application Security Testing (SAST)
bandit -r fugatto_lab/ -f json -o security-report.json

# Dependency vulnerability scanning
safety check --json --output safety-report.json

# Container security scanning
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image fugatto-audio-lab:latest

# Infrastructure as Code scanning
checkov -f Dockerfile --framework dockerfile
```

### Penetration Testing

**Regular Testing Schedule:**
- **Quarterly**: Automated vulnerability scans
- **Semi-annually**: Professional penetration testing
- **Annually**: Comprehensive security audit

**Testing Scope:**
- Web application security
- API endpoint security
- Authentication/authorization
- Input validation
- Session management
- Data protection

## Security Awareness

### Developer Training

- Secure coding practices
- OWASP Top 10 vulnerabilities
- Threat modeling techniques
- Incident response procedures
- Privacy and compliance requirements

### Security Champions Program

- Designated security advocates per team
- Regular security training and updates
- Code review security checklist
- Security tool integration guidance

## Emergency Contacts

### Security Team
- **Security Lead**: security-lead@company.com
- **Incident Response**: incident-response@company.com
- **24/7 Hotline**: +1-555-SEC-RITY

### External Partners
- **Penetration Testing**: pentest-vendor@company.com
- **Security Consulting**: security-consultant@company.com
- **Legal Counsel**: legal@company.com

## Security Configuration Checklist

### Development Environment
- [ ] Use HTTPS for all communications
- [ ] Implement proper authentication
- [ ] Validate all inputs
- [ ] Encrypt sensitive data
- [ ] Enable security logging
- [ ] Regular dependency updates
- [ ] Code review for security issues

### Production Deployment
- [ ] TLS 1.2+ only
- [ ] Strong authentication (MFA)
- [ ] Rate limiting enabled
- [ ] Intrusion detection active
- [ ] Regular security monitoring
- [ ] Backup and recovery tested
- [ ] Incident response plan ready

### Ongoing Maintenance
- [ ] Security patches applied monthly
- [ ] Vulnerability scans run weekly
- [ ] Log analysis performed daily
- [ ] Access reviews conducted quarterly
- [ ] Security training updated annually
- [ ] Incident response tested bi-annually