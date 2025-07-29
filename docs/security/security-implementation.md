# Security Implementation Guide

This document outlines the security measures and compliance setup for Fugatto Audio Lab.

## Security Scanning Setup

### 1. Dependency Vulnerability Scanning

#### pip-audit Integration
```bash
# Install pip-audit for dependency scanning
pip install pip-audit

# Basic vulnerability scan
pip-audit

# Generate reports
pip-audit --format=json --output=security-report.json
pip-audit --format=cyclonedx-json --output=sbom.json
```

#### GitHub Actions Integration
Add to `.github/workflows/security.yml`:

```yaml
name: Security Scan

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * 1'  # Weekly scan

jobs:
  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
        
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
        pip install pip-audit bandit[toml] safety
        
    - name: Run pip-audit
      run: |
        pip-audit --format=json --output=pip-audit-report.json
        
    - name: Run Bandit security linter
      run: |
        bandit -r fugatto_lab/ -f json -o bandit-report.json -c pyproject.toml
        
    - name: Run Safety check
      run: |
        safety check --json --output safety-report.json
        
    - name: Upload security artifacts
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          pip-audit-report.json
          bandit-report.json
          safety-report.json
```

### 2. Static Application Security Testing (SAST)

#### Bandit Configuration
Add to `pyproject.toml`:

```toml
[tool.bandit]
exclude_dirs = ["tests", "docs", "build", "dist"]
skips = ["B101", "B601"]  # Skip assert_used_in_tests, shell_injection_subprocess_popen

[tool.bandit.assert_used]
skips = ["**/test_*.py", "**/conftest.py"]
```

#### CodeQL Analysis
Create `.github/workflows/codeql.yml`:

```yaml
name: CodeQL Security Analysis

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 2'  # Weekly on Tuesday

jobs:
  analyze:
    name: Analyze
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write
      
    strategy:
      fail-fast: false
      matrix:
        language: ['python']
        
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
      
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: ${{ matrix.language }}
        
    - name: Autobuild
      uses: github/codeql-action/autobuild@v2
      
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
```

### 3. Container Security

#### Docker Security Scanning
```dockerfile
# Multi-stage Dockerfile with security focus
FROM python:3.10-slim AS base

# Create non-root user
RUN groupadd --gid 1000 fugatto && \
    useradd --uid 1000 --gid fugatto --shell /bin/bash --create-home fugatto

# Security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        && rm -rf /var/lib/apt/lists/*

FROM base AS builder
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM base AS runtime
COPY --from=builder /home/fugatto/.local /home/fugatto/.local

# Switch to non-root user
USER fugatto
WORKDIR /home/fugatto/app

# Copy application
COPY --chown=fugatto:fugatto . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860
CMD ["python", "-m", "fugatto_lab.server"]
```

#### Container Scanning
```yaml
# Add to CI pipeline
- name: Build and scan Docker image
  run: |
    docker build -t fugatto-audio-lab:latest .
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
      aquasec/trivy image fugatto-audio-lab:latest
```

## Compliance and Governance

### 1. SBOM Generation

Create `scripts/generate-sbom.py`:

```python
#!/usr/bin/env python3
"""Generate Software Bill of Materials (SBOM) for Fugatto Audio Lab."""

import json
import subprocess
from datetime import datetime
from pathlib import Path


def generate_sbom():
    """Generate SBOM in CycloneDX format."""
    
    # Run pip-audit to get dependency info
    result = subprocess.run([
        "pip-audit", "--format=cyclonedx-json", "--output=-"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error generating SBOM: {result.stderr}")
        return
    
    sbom_data = json.loads(result.stdout)
    
    # Add metadata
    sbom_data["metadata"] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tools": [
            {
                "vendor": "pip-audit",
                "name": "pip-audit",
                "version": "2.6.1"
            }
        ]
    }
    
    # Write SBOM file
    sbom_path = Path("sbom.json")
    with open(sbom_path, "w") as f:
        json.dump(sbom_data, f, indent=2)
    
    print(f"SBOM generated: {sbom_path}")


if __name__ == "__main__":
    generate_sbom()
```

### 2. License Compliance

Create `scripts/check-licenses.py`:

```python
#!/usr/bin/env python3
"""Check license compliance for all dependencies."""

import subprocess
import json
from collections import defaultdict


def check_licenses():
    """Check licenses of all dependencies."""
    
    # Get package info
    result = subprocess.run([
        "pip", "list", "--format=json"
    ], capture_output=True, text=True)
    
    packages = json.loads(result.stdout)
    
    # Approved licenses
    approved_licenses = {
        "MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause",
        "ISC", "Python Software Foundation License"
    }
    
    license_groups = defaultdict(list)
    
    for package in packages:
        name = package["name"]
        version = package["version"]
        
        # Get package metadata
        try:
            result = subprocess.run([
                "pip", "show", name
            ], capture_output=True, text=True)
            
            license_line = [line for line in result.stdout.split('\n') 
                          if line.startswith('License:')]
            
            if license_line:
                license_name = license_line[0].replace('License: ', '').strip()
                license_groups[license_name].append(f"{name}=={version}")
            else:
                license_groups["Unknown"].append(f"{name}=={version}")
                
        except Exception as e:
            print(f"Error checking {name}: {e}")
            license_groups["Error"].append(f"{name}=={version}")
    
    # Report findings
    print("License Compliance Report")
    print("=" * 50)
    
    for license_name, packages in license_groups.items():
        status = "✓ APPROVED" if license_name in approved_licenses else "⚠ REVIEW NEEDED"
        print(f"\n{license_name} ({len(packages)} packages) - {status}")
        for package in sorted(packages):
            print(f"  - {package}")
    
    # Check for problematic licenses
    problematic = {k: v for k, v in license_groups.items() 
                  if k not in approved_licenses and k not in ["Unknown", "Error"]}
    
    if problematic:
        print(f"\n⚠ WARNING: {len(problematic)} license types need review")
        return False
    
    return True


if __name__ == "__main__":
    success = check_licenses()
    exit(0 if success else 1)
```

### 3. Secrets Management

#### Environment Variable Security
```python
# fugatto_lab/config/security.py
"""Security configuration and secrets management."""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SecureConfig:
    """Secure configuration management."""
    
    def __init__(self):
        self._sensitive_keys = {
            'api_key', 'secret', 'password', 'token', 'private_key'
        }
    
    def get_env_var(self, key: str, default: Optional[str] = None) -> str:
        """Get environment variable with security checks."""
        value = os.getenv(key, default)
        
        # Log access to sensitive variables (without values)
        if any(sensitive in key.lower() for sensitive in self._sensitive_keys):
            logger.info(f"Accessing sensitive environment variable: {key}")
            
        return value
    
    def validate_secrets(self) -> bool:
        """Validate that required secrets are set."""
        required_secrets = [
            'FUGATTO_MODEL_API_KEY',
            'FUGATTO_ENCRYPTION_KEY'
        ]
        
        missing = []
        for secret in required_secrets:
            if not os.getenv(secret):
                missing.append(secret)
        
        if missing:
            logger.error(f"Missing required secrets: {missing}")
            return False
        
        return True
```

#### Git Secrets Prevention
Add to `.gitignore`:

```gitignore
# Security
.env
.env.local
.env.production
secrets/
*.key
*.pem
*.p12
config/local.yaml
api_keys.json

# Sensitive data
models/private/
datasets/private/
user_data/
```

### 4. Monitoring and Alerting

#### Security Monitoring Setup
```python
# fugatto_lab/monitoring/security.py
"""Security monitoring and alerting."""

import logging
import hashlib
from datetime import datetime
from typing import Dict, Any


class SecurityMonitor:
    """Monitor security events and anomalies."""
    
    def __init__(self):
        self.logger = logging.getLogger("security")
        self.failed_attempts = {}
    
    def log_authentication_attempt(self, user_id: str, success: bool, ip_address: str):
        """Log authentication attempts."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "authentication",
            "user_id": hashlib.sha256(user_id.encode()).hexdigest()[:16],
            "success": success,
            "ip_address": self._anonymize_ip(ip_address)
        }
        
        if success:
            self.logger.info(f"Successful authentication: {event}")
        else:
            self.logger.warning(f"Failed authentication: {event}")
            self._track_failed_attempt(user_id, ip_address)
    
    def _anonymize_ip(self, ip_address: str) -> str:
        """Anonymize IP address for privacy."""
        if ':' in ip_address:  # IPv6
            return ip_address.split(':')[0] + ":xxxx:xxxx:xxxx"
        else:  # IPv4
            parts = ip_address.split('.')
            return f"{parts[0]}.{parts[1]}.xxx.xxx"
    
    def _track_failed_attempt(self, user_id: str, ip_address: str):
        """Track failed authentication attempts."""
        key = f"{user_id}:{ip_address}"
        
        if key not in self.failed_attempts:
            self.failed_attempts[key] = {"count": 0, "first_attempt": datetime.utcnow()}
        
        self.failed_attempts[key]["count"] += 1
        
        # Alert on suspicious activity
        if self.failed_attempts[key]["count"] >= 5:
            self.logger.critical(f"Potential brute force attack detected: {key}")
```

## Implementation Checklist

- [ ] Set up pip-audit for dependency scanning
- [ ] Configure Bandit for static security analysis
- [ ] Enable CodeQL analysis in GitHub Actions
- [ ] Implement container security scanning
- [ ] Set up SBOM generation
- [ ] Configure license compliance checking
- [ ] Implement secrets management
- [ ] Set up security monitoring
- [ ] Configure automated security updates
- [ ] Document incident response procedures

## Security Maintenance

### Weekly Tasks
- Review security scan reports
- Update vulnerable dependencies
- Check license compliance

### Monthly Tasks
- Review security logs and incidents
- Update security configurations
- Conduct security training

### Quarterly Tasks
- Security architecture review
- Penetration testing
- Compliance audit

## Incident Response

1. **Detection**: Automated alerts from monitoring
2. **Assessment**: Determine severity and impact
3. **Containment**: Isolate affected systems
4. **Eradication**: Remove threats and vulnerabilities
5. **Recovery**: Restore normal operations
6. **Lessons Learned**: Document and improve processes