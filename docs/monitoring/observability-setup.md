# Observability and Monitoring Setup

This document provides comprehensive monitoring and observability setup for Fugatto Audio Lab.

## Monitoring Stack Overview

- **Metrics**: Prometheus + Grafana
- **Logging**: Structured logging with JSON format
- **Tracing**: OpenTelemetry (optional)
- **Alerting**: Prometheus Alertmanager
- **Health Checks**: Built-in health endpoints

## 1. Application Metrics

### Custom Metrics Implementation

Create `fugatto_lab/monitoring/metrics.py`:

```python
"""Application metrics collection for Fugatto Audio Lab."""

import time
import logging
from typing import Dict, Any, Optional
from functools import wraps
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]


class MetricsCollector:
    """Collect and expose application metrics."""
    
    def __init__(self):
        self.counters = Counter()
        self.gauges = {}
        self.histograms = defaultdict(list)
        self.logger = logging.getLogger("metrics")
    
    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        key = self._make_key(name, labels or {})
        self.counters[key] += value
        
        self.logger.debug(f"Counter incremented: {key} += {value}")
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        key = self._make_key(name, labels or {})
        self.gauges[key] = value
        
        self.logger.debug(f"Gauge set: {key} = {value}")
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value in a histogram."""
        key = self._make_key(name, labels or {})
        self.histograms[key].append(value)
        
        # Keep only recent observations (last 1000)
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
        
        self.logger.debug(f"Histogram observed: {key} = {value}")
    
    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create metric key with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Counters
        for key, value in self.counters.items():
            lines.append(f"fugatto_{key} {value}")
        
        # Gauges
        for key, value in self.gauges.items():
            lines.append(f"fugatto_{key} {value}")
        
        # Histograms (simplified - just avg, min, max)
        for key, values in self.histograms.items():
            if values:
                lines.append(f"fugatto_{key}_avg {sum(values) / len(values)}")
                lines.append(f"fugatto_{key}_min {min(values)}")
                lines.append(f"fugatto_{key}_max {max(values)}")
                lines.append(f"fugatto_{key}_count {len(values)}")
        
        return "\n".join(lines)


# Global metrics instance
metrics = MetricsCollector()


def track_generation_metrics(func):
    """Decorator to track audio generation metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Extract labels from arguments
        labels = {
            "model": getattr(args[0], "model_name", "unknown") if args else "unknown",
            "duration": str(kwargs.get("duration_seconds", 0))
        }
        
        try:
            result = func(*args, **kwargs)
            
            # Track successful generation
            generation_time = time.time() - start_time
            metrics.increment_counter("audio_generations_total", labels={**labels, "status": "success"})
            metrics.observe_histogram("audio_generation_duration_seconds", generation_time, labels)
            
            # Track audio properties if available
            if hasattr(result, "shape") and result is not None:
                audio_length = result.shape[-1] / 48000  # Assume 48kHz
                metrics.observe_histogram("generated_audio_length_seconds", audio_length, labels)
            
            return result
            
        except Exception as e:
            # Track failed generation
            metrics.increment_counter("audio_generations_total", labels={**labels, "status": "error"})
            metrics.increment_counter("audio_generation_errors_total", labels={**labels, "error_type": type(e).__name__})
            raise
    
    return wrapper


def track_api_metrics(func):
    """Decorator to track API endpoint metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Extract endpoint info
        endpoint = func.__name__
        method = kwargs.get("method", "unknown")
        
        try:
            result = func(*args, **kwargs)
            
            # Track successful request
            duration = time.time() - start_time
            metrics.increment_counter("api_requests_total", labels={
                "endpoint": endpoint,
                "method": method,
                "status": "200"
            })
            metrics.observe_histogram("api_request_duration_seconds", duration, labels={
                "endpoint": endpoint,
                "method": method
            })
            
            return result
            
        except Exception as e:
            # Track failed request
            metrics.increment_counter("api_requests_total", labels={
                "endpoint": endpoint,
                "method": method,
                "status": "error"
            })
            raise
    
    return wrapper
```

### Health Check Endpoint

Create `fugatto_lab/health.py`:

```python
"""Health check endpoints for monitoring."""

import time
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthChecker:
    """Application health monitoring."""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_health_check = None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        now = datetime.utcnow()
        uptime = time.time() - self.start_time
        
        health_data = {
            "status": "healthy",
            "timestamp": now.isoformat() + "Z",
            "uptime_seconds": uptime,
            "version": "0.1.0",
            "checks": {}
        }
        
        # Check model availability
        try:
            # Mock model check - in real implementation, verify model loading
            health_data["checks"]["model"] = {
                "status": "healthy",
                "message": "Model loaded successfully"
            }
        except Exception as e:
            health_data["checks"]["model"] = {
                "status": "unhealthy",
                "message": f"Model check failed: {str(e)}"
            }
            health_data["status"] = "unhealthy"
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_percent = (free / total) * 100
            
            if free_percent < 10:
                health_data["checks"]["disk"] = {
                    "status": "warning",
                    "message": f"Low disk space: {free_percent:.1f}% free"
                }
            else:
                health_data["checks"]["disk"] = {
                    "status": "healthy",
                    "message": f"Disk space OK: {free_percent:.1f}% free"
                }
        except Exception as e:
            health_data["checks"]["disk"] = {
                "status": "unknown",
                "message": f"Disk check failed: {str(e)}"
            }
        
        # Check memory usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                health_data["checks"]["memory"] = {
                    "status": "warning",
                    "message": f"High memory usage: {memory.percent:.1f}%"
                }
            else:
                health_data["checks"]["memory"] = {
                    "status": "healthy",
                    "message": f"Memory usage OK: {memory.percent:.1f}%"
                }
        except Exception as e:
            health_data["checks"]["memory"] = {
                "status": "unknown",
                "message": f"Memory check failed: {str(e)}"
            }
        
        self.last_health_check = now
        return health_data
    
    def get_readiness_status(self) -> Dict[str, Any]:
        """Get readiness status for load balancer."""
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "ready": True
        }


# Global health checker
health_checker = HealthChecker()
```

## 2. Logging Configuration

### Structured Logging Setup

Create `fugatto_lab/logging_config.py`:

```python
"""Structured logging configuration."""

import logging
import logging.config
import json
import sys
from datetime import datetime
from typing import Dict, Any


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", "pathname", 
                          "filename", "module", "exc_info", "exc_text", "stack_info",
                          "lineno", "funcName", "created", "msecs", "relativeCreated",
                          "thread", "threadName", "processName", "process", "getMessage"]:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


def setup_logging(log_level: str = "INFO", json_logs: bool = True) -> None:
    """Setup application logging configuration."""
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JSONFormatter,
            },
            "simple": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "json" if json_logs else "simple",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "json" if json_logs else "simple",
                "filename": "logs/fugatto-lab.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            }
        },
        "loggers": {
            "fugatto_lab": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False
            },
            "security": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "metrics": {
                "level": "DEBUG",
                "handlers": ["file"],
                "propagate": False
            }
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"]
        }
    }
    
    # Create logs directory
    import os
    os.makedirs("logs", exist_ok=True)
    
    logging.config.dictConfig(config)
```

## 3. Prometheus Configuration

Create `configs/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'fugatto-lab'
    static_configs:
      - targets: ['fugatto-lab:8080']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

### Alert Rules

Create `configs/alert_rules.yml`:

```yaml
groups:
  - name: fugatto-lab-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(fugatto_audio_generation_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in audio generation"
          description: "Error rate is {{ $value }} errors per second"
      
      - alert: LongGenerationTime
        expr: fugatto_audio_generation_duration_seconds_avg > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Audio generation taking too long"
          description: "Average generation time is {{ $value }} seconds"
      
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage"
          description: "Memory usage is above 90%"
      
      - alert: ServiceDown
        expr: up{job="fugatto-lab"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Fugatto Lab service is down"
          description: "The Fugatto Lab service has been down for more than 1 minute"
```

## 4. Grafana Dashboard

Create `configs/grafana/dashboards/fugatto-lab.json`:

```json
{
  "dashboard": {
    "id": null,
    "title": "Fugatto Audio Lab",
    "tags": ["fugatto", "audio", "ml"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Audio Generations",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(fugatto_audio_generations_total[5m])",
            "legendFormat": "Generations/sec"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(fugatto_audio_generation_errors_total[5m])",
            "legendFormat": "Errors/sec"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Generation Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "fugatto_audio_generation_duration_seconds_avg",
            "legendFormat": "Average Duration"
          }
        ],
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

## 5. Implementation Steps

1. **Add metrics collection to core functions**:
   ```python
   from fugatto_lab.monitoring.metrics import track_generation_metrics
   
   @track_generation_metrics
   def generate_audio(self, prompt: str, **kwargs):
       # Implementation
   ```

2. **Add health endpoints to web server**:
   ```python
   from fugatto_lab.health import health_checker
   
   @app.route('/health')
   def health():
       return health_checker.get_health_status()
   
   @app.route('/ready')  
   def ready():
       return health_checker.get_readiness_status()
   
   @app.route('/metrics')
   def metrics():
       return metrics.get_prometheus_metrics()
   ```

3. **Deploy monitoring stack**:
   ```bash
   docker-compose --profile monitoring up -d
   ```

4. **Access dashboards**:
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090

## Maintenance

- **Daily**: Check alert status and resolve issues
- **Weekly**: Review metrics trends and capacity planning
- **Monthly**: Update dashboards and alert thresholds
- **Quarterly**: Conduct monitoring system health review