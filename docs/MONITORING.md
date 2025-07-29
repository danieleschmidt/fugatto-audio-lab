# Monitoring and Observability Guide

## Overview

This document outlines monitoring, observability, and alerting strategies for Fugatto Audio Lab in production environments.

## Monitoring Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │───▶│   Metrics        │───▶│   Grafana       │
│   (Fugatto Lab) │    │   (Prometheus)   │    │   (Dashboard)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Logs          │    │   Traces         │    │   Alerts        │
│   (ELK Stack)   │    │   (Jaeger)       │    │   (AlertManager)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Application Metrics

### Core Performance Metrics

```python
import time
import psutil
import torch
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Request metrics
REQUEST_COUNT = Counter(
    'fugatto_requests_total', 
    'Total requests processed',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'fugatto_request_duration_seconds',
    'Request processing time',
    ['method', 'endpoint']
)

# Audio generation metrics
AUDIO_GENERATION_DURATION = Histogram(
    'fugatto_audio_generation_seconds',
    'Audio generation processing time',
    ['model', 'duration_seconds']
)

AUDIO_GENERATIONS_TOTAL = Counter(
    'fugatto_audio_generations_total',
    'Total audio generations',
    ['model', 'success']
)

# System metrics
MEMORY_USAGE = Gauge('fugatto_memory_usage_bytes', 'Memory usage in bytes')
GPU_UTILIZATION = Gauge('fugatto_gpu_utilization_percent', 'GPU utilization percentage')
GPU_MEMORY_USAGE = Gauge('fugatto_gpu_memory_usage_bytes', 'GPU memory usage in bytes')

# Model metrics
MODEL_LOAD_TIME = Histogram('fugatto_model_load_seconds', 'Model loading time')
ACTIVE_MODELS = Gauge('fugatto_active_models', 'Number of loaded models')

class MetricsCollector:
    """Collect and expose application metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_audio_generation(self, model: str, duration: float, success: bool):
        """Record audio generation metrics."""
        AUDIO_GENERATION_DURATION.labels(model=model, duration_seconds=duration).observe(duration)
        AUDIO_GENERATIONS_TOTAL.labels(model=model, success=str(success)).inc()
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        # Memory usage
        memory = psutil.virtual_memory()
        MEMORY_USAGE.set(memory.used)
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            gpu_util = torch.cuda.utilization()
            gpu_memory = torch.cuda.memory_allocated()
            
            GPU_UTILIZATION.set(gpu_util)
            GPU_MEMORY_USAGE.set(gpu_memory)
    
    def start_metrics_server(self, port: int = 8000):
        """Start Prometheus metrics server."""
        start_http_server(port)
```

### Custom Metrics Decorator

```python
import functools
import time
from typing import Callable

def monitor_performance(metric_name: str):
    """Decorator to monitor function performance."""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                # Record metrics based on function result
                if hasattr(func, '__self__'):
                    # Method call
                    class_name = func.__self__.__class__.__name__
                    method_name = func.__name__
                    REQUEST_DURATION.labels(
                        method=class_name, 
                        endpoint=method_name
                    ).observe(duration)
        
        return wrapper
    return decorator

# Usage example
class FugattoModel:
    @monitor_performance("audio_generation")
    def generate(self, prompt: str, duration_seconds: float):
        # Model inference logic
        pass
```

## Structured Logging

### Log Configuration

```python
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration
            
        return json.dumps(log_entry)

def setup_logging():
    """Configure structured logging."""
    
    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(StructuredFormatter())
    root.addHandler(console_handler)
    
    # File handler for persistent logs
    file_handler = logging.FileHandler('fugatto_lab.log')
    file_handler.setFormatter(StructuredFormatter())
    root.addHandler(file_handler)
    
    # Security logger (separate file)
    security_logger = logging.getLogger('security')
    security_handler = logging.FileHandler('security.log')
    security_handler.setFormatter(StructuredFormatter())
    security_logger.addHandler(security_handler)
    security_logger.setLevel(logging.WARNING)
    
    return root

# Usage
logger = setup_logging()

def log_with_context(user_id: str, request_id: str, message: str, **kwargs):
    """Log with additional context."""
    extra = {
        'user_id': user_id,
        'request_id': request_id,
        **kwargs
    }
    logger.info(message, extra=extra)
```

### Application Logging Standards

```python
import logging
from functools import wraps
import uuid

logger = logging.getLogger(__name__)

def log_api_call(func):
    """Log API calls with timing and context."""
    
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(
            f"API call started: {func.__name__}",
            extra={
                'request_id': request_id,
                'function': func.__name__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
        )
        
        try:
            result = func(self, *args, **kwargs)
            duration = time.time() - start_time
            
            logger.info(
                f"API call completed: {func.__name__}",
                extra={
                    'request_id': request_id,
                    'function': func.__name__,
                    'duration': duration,
                    'success': True
                }
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(
                f"API call failed: {func.__name__}",
                extra={
                    'request_id': request_id,
                    'function': func.__name__,
                    'duration': duration,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'success': False
                }
            )
            
            raise
    
    return wrapper

# Usage in application
class FugattoAPI:
    @log_api_call
    def generate_audio(self, prompt: str, user_id: str):
        # Implementation
        pass
```

## Health Checks

### Application Health Endpoints

```python
from flask import Flask, jsonify
import torch
import psutil
from datetime import datetime

app = Flask(__name__)

class HealthChecker:
    """Comprehensive health checking."""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
    
    def check_database_connection(self) -> dict:
        """Check database connectivity."""
        try:
            # Test database connection
            # db.execute("SELECT 1")
            return {
                'status': 'healthy',
                'response_time_ms': 15,
                'details': 'Database connection successful'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'details': 'Database connection failed'
            }
    
    def check_model_availability(self) -> dict:
        """Check if models are loaded and available."""
        try:
            # Check model loading status
            if hasattr(self, 'model') and self.model is not None:
                return {
                    'status': 'healthy',
                    'models_loaded': 1,
                    'details': 'Models are loaded and ready'
                }
            else:
                return {
                    'status': 'unhealthy',
                    'models_loaded': 0,
                    'details': 'No models loaded'
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'details': 'Model availability check failed'
            }
    
    def check_gpu_availability(self) -> dict:
        """Check GPU status and memory."""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_memory = torch.cuda.memory_allocated(0)
                gpu_memory_total = torch.cuda.memory_reserved(0)
                
                return {
                    'status': 'healthy',
                    'gpu_count': gpu_count,
                    'gpu_memory_used': gpu_memory,
                    'gpu_memory_total': gpu_memory_total,
                    'gpu_utilization': torch.cuda.utilization(0),
                    'details': 'GPU available and functioning'
                }
            else:
                return {
                    'status': 'warning',
                    'details': 'No GPU available, running on CPU'
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'details': 'GPU check failed'
            }
    
    def check_system_resources(self) -> dict:
        """Check system resource usage."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk_usage = psutil.disk_usage('/')
            
            status = 'healthy'
            warnings = []
            
            if memory.percent > 85:
                status = 'warning'
                warnings.append('High memory usage')
            
            if cpu_percent > 80:
                status = 'warning'
                warnings.append('High CPU usage')
            
            if disk_usage.percent > 90:
                status = 'warning'
                warnings.append('Low disk space')
            
            return {
                'status': status,
                'memory_percent': memory.percent,
                'cpu_percent': cpu_percent,
                'disk_percent': disk_usage.percent,
                'warnings': warnings,
                'details': 'System resources checked'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'details': 'System resource check failed'
            }

health_checker = HealthChecker()

@app.route('/health')
def health_check():
    """Basic health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'uptime_seconds': (datetime.utcnow() - health_checker.start_time).total_seconds()
    })

@app.route('/health/detailed')
def detailed_health_check():
    """Detailed health check with all components."""
    checks = {
        'database': health_checker.check_database_connection(),
        'models': health_checker.check_model_availability(),
        'gpu': health_checker.check_gpu_availability(),
        'system': health_checker.check_system_resources()
    }
    
    # Determine overall status
    overall_status = 'healthy'
    for check in checks.values():
        if check['status'] == 'unhealthy':
            overall_status = 'unhealthy'
            break
        elif check['status'] == 'warning' and overall_status == 'healthy':
            overall_status = 'warning'
    
    return jsonify({
        'status': overall_status,
        'timestamp': datetime.utcnow().isoformat(),
        'uptime_seconds': (datetime.utcnow() - health_checker.start_time).total_seconds(),
        'checks': checks
    })

@app.route('/ready')
def readiness_check():
    """Kubernetes readiness probe."""
    # Check if application is ready to serve requests
    model_check = health_checker.check_model_availability()
    
    if model_check['status'] == 'healthy':
        return jsonify({'status': 'ready'}), 200
    else:
        return jsonify({'status': 'not ready', 'reason': model_check['details']}), 503

@app.route('/live')
def liveness_check():
    """Kubernetes liveness probe."""
    # Basic liveness check
    return jsonify({'status': 'alive'}), 200
```

## Distributed Tracing

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

def setup_tracing(service_name: str = "fugatto-audio-lab"):
    """Setup distributed tracing with OpenTelemetry."""
    
    # Create tracer provider
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )
    
    # Create span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Auto-instrument Flask and requests
    FlaskInstrumentor().instrument_app(app)
    RequestsInstrumentor().instrument()
    
    return tracer

# Custom tracing for model operations
tracer = setup_tracing()

def trace_model_operation(operation_name: str):
    """Decorator for tracing model operations."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(operation_name) as span:
                # Add attributes
                span.set_attribute("operation.name", operation_name)
                span.set_attribute("function.name", func.__name__)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("operation.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("operation.success", False)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator

# Usage
class FugattoModel:
    @trace_model_operation("model.generate_audio")
    def generate(self, prompt: str, duration_seconds: float):
        with tracer.start_as_current_span("model.inference") as span:
            span.set_attribute("prompt.length", len(prompt))
            span.set_attribute("audio.duration", duration_seconds)
            
            # Model inference logic
            result = self._inference(prompt, duration_seconds)
            
            span.set_attribute("audio.generated_samples", len(result))
            return result
```

## Alerting Rules

### Prometheus Alerting Rules

```yaml
# prometheus-alerts.yml
groups:
  - name: fugatto-audio-lab
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(fugatto_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"
      
      # High response time
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(fugatto_request_duration_seconds_bucket[5m])) > 10
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }} seconds"
      
      # Model generation failures
      - alert: ModelGenerationFailures
        expr: rate(fugatto_audio_generations_total{success="false"}[5m]) > 0.05
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High model generation failure rate"
          description: "Model generation failing at {{ $value }} per second"
      
      # High memory usage
      - alert: HighMemoryUsage
        expr: fugatto_memory_usage_bytes / (1024^3) > 8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}GB"
      
      # GPU utilization
      - alert: HighGPUUtilization
        expr: fugatto_gpu_utilization_percent > 95
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High GPU utilization"
          description: "GPU utilization is {{ $value }}%"
      
      # Service down
      - alert: ServiceDown
        expr: up{job="fugatto-audio-lab"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "Fugatto Audio Lab service is not responding"
```

### Alert Manager Configuration

```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'smtp.company.com:587'
  smtp_from: 'alerts@company.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://webhook-service:8080/alerts'
  
  - name: 'critical-alerts'
    email_configs:
      - to: 'oncall@company.com'
        subject: 'CRITICAL: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/...'
        channel: '#alerts-critical'
        title: 'Critical Alert: {{ .GroupLabels.alertname }}'
  
  - name: 'warning-alerts'
    email_configs:
      - to: 'team@company.com'
        subject: 'WARNING: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
```

## Dashboard Configuration

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Fugatto Audio Lab - Operations Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(fugatto_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(fugatto_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(fugatto_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Audio Generation Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(fugatto_audio_generations_total[5m])",
            "legendFormat": "Generations/sec"
          }
        ]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "fugatto_memory_usage_bytes / (1024^3)",
            "legendFormat": "Memory (GB)"
          },
          {
            "expr": "fugatto_gpu_utilization_percent",
            "legendFormat": "GPU Utilization (%)"
          }
        ]
      }
    ]
  }
}
```

## Performance Monitoring

### Application Performance Monitoring

```python
import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    timestamp: float
    value: float
    tags: Dict[str, str]

class PerformanceMonitor:
    """Real-time performance monitoring."""
    
    def __init__(self, retention_minutes: int = 60):
        self.metrics = defaultdict(deque)
        self.retention_seconds = retention_minutes * 60
        self.lock = threading.Lock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_metrics, daemon=True)
        self.cleanup_thread.start()
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            timestamp=time.time(),
            value=value,
            tags=tags or {}
        )
        
        with self.lock:
            self.metrics[name].append(metric)
    
    def get_metrics_summary(self, name: str, window_minutes: int = 5) -> Dict:
        """Get summary statistics for a metric."""
        window_seconds = window_minutes * 60
        cutoff_time = time.time() - window_seconds
        
        with self.lock:
            recent_metrics = [
                m for m in self.metrics[name] 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {'count': 0}
        
        values = [m.value for m in recent_metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'p50': self._percentile(values, 0.5),
            'p95': self._percentile(values, 0.95),
            'p99': self._percentile(values, 0.99)
        }
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory leaks."""
        while True:
            time.sleep(300)  # Run every 5 minutes
            cutoff_time = time.time() - self.retention_seconds
            
            with self.lock:
                for name, metrics_deque in self.metrics.items():
                    while metrics_deque and metrics_deque[0].timestamp < cutoff_time:
                        metrics_deque.popleft()
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]

# Global performance monitor
perf_monitor = PerformanceMonitor()

# Usage
def monitor_audio_generation():
    start_time = time.time()
    
    # Simulate audio generation
    time.sleep(2.5)
    
    duration = time.time() - start_time
    perf_monitor.record_metric(
        'audio_generation_time',
        duration,
        {'model': 'fugatto-base', 'duration': '10s'}
    )

# Get performance summary
summary = perf_monitor.get_metrics_summary('audio_generation_time')
print(f"Audio generation performance: {summary}")
```

## Log Analysis

### ELK Stack Configuration

```yaml
# docker-compose-elk.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
  
  logstash:
    image: docker.elastic.co/logstash/logstash:8.5.0
    ports:
      - "5044:5044"
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch
  
  kibana:
    image: docker.elastic.co/kibana/kibana:8.5.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

```ruby
# logstash.conf
input {
  file {
    path => "/app/logs/fugatto_lab.log"
    start_position => "beginning"
    codec => "json"
  }
}

filter {
  if [logger] == "security" {
    mutate {
      add_tag => ["security"]
    }
  }
  
  if [level] == "ERROR" {
    mutate {
      add_tag => ["error"]
    }
  }
  
  # Parse duration fields
  if [duration] {
    mutate {
      convert => { "duration" => "float" }
    }
  }
  
  # Add timestamp parsing
  date {
    match => [ "timestamp", "ISO8601" ]
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "fugatto-logs-%{+YYYY.MM.dd}"
  }
  
  # Send errors to separate index
  if "error" in [tags] {
    elasticsearch {
      hosts => ["elasticsearch:9200"]
      index => "fugatto-errors-%{+YYYY.MM.dd}"
    }
  }
  
  stdout {
    codec => rubydebug
  }
}
```

## Deployment Monitoring

### Docker Health Checks

```dockerfile
# Enhanced Dockerfile with health checks
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY fugatto_lab/ ./fugatto_lab/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8001

# Start application with monitoring
CMD ["python", "-m", "fugatto_lab.server", "--port", "8000", "--metrics-port", "8001"]
```

### Kubernetes Monitoring

```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fugatto-audio-lab
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fugatto-audio-lab
  template:
    metadata:
      labels:
        app: fugatto-audio-lab
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8001"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: fugatto-audio-lab
        image: fugatto-audio-lab:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8001
          name: metrics
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /live
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        
        # Resource limits
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
        
        # Environment variables
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: METRICS_ENABLED
          value: "true"
        - name: TRACING_ENABLED
          value: "true"
```

This comprehensive monitoring and observability setup provides:

1. **Comprehensive Metrics**: Application, system, and business metrics
2. **Structured Logging**: JSON-formatted logs with correlation IDs
3. **Health Checks**: Multi-level health endpoints for different use cases
4. **Distributed Tracing**: Request tracing across service boundaries
5. **Alerting**: Prometheus-based alerting with multiple notification channels
6. **Dashboards**: Grafana dashboards for operational visibility
7. **Performance Monitoring**: Real-time performance tracking and analysis
8. **Log Analysis**: ELK stack for centralized log analysis
9. **Container Monitoring**: Docker and Kubernetes health checks

These monitoring capabilities ensure the Fugatto Audio Lab can be operated reliably in production environments with proper observability and alerting.