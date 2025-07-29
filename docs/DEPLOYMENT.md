# Deployment Guide

## Overview

This guide covers deployment strategies, infrastructure requirements, and operational procedures for Fugatto Audio Lab.

## Deployment Options

### 1. Local Development

```bash
# Quick development setup
git clone https://github.com/yourusername/fugatto-audio-lab.git
cd fugatto-audio-lab

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Start development server
python -m fugatto_lab.server --port 8000 --debug
```

### 2. Docker Deployment

#### Single Container

```bash
# Build image
docker build -t fugatto-audio-lab:latest .

# Run container
docker run -d \
  --name fugatto-lab \
  --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -e LOG_LEVEL=INFO \
  fugatto-audio-lab:latest
```

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  fugatto-lab:
    build: .
    ports:
      - "8000:8000"
      - "8001:8001"
    environment:
      - LOG_LEVEL=INFO
      - WORKERS=4
      - MODEL_CACHE_DIR=/app/models
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - fugatto-lab
    restart: unless-stopped

volumes:
  redis_data:
```

### 3. Kubernetes Deployment

#### Namespace and ConfigMap

```yaml
# k8s/namespace.yml
apiVersion: v1
kind: Namespace
metadata:
  name: fugatto-lab

---
# k8s/configmap.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fugatto-config
  namespace: fugatto-lab
data:
  LOG_LEVEL: "INFO"
  WORKERS: "4"
  MODEL_CACHE_DIR: "/app/models"
  REDIS_URL: "redis://redis-service:6379"
```

#### Secrets

```yaml
# k8s/secrets.yml
apiVersion: v1
kind: Secret
metadata:
  name: fugatto-secrets
  namespace: fugatto-lab
type: Opaque
data:
  API_SECRET_KEY: <base64-encoded-secret>
  DATABASE_URL: <base64-encoded-db-url>
  ENCRYPTION_KEY: <base64-encoded-encryption-key>
```

#### Persistent Volumes

```yaml
# k8s/storage.yml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage
  namespace: fugatto-lab
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-storage
  namespace: fugatto-lab
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: standard
```

#### Deployment

```yaml
# k8s/deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fugatto-audio-lab
  namespace: fugatto-lab
  labels:
    app: fugatto-audio-lab
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
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
      serviceAccountName: fugatto-service-account
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      
      initContainers:
      - name: model-downloader
        image: fugatto-audio-lab:latest
        command: ["python", "-m", "fugatto_lab.download_models"]
        env:
        - name: MODEL_CACHE_DIR
          value: "/app/models"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      
      containers:
      - name: fugatto-audio-lab
        image: fugatto-audio-lab:latest
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        - containerPort: 8001
          name: metrics
          protocol: TCP
        
        env:
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: fugatto-config
              key: LOG_LEVEL
        - name: WORKERS
          valueFrom:
            configMapKeyRef:
              name: fugatto-config
              key: WORKERS
        - name: API_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: fugatto-secrets
              key: API_SECRET_KEY
        
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
          readOnly: true
        - name: data-storage
          mountPath: /app/data
        
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "8000m"
            nvidia.com/gpu: 1
        
        livenessProbe:
          httpGet:
            path: /live
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]
      
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage
      - name: data-storage
        persistentVolumeClaim:
          claimName: data-storage
      
      terminationGracePeriodSeconds: 60
      
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - fugatto-audio-lab
              topologyKey: kubernetes.io/hostname
```

#### Service and Ingress

```yaml
# k8s/service.yml
apiVersion: v1
kind: Service
metadata:
  name: fugatto-service
  namespace: fugatto-lab
  labels:
    app: fugatto-audio-lab
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  - port: 8001
    targetPort: 8001
    protocol: TCP
    name: metrics
  selector:
    app: fugatto-audio-lab

---
# k8s/ingress.yml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fugatto-ingress
  namespace: fugatto-lab
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  tls:
  - hosts:
    - api.fugatto-lab.com
    secretName: fugatto-tls
  rules:
  - host: api.fugatto-lab.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: fugatto-service
            port:
              number: 80
```

### 4. Cloud Deployments

#### AWS ECS with Fargate

```json
{
  "family": "fugatto-audio-lab",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "16384",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "fugatto-audio-lab",
      "image": "your-account.dkr.ecr.region.amazonaws.com/fugatto-audio-lab:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        },
        {
          "name": "AWS_DEFAULT_REGION",
          "value": "us-west-2"
        }
      ],
      "secrets": [
        {
          "name": "API_SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:fugatto/api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/fugatto-audio-lab",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### Google Cloud Run

```yaml
# cloudrun.yml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: fugatto-audio-lab
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/execution-environment: gen2
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "8Gi"
        run.googleapis.com/cpu: "4"
        run.googleapis.com/gpu-type: "nvidia-tesla-t4"
        run.googleapis.com/gpu-count: "1"
    spec:
      containerConcurrency: 10
      timeoutSeconds: 300
      containers:
      - image: gcr.io/project-id/fugatto-audio-lab:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: GOOGLE_CLOUD_PROJECT
          value: "your-project-id"
        resources:
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 10
```

## Infrastructure Requirements

### Minimum System Requirements

- **CPU**: 8 cores (Intel Xeon or AMD EPYC recommended)
- **RAM**: 16GB (32GB recommended for production)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3080/A10 or better)
- **Storage**: 100GB SSD for models and cache
- **Network**: 1Gbps bandwidth for API serving

### Production Recommendations

- **CPU**: 16+ cores with high single-thread performance
- **RAM**: 64GB+ for multiple concurrent requests
- **GPU**: Multiple GPUs (A100, H100) for scaling
- **Storage**: NVMe SSD with high IOPS for model loading
- **Network**: Load balancer with SSL termination
- **Monitoring**: Prometheus, Grafana, and alerting setup

### Cloud Instance Types

#### AWS
- **Development**: `g4dn.xlarge` (4 vCPU, 16GB RAM, T4 GPU)
- **Production**: `p4d.2xlarge` (8 vCPU, 64GB RAM, A100 GPU)
- **High-scale**: `p4d.24xlarge` (96 vCPU, 1.1TB RAM, 8x A100)

#### Google Cloud
- **Development**: `n1-standard-4` + `nvidia-tesla-t4`
- **Production**: `a2-highgpu-1g` (12 vCPU, 85GB RAM, A100)
- **High-scale**: `a2-megagpu-16g` (96 vCPU, 1.4TB RAM, 16x A100)

#### Azure
- **Development**: `Standard_NC6s_v3` (6 vCPU, 112GB RAM, V100)
- **Production**: `Standard_ND40rs_v2` (40 vCPU, 672GB RAM, 8x V100)

## Environment Configuration

### Environment Variables

```bash
# Core application settings
LOG_LEVEL=INFO
DEBUG=false
WORKERS=4
HOST=0.0.0.0
PORT=8000
METRICS_PORT=8001

# Model configuration
MODEL_CACHE_DIR=/app/models
DEFAULT_MODEL=nvidia/fugatto-base
MODEL_DOWNLOAD_TIMEOUT=3600

# Performance settings
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=300
MODEL_MEMORY_LIMIT=8GB
ENABLE_MODEL_CACHING=true

# Security settings
API_SECRET_KEY=your-secret-key
CORS_ORIGINS=https://your-domain.com
RATE_LIMIT_PER_MINUTE=60
ENABLE_API_KEY_AUTH=true

# Database (if using)
DATABASE_URL=postgresql://user:pass@host:5432/fugatto
REDIS_URL=redis://localhost:6379/0

# Monitoring and observability
ENABLE_METRICS=true
ENABLE_TRACING=true
JAEGER_ENDPOINT=http://localhost:14268/api/traces
SENTRY_DSN=https://your-sentry-dsn

# Cloud storage (optional)
AWS_S3_BUCKET=fugatto-models
GCS_BUCKET=fugatto-models
AZURE_STORAGE_ACCOUNT=fugattomodels
```

### Configuration File

```yaml
# config/production.yml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 300
  max_concurrent_requests: 10

models:
  cache_dir: "/app/models"
  default_model: "nvidia/fugatto-base"
  memory_limit: "8GB"
  enable_caching: true
  download_timeout: 3600

audio:
  max_duration_seconds: 30
  sample_rate: 48000
  supported_formats: ["wav", "mp3", "flac"]
  max_file_size_mb: 100

security:
  cors_origins:
    - "https://fugatto-lab.com"
    - "https://api.fugatto-lab.com"
  rate_limit:
    requests_per_minute: 60
    burst_size: 10
  api_key_required: true

monitoring:
  enable_metrics: true
  enable_tracing: true
  log_level: "INFO"
  health_check_interval: 30

storage:
  type: "s3"  # or "gcs", "azure", "local"
  bucket: "fugatto-models"
  region: "us-west-2"
```

## Scaling Strategies

### Horizontal Pod Autoscaling (HPA)

```yaml
# k8s/hpa.yml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fugatto-hpa
  namespace: fugatto-lab
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fugatto-audio-lab
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "10"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### Vertical Pod Autoscaling (VPA)

```yaml
# k8s/vpa.yml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: fugatto-vpa
  namespace: fugatto-lab
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fugatto-audio-lab
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: fugatto-audio-lab
      minAllowed:
        cpu: "1000m"
        memory: "2Gi"
      maxAllowed:
        cpu: "8000m"
        memory: "32Gi"
      controlledResources: ["cpu", "memory"]
```

### Load Balancing

#### NGINX Configuration

```nginx
# nginx.conf
upstream fugatto_backend {
    least_conn;
    server fugatto-1:8000 max_fails=3 fail_timeout=30s;
    server fugatto-2:8000 max_fails=3 fail_timeout=30s;
    server fugatto-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    listen 443 ssl http2;
    server_name api.fugatto-lab.com;

    # SSL configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # File upload limits
    client_max_body_size 100M;
    client_body_timeout 300s;

    location / {
        proxy_pass http://fugatto_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        # Health checks
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
    }

    location /health {
        access_log off;
        proxy_pass http://fugatto_backend;
    }

    location /metrics {
        deny all;
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        proxy_pass http://fugatto_backend;
    }
}
```

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy Fugatto Audio Lab

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest --cov=fugatto_lab --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run security scan
      run: |
        pip install bandit[toml] safety
        bandit -r fugatto_lab/
        safety check

  build:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    if: github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
    
    - name: Deploy to staging
      run: |
        kubectl set image deployment/fugatto-audio-lab \
          fugatto-audio-lab=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:main \
          -n fugatto-lab-staging
        kubectl rollout status deployment/fugatto-audio-lab -n fugatto-lab-staging

  deploy-production:
    if: startsWith(github.ref, 'refs/tags/v')
    needs: build
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG_PROD }}
    
    - name: Deploy to production
      run: |
        kubectl set image deployment/fugatto-audio-lab \
          fugatto-audio-lab=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.ref_name }} \
          -n fugatto-lab
        kubectl rollout status deployment/fugatto-audio-lab -n fugatto-lab
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# scripts/backup-database.sh

# Configuration
BACKUP_DIR="/app/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Database backup
pg_dump "$DATABASE_URL" | gzip > "$BACKUP_DIR/database_$TIMESTAMP.sql.gz"

# Upload to cloud storage
aws s3 cp "$BACKUP_DIR/database_$TIMESTAMP.sql.gz" "s3://fugatto-backups/database/"

# Cleanup old backups
find "$BACKUP_DIR" -name "database_*.sql.gz" -mtime +$RETENTION_DAYS -delete

echo "Database backup completed: database_$TIMESTAMP.sql.gz"
```

### Model Backup

```python
# scripts/backup_models.py
import os
import shutil
import boto3
from datetime import datetime

def backup_models():
    """Backup model cache to cloud storage."""
    
    model_dir = os.environ.get('MODEL_CACHE_DIR', '/app/models')
    backup_bucket = os.environ.get('BACKUP_BUCKET', 'fugatto-backups')
    
    # Create S3 client
    s3 = boto3.client('s3')
    
    # Upload all model files
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, model_dir)
            s3_key = f"models/{datetime.now().strftime('%Y%m%d')}/{relative_path}"
            
            print(f"Uploading {local_path} to s3://{backup_bucket}/{s3_key}")
            s3.upload_file(local_path, backup_bucket, s3_key)
    
    print("Model backup completed")

if __name__ == "__main__":
    backup_models()
```

### Disaster Recovery Plan

1. **RTO (Recovery Time Objective)**: 4 hours
2. **RPO (Recovery Point Objective)**: 1 hour

**Recovery Procedures:**

1. **Infrastructure Recovery**
   - Provision new infrastructure using IaC
   - Restore from infrastructure backups
   - Verify network and security configurations

2. **Data Recovery**
   - Restore database from latest backup
   - Download model files from backup storage
   - Verify data integrity

3. **Application Recovery**
   - Deploy latest known-good image
   - Configure environment variables
   - Run health checks and smoke tests

4. **DNS and Traffic Recovery**
   - Update DNS records if needed
   - Gradually shift traffic to recovered environment
   - Monitor for issues

## Troubleshooting

### Common Issues

#### Model Loading Failures

```bash
# Check model files
ls -la /app/models/
du -sh /app/models/*

# Verify permissions
chmod -R 755 /app/models/

# Check disk space
df -h

# Check memory usage
free -h
nvidia-smi
```

#### High Memory Usage

```bash
# Monitor memory usage
watch -n 1 free -h

# Check process memory usage
ps aux --sort=-%mem | head -10

# Clear model cache if needed
curl -X POST http://localhost:8000/admin/clear-cache
```

#### GPU Issues

```bash
# Check GPU status
nvidia-smi

# Reset GPU
sudo nvidia-smi -r

# Check CUDA version
nvcc --version

# Verify PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available())"
```

#### Network Connectivity

```bash
# Test external connectivity
curl -I https://huggingface.co/

# Check internal service connectivity
curl -f http://localhost:8000/health

# Verify DNS resolution
nslookup api.fugatto-lab.com
```

### Log Analysis

```bash
# Application logs
tail -f /app/logs/fugatto_lab.log | jq .

# Error logs only
grep -E "(ERROR|CRITICAL)" /app/logs/fugatto_lab.log | tail -20

# Performance logs
grep "duration" /app/logs/fugatto_lab.log | tail -10

# Security logs
tail -f /app/logs/security.log | jq .
```

### Performance Debugging

```bash
# Check API response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# Monitor system resources
htop
iotop
nethogs

# Profile Python application
py-spy top --pid $(pgrep -f fugatto_lab)
```

This comprehensive deployment guide covers all aspects of deploying and operating Fugatto Audio Lab in various environments, from development to production-scale deployments with proper monitoring, backup, and recovery procedures.