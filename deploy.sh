#!/bin/bash

# Fugatto Audio Lab Production Deployment Script
# This script handles the complete deployment process

set -e  # Exit on any error

# Configuration
PROJECT_NAME="fugatto-audio-lab"
ENVIRONMENT=${ENVIRONMENT:-production}
VERSION=${VERSION:-latest}
REGISTRY=${REGISTRY:-""}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed"
    fi
    
    # Check available disk space (at least 10GB)
    AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
    if [ "$AVAILABLE_SPACE" -lt 10485760 ]; then  # 10GB in KB
        warn "Less than 10GB available disk space. Deployment may fail."
    fi
    
    success "Prerequisites check completed"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    # Build main application image
    docker build -f Dockerfile.production -t ${PROJECT_NAME}:${VERSION} .
    
    # Build worker image
    docker build -f Dockerfile.worker -t ${PROJECT_NAME}-worker:${VERSION} .
    
    # Tag images if registry is specified
    if [ -n "$REGISTRY" ]; then
        docker tag ${PROJECT_NAME}:${VERSION} ${REGISTRY}/${PROJECT_NAME}:${VERSION}
        docker tag ${PROJECT_NAME}-worker:${VERSION} ${REGISTRY}/${PROJECT_NAME}-worker:${VERSION}
    fi
    
    success "Docker images built successfully"
}

# Push images to registry
push_images() {
    if [ -z "$REGISTRY" ]; then
        log "No registry specified, skipping image push"
        return
    fi
    
    log "Pushing images to registry..."
    
    docker push ${REGISTRY}/${PROJECT_NAME}:${VERSION}
    docker push ${REGISTRY}/${PROJECT_NAME}-worker:${VERSION}
    
    success "Images pushed to registry"
}

# Setup configuration
setup_config() {
    log "Setting up configuration..."
    
    # Create necessary directories
    mkdir -p data logs models cache config ssl
    
    # Copy configuration files
    if [ ! -f "config/.env.production" ]; then
        log "Creating production environment file..."
        cat > config/.env.production << EOF
# Fugatto Audio Lab Production Configuration
ENVIRONMENT=production
LOG_LEVEL=info
DEBUG=false

# Database
POSTGRES_URL=postgresql://fugatto:fugatto_secure_2025@postgres:5432/fugatto_db

# Redis
REDIS_URL=redis://redis:6379

# Performance
CACHE_SIZE_MB=2048
MAX_CONCURRENT_TASKS=8
WORKERS=4

# Security
SECRET_KEY=$(openssl rand -hex 32)
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Monitoring
ENABLE_MONITORING=true
PROMETHEUS_PORT=8080

# Features
ENABLE_AUTO_SCALING=true
ENABLE_SECURITY=true
EOF
        warn "Created default .env.production file. Please review and update as needed."
    fi
    
    # Generate SSL certificates if they don't exist
    if [ ! -f "ssl/cert.pem" ]; then
        log "Generating self-signed SSL certificates..."
        openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes \
            -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=fugatto-lab.local"
        warn "Generated self-signed SSL certificates. Use proper certificates in production."
    fi
    
    success "Configuration setup completed"
}

# Pre-flight checks
preflight_checks() {
    log "Running pre-flight checks..."
    
    # Check if required ports are available
    PORTS=(80 443 5432 6379 9090 3000)
    for port in "${PORTS[@]}"; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null; then
            error "Port $port is already in use"
        fi
    done
    
    # Validate docker-compose file
    if command -v docker-compose &> /dev/null; then
        docker-compose -f docker-compose.production.yml config -q
    else
        docker compose -f docker-compose.production.yml config -q
    fi
    
    success "Pre-flight checks completed"
}

# Deploy the application
deploy() {
    log "Deploying Fugatto Audio Lab..."
    
    # Pull latest images if using registry
    if [ -n "$REGISTRY" ]; then
        log "Pulling latest images from registry..."
        if command -v docker-compose &> /dev/null; then
            docker-compose -f docker-compose.production.yml pull
        else
            docker compose -f docker-compose.production.yml pull
        fi
    fi
    
    # Start services
    log "Starting services..."
    if command -v docker-compose &> /dev/null; then
        docker-compose -f docker-compose.production.yml up -d
    else
        docker compose -f docker-compose.production.yml up -d
    fi
    
    # Wait for services to be healthy
    log "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    check_health
    
    success "Deployment completed successfully!"
}

# Check service health
check_health() {
    log "Checking service health..."
    
    # Wait for API to be ready
    for i in {1..30}; do
        if curl -f http://localhost/health >/dev/null 2>&1; then
            success "API service is healthy"
            break
        fi
        if [ $i -eq 30 ]; then
            error "API service failed to become healthy"
        fi
        sleep 2
    done
    
    # Check database connectivity
    if docker exec fugatto-postgres pg_isready -U fugatto >/dev/null 2>&1; then
        success "Database is healthy"
    else
        warn "Database health check failed"
    fi
    
    # Check Redis connectivity
    if docker exec fugatto-redis redis-cli ping | grep -q PONG; then
        success "Redis is healthy"
    else
        warn "Redis health check failed"
    fi
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    # Wait for database to be ready
    sleep 10
    
    # Run migrations
    docker exec fugatto-api python3 -c "
from fugatto_lab.database.connection import create_tables
import asyncio
asyncio.run(create_tables())
"
    
    success "Database migrations completed"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Import Grafana dashboards
    if [ -d "config/grafana/dashboards" ]; then
        log "Grafana dashboards will be automatically imported"
    fi
    
    # Check Prometheus targets
    sleep 15
    PROMETHEUS_TARGETS=$(curl -s http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets[].health' 2>/dev/null | grep -c "up" || echo "0")
    log "Prometheus has $PROMETHEUS_TARGETS healthy targets"
    
    success "Monitoring setup completed"
}

# Backup function
backup() {
    log "Creating backup..."
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup database
    docker exec fugatto-postgres pg_dump -U fugatto fugatto_db > "$BACKUP_DIR/database.sql"
    
    # Backup Redis data
    docker exec fugatto-redis redis-cli BGSAVE
    docker cp fugatto-redis:/data/dump.rdb "$BACKUP_DIR/redis.rdb"
    
    # Backup application data
    tar -czf "$BACKUP_DIR/app_data.tar.gz" data/ logs/ models/
    
    success "Backup created in $BACKUP_DIR"
}

# Rollback function
rollback() {
    warn "Rolling back deployment..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f docker-compose.production.yml down
    else
        docker compose -f docker-compose.production.yml down
    fi
    
    # Restore from backup if specified
    if [ -n "$BACKUP_PATH" ] && [ -d "$BACKUP_PATH" ]; then
        log "Restoring from backup: $BACKUP_PATH"
        
        # Restore database
        if [ -f "$BACKUP_PATH/database.sql" ]; then
            docker exec -i fugatto-postgres psql -U fugatto fugatto_db < "$BACKUP_PATH/database.sql"
        fi
        
        # Restore Redis
        if [ -f "$BACKUP_PATH/redis.rdb" ]; then
            docker cp "$BACKUP_PATH/redis.rdb" fugatto-redis:/data/dump.rdb
            docker restart fugatto-redis
        fi
        
        # Restore application data
        if [ -f "$BACKUP_PATH/app_data.tar.gz" ]; then
            tar -xzf "$BACKUP_PATH/app_data.tar.gz"
        fi
    fi
    
    warn "Rollback completed"
}

# Main deployment process
main() {
    log "Starting Fugatto Audio Lab deployment..."
    log "Environment: $ENVIRONMENT"
    log "Version: $VERSION"
    
    case "${1:-deploy}" in
        "build")
            check_prerequisites
            build_images
            ;;
        "push")
            push_images
            ;;
        "deploy")
            check_prerequisites
            setup_config
            preflight_checks
            build_images
            deploy
            run_migrations
            setup_monitoring
            ;;
        "health")
            check_health
            ;;
        "backup")
            backup
            ;;
        "rollback")
            rollback
            ;;
        "logs")
            if command -v docker-compose &> /dev/null; then
                docker-compose -f docker-compose.production.yml logs -f
            else
                docker compose -f docker-compose.production.yml logs -f
            fi
            ;;
        "status")
            if command -v docker-compose &> /dev/null; then
                docker-compose -f docker-compose.production.yml ps
            else
                docker compose -f docker-compose.production.yml ps
            fi
            ;;
        "stop")
            if command -v docker-compose &> /dev/null; then
                docker-compose -f docker-compose.production.yml down
            else
                docker compose -f docker-compose.production.yml down
            fi
            ;;
        "restart")
            if command -v docker-compose &> /dev/null; then
                docker-compose -f docker-compose.production.yml restart
            else
                docker compose -f docker-compose.production.yml restart
            fi
            ;;
        *)
            echo "Usage: $0 {deploy|build|push|health|backup|rollback|logs|status|stop|restart}"
            echo
            echo "Commands:"
            echo "  deploy   - Full deployment (default)"
            echo "  build    - Build Docker images only"
            echo "  push     - Push images to registry"
            echo "  health   - Check service health"
            echo "  backup   - Create backup"
            echo "  rollback - Rollback deployment"
            echo "  logs     - Show service logs"
            echo "  status   - Show service status"
            echo "  stop     - Stop all services"
            echo "  restart  - Restart all services"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"