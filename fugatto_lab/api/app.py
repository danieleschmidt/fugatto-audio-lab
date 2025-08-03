"""FastAPI application factory and configuration."""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None
    HTTPException = None
    FASTAPI_AVAILABLE = False

from ..core import FugattoModel, AudioProcessor
from ..services import AudioGenerationService
from ..database import get_db_manager
from ..monitoring import get_monitor
from .routes import register_routes
from .middleware import setup_middleware

logger = logging.getLogger(__name__)


def create_app(config: Optional[Dict[str, Any]] = None) -> Any:
    """Create and configure FastAPI application.
    
    Args:
        config: Application configuration dictionary
        
    Returns:
        Configured FastAPI application
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    # Default configuration
    default_config = {
        'title': 'Fugatto Audio Lab API',
        'description': 'REST API for AI-powered audio generation and transformation',
        'version': '1.0.0',
        'debug': os.getenv('DEBUG', 'false').lower() == 'true',
        'host': os.getenv('API_HOST', '0.0.0.0'),
        'port': int(os.getenv('API_PORT', '8000')),
        'workers': int(os.getenv('API_WORKERS', '1')),
        'cors_origins': os.getenv('CORS_ORIGINS', '*').split(','),
        'max_upload_size': int(os.getenv('MAX_UPLOAD_SIZE_MB', '100')) * 1024 * 1024,
        'enable_docs': os.getenv('ENABLE_API_DOCS', 'true').lower() == 'true'
    }
    
    if config:
        default_config.update(config)
    
    # Create FastAPI app
    app = FastAPI(
        title=default_config['title'],
        description=default_config['description'],
        version=default_config['version'],
        debug=default_config['debug'],
        docs_url='/docs' if default_config['enable_docs'] else None,
        redoc_url='/redoc' if default_config['enable_docs'] else None
    )
    
    # Setup middleware
    setup_middleware(app, default_config)
    
    # Register routes
    register_routes(app)
    
    # Store config for access in routes
    app.state.config = default_config
    
    # Initialize services
    app.state.services = _initialize_services()
    
    # Add startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting Fugatto Audio Lab API")
        
        # Initialize database
        db_manager = get_db_manager()
        db_manager.initialize()
        
        # Health check
        monitor = get_monitor()
        health = monitor.health_check()
        logger.info(f"System health: {health['status']}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down Fugatto Audio Lab API")
        
        # Cleanup services
        if hasattr(app.state, 'services'):
            for service in app.state.services.values():
                if hasattr(service, 'cleanup'):
                    try:
                        service.cleanup()
                    except Exception as e:
                        logger.error(f"Error cleaning up service: {e}")
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": str(exc) if default_config['debug'] else "An unexpected error occurred",
                "type": "internal_error"
            }
        )
    
    # HTTP exception handler
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "type": "http_error"
            }
        )
    
    logger.info(f"FastAPI application created: {app.title} v{app.version}")
    return app


def _initialize_services() -> Dict[str, Any]:
    """Initialize application services.
    
    Returns:
        Dictionary of initialized services
    """
    try:
        services = {
            'audio_generation': AudioGenerationService(),
            'audio_processor': AudioProcessor(),
            'db_manager': get_db_manager(),
            'monitor': get_monitor()
        }
        
        logger.info("Application services initialized successfully")
        return services
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


def run_server(app: FastAPI, config: Optional[Dict[str, Any]] = None):
    """Run the FastAPI server.
    
    Args:
        app: FastAPI application instance
        config: Server configuration
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI/uvicorn not available")
    
    # Get config from app state or use defaults
    if hasattr(app.state, 'config'):
        server_config = app.state.config
    else:
        server_config = config or {}
    
    # Server configuration
    host = server_config.get('host', '0.0.0.0')
    port = server_config.get('port', 8000)
    workers = server_config.get('workers', 1)
    debug = server_config.get('debug', False)
    
    logger.info(f"Starting server on {host}:{port} with {workers} workers")
    
    if workers > 1:
        # Multi-worker production setup
        uvicorn.run(
            "fugatto_lab.api.app:create_app",
            host=host,
            port=port,
            workers=workers,
            factory=True,
            log_level="info" if not debug else "debug"
        )
    else:
        # Single worker development setup
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=debug,
            log_level="info" if not debug else "debug"
        )


class APIError(Exception):
    """Base exception for API errors."""
    
    def __init__(self, message: str, status_code: int = 500, error_type: str = "api_error"):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(message)


class ValidationError(APIError):
    """Exception for validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super().__init__(message, 400, "validation_error")


class NotFoundError(APIError):
    """Exception for resource not found errors."""
    
    def __init__(self, resource: str, identifier: str):
        message = f"{resource} with id '{identifier}' not found"
        super().__init__(message, 404, "not_found")


class RateLimitError(APIError):
    """Exception for rate limiting errors."""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, 429, "rate_limit_exceeded")


# Development server entry point
if __name__ == "__main__":
    app = create_app()
    run_server(app)