"""Middleware components for the Fugatto Audio Lab API."""

import time
import logging
import json
from typing import Dict, Any, Callable
from pathlib import Path

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    BaseHTTPMiddleware = object

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    def __init__(self, app, log_requests: bool = True, log_responses: bool = False):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Start timing
        start_time = time.time()
        
        # Log request
        if self.log_requests:
            logger.info(f"Request: {request.method} {request.url.path}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        if self.log_responses:
            logger.info(f"Response: {response.status_code} in {process_time:.3f}s")
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}
        self.window_start = {}
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Get current time
        current_time = time.time()
        
        # Initialize or reset window for this IP
        if client_ip not in self.window_start or current_time - self.window_start[client_ip] >= 60:
            self.window_start[client_ip] = current_time
            self.request_counts[client_ip] = 0
        
        # Check rate limit
        if self.request_counts[client_ip] >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute allowed",
                    "retry_after": 60 - (current_time - self.window_start[client_ip])
                }
            )
        
        # Increment counter
        self.request_counts[client_ip] += 1
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_minute - self.request_counts[client_ip])
        )
        response.headers["X-RateLimit-Reset"] = str(int(self.window_start[client_ip] + 60))
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response


class FileSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit file upload sizes."""
    
    def __init__(self, app, max_size: int = 100 * 1024 * 1024):  # 100MB default
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Check content length
        if request.headers.get("content-length"):
            content_length = int(request.headers["content-length"])
            if content_length > self.max_size:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "File too large",
                        "message": f"Maximum file size is {self.max_size / (1024*1024):.1f}MB",
                        "max_size_bytes": self.max_size
                    }
                )
        
        return await call_next(request)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics."""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.request_times = []
        self.status_counts = {}
    
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Collect metrics
        self.request_count += 1
        process_time = time.time() - start_time
        self.request_times.append(process_time)
        
        # Keep only recent times (last 1000 requests)
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]
        
        # Count status codes
        status_code = response.status_code
        self.status_counts[status_code] = self.status_counts.get(status_code, 0) + 1
        
        return response
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        if not self.request_times:
            return {
                "request_count": 0,
                "avg_response_time": 0,
                "status_counts": {}
            }
        
        return {
            "request_count": self.request_count,
            "avg_response_time": sum(self.request_times) / len(self.request_times),
            "min_response_time": min(self.request_times),
            "max_response_time": max(self.request_times),
            "status_counts": self.status_counts.copy()
        }


def setup_middleware(app: FastAPI, config: Dict[str, Any]) -> None:
    """Setup all middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        config: Configuration dictionary
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available, skipping middleware setup")
        return
    
    # CORS middleware
    if config.get('cors_origins'):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config['cors_origins'],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
        logger.info(f"CORS enabled for origins: {config['cors_origins']}")
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # File size limiting
    if config.get('max_upload_size'):
        app.add_middleware(FileSizeLimitMiddleware, max_size=config['max_upload_size'])
        logger.info(f"File upload limit: {config['max_upload_size'] / (1024*1024):.1f}MB")
    
    # Rate limiting (if enabled)
    if config.get('enable_rate_limiting', True):
        rate_limit = config.get('rate_limit_per_minute', 60)
        app.add_middleware(RateLimitingMiddleware, requests_per_minute=rate_limit)
        logger.info(f"Rate limiting enabled: {rate_limit} requests/minute")
    
    # Request logging
    log_requests = config.get('log_requests', True)
    app.add_middleware(RequestLoggingMiddleware, log_requests=log_requests)
    
    # Metrics collection
    metrics_middleware = MetricsMiddleware(app)
    app.add_middleware(type(metrics_middleware), app=app)
    app.state.metrics = metrics_middleware
    
    # Trusted host middleware (production)
    if not config.get('debug', False):
        allowed_hosts = config.get('allowed_hosts', ['*'])
        if allowed_hosts != ['*']:
            app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
            logger.info(f"Trusted hosts: {allowed_hosts}")
    
    logger.info("All middleware configured successfully")