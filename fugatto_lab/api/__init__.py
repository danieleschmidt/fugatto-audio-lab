"""API layer for Fugatto Audio Lab REST endpoints."""

from .app import create_app
from .routes import register_routes
from .middleware import setup_middleware

__all__ = [
    'create_app',
    'register_routes', 
    'setup_middleware'
]