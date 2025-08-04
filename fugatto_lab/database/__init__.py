"""Database layer for Fugatto Audio Lab."""

from .connection import DatabaseManager, get_db_session
from .models import AudioRecord, UserSession, ExperimentRun
from .repositories import AudioRepository, SessionRepository, ExperimentRepository

def get_db_manager():
    """Get database manager instance."""
    return DatabaseManager()

__all__ = [
    'DatabaseManager',
    'get_db_session',
    'get_db_manager',
    'AudioRecord',
    'UserSession', 
    'ExperimentRun',
    'AudioRepository',
    'SessionRepository',
    'ExperimentRepository'
]