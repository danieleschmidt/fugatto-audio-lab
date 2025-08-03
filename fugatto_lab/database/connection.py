"""Database connection and session management."""

import logging
import sqlite3
from typing import Optional, Generator, Any, Dict
from pathlib import Path
from contextlib import contextmanager
import json
import time

try:
    import sqlalchemy as sa
    from sqlalchemy import create_engine, MetaData
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.ext.declarative import declarative_base
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    sa = None
    Session = None
    declarative_base = None

logger = logging.getLogger(__name__)

# Base class for ORM models
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
else:
    Base = None


class DatabaseManager:
    """Database connection and session manager."""
    
    def __init__(self, database_url: str = "sqlite:///./fugatto_lab.db",
                 echo: bool = False, pool_size: int = 5):
        """Initialize database manager.
        
        Args:
            database_url: Database connection URL
            echo: Whether to echo SQL statements
            pool_size: Connection pool size
        """
        self.database_url = database_url
        self.echo = echo
        self.pool_size = pool_size
        self._engine = None
        self._session_factory = None
        self._initialized = False
        
        logger.info(f"DatabaseManager initialized with URL: {database_url}")
    
    def initialize(self) -> None:
        """Initialize database connection and create tables."""
        if self._initialized:
            return
            
        if SQLALCHEMY_AVAILABLE:
            self._initialize_sqlalchemy()
        else:
            self._initialize_sqlite()
            
        self._initialized = True
        logger.info("Database initialized successfully")
    
    def _initialize_sqlalchemy(self) -> None:
        """Initialize SQLAlchemy engine and session factory."""
        self._engine = create_engine(
            self.database_url,
            echo=self.echo,
            pool_size=self.pool_size,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600
        )
        
        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False
        )
        
        # Create tables
        from .models import AudioRecord, UserSession, ExperimentRun
        Base.metadata.create_all(self._engine)
        
        logger.info("SQLAlchemy database initialized")
    
    def _initialize_sqlite(self) -> None:
        """Initialize SQLite database with fallback implementation."""
        # Extract file path from URL
        if self.database_url.startswith("sqlite:///"):
            db_path = self.database_url[10:]  # Remove "sqlite:///"
        else:
            db_path = "fugatto_lab.db"
            
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create tables
        self._create_sqlite_tables()
        
        logger.info(f"SQLite database initialized at {self.db_path}")
    
    def _create_sqlite_tables(self) -> None:
        """Create SQLite tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Audio records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audio_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    audio_path TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    sample_rate INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    generation_time_ms REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}',
                    tags TEXT DEFAULT '[]'
                )
            """)
            
            # User sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    user_agent TEXT,
                    ip_address TEXT,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    session_data TEXT DEFAULT '{}'
                )
            """)
            
            # Experiment runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    results TEXT DEFAULT '{}',
                    status TEXT DEFAULT 'running',
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    metrics TEXT DEFAULT '{}'
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audio_created_at ON audio_records(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audio_model ON audio_records(model_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON user_sessions(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiment_name ON experiment_runs(experiment_name)")
            
            conn.commit()
    
    @contextmanager
    def get_session(self) -> Generator[Any, None, None]:
        """Get database session context manager."""
        if not self._initialized:
            self.initialize()
            
        if SQLALCHEMY_AVAILABLE and self._session_factory:
            session = self._session_factory()
            try:
                yield session
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                session.close()
        else:
            # Fallback to SQLite connection
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable dict-like access
                yield conn
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a database query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results
        """
        with self.get_session() as session:
            if SQLALCHEMY_AVAILABLE:
                result = session.execute(sa.text(query), params or {})
                return result.fetchall()
            else:
                cursor = session.cursor()
                cursor.execute(query, params or {})
                return cursor.fetchall()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get database health status.
        
        Returns:
            Health status information
        """
        try:
            # Test basic connectivity
            start_time = time.time()
            with self.get_session() as session:
                if SQLALCHEMY_AVAILABLE:
                    session.execute(sa.text("SELECT 1"))
                else:
                    session.execute("SELECT 1")
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time,
                'database_url': self.database_url.split('@')[-1] if '@' in self.database_url else self.database_url,
                'sqlalchemy_available': SQLALCHEMY_AVAILABLE,
                'initialized': self._initialized
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'database_url': self.database_url.split('@')[-1] if '@' in self.database_url else self.database_url,
                'sqlalchemy_available': SQLALCHEMY_AVAILABLE,
                'initialized': self._initialized
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Database statistics
        """
        try:
            with self.get_session() as session:
                stats = {}
                
                # Count records in each table
                tables = ['audio_records', 'user_sessions', 'experiment_runs']
                for table in tables:
                    try:
                        if SQLALCHEMY_AVAILABLE:
                            result = session.execute(sa.text(f"SELECT COUNT(*) FROM {table}"))
                            count = result.scalar()
                        else:
                            cursor = session.cursor()
                            cursor.execute(f"SELECT COUNT(*) FROM {table}")
                            count = cursor.fetchone()[0]
                        stats[f"{table}_count"] = count
                    except Exception as e:
                        logger.warning(f"Failed to count {table}: {e}")
                        stats[f"{table}_count"] = 0
                
                # Database size (SQLite only)
                if not SQLALCHEMY_AVAILABLE and hasattr(self, 'db_path'):
                    try:
                        stats['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
                    except Exception:
                        stats['database_size_mb'] = 0
                
                return stats
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {'error': str(e)}
    
    def cleanup_old_records(self, table: str, days_old: int = 30) -> int:
        """Clean up old records from a table.
        
        Args:
            table: Table name
            days_old: Delete records older than this many days
            
        Returns:
            Number of deleted records
        """
        try:
            with self.get_session() as session:
                if SQLALCHEMY_AVAILABLE:
                    query = sa.text(f"""
                        DELETE FROM {table} 
                        WHERE created_at < datetime('now', '-{days_old} days')
                    """)
                    result = session.execute(query)
                    deleted_count = result.rowcount
                else:
                    cursor = session.cursor()
                    cursor.execute(f"""
                        DELETE FROM {table} 
                        WHERE created_at < datetime('now', '-{days_old} days')
                    """)
                    deleted_count = cursor.rowcount
                    session.commit()
                
                logger.info(f"Cleaned up {deleted_count} old records from {table}")
                return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup {table}: {e}")
            return 0


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create global database manager instance."""
    global _db_manager
    if _db_manager is None:
        # Try to get URL from environment
        import os
        database_url = os.getenv('DATABASE_URL', 'sqlite:///./fugatto_lab.db')
        _db_manager = DatabaseManager(database_url)
    return _db_manager


def get_db_session():
    """Get database session for dependency injection."""
    manager = get_db_manager()
    return manager.get_session()


class SQLiteRepository:
    """Base repository class for SQLite operations."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize repository.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager or get_db_manager()
    
    def _execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a query and return results."""
        return self.db_manager.execute_query(query, params)
    
    def _execute_insert(self, query: str, params: Dict[str, Any]) -> int:
        """Execute an insert query and return the last row ID."""
        with self.db_manager.get_session() as session:
            cursor = session.cursor()
            cursor.execute(query, params)
            session.commit()
            return cursor.lastrowid
    
    def _execute_update(self, query: str, params: Dict[str, Any]) -> int:
        """Execute an update query and return the number of affected rows."""
        with self.db_manager.get_session() as session:
            cursor = session.cursor()
            cursor.execute(query, params)
            session.commit()
            return cursor.rowcount