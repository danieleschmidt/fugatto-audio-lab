"""Database repositories for data access operations."""

import json
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pathlib import Path

from .connection import SQLiteRepository, get_db_manager
from .models import AudioRecordData, UserSessionData, ExperimentRunData

try:
    import sqlalchemy as sa
    from sqlalchemy.orm import Session
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Session = None

logger = logging.getLogger(__name__)


class AudioRepository(SQLiteRepository):
    """Repository for audio generation records."""
    
    def create_record(self, record_data: AudioRecordData) -> int:
        """Create a new audio record.
        
        Args:
            record_data: Audio record data
            
        Returns:
            Created record ID
        """
        if SQLALCHEMY_AVAILABLE:
            return self._create_record_sqlalchemy(record_data)
        else:
            return self._create_record_sqlite(record_data)
    
    def _create_record_sqlalchemy(self, record_data: AudioRecordData) -> int:
        """Create record using SQLAlchemy."""
        from .models import AudioRecord
        
        with self.db_manager.get_session() as session:
            record = AudioRecord.from_data_class(record_data)
            session.add(record)
            session.flush()  # Get the ID
            record_id = record.id
            session.commit()
            
        logger.info(f"Created audio record {record_id}")
        return record_id
    
    def _create_record_sqlite(self, record_data: AudioRecordData) -> int:
        """Create record using SQLite."""
        query = """
            INSERT INTO audio_records 
            (prompt, audio_path, duration_seconds, sample_rate, model_name, 
             temperature, generation_time_ms, metadata, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            record_data.prompt,
            record_data.audio_path,
            record_data.duration_seconds,
            record_data.sample_rate,
            record_data.model_name,
            record_data.temperature,
            record_data.generation_time_ms,
            json.dumps(record_data.metadata),
            json.dumps(record_data.tags)
        )
        
        record_id = self._execute_insert(query, params)
        logger.info(f"Created audio record {record_id}")
        return record_id
    
    def get_record(self, record_id: int) -> Optional[AudioRecordData]:
        """Get audio record by ID.
        
        Args:
            record_id: Record ID
            
        Returns:
            Audio record data or None if not found
        """
        if SQLALCHEMY_AVAILABLE:
            return self._get_record_sqlalchemy(record_id)
        else:
            return self._get_record_sqlite(record_id)
    
    def _get_record_sqlalchemy(self, record_id: int) -> Optional[AudioRecordData]:
        """Get record using SQLAlchemy."""
        from .models import AudioRecord
        
        with self.db_manager.get_session() as session:
            record = session.query(AudioRecord).filter(AudioRecord.id == record_id).first()
            return record.to_data_class() if record else None
    
    def _get_record_sqlite(self, record_id: int) -> Optional[AudioRecordData]:
        """Get record using SQLite."""
        query = "SELECT * FROM audio_records WHERE id = ?"
        
        with self.db_manager.get_session() as session:
            cursor = session.cursor()
            cursor.execute(query, (record_id,))
            row = cursor.fetchone()
            
            if row:
                return AudioRecordData(
                    id=row['id'],
                    prompt=row['prompt'],
                    audio_path=row['audio_path'],
                    duration_seconds=row['duration_seconds'],
                    sample_rate=row['sample_rate'],
                    model_name=row['model_name'],
                    temperature=row['temperature'],
                    generation_time_ms=row['generation_time_ms'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    tags=json.loads(row['tags']) if row['tags'] else []
                )
            return None
    
    def list_records(self, limit: int = 100, offset: int = 0,
                    model_name: Optional[str] = None,
                    order_by: str = 'created_at',
                    order_desc: bool = True) -> List[AudioRecordData]:
        """List audio records with filtering and pagination.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            model_name: Filter by model name
            order_by: Field to order by
            order_desc: Whether to order in descending order
            
        Returns:
            List of audio record data
        """
        if SQLALCHEMY_AVAILABLE:
            return self._list_records_sqlalchemy(limit, offset, model_name, order_by, order_desc)
        else:
            return self._list_records_sqlite(limit, offset, model_name, order_by, order_desc)
    
    def _list_records_sqlalchemy(self, limit: int, offset: int,
                                model_name: Optional[str],
                                order_by: str, order_desc: bool) -> List[AudioRecordData]:
        """List records using SQLAlchemy."""
        from .models import AudioRecord
        
        with self.db_manager.get_session() as session:
            query = session.query(AudioRecord)
            
            if model_name:
                query = query.filter(AudioRecord.model_name == model_name)
            
            # Order by
            order_field = getattr(AudioRecord, order_by, AudioRecord.created_at)
            if order_desc:
                query = query.order_by(order_field.desc())
            else:
                query = query.order_by(order_field)
            
            # Pagination
            query = query.offset(offset).limit(limit)
            
            records = query.all()
            return [record.to_data_class() for record in records]
    
    def _list_records_sqlite(self, limit: int, offset: int,
                           model_name: Optional[str],
                           order_by: str, order_desc: bool) -> List[AudioRecordData]:
        """List records using SQLite."""
        # Build query
        where_clause = ""
        params = []
        
        if model_name:
            where_clause = "WHERE model_name = ?"
            params.append(model_name)
        
        order_direction = "DESC" if order_desc else "ASC"
        
        query = f"""
            SELECT * FROM audio_records 
            {where_clause}
            ORDER BY {order_by} {order_direction}
            LIMIT ? OFFSET ?
        """
        
        params.extend([limit, offset])
        
        with self.db_manager.get_session() as session:
            cursor = session.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                records.append(AudioRecordData(
                    id=row['id'],
                    prompt=row['prompt'],
                    audio_path=row['audio_path'],
                    duration_seconds=row['duration_seconds'],
                    sample_rate=row['sample_rate'],
                    model_name=row['model_name'],
                    temperature=row['temperature'],
                    generation_time_ms=row['generation_time_ms'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    tags=json.loads(row['tags']) if row['tags'] else []
                ))
            
            return records
    
    def search_records(self, search_term: str, limit: int = 50) -> List[AudioRecordData]:
        """Search audio records by prompt text.
        
        Args:
            search_term: Search term to match in prompts
            limit: Maximum number of results
            
        Returns:
            List of matching audio records
        """
        if SQLALCHEMY_AVAILABLE:
            return self._search_records_sqlalchemy(search_term, limit)
        else:
            return self._search_records_sqlite(search_term, limit)
    
    def _search_records_sqlalchemy(self, search_term: str, limit: int) -> List[AudioRecordData]:
        """Search records using SQLAlchemy."""
        from .models import AudioRecord
        
        with self.db_manager.get_session() as session:
            query = session.query(AudioRecord).filter(
                AudioRecord.prompt.contains(search_term)
            ).order_by(AudioRecord.created_at.desc()).limit(limit)
            
            records = query.all()
            return [record.to_data_class() for record in records]
    
    def _search_records_sqlite(self, search_term: str, limit: int) -> List[AudioRecordData]:
        """Search records using SQLite."""
        query = """
            SELECT * FROM audio_records 
            WHERE prompt LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """
        
        with self.db_manager.get_session() as session:
            cursor = session.cursor()
            cursor.execute(query, (f"%{search_term}%", limit))
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                records.append(AudioRecordData(
                    id=row['id'],
                    prompt=row['prompt'],
                    audio_path=row['audio_path'],
                    duration_seconds=row['duration_seconds'],
                    sample_rate=row['sample_rate'],
                    model_name=row['model_name'],
                    temperature=row['temperature'],
                    generation_time_ms=row['generation_time_ms'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    tags=json.loads(row['tags']) if row['tags'] else []
                ))
            
            return records
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audio generation statistics.
        
        Returns:
            Statistics dictionary
        """
        query = """
            SELECT 
                COUNT(*) as total_records,
                AVG(duration_seconds) as avg_duration,
                AVG(generation_time_ms) as avg_generation_time,
                COUNT(DISTINCT model_name) as unique_models,
                MIN(created_at) as first_record,
                MAX(created_at) as latest_record
            FROM audio_records
        """
        
        with self.db_manager.get_session() as session:
            cursor = session.cursor()
            cursor.execute(query)
            row = cursor.fetchone()
            
            return {
                'total_records': row['total_records'],
                'avg_duration_seconds': row['avg_duration'],
                'avg_generation_time_ms': row['avg_generation_time'],
                'unique_models': row['unique_models'],
                'first_record': row['first_record'],
                'latest_record': row['latest_record']
            }
    
    def delete_record(self, record_id: int) -> bool:
        """Delete an audio record.
        
        Args:
            record_id: Record ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        query = "DELETE FROM audio_records WHERE id = ?"
        
        affected_rows = self._execute_update(query, {'id': record_id})
        
        if affected_rows > 0:
            logger.info(f"Deleted audio record {record_id}")
            return True
        else:
            logger.warning(f"Audio record {record_id} not found for deletion")
            return False


class SessionRepository(SQLiteRepository):
    """Repository for user session management."""
    
    def create_session(self, session_data: UserSessionData) -> int:
        """Create a new user session.
        
        Args:
            session_data: Session data
            
        Returns:
            Created session ID
        """
        query = """
            INSERT INTO user_sessions 
            (session_id, user_agent, ip_address, session_data)
            VALUES (?, ?, ?, ?)
        """
        
        params = (
            session_data.session_id,
            session_data.user_agent,
            session_data.ip_address,
            json.dumps(session_data.session_data)
        )
        
        session_id = self._execute_insert(query, params)
        logger.info(f"Created user session {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSessionData]:
        """Get session by session ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found
        """
        query = "SELECT * FROM user_sessions WHERE session_id = ?"
        
        with self.db_manager.get_session() as session:
            cursor = session.cursor()
            cursor.execute(query, (session_id,))
            row = cursor.fetchone()
            
            if row:
                return UserSessionData(
                    id=row['id'],
                    session_id=row['session_id'],
                    user_agent=row['user_agent'],
                    ip_address=row['ip_address'],
                    started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                    last_activity=datetime.fromisoformat(row['last_activity']) if row['last_activity'] else None,
                    is_active=bool(row['is_active']),
                    session_data=json.loads(row['session_data']) if row['session_data'] else {}
                )
            return None
    
    def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity timestamp.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if updated, False if not found
        """
        query = """
            UPDATE user_sessions 
            SET last_activity = CURRENT_TIMESTAMP 
            WHERE session_id = ?
        """
        
        affected_rows = self._execute_update(query, {'session_id': session_id})
        return affected_rows > 0
    
    def deactivate_session(self, session_id: str) -> bool:
        """Deactivate a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deactivated, False if not found
        """
        query = """
            UPDATE user_sessions 
            SET is_active = 0 
            WHERE session_id = ?
        """
        
        affected_rows = self._execute_update(query, {'session_id': session_id})
        
        if affected_rows > 0:
            logger.info(f"Deactivated session {session_id}")
            return True
        return False


class ExperimentRepository(SQLiteRepository):
    """Repository for experiment run tracking."""
    
    def create_experiment(self, experiment_data: ExperimentRunData) -> int:
        """Create a new experiment run.
        
        Args:
            experiment_data: Experiment data
            
        Returns:
            Created experiment ID
        """
        query = """
            INSERT INTO experiment_runs 
            (experiment_name, parameters, status, metrics)
            VALUES (?, ?, ?, ?)
        """
        
        params = (
            experiment_data.experiment_name,
            json.dumps(experiment_data.parameters),
            experiment_data.status,
            json.dumps(experiment_data.metrics)
        )
        
        experiment_id = self._execute_insert(query, params)
        logger.info(f"Created experiment run {experiment_id}: {experiment_data.experiment_name}")
        return experiment_id
    
    def update_experiment_status(self, experiment_id: int, status: str,
                               results: Optional[Dict[str, Any]] = None,
                               error_message: Optional[str] = None) -> bool:
        """Update experiment status and results.
        
        Args:
            experiment_id: Experiment ID
            status: New status (running, completed, failed)
            results: Experiment results
            error_message: Error message if failed
            
        Returns:
            True if updated, False if not found
        """
        if status == 'completed':
            query = """
                UPDATE experiment_runs 
                SET status = ?, results = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """
            params = (status, json.dumps(results or {}), experiment_id)
        elif status == 'failed':
            query = """
                UPDATE experiment_runs 
                SET status = ?, error_message = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """
            params = (status, error_message, experiment_id)
        else:
            query = "UPDATE experiment_runs SET status = ? WHERE id = ?"
            params = (status, experiment_id)
        
        affected_rows = self._execute_update(query, params)
        
        if affected_rows > 0:
            logger.info(f"Updated experiment {experiment_id} status to {status}")
            return True
        return False
    
    def get_experiment(self, experiment_id: int) -> Optional[ExperimentRunData]:
        """Get experiment by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment data or None if not found
        """
        query = "SELECT * FROM experiment_runs WHERE id = ?"
        
        with self.db_manager.get_session() as session:
            cursor = session.cursor()
            cursor.execute(query, (experiment_id,))
            row = cursor.fetchone()
            
            if row:
                return ExperimentRunData(
                    id=row['id'],
                    experiment_name=row['experiment_name'],
                    parameters=json.loads(row['parameters']) if row['parameters'] else {},
                    results=json.loads(row['results']) if row['results'] else {},
                    status=row['status'],
                    started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                    completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                    error_message=row['error_message'],
                    metrics=json.loads(row['metrics']) if row['metrics'] else {}
                )
            return None
    
    def list_experiments(self, experiment_name: Optional[str] = None,
                        status: Optional[str] = None,
                        limit: int = 100) -> List[ExperimentRunData]:
        """List experiments with filtering.
        
        Args:
            experiment_name: Filter by experiment name
            status: Filter by status
            limit: Maximum number of results
            
        Returns:
            List of experiment data
        """
        where_clauses = []
        params = []
        
        if experiment_name:
            where_clauses.append("experiment_name = ?")
            params.append(experiment_name)
        
        if status:
            where_clauses.append("status = ?")
            params.append(status)
        
        where_clause = ""
        if where_clauses:
            where_clause = "WHERE " + " AND ".join(where_clauses)
        
        query = f"""
            SELECT * FROM experiment_runs
            {where_clause}
            ORDER BY started_at DESC
            LIMIT ?
        """
        
        params.append(limit)
        
        with self.db_manager.get_session() as session:
            cursor = session.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            experiments = []
            for row in rows:
                experiments.append(ExperimentRunData(
                    id=row['id'],
                    experiment_name=row['experiment_name'],
                    parameters=json.loads(row['parameters']) if row['parameters'] else {},
                    results=json.loads(row['results']) if row['results'] else {},
                    status=row['status'],
                    started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                    completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                    error_message=row['error_message'],
                    metrics=json.loads(row['metrics']) if row['metrics'] else {}
                ))
            
            return experiments