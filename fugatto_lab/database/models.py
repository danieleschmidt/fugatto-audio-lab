"""Database models for Fugatto Audio Lab."""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import sqlalchemy as sa
    from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Boolean
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.sql import func
    SQLALCHEMY_AVAILABLE = True
    
    from .connection import Base
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    sa = None
    Column = None
    Base = None


@dataclass
class AudioRecordData:
    """Data class for audio record information."""
    prompt: str
    audio_path: str
    duration_seconds: float
    sample_rate: int
    model_name: str
    temperature: float
    generation_time_ms: float
    metadata: Dict[str, Any]
    tags: List[str]
    created_at: Optional[datetime] = None
    id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudioRecordData':
        """Create from dictionary."""
        return cls(**data)


@dataclass 
class UserSessionData:
    """Data class for user session information."""
    session_id: str
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    started_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    is_active: bool = True
    session_data: Dict[str, Any] = None
    id: Optional[int] = None
    
    def __post_init__(self):
        if self.session_data is None:
            self.session_data = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserSessionData':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ExperimentRunData:
    """Data class for experiment run information."""
    experiment_name: str
    parameters: Dict[str, Any]
    results: Dict[str, Any] = None
    status: str = 'running'
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None
    id: Optional[int] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = {}
        if self.metrics is None:
            self.metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentRunData':
        """Create from dictionary."""
        return cls(**data)


if SQLALCHEMY_AVAILABLE:
    
    class AudioRecord(Base):
        """SQLAlchemy model for audio generation records."""
        __tablename__ = 'audio_records'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        prompt = Column(Text, nullable=False)
        audio_path = Column(String(500), nullable=False)
        duration_seconds = Column(Float, nullable=False)
        sample_rate = Column(Integer, nullable=False)
        model_name = Column(String(100), nullable=False)
        temperature = Column(Float, nullable=False)
        generation_time_ms = Column(Float, nullable=False)
        created_at = Column(DateTime, default=func.now())
        metadata = Column(Text, default='{}')  # JSON string
        tags = Column(Text, default='[]')  # JSON string
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary."""
            return {
                'id': self.id,
                'prompt': self.prompt,
                'audio_path': self.audio_path,
                'duration_seconds': self.duration_seconds,
                'sample_rate': self.sample_rate,
                'model_name': self.model_name,
                'temperature': self.temperature,
                'generation_time_ms': self.generation_time_ms,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'metadata': json.loads(self.metadata) if self.metadata else {},
                'tags': json.loads(self.tags) if self.tags else []
            }
        
        def to_data_class(self) -> AudioRecordData:
            """Convert to data class."""
            return AudioRecordData(
                id=self.id,
                prompt=self.prompt,
                audio_path=self.audio_path,
                duration_seconds=self.duration_seconds,
                sample_rate=self.sample_rate,
                model_name=self.model_name,
                temperature=self.temperature,
                generation_time_ms=self.generation_time_ms,
                created_at=self.created_at,
                metadata=json.loads(self.metadata) if self.metadata else {},
                tags=json.loads(self.tags) if self.tags else []
            )
        
        @classmethod
        def from_data_class(cls, data: AudioRecordData) -> 'AudioRecord':
            """Create from data class."""
            return cls(
                prompt=data.prompt,
                audio_path=data.audio_path,
                duration_seconds=data.duration_seconds,
                sample_rate=data.sample_rate,
                model_name=data.model_name,
                temperature=data.temperature,
                generation_time_ms=data.generation_time_ms,
                metadata=json.dumps(data.metadata),
                tags=json.dumps(data.tags)
            )
    
    
    class UserSession(Base):
        """SQLAlchemy model for user sessions."""
        __tablename__ = 'user_sessions'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        session_id = Column(String(100), unique=True, nullable=False)
        user_agent = Column(Text)
        ip_address = Column(String(45))  # IPv6 compatible
        started_at = Column(DateTime, default=func.now())
        last_activity = Column(DateTime, default=func.now())
        is_active = Column(Boolean, default=True)
        session_data = Column(Text, default='{}')  # JSON string
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary."""
            return {
                'id': self.id,
                'session_id': self.session_id,
                'user_agent': self.user_agent,
                'ip_address': self.ip_address,
                'started_at': self.started_at.isoformat() if self.started_at else None,
                'last_activity': self.last_activity.isoformat() if self.last_activity else None,
                'is_active': self.is_active,
                'session_data': json.loads(self.session_data) if self.session_data else {}
            }
        
        def to_data_class(self) -> UserSessionData:
            """Convert to data class."""
            return UserSessionData(
                id=self.id,
                session_id=self.session_id,
                user_agent=self.user_agent,
                ip_address=self.ip_address,
                started_at=self.started_at,
                last_activity=self.last_activity,
                is_active=self.is_active,
                session_data=json.loads(self.session_data) if self.session_data else {}
            )
    
    
    class ExperimentRun(Base):
        """SQLAlchemy model for experiment runs."""
        __tablename__ = 'experiment_runs'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        experiment_name = Column(String(200), nullable=False)
        parameters = Column(Text, nullable=False)  # JSON string
        results = Column(Text, default='{}')  # JSON string
        status = Column(String(50), default='running')
        started_at = Column(DateTime, default=func.now())
        completed_at = Column(DateTime)
        error_message = Column(Text)
        metrics = Column(Text, default='{}')  # JSON string
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary."""
            return {
                'id': self.id,
                'experiment_name': self.experiment_name,
                'parameters': json.loads(self.parameters) if self.parameters else {},
                'results': json.loads(self.results) if self.results else {},
                'status': self.status,
                'started_at': self.started_at.isoformat() if self.started_at else None,
                'completed_at': self.completed_at.isoformat() if self.completed_at else None,
                'error_message': self.error_message,
                'metrics': json.loads(self.metrics) if self.metrics else {}
            }
        
        def to_data_class(self) -> ExperimentRunData:
            """Convert to data class."""
            return ExperimentRunData(
                id=self.id,
                experiment_name=self.experiment_name,
                parameters=json.loads(self.parameters) if self.parameters else {},
                results=json.loads(self.results) if self.results else {},
                status=self.status,
                started_at=self.started_at,
                completed_at=self.completed_at,
                error_message=self.error_message,
                metrics=json.loads(self.metrics) if self.metrics else {}
            )
    
else:
    # Fallback classes when SQLAlchemy is not available
    class AudioRecord:
        """Fallback audio record model."""
        
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def to_dict(self) -> Dict[str, Any]:
            return self.__dict__
        
        def to_data_class(self) -> AudioRecordData:
            return AudioRecordData(**self.__dict__)
    
    
    class UserSession:
        """Fallback user session model."""
        
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def to_dict(self) -> Dict[str, Any]:
            return self.__dict__
        
        def to_data_class(self) -> UserSessionData:
            return UserSessionData(**self.__dict__)
    
    
    class ExperimentRun:
        """Fallback experiment run model."""
        
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def to_dict(self) -> Dict[str, Any]:
            return self.__dict__
        
        def to_data_class(self) -> ExperimentRunData:
            return ExperimentRunData(**self.__dict__)