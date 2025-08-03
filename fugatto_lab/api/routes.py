"""API routes for Fugatto Audio Lab."""

import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
import io

try:
    from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
    from fastapi.responses import FileResponse, StreamingResponse
    from pydantic import BaseModel, Field, validator
    FASTAPI_AVAILABLE = True
except ImportError:
    APIRouter = None
    BaseModel = object
    FASTAPI_AVAILABLE = False

import numpy as np

from ..services import AudioGenerationService, VoiceCloneService
from ..database import get_db_manager, AudioRepository
from ..database.models import AudioRecordData
from ..monitoring import get_monitor

logger = logging.getLogger(__name__)


# Pydantic models for request/response validation
if FASTAPI_AVAILABLE:
    
    class AudioGenerationRequest(BaseModel):
        """Request model for audio generation."""
        prompt: str = Field(..., min_length=1, max_length=500, description="Text prompt for audio generation")
        duration_seconds: float = Field(10.0, ge=1.0, le=30.0, description="Duration of generated audio")
        temperature: float = Field(0.8, ge=0.1, le=1.5, description="Generation temperature")
        model_name: Optional[str] = Field("nvidia/fugatto-base", description="Model to use for generation")
        save_to_db: bool = Field(True, description="Whether to save generation to database")
        
        @validator('prompt')
        def validate_prompt(cls, v):
            if not v.strip():
                raise ValueError('Prompt cannot be empty')
            return v.strip()
    
    
    class AudioTransformRequest(BaseModel):
        """Request model for audio transformation."""
        prompt: str = Field(..., min_length=1, max_length=500, description="Transformation description")
        strength: float = Field(0.7, ge=0.0, le=1.0, description="Transformation strength")
        preserve_length: bool = Field(True, description="Whether to preserve original length")
    
    
    class VoiceCloneRequest(BaseModel):
        """Request model for voice cloning."""
        text: str = Field(..., min_length=1, max_length=1000, description="Text to synthesize")
        speaker_id: Optional[str] = Field(None, description="Speaker ID for cached embedding")
        prosody_transfer: bool = Field(True, description="Whether to transfer prosody")
        emotion: Optional[str] = Field(None, description="Emotion to apply")
    
    
    class HealthResponse(BaseModel):
        """Response model for health check."""
        status: str
        timestamp: str
        version: str
        services: Dict[str, Any]
        system_info: Dict[str, Any]
    
    
    class AudioGenerationResponse(BaseModel):
        """Response model for audio generation."""
        generation_id: str
        prompt: str
        duration_seconds: float
        generation_time_ms: float
        model_name: str
        audio_url: str
        metadata: Dict[str, Any]
    
    
    class GenerationListResponse(BaseModel):
        """Response model for generation list."""
        generations: List[Dict[str, Any]]
        total_count: int
        page: int
        page_size: int
        has_next: bool


# API Routes
def get_audio_service() -> AudioGenerationService:
    """Dependency to get audio generation service."""
    # In a real app, this would be injected from app state
    return AudioGenerationService()


def get_voice_service() -> VoiceCloneService:
    """Dependency to get voice cloning service."""
    return VoiceCloneService()


def get_audio_repository() -> AudioRepository:
    """Dependency to get audio repository."""
    return AudioRepository()


def create_health_router() -> APIRouter:
    """Create health check router."""
    router = APIRouter(prefix="/health", tags=["health"])
    
    @router.get("/", response_model=HealthResponse if FASTAPI_AVAILABLE else None)
    async def health_check():
        """Get system health status."""
        monitor = get_monitor()
        health_data = monitor.health_check()
        
        # Get service statuses
        services = {}
        try:
            audio_service = get_audio_service()
            services['audio_generation'] = audio_service.get_service_stats()
        except Exception as e:
            services['audio_generation'] = {'error': str(e)}
        
        try:
            db_manager = get_db_manager()
            services['database'] = db_manager.get_health_status()
        except Exception as e:
            services['database'] = {'error': str(e)}
        
        return {
            "status": health_data['status'],
            "timestamp": health_data['timestamp'],
            "version": "1.0.0",
            "services": services,
            "system_info": health_data.get('system_health', {})
        }
    
    @router.get("/metrics")
    async def get_metrics(request: Request):
        """Get system metrics."""
        monitor = get_monitor()
        metrics = monitor.get_performance_summary()
        
        # Add API metrics if available
        if hasattr(request.app.state, 'metrics'):
            api_metrics = request.app.state.metrics.get_metrics()
            metrics['api'] = api_metrics
        
        return metrics
    
    return router


def create_generation_router() -> APIRouter:
    """Create audio generation router."""
    router = APIRouter(prefix="/generate", tags=["generation"])
    
    @router.post("/", response_model=AudioGenerationResponse if FASTAPI_AVAILABLE else None)
    async def generate_audio(
        request: AudioGenerationRequest,
        audio_service: AudioGenerationService = Depends(get_audio_service),
        audio_repo: AudioRepository = Depends(get_audio_repository)
    ):
        """Generate audio from text prompt."""
        try:
            # Generate audio
            result = audio_service.generate_audio(
                prompt=request.prompt,
                duration_seconds=request.duration_seconds,
                temperature=request.temperature,
                cache_key=f"api_{hash(request.prompt)}_{request.duration_seconds}"
            )
            
            # Save to database if requested
            generation_id = None
            if request.save_to_db:
                record_data = AudioRecordData(
                    prompt=request.prompt,
                    audio_path="",  # Will be set after file save
                    duration_seconds=result['duration_seconds'],
                    sample_rate=result['sample_rate'],
                    model_name=result['model_name'],
                    temperature=request.temperature,
                    generation_time_ms=result['generation_time_ms'],
                    metadata=result['audio_stats'],
                    tags=['api_generated']
                )
                generation_id = audio_repo.create_record(record_data)
            
            return {
                "generation_id": str(generation_id) if generation_id else "temp",
                "prompt": request.prompt,
                "duration_seconds": result['duration_seconds'],
                "generation_time_ms": result['generation_time_ms'],
                "model_name": result['model_name'],
                "audio_url": f"/api/audio/{generation_id}" if generation_id else "#",
                "metadata": result['audio_stats']
            }
            
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    @router.post("/transform")
    async def transform_audio(
        transform_request: AudioTransformRequest = Form(...),
        audio_file: UploadFile = File(...),
        audio_service: AudioGenerationService = Depends(get_audio_service)
    ):
        """Transform uploaded audio with text conditioning."""
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                content = await audio_file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            # Transform audio
            result = audio_service.transform_audio(
                input_audio=tmp_file_path,
                prompt=transform_request.prompt,
                strength=transform_request.strength
            )
            
            # Clean up temp file
            Path(tmp_file_path).unlink(missing_ok=True)
            
            return {
                "transformation_id": f"transform_{int(time.time())}",
                "prompt": transform_request.prompt,
                "strength": transform_request.strength,
                "transformation_time_ms": result['transformation_time_ms'],
                "duration_seconds": result['duration_seconds'],
                "metadata": result['audio_stats']
            }
            
        except Exception as e:
            logger.error(f"Audio transformation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Transformation failed: {str(e)}")
    
    @router.get("/history")
    async def get_generation_history(
        page: int = 1,
        page_size: int = 20,
        model_name: Optional[str] = None,
        audio_repo: AudioRepository = Depends(get_audio_repository)
    ):
        """Get generation history with pagination."""
        try:
            offset = (page - 1) * page_size
            records = audio_repo.list_records(
                limit=page_size,
                offset=offset,
                model_name=model_name,
                order_by='created_at',
                order_desc=True
            )
            
            # Get total count (simplified)
            total_records = audio_repo.list_records(limit=1000)  # Simplified count
            total_count = len(total_records)
            
            generations = []
            for record in records:
                generations.append({
                    "id": record.id,
                    "prompt": record.prompt,
                    "duration_seconds": record.duration_seconds,
                    "model_name": record.model_name,
                    "temperature": record.temperature,
                    "created_at": record.created_at.isoformat() if record.created_at else None,
                    "generation_time_ms": record.generation_time_ms,
                    "audio_url": f"/api/audio/{record.id}"
                })
            
            return {
                "generations": generations,
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "has_next": len(records) == page_size
            }
            
        except Exception as e:
            logger.error(f"Failed to get generation history: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve history")
    
    return router


def create_voice_router() -> APIRouter:
    """Create voice cloning router."""
    router = APIRouter(prefix="/voice", tags=["voice"])
    
    @router.post("/clone")
    async def clone_voice(
        clone_request: VoiceCloneRequest = Form(...),
        reference_audio: UploadFile = File(...),
        voice_service: VoiceCloneService = Depends(get_voice_service)
    ):
        """Clone voice from reference audio."""
        if not reference_audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Reference must be an audio file")
        
        try:
            # Save reference audio temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                content = await reference_audio.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            # Clone voice
            result = voice_service.clone_voice(
                reference_audio=tmp_file_path,
                text=clone_request.text,
                speaker_id=clone_request.speaker_id,
                prosody_transfer=clone_request.prosody_transfer,
                emotion_control=clone_request.emotion
            )
            
            # Clean up temp file
            Path(tmp_file_path).unlink(missing_ok=True)
            
            return {
                "clone_id": f"clone_{int(time.time())}",
                "text": clone_request.text,
                "speaker_id": clone_request.speaker_id,
                "cloning_time_ms": result['cloning_time_ms'],
                "duration_seconds": result['duration_seconds'],
                "metadata": result['audio_stats']
            }
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")
    
    @router.get("/speakers")
    async def list_speakers(voice_service: VoiceCloneService = Depends(get_voice_service)):
        """List cached speaker embeddings."""
        try:
            speakers = voice_service.get_speaker_list()
            return {"speakers": speakers}
        except Exception as e:
            logger.error(f"Failed to list speakers: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve speakers")
    
    return router


def create_audio_router() -> APIRouter:
    """Create audio file serving router."""
    router = APIRouter(prefix="/audio", tags=["audio"])
    
    @router.get("/{generation_id}")
    async def get_audio_file(
        generation_id: int,
        audio_repo: AudioRepository = Depends(get_audio_repository)
    ):
        """Download generated audio file."""
        try:
            record = audio_repo.get_record(generation_id)
            if not record:
                raise HTTPException(status_code=404, detail="Audio not found")
            
            audio_path = Path(record.audio_path)
            if not audio_path.exists():
                raise HTTPException(status_code=404, detail="Audio file not found")
            
            return FileResponse(
                path=audio_path,
                media_type="audio/wav",
                filename=f"generated_{generation_id}.wav"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to serve audio {generation_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve audio")
    
    return router


def register_routes(app) -> None:
    """Register all API routes with the FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available, skipping route registration")
        return
    
    # API router
    api_router = APIRouter(prefix="/api/v1")
    
    # Register sub-routers
    api_router.include_router(create_health_router())
    api_router.include_router(create_generation_router())
    api_router.include_router(create_voice_router())
    api_router.include_router(create_audio_router())
    
    # Include API router in main app
    app.include_router(api_router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Fugatto Audio Lab API",
            "version": "1.0.0",
            "description": "REST API for AI-powered audio generation and transformation",
            "docs_url": "/docs",
            "health_url": "/api/v1/health/",
            "endpoints": {
                "generation": "/api/v1/generate/",
                "voice_cloning": "/api/v1/voice/",
                "audio_files": "/api/v1/audio/",
                "health": "/api/v1/health/"
            }
        }
    
    logger.info("All API routes registered successfully")