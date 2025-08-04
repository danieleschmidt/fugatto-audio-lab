"""Audio generation and processing API endpoints."""

import logging
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

try:
    from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
    from fastapi.responses import FileResponse, StreamingResponse
    from pydantic import BaseModel, Field, validator
    FASTAPI_AVAILABLE = True
except ImportError:
    # Fallback definitions when FastAPI is not available
    FASTAPI_AVAILABLE = False
    APIRouter = None
    HTTPException = None
    BaseModel = object
    Field = lambda **kwargs: None

from ...core import FugattoModel, AudioProcessor
from ...services import AudioGenerationService, VoiceCloneService
from ...analyzers import AudioAnalyzer
from ...processors import ProcessingPipeline
from ...database.repositories import AudioRepository
from ...database.models import AudioRecordData

logger = logging.getLogger(__name__)

# Pydantic models for API requests and responses
if FASTAPI_AVAILABLE:
    
    class AudioGenerationRequest(BaseModel):
        """Request model for audio generation."""
        prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for audio generation")
        duration_seconds: float = Field(10.0, ge=1.0, le=30.0, description="Duration of generated audio")
        temperature: float = Field(0.8, ge=0.1, le=1.5, description="Sampling temperature")
        top_p: float = Field(0.95, ge=0.1, le=1.0, description="Nucleus sampling parameter")
        guidance_scale: float = Field(3.0, ge=1.0, le=10.0, description="Classifier-free guidance scale")
        model_name: str = Field("nvidia/fugatto-base", description="Model to use for generation")
        save_output: bool = Field(True, description="Whether to save generated audio")
        output_format: str = Field("wav", regex="^(wav|mp3|flac)$", description="Output audio format")
        
        @validator('prompt')
        def validate_prompt(cls, v):
            if not v.strip():
                raise ValueError("Prompt cannot be empty")
            return v.strip()
    
    
    class AudioTransformRequest(BaseModel):
        """Request model for audio transformation."""
        prompt: str = Field(..., min_length=1, max_length=500, description="Transformation description")
        strength: float = Field(0.7, ge=0.0, le=1.0, description="Transformation strength")
        preserve_length: bool = Field(True, description="Whether to preserve original length")
        output_format: str = Field("wav", regex="^(wav|mp3|flac)$", description="Output audio format")
        
        @validator('prompt')
        def validate_prompt(cls, v):
            if not v.strip():
                raise ValueError("Transformation prompt cannot be empty")
            return v.strip()
    
    
    class VoiceCloneRequest(BaseModel):
        """Request model for voice cloning."""
        text: str = Field(..., min_length=1, max_length=1000, description="Text to synthesize")
        speaker_id: Optional[str] = Field(None, description="Cached speaker ID")
        prosody_transfer: bool = Field(True, description="Transfer prosody patterns")
        emotion_control: Optional[str] = Field(None, description="Emotion modification")
        output_format: str = Field("wav", regex="^(wav|mp3|flac)$", description="Output format")
        
        @validator('text')  
        def validate_text(cls, v):
            if not v.strip():
                raise ValueError("Text cannot be empty")
            return v.strip()
    
    
    class AudioAnalysisRequest(BaseModel):
        """Request model for audio analysis."""
        include_features: List[str] = Field(default_factory=lambda: ["basic", "spectral", "temporal"], 
                                          description="Feature types to include")
        detailed_analysis: bool = Field(False, description="Include detailed analysis")
        compare_reference: bool = Field(False, description="Compare with reference audio if provided")
        
        @validator('include_features')
        def validate_features(cls, v):
            valid_features = ["basic", "spectral", "temporal", "perceptual", "content"]
            for feature in v:
                if feature not in valid_features:
                    raise ValueError(f"Invalid feature type: {feature}")
            return v
    
    
    class BatchProcessingRequest(BaseModel):
        """Request model for batch processing."""
        operation: str = Field(..., regex="^(generate|transform|analyze|clone)$", description="Batch operation type")
        parameters: Dict[str, Any] = Field(..., description="Parameters for each operation")
        parallel_workers: int = Field(4, ge=1, le=16, description="Number of parallel workers")
        save_individual: bool = Field(True, description="Save individual results")
        create_manifest: bool = Field(True, description="Create result manifest")
    
    
    class AudioGenerationResponse(BaseModel):
        """Response model for audio generation."""
        task_id: str
        audio_path: Optional[str] = None
        duration_seconds: float
        generation_time_ms: float
        model_used: str
        parameters: Dict[str, Any]
        audio_stats: Dict[str, Any]
        status: str
        created_at: str
    
    
    class AudioAnalysisResponse(BaseModel):
        """Response model for audio analysis."""
        task_id: str
        analysis_results: Dict[str, Any]
        analysis_time_ms: float
        audio_stats: Dict[str, Any]
        status: str
        created_at: str
    
    
    class BatchProcessingResponse(BaseModel):
        """Response model for batch processing."""
        batch_id: str
        operation: str
        total_tasks: int
        completed_tasks: int
        failed_tasks: int
        progress_percentage: float
        estimated_completion_time: Optional[str] = None
        results_manifest: Optional[str] = None
        status: str
        created_at: str


# Dependency injection functions
def get_audio_generation_service() -> AudioGenerationService:
    """Get audio generation service instance."""
    return AudioGenerationService()


def get_voice_clone_service() -> VoiceCloneService:
    """Get voice cloning service instance."""
    return VoiceCloneService()


def get_audio_analyzer() -> AudioAnalyzer:
    """Get audio analyzer instance."""
    return AudioAnalyzer()


def get_audio_processor() -> AudioProcessor:
    """Get audio processor instance."""
    return AudioProcessor()


def get_audio_repository() -> AudioRepository:
    """Get audio repository instance."""
    return AudioRepository()


# API Router
if FASTAPI_AVAILABLE:
    router = APIRouter(prefix="/audio", tags=["audio"])
    
    # In-memory storage for batch jobs (in production, use Redis or database)
    batch_jobs = {}
    
    
    @router.post("/generate", response_model=AudioGenerationResponse)
    async def generate_audio(
        request: AudioGenerationRequest,
        background_tasks: BackgroundTasks,
        generation_service: AudioGenerationService = Depends(get_audio_generation_service),
        repository: AudioRepository = Depends(get_audio_repository)
    ):
        """Generate audio from text prompt.
        
        Generate high-quality audio from text descriptions using Fugatto models.
        Supports various parameters for controlling generation quality and style.
        """
        try:
            task_id = str(uuid.uuid4())
            start_time = time.time()
            
            logger.info(f"Starting audio generation task {task_id}: '{request.prompt[:50]}...'")
            
            # Generate audio
            result = generation_service.generate_audio(
                prompt=request.prompt,
                duration_seconds=request.duration_seconds,
                temperature=request.temperature,
                top_p=request.top_p,
                guidance_scale=request.guidance_scale,
                model_name=request.model_name
            )
            
            generation_time = (time.time() - start_time) * 1000
            
            # Save to storage if requested
            audio_path = None
            if request.save_output:
                output_dir = Path("outputs") / "generated"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_{timestamp}_{task_id[:8]}.{request.output_format}"
                audio_path = output_dir / filename
                
                # Save audio file
                processor = AudioProcessor()
                processor.save_audio(
                    result['generated_audio'], 
                    audio_path, 
                    format=request.output_format
                )
                
                # Store record in database
                record_data = AudioRecordData(
                    prompt=request.prompt,
                    audio_path=str(audio_path),
                    duration_seconds=result['duration_seconds'],
                    sample_rate=result['sample_rate'],
                    model_name=request.model_name,
                    temperature=request.temperature,
                    generation_time_ms=generation_time,
                    metadata={\n                        'top_p': request.top_p,\n                        'guidance_scale': request.guidance_scale,\n                        'output_format': request.output_format,\n                        'task_id': task_id\n                    },\n                    tags=['generated', 'api'],\n                    created_at=datetime.now()\n                )\n                \n                # Save to database in background\n                background_tasks.add_task(repository.create_record, record_data)\n            \n            response = AudioGenerationResponse(\n                task_id=task_id,\n                audio_path=str(audio_path) if audio_path else None,\n                duration_seconds=result['duration_seconds'],\n                generation_time_ms=generation_time,\n                model_used=request.model_name,\n                parameters={\n                    'prompt': request.prompt,\n                    'temperature': request.temperature,\n                    'top_p': request.top_p,\n                    'guidance_scale': request.guidance_scale\n                },\n                audio_stats=result.get('audio_stats', {}),\n                status='completed',\n                created_at=datetime.now().isoformat()\n            )\n            \n            logger.info(f\"Audio generation completed: {task_id} ({generation_time:.1f}ms)\")\n            return response\n            \n        except Exception as e:\n            logger.error(f\"Audio generation failed for task {task_id}: {e}\")\n            raise HTTPException(status_code=500, detail=f\"Audio generation failed: {str(e)}\")\n    \n    \n    @router.post(\"/transform\")\n    async def transform_audio(\n        audio_file: UploadFile = File(..., description=\"Audio file to transform\"),\n        prompt: str = Form(..., description=\"Transformation description\"),\n        strength: float = Form(0.7, description=\"Transformation strength\"),\n        preserve_length: bool = Form(True, description=\"Preserve original length\"),\n        output_format: str = Form(\"wav\", description=\"Output format\"),\n        generation_service: AudioGenerationService = Depends(get_audio_generation_service),\n        processor: AudioProcessor = Depends(get_audio_processor)\n    ):\n        \"\"\"Transform uploaded audio file with text conditioning.\n        \n        Apply text-guided transformations to uploaded audio files.\n        Supports various transformation types like style transfer, effects, and voice conversion.\n        \"\"\"\n        try:\n            task_id = str(uuid.uuid4())\n            start_time = time.time()\n            \n            logger.info(f\"Starting audio transformation task {task_id}: '{prompt[:50]}...'\")\n            \n            # Validate file type\n            if not audio_file.content_type.startswith('audio/'):\n                raise HTTPException(status_code=400, detail=\"File must be an audio file\")\n            \n            # Save uploaded file temporarily\n            temp_dir = Path(\"temp\")\n            temp_dir.mkdir(exist_ok=True)\n            \n            temp_path = temp_dir / f\"upload_{task_id}_{audio_file.filename}\"\n            with open(temp_path, \"wb\") as f:\n                content = await audio_file.read()\n                f.write(content)\n            \n            try:\n                # Load and process audio\n                input_audio = processor.load_audio(temp_path)\n                \n                # Transform audio\n                result = generation_service.transform_audio(\n                    audio=input_audio,\n                    prompt=prompt,\n                    strength=strength,\n                    preserve_length=preserve_length\n                )\n                \n                # Save transformed audio\n                output_dir = Path(\"outputs\") / \"transformed\"\n                output_dir.mkdir(parents=True, exist_ok=True)\n                \n                timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n                output_filename = f\"transformed_{timestamp}_{task_id[:8]}.{output_format}\"\n                output_path = output_dir / output_filename\n                \n                processor.save_audio(\n                    result['transformed_audio'],\n                    output_path,\n                    format=output_format\n                )\n                \n                transformation_time = (time.time() - start_time) * 1000\n                \n                response = {\n                    \"task_id\": task_id,\n                    \"input_filename\": audio_file.filename,\n                    \"output_path\": str(output_path),\n                    \"transformation_prompt\": prompt,\n                    \"strength\": strength,\n                    \"duration_seconds\": result['duration_seconds'],\n                    \"transformation_time_ms\": transformation_time,\n                    \"audio_stats\": result.get('audio_stats', {}),\n                    \"status\": \"completed\",\n                    \"created_at\": datetime.now().isoformat()\n                }\n                \n                logger.info(f\"Audio transformation completed: {task_id} ({transformation_time:.1f}ms)\")\n                return response\n                \n            finally:\n                # Clean up temp file\n                if temp_path.exists():\n                    temp_path.unlink()\n                    \n        except Exception as e:\n            logger.error(f\"Audio transformation failed for task {task_id}: {e}\")\n            raise HTTPException(status_code=500, detail=f\"Audio transformation failed: {str(e)}\")\n    \n    \n    @router.post(\"/clone\", response_model=Dict[str, Any])\n    async def clone_voice(\n        reference_audio: UploadFile = File(..., description=\"Reference audio for voice cloning\"),\n        text: str = Form(..., description=\"Text to synthesize\"),\n        speaker_id: Optional[str] = Form(None, description=\"Speaker identifier for caching\"),\n        prosody_transfer: bool = Form(True, description=\"Transfer prosody patterns\"),\n        emotion_control: Optional[str] = Form(None, description=\"Emotion control\"),\n        output_format: str = Form(\"wav\", description=\"Output format\"),\n        voice_service: VoiceCloneService = Depends(get_voice_clone_service),\n        processor: AudioProcessor = Depends(get_audio_processor)\n    ):\n        \"\"\"Clone voice from reference audio and synthesize new text.\n        \n        Create speech in a target voice using a reference audio sample.\n        Supports prosody transfer and emotion control for natural-sounding results.\n        \"\"\"\n        try:\n            task_id = str(uuid.uuid4())\n            start_time = time.time()\n            \n            logger.info(f\"Starting voice cloning task {task_id}: '{text[:50]}...'\")\n            \n            # Validate inputs\n            if not reference_audio.content_type.startswith('audio/'):\n                raise HTTPException(status_code=400, detail=\"Reference file must be an audio file\")\n            \n            if len(text.strip()) == 0:\n                raise HTTPException(status_code=400, detail=\"Text cannot be empty\")\n            \n            # Save reference audio temporarily\n            temp_dir = Path(\"temp\")\n            temp_dir.mkdir(exist_ok=True)\n            \n            ref_temp_path = temp_dir / f\"ref_{task_id}_{reference_audio.filename}\"\n            with open(ref_temp_path, \"wb\") as f:\n                content = await reference_audio.read()\n                f.write(content)\n            \n            try:\n                # Perform voice cloning\n                result = voice_service.clone_voice(\n                    reference_audio=ref_temp_path,\n                    text=text,\n                    speaker_id=speaker_id,\n                    prosody_transfer=prosody_transfer,\n                    emotion_control=emotion_control\n                )\n                \n                # Save cloned audio\n                output_dir = Path(\"outputs\") / \"cloned\"\n                output_dir.mkdir(parents=True, exist_ok=True)\n                \n                timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n                output_filename = f\"cloned_{timestamp}_{task_id[:8]}.{output_format}\"\n                output_path = output_dir / output_filename\n                \n                processor.save_audio(\n                    result['cloned_audio'],\n                    output_path,\n                    format=output_format\n                )\n                \n                cloning_time = (time.time() - start_time) * 1000\n                \n                response = {\n                    \"task_id\": task_id,\n                    \"reference_filename\": reference_audio.filename,\n                    \"synthesized_text\": text,\n                    \"speaker_id\": result.get('speaker_id'),\n                    \"output_path\": str(output_path),\n                    \"duration_seconds\": result['duration_seconds'],\n                    \"cloning_time_ms\": cloning_time,\n                    \"prosody_transfer\": prosody_transfer,\n                    \"emotion_control\": emotion_control,\n                    \"audio_stats\": result.get('audio_stats', {}),\n                    \"status\": \"completed\",\n                    \"created_at\": datetime.now().isoformat()\n                }\n                \n                logger.info(f\"Voice cloning completed: {task_id} ({cloning_time:.1f}ms)\")\n                return response\n                \n            finally:\n                # Clean up temp file\n                if ref_temp_path.exists():\n                    ref_temp_path.unlink()\n                    \n        except Exception as e:\n            logger.error(f\"Voice cloning failed for task {task_id}: {e}\")\n            raise HTTPException(status_code=500, detail=f\"Voice cloning failed: {str(e)}\")\n    \n    \n    @router.post(\"/analyze\", response_model=AudioAnalysisResponse)\n    async def analyze_audio(\n        audio_file: UploadFile = File(..., description=\"Audio file to analyze\"),\n        include_features: List[str] = Form(default=[\"basic\", \"spectral\"], description=\"Features to include\"),\n        detailed_analysis: bool = Form(False, description=\"Include detailed analysis\"),\n        compare_reference: bool = Form(False, description=\"Compare with reference\"),\n        analyzer: AudioAnalyzer = Depends(get_audio_analyzer),\n        processor: AudioProcessor = Depends(get_audio_processor)\n    ):\n        \"\"\"Analyze uploaded audio file and extract comprehensive features.\n        \n        Perform detailed audio analysis including spectral, temporal, and perceptual features.\n        Useful for understanding audio characteristics and quality assessment.\n        \"\"\"\n        try:\n            task_id = str(uuid.uuid4())\n            start_time = time.time()\n            \n            logger.info(f\"Starting audio analysis task {task_id}: {audio_file.filename}\")\n            \n            # Validate file\n            if not audio_file.content_type.startswith('audio/'):\n                raise HTTPException(status_code=400, detail=\"File must be an audio file\")\n            \n            # Save uploaded file temporarily\n            temp_dir = Path(\"temp\")\n            temp_dir.mkdir(exist_ok=True)\n            \n            temp_path = temp_dir / f\"analyze_{task_id}_{audio_file.filename}\"\n            with open(temp_path, \"wb\") as f:\n                content = await audio_file.read()\n                f.write(content)\n            \n            try:\n                # Load audio\n                audio_data = processor.load_audio(temp_path)\n                \n                # Perform analysis\n                if detailed_analysis:\n                    analysis_results = analyzer.analyze_comprehensive(audio_data)\n                else:\n                    # Basic analysis only\n                    analysis_results = {}\n                    if \"basic\" in include_features:\n                        analysis_results[\"basic\"] = analyzer._analyze_basic_stats(audio_data)\n                    if \"spectral\" in include_features:\n                        analysis_results[\"spectral\"] = analyzer._analyze_spectral_features(audio_data)\n                    if \"temporal\" in include_features:\n                        analysis_results[\"temporal\"] = analyzer._analyze_temporal_features(audio_data)\n                \n                # Add quality analysis\n                quality_analysis = processor.analyze_audio_quality(audio_data)\n                analysis_results[\"quality\"] = quality_analysis\n                \n                # Get audio statistics\n                audio_stats = processor.get_audio_stats(audio_data)\n                \n                analysis_time = (time.time() - start_time) * 1000\n                \n                response = AudioAnalysisResponse(\n                    task_id=task_id,\n                    analysis_results=analysis_results,\n                    analysis_time_ms=analysis_time,\n                    audio_stats=audio_stats,\n                    status=\"completed\",\n                    created_at=datetime.now().isoformat()\n                )\n                \n                logger.info(f\"Audio analysis completed: {task_id} ({analysis_time:.1f}ms)\")\n                return response\n                \n            finally:\n                # Clean up temp file\n                if temp_path.exists():\n                    temp_path.unlink()\n                    \n        except Exception as e:\n            logger.error(f\"Audio analysis failed for task {task_id}: {e}\")\n            raise HTTPException(status_code=500, detail=f\"Audio analysis failed: {str(e)}\")\n    \n    \n    @router.get(\"/download/{task_id}\")\n    async def download_audio(task_id: str):\n        \"\"\"Download generated audio file by task ID.\n        \n        Retrieve and download audio files generated by previous API calls.\n        Supports various audio formats and includes proper content headers.\n        \"\"\"\n        try:\n            # Search for audio file in output directories\n            search_dirs = [\"outputs/generated\", \"outputs/transformed\", \"outputs/cloned\"]\n            \n            audio_file = None\n            for search_dir in search_dirs:\n                output_dir = Path(search_dir)\n                if output_dir.exists():\n                    # Look for files containing the task ID\n                    for file_path in output_dir.glob(f\"*{task_id[:8]}*\"):\n                        if file_path.is_file():\n                            audio_file = file_path\n                            break\n                \n                if audio_file:\n                    break\n            \n            if not audio_file or not audio_file.exists():\n                raise HTTPException(status_code=404, detail=f\"Audio file not found for task {task_id}\")\n            \n            # Determine content type based on file extension\n            content_type_map = {\n                '.wav': 'audio/wav',\n                '.mp3': 'audio/mpeg',\n                '.flac': 'audio/flac',\n                '.ogg': 'audio/ogg'\n            }\n            content_type = content_type_map.get(audio_file.suffix.lower(), 'audio/wav')\n            \n            logger.info(f\"Serving audio file: {audio_file} for task {task_id}\")\n            \n            return FileResponse(\n                path=audio_file,\n                media_type=content_type,\n                filename=audio_file.name,\n                headers={\"Content-Disposition\": f\"attachment; filename={audio_file.name}\"}\n            )\n            \n        except HTTPException:\n            raise\n        except Exception as e:\n            logger.error(f\"Download failed for task {task_id}: {e}\")\n            raise HTTPException(status_code=500, detail=f\"Download failed: {str(e)}\")\n    \n    \n    @router.post(\"/batch\", response_model=BatchProcessingResponse)\n    async def start_batch_processing(\n        request: BatchProcessingRequest,\n        background_tasks: BackgroundTasks\n    ):\n        \"\"\"Start batch processing operation.\n        \n        Process multiple audio operations in parallel for efficient bulk processing.\n        Supports generation, transformation, analysis, and voice cloning operations.\n        \"\"\"\n        try:\n            batch_id = str(uuid.uuid4())\n            \n            logger.info(f\"Starting batch processing {batch_id}: {request.operation} with {request.parallel_workers} workers\")\n            \n            # Initialize batch job tracking\n            batch_jobs[batch_id] = {\n                'batch_id': batch_id,\n                'operation': request.operation,\n                'total_tasks': 0,\n                'completed_tasks': 0,\n                'failed_tasks': 0,\n                'progress_percentage': 0.0,\n                'status': 'initializing',\n                'created_at': datetime.now().isoformat(),\n                'results': []\n            }\n            \n            # Start batch processing in background\n            background_tasks.add_task(\n                _process_batch,\n                batch_id,\n                request.operation,\n                request.parameters,\n                request.parallel_workers,\n                request.save_individual,\n                request.create_manifest\n            )\n            \n            response = BatchProcessingResponse(\n                batch_id=batch_id,\n                operation=request.operation,\n                total_tasks=0,  # Will be updated as tasks are processed\n                completed_tasks=0,\n                failed_tasks=0,\n                progress_percentage=0.0,\n                status='initializing',\n                created_at=datetime.now().isoformat()\n            )\n            \n            return response\n            \n        except Exception as e:\n            logger.error(f\"Batch processing initialization failed: {e}\")\n            raise HTTPException(status_code=500, detail=f\"Batch processing failed: {str(e)}\")\n    \n    \n    @router.get(\"/batch/{batch_id}\", response_model=BatchProcessingResponse)\n    async def get_batch_status(batch_id: str):\n        \"\"\"Get status of batch processing operation.\n        \n        Monitor the progress of batch processing jobs and retrieve completion status.\n        \"\"\"\n        if batch_id not in batch_jobs:\n            raise HTTPException(status_code=404, detail=f\"Batch job {batch_id} not found\")\n        \n        job_info = batch_jobs[batch_id]\n        \n        response = BatchProcessingResponse(\n            batch_id=batch_id,\n            operation=job_info['operation'],\n            total_tasks=job_info['total_tasks'],\n            completed_tasks=job_info['completed_tasks'],\n            failed_tasks=job_info['failed_tasks'],\n            progress_percentage=job_info['progress_percentage'],\n            estimated_completion_time=job_info.get('estimated_completion_time'),\n            results_manifest=job_info.get('results_manifest'),\n            status=job_info['status'],\n            created_at=job_info['created_at']\n        )\n        \n        return response\n    \n    \n    @router.get(\"/records\")\n    async def list_audio_records(\n        limit: int = 100,\n        offset: int = 0,\n        model_name: Optional[str] = None,\n        min_duration: Optional[float] = None,\n        max_duration: Optional[float] = None,\n        repository: AudioRepository = Depends(get_audio_repository)\n    ):\n        \"\"\"List audio generation records with filtering and pagination.\n        \n        Retrieve historical audio generation records with support for filtering\n        by model, duration, and other parameters.\n        \"\"\"\n        try:\n            # Build filters\n            filters = {}\n            if model_name:\n                filters['model_name'] = model_name\n            if min_duration is not None:\n                filters['duration_seconds'] = {'op': 'gte', 'value': min_duration}\n            if max_duration is not None:\n                if 'duration_seconds' in filters:\n                    # Combine with existing duration filter\n                    filters['duration_seconds'] = {\n                        'op': 'and',\n                        'conditions': [\n                            filters['duration_seconds'],\n                            {'op': 'lte', 'value': max_duration}\n                        ]\n                    }\n                else:\n                    filters['duration_seconds'] = {'op': 'lte', 'value': max_duration}\n            \n            # Get records\n            records = repository.list_records(\n                limit=limit,\n                offset=offset,\n                filters=filters,\n                order_by='created_at',\n                order_desc=True\n            )\n            \n            # Convert to response format\n            record_dicts = [record.to_dict() for record in records]\n            \n            # Get total count for pagination\n            total_count = len(record_dicts)  # Simplified - in production, get actual count\n            \n            response = {\n                'records': record_dicts,\n                'pagination': {\n                    'limit': limit,\n                    'offset': offset,\n                    'total_count': total_count,\n                    'has_more': total_count == limit\n                },\n                'filters_applied': filters\n            }\n            \n            return response\n            \n        except Exception as e:\n            logger.error(f\"Failed to list audio records: {e}\")\n            raise HTTPException(status_code=500, detail=f\"Failed to list records: {str(e)}\")\n    \n    \n    @router.get(\"/stats\")\n    async def get_audio_stats(\n        repository: AudioRepository = Depends(get_audio_repository)\n    ):\n        \"\"\"Get comprehensive audio generation statistics.\n        \n        Retrieve statistics about audio generation usage, performance,\n        and system health metrics.\n        \"\"\"\n        try:\n            # Get repository statistics\n            repo_stats = repository.get_statistics()\n            \n            # Get aggregated statistics by model\n            model_stats = repository.get_aggregated_stats(group_by='model_name')\n            \n            # Get recent activity\n            from datetime import timedelta\n            week_ago = datetime.now() - timedelta(days=7)\n            recent_stats = repository.get_aggregated_stats(\n                date_range=(week_ago, datetime.now())\n            )\n            \n            response = {\n                'overall_statistics': repo_stats,\n                'model_statistics': model_stats,\n                'recent_activity_7days': recent_stats,\n                'system_info': {\n                    'api_version': '1.0.0',\n                    'timestamp': datetime.now().isoformat()\n                }\n            }\n            \n            return response\n            \n        except Exception as e:\n            logger.error(f\"Failed to get audio statistics: {e}\")\n            raise HTTPException(status_code=500, detail=f\"Failed to get statistics: {str(e)}\")\n\n\n# Background task functions\nasync def _process_batch(batch_id: str, operation: str, parameters: Dict[str, Any],\n                        parallel_workers: int, save_individual: bool, create_manifest: bool):\n    \"\"\"Process batch operation in background.\"\"\"\n    try:\n        logger.info(f\"Processing batch {batch_id}: {operation}\")\n        \n        # Update status\n        batch_jobs[batch_id]['status'] = 'processing'\n        \n        # Simulate batch processing (in real implementation, use actual processing logic)\n        import asyncio\n        \n        # Mock task list based on parameters\n        if operation == 'generate':\n            tasks = parameters.get('prompts', [])\n        elif operation == 'transform':\n            tasks = parameters.get('transformations', [])\n        else:\n            tasks = parameters.get('items', [])\n        \n        batch_jobs[batch_id]['total_tasks'] = len(tasks)\n        \n        # Process tasks (simplified simulation)\n        for i, task in enumerate(tasks):\n            await asyncio.sleep(0.1)  # Simulate processing time\n            \n            # Update progress\n            batch_jobs[batch_id]['completed_tasks'] = i + 1\n            batch_jobs[batch_id]['progress_percentage'] = ((i + 1) / len(tasks)) * 100\n            \n            # Add mock result\n            batch_jobs[batch_id]['results'].append({\n                'task_index': i,\n                'status': 'completed',\n                'output_path': f'outputs/batch_{batch_id}/task_{i}.wav'\n            })\n        \n        # Create manifest if requested\n        if create_manifest:\n            manifest_path = f'outputs/batch_{batch_id}/manifest.json'\n            batch_jobs[batch_id]['results_manifest'] = manifest_path\n        \n        batch_jobs[batch_id]['status'] = 'completed'\n        logger.info(f\"Batch processing completed: {batch_id}\")\n        \n    except Exception as e:\n        logger.error(f\"Batch processing failed for {batch_id}: {e}\")\n        batch_jobs[batch_id]['status'] = 'failed'\n        batch_jobs[batch_id]['error'] = str(e)\n\n\n# Export router if FastAPI is available\nif FASTAPI_AVAILABLE:\n    __all__ = ['router']\nelse:\n    __all__ = []