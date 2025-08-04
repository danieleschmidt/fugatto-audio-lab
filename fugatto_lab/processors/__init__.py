"""Advanced audio processing pipelines and algorithms."""

from .pipeline import ProcessingPipeline, ProcessingStage, create_default_pipeline

__all__ = [
    "ProcessingPipeline",
    "ProcessingStage", 
    "create_default_pipeline"
]