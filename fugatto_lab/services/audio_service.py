"""Core audio generation service with business logic."""

import logging
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import numpy as np

from ..core import FugattoModel, AudioProcessor
from ..monitoring import get_monitor


logger = logging.getLogger(__name__)


class AudioGenerationService:
    """High-level service for audio generation operations."""
    
    def __init__(self, model_name: str = "nvidia/fugatto-base", 
                 cache_enabled: bool = True, max_cache_size: int = 100):
        """Initialize audio generation service.
        
        Args:
            model_name: Model identifier
            cache_enabled: Whether to cache generations
            max_cache_size: Maximum number of cached generations
        """
        self.model_name = model_name
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size
        
        # Lazy initialization
        self._model = None
        self._processor = None
        self._generation_cache = {}
        self.monitor = get_monitor()
        
        logger.info(f"AudioGenerationService initialized: {model_name}")
    
    @property
    def model(self) -> FugattoModel:
        """Get or create model instance."""
        if self._model is None:
            self._model = FugattoModel.from_pretrained(self.model_name)
        return self._model
    
    @property
    def processor(self) -> AudioProcessor:
        """Get or create audio processor."""
        if self._processor is None:
            self._processor = AudioProcessor()
        return self._processor
    
    def generate_audio(self, prompt: str, duration_seconds: float = 10.0,
                      temperature: float = 0.8, output_path: Optional[Union[str, Path]] = None,
                      normalize: bool = True, cache_key: Optional[str] = None) -> Dict[str, Any]:
        """Generate audio with comprehensive options.
        
        Args:
            prompt: Text description for generation
            duration_seconds: Duration of output audio
            temperature: Generation randomness
            output_path: Optional file save path
            normalize: Whether to normalize output
            cache_key: Optional cache key for reuse
            
        Returns:
            Dictionary with audio data and metadata
        """
        start_time = time.time()
        
        # Check cache first
        if self.cache_enabled and cache_key:
            cached_result = self._get_cached_generation(cache_key)
            if cached_result:
                logger.info(f"Returning cached generation for: {cache_key}")
                return cached_result
        
        logger.info(f"Generating audio: '{prompt}' ({duration_seconds}s)")
        
        try:
            # Generate audio
            audio_data = self.model.generate(
                prompt=prompt,
                duration_seconds=duration_seconds,
                temperature=temperature
            )
            
            # Post-process audio
            if normalize:
                audio_data = self.processor.normalize_loudness(audio_data)
            
            # Prepare result
            result = {
                'audio_data': audio_data,
                'sample_rate': self.processor.sample_rate,
                'duration_seconds': len(audio_data) / self.processor.sample_rate,
                'prompt': prompt,
                'temperature': temperature,
                'generation_time_ms': (time.time() - start_time) * 1000,
                'model_name': self.model_name,
                'audio_stats': self.processor.get_audio_stats(audio_data)
            }
            
            # Save to file if requested
            if output_path:
                self.processor.save_audio(audio_data, output_path)
                result['output_path'] = str(output_path)
            
            # Cache result
            if self.cache_enabled and cache_key:
                self._cache_generation(cache_key, result)
            
            # Record metrics
            self.monitor.record_generation_metrics(
                start_time, prompt, duration_seconds, audio_data, self.model_name
            )
            
            logger.info(f"Audio generation completed in {result['generation_time_ms']:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            raise AudioGenerationError(f"Generation failed: {e}") from e
    
    def transform_audio(self, input_audio: Union[np.ndarray, str, Path], 
                       prompt: str, strength: float = 0.7,
                       output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Transform existing audio with text conditioning.
        
        Args:
            input_audio: Audio data or file path
            prompt: Transformation description
            strength: Transformation intensity
            output_path: Optional save path
            
        Returns:
            Dictionary with transformed audio and metadata
        """
        start_time = time.time()
        
        # Load audio if path provided
        if isinstance(input_audio, (str, Path)):
            audio_data = self.processor.load_audio(input_audio)
            input_path = str(input_audio)
        else:
            audio_data = input_audio
            input_path = None
        
        logger.info(f"Transforming audio: '{prompt}' (strength={strength})")
        
        try:
            # Transform audio
            transformed_data = self.model.transform(
                audio=audio_data,
                prompt=prompt,
                strength=strength
            )
            
            # Prepare result
            result = {
                'audio_data': transformed_data,
                'original_audio': audio_data,
                'sample_rate': self.processor.sample_rate,
                'duration_seconds': len(transformed_data) / self.processor.sample_rate,
                'prompt': prompt,
                'strength': strength,
                'transformation_time_ms': (time.time() - start_time) * 1000,
                'input_path': input_path,
                'audio_stats': self.processor.get_audio_stats(transformed_data)
            }
            
            # Save to file if requested
            if output_path:
                self.processor.save_audio(transformed_data, output_path)
                result['output_path'] = str(output_path)
            
            logger.info(f"Audio transformation completed in {result['transformation_time_ms']:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Audio transformation failed: {e}")
            raise AudioTransformationError(f"Transformation failed: {e}") from e
    
    def batch_generate(self, prompts: List[str], duration_seconds: float = 10.0,
                      temperature: float = 0.8, output_dir: Optional[Union[str, Path]] = None,
                      parallel: bool = False) -> List[Dict[str, Any]]:
        """Generate multiple audio clips in batch.
        
        Args:
            prompts: List of text prompts
            duration_seconds: Duration per clip
            temperature: Generation temperature
            output_dir: Directory to save outputs
            parallel: Whether to use parallel processing (future feature)
            
        Returns:
            List of generation results
        """
        logger.info(f"Batch generating {len(prompts)} audio clips")
        
        results = []
        output_dir = Path(output_dir) if output_dir else None
        
        for i, prompt in enumerate(prompts):
            try:
                # Prepare output path
                output_path = None
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    safe_filename = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    output_path = output_dir / f"{i:03d}_{safe_filename}.wav"
                
                # Generate audio
                result = self.generate_audio(
                    prompt=prompt,
                    duration_seconds=duration_seconds,
                    temperature=temperature,
                    output_path=output_path,
                    cache_key=f"batch_{i}_{prompt}"
                )
                
                result['batch_index'] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch generation failed for prompt {i}: {e}")
                results.append({
                    'batch_index': i,
                    'prompt': prompt,
                    'error': str(e),
                    'success': False
                })
        
        successful = sum(1 for r in results if 'error' not in r)
        logger.info(f"Batch generation completed: {successful}/{len(prompts)} successful")
        
        return results
    
    def compare_generations(self, prompt: str, temperatures: List[float],
                           duration_seconds: float = 10.0) -> Dict[str, Any]:
        """Generate multiple versions with different temperatures for comparison.
        
        Args:
            prompt: Text prompt
            temperatures: List of temperature values to try
            duration_seconds: Duration for each generation
            
        Returns:
            Comparison results with statistics
        """
        logger.info(f"Comparing generations for '{prompt}' with {len(temperatures)} temperatures")
        
        generations = []
        for temp in temperatures:
            try:
                result = self.generate_audio(
                    prompt=prompt,
                    duration_seconds=duration_seconds,
                    temperature=temp,
                    cache_key=f"compare_{prompt}_{temp}"
                )
                result['temperature'] = temp
                generations.append(result)
            except Exception as e:
                logger.error(f"Generation failed for temperature {temp}: {e}")
        
        # Calculate comparison statistics
        if generations:
            generation_times = [g['generation_time_ms'] for g in generations]
            audio_stats = [g['audio_stats']['rms'] for g in generations]
            
            comparison_stats = {
                'prompt': prompt,
                'num_generations': len(generations),
                'temperatures': temperatures,
                'avg_generation_time_ms': sum(generation_times) / len(generation_times),
                'avg_rms': sum(audio_stats) / len(audio_stats),
                'generations': generations
            }
        else:
            comparison_stats = {
                'prompt': prompt,
                'num_generations': 0,
                'error': 'All generations failed'
            }
        
        return comparison_stats
    
    def _get_cached_generation(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached generation result."""
        return self._generation_cache.get(cache_key)
    
    def _cache_generation(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache generation result."""
        if len(self._generation_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._generation_cache))
            del self._generation_cache[oldest_key]
        
        # Store without audio data to save memory
        cache_result = result.copy()
        cache_result['audio_data'] = None  # Don't cache large audio arrays
        self._generation_cache[cache_key] = cache_result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_enabled': self.cache_enabled,
            'cache_size': len(self._generation_cache),
            'max_cache_size': self.max_cache_size,
            'cache_keys': list(self._generation_cache.keys())
        }
    
    def clear_cache(self) -> None:
        """Clear generation cache."""
        self._generation_cache.clear()
        logger.info("Generation cache cleared")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        model_info = self.model.get_model_info() if self._model else {}
        cache_stats = self.get_cache_stats()
        monitor_stats = self.monitor.get_performance_summary()
        
        return {
            'service': 'AudioGenerationService',
            'model_name': self.model_name,
            'model_info': model_info,
            'cache_stats': cache_stats,
            'performance_stats': monitor_stats
        }


class AudioGenerationError(Exception):
    """Exception raised for audio generation errors."""
    pass


class AudioTransformationError(Exception):
    """Exception raised for audio transformation errors."""
    pass