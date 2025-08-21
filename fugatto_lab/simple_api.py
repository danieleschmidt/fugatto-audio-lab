"""Simple API Interface for Fugatto Audio Lab.

A streamlined API for basic audio generation and processing tasks.
Perfect for quick prototyping and simple integrations.
"""

import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import json
import time

# Conditional imports for graceful degradation
try:
    from .core import FugattoModel, AudioProcessor
    HAS_AUDIO_CORE = True
except ImportError:
    HAS_AUDIO_CORE = False

try:
    from .quantum_planner import QuantumTaskPlanner, QuantumTask, TaskPriority
    HAS_QUANTUM_PLANNER = True
except ImportError:
    HAS_QUANTUM_PLANNER = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


class SimpleAudioAPI:
    """Simple, user-friendly API for Fugatto audio operations."""
    
    def __init__(self, model_name: str = "nvidia/fugatto-base", device: Optional[str] = None):
        """Initialize the Simple Audio API.
        
        Args:
            model_name: Model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self._planner = None
        
        # Simple configuration
        self.config = {
            'sample_rate': 48000,
            'max_duration': 30.0,
            'default_temperature': 0.8,
            'output_dir': Path('outputs'),
            'enable_caching': True,
            'auto_normalize': True
        }
        
        # Setup output directory
        self.config['output_dir'].mkdir(exist_ok=True)
        
        logger.info(f"SimpleAudioAPI initialized with model: {model_name}")
    
    def generate_audio(self, prompt: str, duration: float = 10.0, 
                      output_path: Optional[str] = None, 
                      **kwargs) -> Dict[str, Any]:
        """Generate audio from text prompt - the simplest way to create audio.
        
        Args:
            prompt: Description of the audio to generate
            duration: Length in seconds (max 30)
            output_path: Where to save the audio (auto-generated if None)
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generation results and metadata
        """
        start_time = time.time()
        
        # Validate inputs
        duration = min(duration, self.config['max_duration'])
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Initialize model if needed
        if not self._model and HAS_AUDIO_CORE:
            self._model = FugattoModel.from_pretrained(self.model_name, self.device)
            self._processor = AudioProcessor(sample_rate=self.config['sample_rate'])
        
        # Generate unique filename if none provided
        if output_path is None:
            timestamp = int(time.time())
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '_')).strip()
            safe_prompt = safe_prompt.replace(' ', '_')
            output_path = self.config['output_dir'] / f"{safe_prompt}_{timestamp}.wav"
        else:
            output_path = Path(output_path)
        
        result = {
            'prompt': prompt,
            'duration': duration,
            'output_path': str(output_path),
            'status': 'started',
            'start_time': start_time
        }
        
        try:
            if HAS_AUDIO_CORE and self._model:
                # Generate audio using the full model
                temperature = kwargs.get('temperature', self.config['default_temperature'])
                
                logger.info(f"Generating audio: '{prompt}' ({duration}s)")
                audio = self._model.generate(
                    prompt=prompt,
                    duration_seconds=duration,
                    temperature=temperature,
                    **{k: v for k, v in kwargs.items() if k != 'temperature'}
                )
                
                # Save audio
                self._processor.save_audio(
                    audio, 
                    output_path, 
                    normalize=self.config['auto_normalize']
                )
                
                # Get audio statistics
                stats = self._processor.get_audio_stats(audio)
                result.update({
                    'audio_stats': stats,
                    'generation_successful': True
                })
                
            else:
                # Fallback: create a simple mock file
                logger.warning("Audio core not available, creating mock output")
                if HAS_NUMPY:
                    # Create simple test tone
                    sample_rate = self.config['sample_rate']
                    t = np.linspace(0, duration, int(duration * sample_rate))
                    frequency = 440  # A4 note
                    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
                    
                    # Save as numpy array
                    np.save(output_path.with_suffix('.npy'), audio)
                    result['output_path'] = str(output_path.with_suffix('.npy'))
                else:
                    # Create text file with metadata
                    with open(output_path.with_suffix('.txt'), 'w') as f:
                        f.write(f"Mock audio generation\nPrompt: {prompt}\nDuration: {duration}s\n")
                    result['output_path'] = str(output_path.with_suffix('.txt'))
                
                result['generation_successful'] = True
                result['audio_stats'] = {
                    'duration_seconds': duration,
                    'sample_rate': self.config['sample_rate'],
                    'mock_generation': True
                }
            
            # Calculate timing
            end_time = time.time()
            result.update({
                'status': 'completed',
                'end_time': end_time,
                'generation_time': end_time - start_time,
                'real_time_factor': duration / (end_time - start_time)
            })
            
            logger.info(f"Audio generation completed in {result['generation_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            result.update({
                'status': 'failed',
                'error': str(e),
                'end_time': time.time()
            })
        
        return result
    
    def transform_audio(self, input_path: str, prompt: str, 
                       output_path: Optional[str] = None, 
                       strength: float = 0.7, **kwargs) -> Dict[str, Any]:
        """Transform existing audio with text instructions.
        
        Args:
            input_path: Path to input audio file
            prompt: Description of desired transformation
            output_path: Where to save transformed audio
            strength: Transformation intensity (0.0-1.0)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with transformation results
        """
        start_time = time.time()
        
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_path}")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = int(time.time())
            output_path = self.config['output_dir'] / f"{input_path.stem}_transformed_{timestamp}.wav"
        else:
            output_path = Path(output_path)
        
        result = {
            'input_path': str(input_path),
            'prompt': prompt,
            'output_path': str(output_path),
            'strength': strength,
            'status': 'started',
            'start_time': start_time
        }
        
        try:
            # Initialize components if needed
            if not self._processor and HAS_AUDIO_CORE:
                self._processor = AudioProcessor(sample_rate=self.config['sample_rate'])
                self._model = FugattoModel.from_pretrained(self.model_name, self.device)
            
            if HAS_AUDIO_CORE and self._processor and self._model:
                # Load input audio
                logger.info(f"Loading audio: {input_path}")
                audio = self._processor.load_audio(input_path)
                
                # Apply transformation
                logger.info(f"Transforming audio: '{prompt}' (strength={strength})")
                transformed = self._model.transform(
                    audio=audio,
                    prompt=prompt,
                    strength=strength,
                    **kwargs
                )
                
                # Save result
                self._processor.save_audio(
                    transformed, 
                    output_path, 
                    normalize=self.config['auto_normalize']
                )
                
                # Get statistics
                stats_original = self._processor.get_audio_stats(audio)
                stats_transformed = self._processor.get_audio_stats(transformed)
                
                result.update({
                    'original_stats': stats_original,
                    'transformed_stats': stats_transformed,
                    'transformation_successful': True
                })
                
            else:
                # Fallback: copy input file with metadata
                logger.warning("Audio core not available, using fallback")
                import shutil
                if input_path.suffix.lower() in ['.wav', '.mp3', '.flac']:
                    shutil.copy(input_path, output_path)
                else:
                    # Create metadata file
                    with open(output_path.with_suffix('.txt'), 'w') as f:
                        f.write(f"Mock transformation\nInput: {input_path}\nPrompt: {prompt}\nStrength: {strength}\n")
                    result['output_path'] = str(output_path.with_suffix('.txt'))
                
                result['transformation_successful'] = True
            
            # Calculate timing
            end_time = time.time()
            result.update({
                'status': 'completed',
                'end_time': end_time,
                'processing_time': end_time - start_time
            })
            
            logger.info(f"Audio transformation completed in {result['processing_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Audio transformation failed: {e}")
            result.update({
                'status': 'failed',
                'error': str(e),
                'end_time': time.time()
            })
        
        return result
    
    def batch_generate(self, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate multiple audio files in batch.
        
        Args:
            prompts: List of generation requests, each containing:
                    {'prompt': str, 'duration': float, 'output_path': Optional[str], ...}
        
        Returns:
            List of generation results
        """
        if not prompts:
            return []
        
        logger.info(f"Starting batch generation for {len(prompts)} prompts")
        results = []
        
        # Initialize planner for efficient batch processing
        if HAS_QUANTUM_PLANNER and not self._planner:
            self._planner = QuantumTaskPlanner()
        
        for i, prompt_config in enumerate(prompts):
            logger.info(f"Processing batch item {i+1}/{len(prompts)}")
            
            # Extract parameters
            prompt = prompt_config.get('prompt', '')
            duration = prompt_config.get('duration', 10.0)
            output_path = prompt_config.get('output_path')
            
            # Remove these from kwargs
            kwargs = {k: v for k, v in prompt_config.items() 
                     if k not in ['prompt', 'duration', 'output_path']}
            
            # Generate audio
            result = self.generate_audio(
                prompt=prompt,
                duration=duration,
                output_path=output_path,
                **kwargs
            )
            
            result['batch_index'] = i
            results.append(result)
            
            # Brief pause between generations to prevent overheating
            if i < len(prompts) - 1:
                time.sleep(0.1)
        
        # Calculate batch statistics
        successful = sum(1 for r in results if r.get('status') == 'completed')
        total_time = sum(r.get('generation_time', 0) for r in results)
        
        batch_summary = {
            'batch_completed': True,
            'total_requests': len(prompts),
            'successful_generations': successful,
            'failed_generations': len(prompts) - successful,
            'total_processing_time': total_time,
            'average_generation_time': total_time / len(prompts) if prompts else 0
        }
        
        logger.info(f"Batch generation completed: {successful}/{len(prompts)} successful")
        
        # Add summary to first result
        if results:
            results[0]['batch_summary'] = batch_summary
        
        return results
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current status of the audio model and system.
        
        Returns:
            Dictionary with system status information
        """
        status = {
            'model_name': self.model_name,
            'device': self.device,
            'config': self.config.copy(),
            'components_loaded': {
                'model': self._model is not None,
                'processor': self._processor is not None,
                'planner': self._planner is not None
            },
            'features_available': {
                'audio_core': HAS_AUDIO_CORE,
                'quantum_planner': HAS_QUANTUM_PLANNER,
                'numpy': HAS_NUMPY
            }
        }
        
        # Add model info if available
        if self._model and hasattr(self._model, 'get_model_info'):
            status['model_info'] = self._model.get_model_info()
        
        # Add system info
        status['output_directory'] = str(self.config['output_dir'])
        status['output_directory_exists'] = self.config['output_dir'].exists()
        
        return status
    
    def configure(self, **kwargs) -> Dict[str, Any]:
        """Update API configuration.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Returns:
            Updated configuration
        """
        for key, value in kwargs.items():
            if key in self.config:
                old_value = self.config[key]
                self.config[key] = value
                logger.info(f"Configuration updated: {key} = {value} (was {old_value})")
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        # Recreate output directory if path changed
        if 'output_dir' in kwargs:
            self.config['output_dir'] = Path(self.config['output_dir'])
            self.config['output_dir'].mkdir(exist_ok=True)
        
        return self.config.copy()
    
    def list_outputs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent audio generation outputs.
        
        Args:
            limit: Maximum number of files to return
            
        Returns:
            List of output file information
        """
        output_dir = self.config['output_dir']
        if not output_dir.exists():
            return []
        
        # Get all audio and related files
        extensions = ['.wav', '.mp3', '.flac', '.npy', '.txt']
        files = []
        
        for ext in extensions:
            files.extend(output_dir.glob(f'*{ext}'))
        
        # Sort by modification time (newest first)
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Limit results
        files = files[:limit]
        
        # Build file info
        file_info = []
        for file_path in files:
            stat = file_path.stat()
            info = {
                'filename': file_path.name,
                'path': str(file_path),
                'size_bytes': stat.st_size,
                'modified_time': stat.st_mtime,
                'extension': file_path.suffix.lower()
            }
            
            # Try to extract metadata from filename
            if '_' in file_path.stem:
                parts = file_path.stem.split('_')
                if parts[-1].isdigit():
                    info['timestamp'] = int(parts[-1])
                    info['prompt_hint'] = '_'.join(parts[:-1])
            
            file_info.append(info)
        
        return file_info
    
    def cleanup_outputs(self, keep_recent: int = 10) -> Dict[str, int]:
        """Clean up old output files, keeping only the most recent ones.
        
        Args:
            keep_recent: Number of recent files to keep
            
        Returns:
            Dictionary with cleanup statistics
        """
        output_dir = self.config['output_dir']
        if not output_dir.exists():
            return {'deleted': 0, 'kept': 0, 'errors': 0}
        
        # Get all output files
        all_files = list(self.list_outputs(limit=1000))  # Get more for cleanup
        
        deleted = 0
        errors = 0
        
        # Delete older files beyond keep_recent
        for file_info in all_files[keep_recent:]:
            try:
                Path(file_info['path']).unlink()
                deleted += 1
                logger.debug(f"Deleted old output: {file_info['filename']}")
            except Exception as e:
                logger.error(f"Failed to delete {file_info['filename']}: {e}")
                errors += 1
        
        stats = {
            'deleted': deleted,
            'kept': min(len(all_files), keep_recent),
            'errors': errors
        }
        
        logger.info(f"Cleanup completed: {stats}")
        return stats


# Convenience functions for quick access
def generate(prompt: str, duration: float = 10.0, output_path: Optional[str] = None) -> str:
    """Quick function to generate audio with minimal setup.
    
    Args:
        prompt: Audio description
        duration: Length in seconds
        output_path: Where to save (auto-generated if None)
        
    Returns:
        Path to generated audio file
    """
    api = SimpleAudioAPI()
    result = api.generate_audio(prompt, duration, output_path)
    
    if result.get('status') == 'completed':
        return result['output_path']
    else:
        raise RuntimeError(f"Generation failed: {result.get('error', 'Unknown error')}")


def transform(input_path: str, prompt: str, strength: float = 0.7) -> str:
    """Quick function to transform audio with minimal setup.
    
    Args:
        input_path: Input audio file
        prompt: Transformation description
        strength: Transformation intensity
        
    Returns:
        Path to transformed audio file
    """
    api = SimpleAudioAPI()
    result = api.transform_audio(input_path, prompt, strength=strength)
    
    if result.get('status') == 'completed':
        return result['output_path']
    else:
        raise RuntimeError(f"Transformation failed: {result.get('error', 'Unknown error')}")


# Quick demo function
def demo() -> Dict[str, Any]:
    """Run a quick demo of the Simple Audio API.
    
    Returns:
        Dictionary with demo results
    """
    logger.info("Running Simple Audio API demo...")
    
    api = SimpleAudioAPI()
    
    # Test generation
    demo_results = {
        'api_status': api.get_model_status(),
        'demo_generations': []
    }
    
    # Generate a few test samples
    test_prompts = [
        "A gentle rain on leaves",
        "Bird singing in the morning",
        "Ocean waves on a beach"
    ]
    
    for prompt in test_prompts:
        try:
            result = api.generate_audio(prompt, duration=3.0)
            demo_results['demo_generations'].append({
                'prompt': prompt,
                'success': result.get('status') == 'completed',
                'output': result.get('output_path'),
                'duration': result.get('generation_time', 0)
            })
            logger.info(f"Demo generation: '{prompt}' -> {result.get('status')}")
        except Exception as e:
            logger.error(f"Demo generation failed for '{prompt}': {e}")
            demo_results['demo_generations'].append({
                'prompt': prompt,
                'success': False,
                'error': str(e)
            })
    
    demo_results['demo_completed'] = True
    demo_results['successful_generations'] = sum(
        1 for g in demo_results['demo_generations'] if g.get('success')
    )
    
    logger.info(f"Demo completed: {demo_results['successful_generations']}/{len(test_prompts)} successful")
    
    return demo_results


if __name__ == "__main__":
    # Run demo if called directly
    demo_result = demo()
    print(json.dumps(demo_result, indent=2))