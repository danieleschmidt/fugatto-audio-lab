"""Audio processing pipeline for chaining multiple processing stages."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class ProcessingStage(ABC):
    """Abstract base class for processing stages."""
    
    def __init__(self, name: str, enabled: bool = True):
        """Initialize processing stage.
        
        Args:
            name: Name of the processing stage
            enabled: Whether this stage is enabled
        """
        self.name = name
        self.enabled = enabled
        
    def process(self, audio: np.ndarray, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Process audio through this stage.
        
        Args:
            audio: Input audio array
            params: Optional parameters for processing
            
        Returns:
            Processed audio array
        """
        if not self.enabled:
            return audio
            
        if params is None:
            params = {}
            
        try:
            result = self._process_impl(audio, params)
            logger.debug(f"Stage '{self.name}' processed {len(audio)} samples")
            return result
        except Exception as e:
            logger.error(f"Error in stage '{self.name}': {e}")
            return audio  # Return original audio on error
    
    @abstractmethod
    def _process_impl(self, audio: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Implement the actual processing logic."""
        pass


class ProcessingPipeline:
    """Audio processing pipeline that chains multiple stages."""
    
    def __init__(self, stages: Optional[List[ProcessingStage]] = None):
        """Initialize processing pipeline.
        
        Args:
            stages: List of processing stages to include
        """
        self.stages = stages or []
        self.processing_stats = {}
        
        logger.info(f"ProcessingPipeline initialized with {len(self.stages)} stages")
    
    def add_stage(self, stage: ProcessingStage) -> None:
        """Add a processing stage to the pipeline.
        
        Args:
            stage: Processing stage to add
        """
        self.stages.append(stage)
        logger.debug(f"Added stage '{stage.name}' to pipeline")
    
    def remove_stage(self, name: str) -> bool:
        """Remove a processing stage by name.
        
        Args:
            name: Name of stage to remove
            
        Returns:
            True if stage was removed, False if not found
        """
        for i, stage in enumerate(self.stages):
            if stage.name == name:
                removed_stage = self.stages.pop(i)
                logger.debug(f"Removed stage '{removed_stage.name}' from pipeline")
                return True
        return False
    
    def enable_stage(self, name: str) -> bool:
        """Enable a processing stage by name.
        
        Args:
            name: Name of stage to enable
            
        Returns:
            True if stage was found and enabled
        """
        for stage in self.stages:
            if stage.name == name:
                stage.enabled = True
                logger.debug(f"Enabled stage '{name}'")
                return True
        return False
    
    def disable_stage(self, name: str) -> bool:
        """Disable a processing stage by name.
        
        Args:
            name: Name of stage to disable
            
        Returns:
            True if stage was found and disabled
        """
        for stage in self.stages:
            if stage.name == name:
                stage.enabled = False
                logger.debug(f"Disabled stage '{name}'")
                return True
        return False
    
    def process(self, audio: np.ndarray, stage_params: Optional[Dict[str, Dict[str, Any]]] = None) -> np.ndarray:
        """Process audio through the entire pipeline.
        
        Args:
            audio: Input audio array
            stage_params: Optional parameters for each stage
            
        Returns:
            Processed audio array
        """
        if stage_params is None:
            stage_params = {}
        
        result = audio.copy()
        
        for stage in self.stages:
            if stage.enabled:
                params = stage_params.get(stage.name, {})
                result = stage.process(result, params)
        
        logger.debug(f"Pipeline processed {len(audio)} samples through {len([s for s in self.stages if s.enabled])} enabled stages")
        return result
    
    def get_stage_names(self) -> List[str]:
        """Get list of stage names in pipeline."""
        return [stage.name for stage in self.stages]
    
    def get_enabled_stages(self) -> List[str]:
        """Get list of enabled stage names."""
        return [stage.name for stage in self.stages if stage.enabled]
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline."""
        return {
            'total_stages': len(self.stages),
            'enabled_stages': len([s for s in self.stages if s.enabled]),
            'stage_names': self.get_stage_names(),
            'enabled_stage_names': self.get_enabled_stages()
        }


class NoiseReductionStage(ProcessingStage):
    """Noise reduction processing stage."""
    
    def __init__(self, enabled: bool = True):
        super().__init__("noise_reduction", enabled)
        self.reduction_strength = 0.5
        
    def _process_impl(self, audio: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Implement noise reduction using spectral subtraction."""
        strength = params.get('strength', self.reduction_strength)
        
        if len(audio) < 1024:  # Too short for FFT processing
            return audio
            
        try:
            # Simple spectral subtraction noise reduction
            fft = np.fft.fft(audio)
            magnitude = np.abs(fft)
            phase = np.angle(fft)
            
            # Estimate noise floor from first 10% of signal
            noise_samples = int(0.1 * len(audio))
            noise_magnitude = np.mean(magnitude[:noise_samples])
            
            # Apply spectral subtraction
            alpha = strength * 2.0  # Over-subtraction factor
            reduced_magnitude = magnitude - alpha * noise_magnitude
            
            # Prevent over-subtraction
            reduced_magnitude = np.maximum(reduced_magnitude, 0.1 * magnitude)
            
            # Reconstruct signal
            reduced_fft = reduced_magnitude * np.exp(1j * phase)
            result = np.real(np.fft.ifft(reduced_fft))
            
            return result.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Noise reduction error: {e}")
            return audio


class NormalizationStage(ProcessingStage):
    """Audio normalization stage."""
    
    def __init__(self, enabled: bool = True):
        super().__init__("normalization", enabled)
        self.target_lufs = -14.0
        self.peak_limit = 0.95
        
    def _process_impl(self, audio: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Implement loudness normalization."""
        target_lufs = params.get('target_lufs', self.target_lufs)
        peak_limit = params.get('peak_limit', self.peak_limit)
        
        try:
            # RMS-based normalization (approximates LUFS)
            rms = np.sqrt(np.mean(audio ** 2))
            if rms <= 0:
                return audio
                
            # Convert target LUFS to linear scale
            target_rms = 10 ** (target_lufs / 20) * 0.1
            scale_factor = target_rms / rms
            
            # Apply scaling
            normalized = audio * scale_factor
            
            # Apply peak limiting
            peak = np.max(np.abs(normalized))
            if peak > peak_limit:
                normalized = normalized * (peak_limit / peak)
                
            return normalized.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Normalization error: {e}")
            return audio


class FilterStage(ProcessingStage):
    """Audio filtering stage."""
    
    def __init__(self, enabled: bool = True):
        super().__init__("filter", enabled)
        self.highpass_freq = 80.0
        self.lowpass_freq = 16000.0
        
    def _process_impl(self, audio: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Implement basic filtering."""
        highpass = params.get('highpass_freq', self.highpass_freq)
        lowpass = params.get('lowpass_freq', self.lowpass_freq)
        
        try:
            # Simple high-pass filter (removes DC and very low frequencies)
            if len(audio) > 1 and highpass > 0:
                filtered = np.zeros_like(audio)
                alpha = 0.99  # High-pass coefficient
                filtered[0] = audio[0]
                
                for i in range(1, len(audio)):
                    filtered[i] = alpha * filtered[i-1] + alpha * (audio[i] - audio[i-1])
                
                return filtered.astype(np.float32)
            else:
                return audio
                
        except Exception as e:
            logger.warning(f"Filter error: {e}")
            return audio


class CompressorStage(ProcessingStage):
    """Dynamic range compression stage."""
    
    def __init__(self, enabled: bool = True):
        super().__init__("compressor", enabled)
        self.threshold_db = -20.0
        self.ratio = 4.0
        self.attack_time = 0.003  # 3ms
        self.release_time = 0.1   # 100ms
        
    def _process_impl(self, audio: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Implement simple dynamic range compression."""
        threshold_db = params.get('threshold_db', self.threshold_db)
        ratio = params.get('ratio', self.ratio)
        
        try:
            threshold_linear = 10 ** (threshold_db / 20)
            
            compressed = audio.copy()
            
            # Simple sample-by-sample compression
            for i in range(len(audio)):
                sample_abs = abs(audio[i])
                
                if sample_abs > threshold_linear:
                    # Calculate compression
                    excess_db = 20 * np.log10(sample_abs / threshold_linear)
                    compressed_excess_db = excess_db / ratio
                    new_amplitude = threshold_linear * (10 ** (compressed_excess_db / 20))
                    
                    # Apply compression while preserving sign
                    compressed[i] = np.sign(audio[i]) * new_amplitude
            
            return compressed.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Compressor error: {e}")
            return audio


def create_default_pipeline() -> ProcessingPipeline:
    """Create a default processing pipeline with common stages."""
    pipeline = ProcessingPipeline()
    
    # Add stages in typical processing order
    pipeline.add_stage(FilterStage(enabled=True))
    pipeline.add_stage(NoiseReductionStage(enabled=False))  # Disabled by default
    pipeline.add_stage(CompressorStage(enabled=False))      # Disabled by default
    pipeline.add_stage(NormalizationStage(enabled=True))
    
    logger.info("Created default processing pipeline")
    return pipeline