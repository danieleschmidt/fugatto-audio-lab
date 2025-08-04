"""Advanced audio processing pipeline with real-time capabilities."""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import threading
import queue
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Audio processing modes."""
    BATCH = "batch"
    STREAMING = "streaming" 
    REALTIME = "realtime"


@dataclass
class ProcessingConfig:
    """Configuration for audio processing pipeline."""
    sample_rate: int = 48000
    buffer_size: int = 1024
    overlap: float = 0.5
    mode: ProcessingMode = ProcessingMode.BATCH
    max_latency_ms: float = 100.0
    enable_caching: bool = True
    cache_size_mb: int = 256
    parallel_workers: int = 1


class ProcessingStage:
    """Base class for processing stages."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.processing_time = 0.0
        self.processed_samples = 0
        
    def process(self, audio: np.ndarray, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Process audio data.
        
        Args:
            audio: Input audio array
            params: Processing parameters
            
        Returns:
            Processed audio array
        """
        if not self.enabled:
            return audio
            
        start_time = time.time()
        result = self._process_impl(audio, params or {})
        
        # Update performance metrics
        self.processing_time += time.time() - start_time
        self.processed_samples += len(audio)
        
        return result
    
    def _process_impl(self, audio: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Implementation of processing logic."""
        raise NotImplementedError
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if self.processed_samples > 0:
            avg_processing_time = self.processing_time / (self.processed_samples / 48000)  # per second of audio
        else:
            avg_processing_time = 0.0
            
        return {
            'total_processing_time': self.processing_time,
            'processed_samples': self.processed_samples,
            'avg_processing_time_per_second': avg_processing_time,
            'enabled': self.enabled
        }


class NoiseReductionStage(ProcessingStage):
    """Noise reduction processing stage."""
    
    def __init__(self, enabled: bool = True):
        super().__init__("noise_reduction", enabled)
        self.noise_profile = None
        self.reduction_strength = 0.5
        
    def _process_impl(self, audio: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Implement noise reduction using spectral subtraction."""
        strength = params.get('strength', self.reduction_strength)
        
        if len(audio) < 1024:  # Too short for FFT processing
            return audio
            
        try:\n            # Simple spectral subtraction noise reduction\n            fft = np.fft.fft(audio)\n            magnitude = np.abs(fft)\n            phase = np.angle(fft)\n            \n            # Estimate noise floor from first 10% of signal\n            noise_samples = int(0.1 * len(audio))\n            noise_magnitude = np.mean(magnitude[:noise_samples])\n            \n            # Apply spectral subtraction\n            alpha = strength * 2.0  # Over-subtraction factor\n            reduced_magnitude = magnitude - alpha * noise_magnitude\n            \n            # Prevent over-subtraction\n            reduced_magnitude = np.maximum(reduced_magnitude, 0.1 * magnitude)\n            \n            # Reconstruct signal\n            reduced_fft = reduced_magnitude * np.exp(1j * phase)\n            result = np.real(np.fft.ifft(reduced_fft))\n            \n            return result.astype(np.float32)\n            \n        except Exception as e:\n            logger.warning(f\"Noise reduction error: {e}\")\n            return audio


class NormalizationStage(ProcessingStage):\n    \"\"\"Audio normalization stage.\"\"\"\n    \n    def __init__(self, enabled: bool = True):\n        super().__init__(\"normalization\", enabled)\n        self.target_lufs = -14.0\n        self.peak_limit = 0.95\n        \n    def _process_impl(self, audio: np.ndarray, params: Dict[str, Any]) -> np.ndarray:\n        \"\"\"Implement loudness normalization.\"\"\"\n        target_lufs = params.get('target_lufs', self.target_lufs)\n        peak_limit = params.get('peak_limit', self.peak_limit)\n        \n        try:\n            # RMS-based normalization (approximates LUFS)\n            rms = np.sqrt(np.mean(audio ** 2))\n            if rms <= 0:\n                return audio\n                \n            # Convert target LUFS to linear scale\n            target_rms = 10 ** (target_lufs / 20) * 0.1\n            scale_factor = target_rms / rms\n            \n            # Apply scaling\n            normalized = audio * scale_factor\n            \n            # Apply peak limiting\n            peak = np.max(np.abs(normalized))\n            if peak > peak_limit:\n                normalized = normalized * (peak_limit / peak)\n                \n            return normalized.astype(np.float32)\n            \n        except Exception as e:\n            logger.warning(f\"Normalization error: {e}\")\n            return audio


class EqualizationStage(ProcessingStage):\n    \"\"\"Multi-band equalizer stage.\"\"\"\n    \n    def __init__(self, enabled: bool = True):\n        super().__init__(\"equalization\", enabled)\n        self.eq_bands = {\n            'low': {'freq': 100, 'gain': 0, 'q': 0.7},\n            'low_mid': {'freq': 500, 'gain': 0, 'q': 0.7},\n            'mid': {'freq': 1500, 'gain': 0, 'q': 0.7},\n            'high_mid': {'freq': 4000, 'gain': 0, 'q': 0.7},\n            'high': {'freq': 10000, 'gain': 0, 'q': 0.7}\n        }\n        \n    def _process_impl(self, audio: np.ndarray, params: Dict[str, Any]) -> np.ndarray:\n        \"\"\"Implement multi-band equalization.\"\"\"\n        eq_gains = params.get('eq_gains', {})\n        \n        if not eq_gains or len(audio) < 256:\n            return audio\n            \n        try:\n            # Apply EQ in frequency domain\n            fft = np.fft.fft(audio)\n            freqs = np.fft.fftfreq(len(fft), 1/48000)  # Assume 48kHz\n            \n            # Apply gains to frequency bands\n            for band_name, gain_db in eq_gains.items():\n                if band_name in self.eq_bands and gain_db != 0:\n                    band_config = self.eq_bands[band_name]\n                    center_freq = band_config['freq']\n                    q = band_config['q']\n                    \n                    # Bell filter implementation\n                    gain_linear = 10 ** (gain_db / 20)\n                    bandwidth = center_freq / q\n                    \n                    # Create frequency mask\n                    freq_mask = (np.abs(freqs) >= center_freq - bandwidth/2) & \\\n                               (np.abs(freqs) <= center_freq + bandwidth/2)\n                    \n                    # Apply gain\n                    fft[freq_mask] *= gain_linear\n            \n            # Convert back to time domain\n            result = np.real(np.fft.ifft(fft))\n            return result.astype(np.float32)\n            \n        except Exception as e:\n            logger.warning(f\"Equalization error: {e}\")\n            return audio


class CompressionStage(ProcessingStage):\n    \"\"\"Dynamic range compression stage.\"\"\"\n    \n    def __init__(self, enabled: bool = True):\n        super().__init__(\"compression\", enabled)\n        self.threshold_db = -20.0\n        self.ratio = 4.0\n        self.attack_ms = 5.0\n        self.release_ms = 50.0\n        \n    def _process_impl(self, audio: np.ndarray, params: Dict[str, Any]) -> np.ndarray:\n        \"\"\"Implement dynamic range compression.\"\"\"\n        threshold_db = params.get('threshold_db', self.threshold_db)\n        ratio = params.get('ratio', self.ratio)\n        \n        try:\n            threshold_linear = 10 ** (threshold_db / 20)\n            \n            # Simple peak compression\n            compressed = audio.copy()\n            over_threshold = np.abs(audio) > threshold_linear\n            \n            # Apply compression to samples over threshold\n            for i in range(len(audio)):\n                if over_threshold[i]:\n                    excess_db = 20 * np.log10(np.abs(audio[i]) / threshold_linear)\n                    compressed_excess_db = excess_db / ratio\n                    new_amplitude = threshold_linear * (10 ** (compressed_excess_db / 20))\n                    compressed[i] = np.sign(audio[i]) * new_amplitude\n            \n            return compressed.astype(np.float32)\n            \n        except Exception as e:\n            logger.warning(f\"Compression error: {e}\")\n            return audio


class ProcessingPipeline:\n    \"\"\"Advanced audio processing pipeline with configurable stages.\"\"\"\n    \n    def __init__(self, config: Optional[ProcessingConfig] = None):\n        \"\"\"Initialize processing pipeline.\n        \n        Args:\n            config: Pipeline configuration\n        \"\"\"\n        self.config = config or ProcessingConfig()\n        self.stages: List[ProcessingStage] = []\n        self.cache = {} if self.config.enable_caching else None\n        self.cache_size = 0\n        self.processing_stats = {}\n        \n        # Initialize default stages\n        self._initialize_default_stages()\n        \n        logger.info(f\"ProcessingPipeline initialized with {len(self.stages)} stages\")\n    \n    def _initialize_default_stages(self):\n        \"\"\"Initialize default processing stages.\"\"\"\n        self.stages = [\n            NoiseReductionStage(enabled=False),  # Disabled by default\n            NormalizationStage(enabled=True),\n            EqualizationStage(enabled=False),     # Disabled by default\n            CompressionStage(enabled=False)       # Disabled by default\n        ]\n    \n    def add_stage(self, stage: ProcessingStage, position: Optional[int] = None):\n        \"\"\"Add a processing stage to the pipeline.\n        \n        Args:\n            stage: Processing stage to add\n            position: Position to insert stage (None for append)\n        \"\"\"\n        if position is None:\n            self.stages.append(stage)\n        else:\n            self.stages.insert(position, stage)\n            \n        logger.info(f\"Added stage '{stage.name}' to pipeline\")\n    \n    def remove_stage(self, stage_name: str) -> bool:\n        \"\"\"Remove a processing stage by name.\n        \n        Args:\n            stage_name: Name of stage to remove\n            \n        Returns:\n            True if stage was removed, False otherwise\n        \"\"\"\n        for i, stage in enumerate(self.stages):\n            if stage.name == stage_name:\n                self.stages.pop(i)\n                logger.info(f\"Removed stage '{stage_name}' from pipeline\")\n                return True\n        return False\n    \n    def enable_stage(self, stage_name: str, enabled: bool = True):\n        \"\"\"Enable or disable a processing stage.\n        \n        Args:\n            stage_name: Name of stage to modify\n            enabled: Whether to enable the stage\n        \"\"\"\n        for stage in self.stages:\n            if stage.name == stage_name:\n                stage.enabled = enabled\n                logger.info(f\"Stage '{stage_name}' {'enabled' if enabled else 'disabled'}\")\n                return\n        logger.warning(f\"Stage '{stage_name}' not found\")\n    \n    def process(self, audio: np.ndarray, params: Optional[Dict[str, Any]] = None) -> np.ndarray:\n        \"\"\"Process audio through the pipeline.\n        \n        Args:\n            audio: Input audio array\n            params: Processing parameters for stages\n            \n        Returns:\n            Processed audio array\n        \"\"\"\n        if params is None:\n            params = {}\n            \n        # Check cache first\n        cache_key = self._generate_cache_key(audio, params)\n        if self.cache and cache_key in self.cache:\n            logger.debug(\"Cache hit for audio processing\")\n            return self.cache[cache_key]\n        \n        # Process through stages\n        processed = audio.copy()\n        stage_times = {}\n        \n        start_time = time.time()\n        \n        for stage in self.stages:\n            if stage.enabled:\n                stage_start = time.time()\n                stage_params = params.get(stage.name, {})\n                processed = stage.process(processed, stage_params)\n                stage_times[stage.name] = time.time() - stage_start\n        \n        total_time = time.time() - start_time\n        \n        # Update processing stats\n        self.processing_stats = {\n            'total_time': total_time,\n            'stage_times': stage_times,\n            'throughput_ratio': (len(audio) / self.config.sample_rate) / total_time if total_time > 0 else 0,\n            'latency_ms': total_time * 1000\n        }\n        \n        # Cache result if enabled and within size limits\n        if self.cache and self._should_cache(processed):\n            self.cache[cache_key] = processed.copy()\n            self.cache_size += processed.nbytes\n            \n            # Evict old entries if cache is too large\n            self._evict_cache_if_needed()\n        \n        logger.debug(f\"Processed {len(audio)} samples in {total_time*1000:.2f}ms\")\n        return processed\n    \n    def process_streaming(self, audio_generator, params: Optional[Dict[str, Any]] = None):\n        \"\"\"Process audio in streaming mode.\n        \n        Args:\n            audio_generator: Generator yielding audio chunks\n            params: Processing parameters\n            \n        Yields:\n            Processed audio chunks\n        \"\"\"\n        overlap_samples = int(self.config.buffer_size * self.config.overlap)\n        overlap_buffer = np.zeros(overlap_samples, dtype=np.float32)\n        \n        for chunk in audio_generator:\n            # Add overlap from previous chunk\n            if len(overlap_buffer) > 0:\n                padded_chunk = np.concatenate([overlap_buffer, chunk])\n            else:\n                padded_chunk = chunk\n                \n            # Process chunk\n            processed_chunk = self.process(padded_chunk, params)\n            \n            # Extract main output (without overlap)\n            if len(processed_chunk) > overlap_samples:\n                output = processed_chunk[overlap_samples:]\n                # Save overlap for next chunk\n                overlap_buffer = processed_chunk[-overlap_samples:] if overlap_samples > 0 else np.array([])\n            else:\n                output = processed_chunk\n                overlap_buffer = np.array([])\n            \n            yield output\n    \n    def _generate_cache_key(self, audio: np.ndarray, params: Dict[str, Any]) -> str:\n        \"\"\"Generate cache key for audio and parameters.\"\"\"\n        import hashlib\n        \n        # Create hash from audio statistics and parameters\n        audio_hash = hashlib.md5()\n        audio_hash.update(audio.tobytes())\n        \n        params_str = str(sorted(params.items()))\n        params_hash = hashlib.md5(params_str.encode()).hexdigest()\n        \n        return f\"{audio_hash.hexdigest()}_{params_hash}\"\n    \n    def _should_cache(self, audio: np.ndarray) -> bool:\n        \"\"\"Determine if audio should be cached.\"\"\"\n        if not self.config.enable_caching:\n            return False\n            \n        # Don't cache very large audio files\n        max_cache_size_bytes = self.config.cache_size_mb * 1024 * 1024\n        return audio.nbytes < max_cache_size_bytes * 0.1  # Max 10% of cache per item\n    \n    def _evict_cache_if_needed(self):\n        \"\"\"Evict cache entries if cache is too large.\"\"\"\n        max_cache_size_bytes = self.config.cache_size_mb * 1024 * 1024\n        \n        if self.cache_size > max_cache_size_bytes:\n            # Simple FIFO eviction\n            keys_to_remove = list(self.cache.keys())[:len(self.cache)//2]\n            \n            for key in keys_to_remove:\n                if key in self.cache:\n                    self.cache_size -= self.cache[key].nbytes\n                    del self.cache[key]\n            \n            logger.debug(f\"Evicted {len(keys_to_remove)} cache entries\")\n    \n    def get_pipeline_stats(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive pipeline statistics.\"\"\"\n        stats = {\n            'config': {\n                'sample_rate': self.config.sample_rate,\n                'buffer_size': self.config.buffer_size,\n                'mode': self.config.mode.value,\n                'max_latency_ms': self.config.max_latency_ms\n            },\n            'stages': {}\n        }\n        \n        # Get stats for each stage\n        for stage in self.stages:\n            stats['stages'][stage.name] = stage.get_performance_stats()\n        \n        # Add processing stats\n        if self.processing_stats:\n            stats['processing'] = self.processing_stats\n        \n        # Add cache stats\n        if self.cache is not None:\n            stats['cache'] = {\n                'entries': len(self.cache),\n                'size_mb': self.cache_size / (1024 * 1024),\n                'max_size_mb': self.config.cache_size_mb\n            }\n        \n        return stats\n    \n    def reset_stats(self):\n        \"\"\"Reset all pipeline statistics.\"\"\"\n        for stage in self.stages:\n            stage.processing_time = 0.0\n            stage.processed_samples = 0\n        \n        self.processing_stats = {}\n        logger.info(\"Pipeline statistics reset\")\n    \n    def clear_cache(self):\n        \"\"\"Clear processing cache.\"\"\"\n        if self.cache:\n            self.cache.clear()\n            self.cache_size = 0\n            logger.info(\"Processing cache cleared\")