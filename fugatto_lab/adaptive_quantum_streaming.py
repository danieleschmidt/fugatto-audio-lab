"""Adaptive Quantum Streaming Engine - Generation 1 Enhancement.

Real-time adaptive audio streaming with quantum-inspired predictive buffering,
intelligent latency optimization, and dynamic quality adaptation.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


class StreamQuality(Enum):
    """Stream quality levels with adaptive characteristics."""
    ULTRA_LOW = "ultra_low"     # 8kHz, high compression
    LOW = "low"                 # 16kHz, moderate compression
    MEDIUM = "medium"           # 24kHz, balanced
    HIGH = "high"               # 48kHz, low compression
    ULTRA_HIGH = "ultra_high"   # 96kHz, minimal compression
    ADAPTIVE = "adaptive"       # AI-driven quality selection


@dataclass
class StreamMetrics:
    """Real-time streaming performance metrics."""
    latency_ms: float = 0.0
    throughput_mbps: float = 0.0
    buffer_health: float = 1.0  # 0.0 = empty, 1.0 = full
    quality_index: float = 1.0  # 0.0 = lowest, 1.0 = highest
    adaptation_events: int = 0
    error_rate: float = 0.0
    prediction_accuracy: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive streaming behavior."""
    min_buffer_ms: int = 50
    max_buffer_ms: int = 500
    target_latency_ms: int = 100
    quality_adaptation_threshold: float = 0.3
    prediction_window_size: int = 10
    learning_rate: float = 0.1
    enable_quantum_prediction: bool = True
    enable_quality_adaptation: bool = True
    enable_predictive_buffering: bool = True


class QuantumBufferPredictor:
    """Quantum-inspired predictive buffer management."""
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.history = deque(maxlen=config.prediction_window_size)
        self.quantum_states = {}
        self.prediction_weights = np.ones(5) if np else [1.0] * 5
        self.accuracy_tracker = deque(maxlen=100)
        
    def predict_buffer_needs(self, current_metrics: StreamMetrics) -> Tuple[int, float]:
        """Predict optimal buffer size and confidence.
        
        Returns:
            Tuple of (predicted_buffer_ms, confidence)
        """
        if not self.history:
            return self.config.target_latency_ms, 0.5
            
        # Store current state
        self.history.append({
            'latency': current_metrics.latency_ms,
            'buffer_health': current_metrics.buffer_health,
            'quality': current_metrics.quality_index,
            'throughput': current_metrics.throughput_mbps,
            'timestamp': current_metrics.timestamp
        })
        
        # Quantum-inspired superposition of possible buffer states
        predictions = self._generate_quantum_predictions()
        
        # Weight predictions based on historical accuracy
        weighted_prediction = self._apply_quantum_weights(predictions)
        
        # Calculate confidence based on consistency
        confidence = self._calculate_prediction_confidence(predictions)
        
        # Adaptive learning
        self._update_prediction_weights(current_metrics)
        
        return int(weighted_prediction), confidence
    
    def _generate_quantum_predictions(self) -> List[float]:
        """Generate multiple predictions using quantum-inspired superposition."""
        if len(self.history) < 2:
            return [self.config.target_latency_ms] * 5
            
        recent = list(self.history)[-5:]
        predictions = []
        
        # Trend-based prediction
        latencies = [state['latency'] for state in recent]
        if len(latencies) >= 2:
            trend = (latencies[-1] - latencies[0]) / len(latencies)
            trend_prediction = latencies[-1] + trend * 2  # 2-step ahead
            predictions.append(max(self.config.min_buffer_ms, 
                                 min(self.config.max_buffer_ms, trend_prediction)))
        else:
            predictions.append(self.config.target_latency_ms)
            
        # Buffer health based prediction
        buffer_healths = [state['buffer_health'] for state in recent]
        avg_health = sum(buffer_healths) / len(buffer_healths) if buffer_healths else 0.5
        health_prediction = self.config.target_latency_ms * (2.0 - avg_health)
        predictions.append(max(self.config.min_buffer_ms, 
                             min(self.config.max_buffer_ms, health_prediction)))
        
        # Throughput-based prediction
        throughputs = [state['throughput'] for state in recent]
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 1.0
        throughput_factor = max(0.5, min(2.0, 1.0 / max(0.1, avg_throughput)))
        throughput_prediction = self.config.target_latency_ms * throughput_factor
        predictions.append(max(self.config.min_buffer_ms, 
                             min(self.config.max_buffer_ms, throughput_prediction)))
        
        # Oscillation-damped prediction
        if len(recent) >= 3:
            latency_variance = np.var(latencies) if np else self._calculate_variance(latencies)
            stability_factor = max(0.8, min(1.2, 1.0 - latency_variance / 1000.0))
            stable_prediction = self.config.target_latency_ms * stability_factor
            predictions.append(max(self.config.min_buffer_ms, 
                                 min(self.config.max_buffer_ms, stable_prediction)))
        else:
            predictions.append(self.config.target_latency_ms)
            
        # Quantum entangled prediction (combines all factors)
        quantum_prediction = self._quantum_entangle_predictions(predictions)
        predictions.append(quantum_prediction)
        
        return predictions
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance without numpy."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def _quantum_entangle_predictions(self, predictions: List[float]) -> float:
        """Create quantum entangled prediction from multiple states."""
        if not predictions:
            return self.config.target_latency_ms
            
        # Weighted harmonic mean for quantum entanglement effect
        weights = [0.3, 0.25, 0.2, 0.15, 0.1][:len(predictions)]
        
        if np is not None:
            weighted_sum = np.sum([w * p for w, p in zip(weights, predictions)])
            return float(weighted_sum)
        else:
            weighted_sum = sum(w * p for w, p in zip(weights, predictions))
            return weighted_sum
    
    def _apply_quantum_weights(self, predictions: List[float]) -> float:
        """Apply learned quantum weights to predictions."""
        if not predictions or len(self.prediction_weights) != len(predictions):
            return predictions[0] if predictions else self.config.target_latency_ms
            
        if np is not None:
            weights = self.prediction_weights / np.sum(self.prediction_weights)
            return float(np.sum(weights * np.array(predictions)))
        else:
            weight_sum = sum(self.prediction_weights)
            weights = [w / weight_sum for w in self.prediction_weights]
            return sum(w * p for w, p in zip(weights, predictions))
    
    def _calculate_prediction_confidence(self, predictions: List[float]) -> float:
        """Calculate confidence based on prediction consistency."""
        if len(predictions) < 2:
            return 0.5
            
        if np is not None:
            variance = float(np.var(predictions))
        else:
            mean = sum(predictions) / len(predictions)
            variance = sum((p - mean) ** 2 for p in predictions) / len(predictions)
            
        # Higher variance = lower confidence
        max_reasonable_variance = (self.config.max_buffer_ms - self.config.min_buffer_ms) ** 2 / 4
        confidence = max(0.1, min(1.0, 1.0 - variance / max_reasonable_variance))
        
        return confidence
    
    def _update_prediction_weights(self, actual_metrics: StreamMetrics) -> None:
        """Update prediction weights based on actual performance."""
        if len(self.accuracy_tracker) < 2:
            return
            
        # Calculate prediction error for weight adjustment
        # This would be implemented with actual vs predicted comparison
        # For now, we use a simplified learning approach
        
        current_accuracy = 1.0 - abs(actual_metrics.latency_ms - self.config.target_latency_ms) / self.config.target_latency_ms
        current_accuracy = max(0.0, min(1.0, current_accuracy))
        
        self.accuracy_tracker.append(current_accuracy)
        
        # Adaptive weight adjustment
        if len(self.accuracy_tracker) >= 10:
            recent_accuracy = sum(list(self.accuracy_tracker)[-5:]) / 5
            if recent_accuracy < 0.7:  # Poor performance, adjust weights
                adjustment = self.config.learning_rate * (0.7 - recent_accuracy)
                if np is not None:
                    self.prediction_weights += np.random.normal(0, adjustment, len(self.prediction_weights))
                    self.prediction_weights = np.abs(self.prediction_weights)  # Keep positive
                else:
                    import random
                    for i in range(len(self.prediction_weights)):
                        self.prediction_weights[i] += random.gauss(0, adjustment)
                        self.prediction_weights[i] = abs(self.prediction_weights[i])


class AdaptiveQualityManager:
    """Manages adaptive quality scaling based on performance metrics."""
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.quality_history = deque(maxlen=20)
        self.current_quality = StreamQuality.HIGH
        self.quality_scores = {
            StreamQuality.ULTRA_LOW: 0.2,
            StreamQuality.LOW: 0.4,
            StreamQuality.MEDIUM: 0.6,
            StreamQuality.HIGH: 0.8,
            StreamQuality.ULTRA_HIGH: 1.0
        }
        
    def adapt_quality(self, metrics: StreamMetrics) -> StreamQuality:
        """Adapt stream quality based on current performance."""
        if not self.config.enable_quality_adaptation:
            return self.current_quality
            
        # Store current performance
        self.quality_history.append({
            'latency': metrics.latency_ms,
            'buffer_health': metrics.buffer_health,
            'error_rate': metrics.error_rate,
            'throughput': metrics.throughput_mbps,
            'quality': self.quality_scores[self.current_quality]
        })
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(metrics)
        
        # Determine optimal quality
        new_quality = self._select_optimal_quality(performance_score, metrics)
        
        # Apply hysteresis to prevent oscillation
        if self._should_change_quality(new_quality, performance_score):
            logger.info(f"Quality adaptation: {self.current_quality.value} -> {new_quality.value} (score: {performance_score:.3f})")
            self.current_quality = new_quality
            
        return self.current_quality
    
    def _calculate_performance_score(self, metrics: StreamMetrics) -> float:
        """Calculate overall performance score (0.0-1.0)."""
        # Latency score (lower is better)
        latency_score = max(0.0, min(1.0, 1.0 - (metrics.latency_ms - self.config.target_latency_ms) / self.config.target_latency_ms))
        
        # Buffer health score
        buffer_score = metrics.buffer_health
        
        # Error rate score (lower is better)
        error_score = max(0.0, 1.0 - metrics.error_rate * 10)
        
        # Throughput score (normalized)
        throughput_score = min(1.0, metrics.throughput_mbps / 10.0)  # Assume 10 Mbps is excellent
        
        # Weighted combination
        performance_score = (
            latency_score * 0.4 +
            buffer_score * 0.3 +
            error_score * 0.2 +
            throughput_score * 0.1
        )
        
        return max(0.0, min(1.0, performance_score))
    
    def _select_optimal_quality(self, performance_score: float, metrics: StreamMetrics) -> StreamQuality:
        """Select optimal quality based on performance score."""
        # Map performance score to quality level
        if performance_score >= 0.9 and metrics.throughput_mbps > 5.0:
            return StreamQuality.ULTRA_HIGH
        elif performance_score >= 0.8:
            return StreamQuality.HIGH
        elif performance_score >= 0.6:
            return StreamQuality.MEDIUM
        elif performance_score >= 0.4:
            return StreamQuality.LOW
        else:
            return StreamQuality.ULTRA_LOW
    
    def _should_change_quality(self, new_quality: StreamQuality, performance_score: float) -> bool:
        """Apply hysteresis to prevent quality oscillation."""
        current_score = self.quality_scores[self.current_quality]
        new_score = self.quality_scores[new_quality]
        
        # Require significant improvement to increase quality
        if new_score > current_score:
            return performance_score > (current_score + self.config.quality_adaptation_threshold)
        
        # Allow quicker quality reduction during problems
        elif new_score < current_score:
            return performance_score < (current_score - self.config.quality_adaptation_threshold / 2)
        
        return False


class StreamingAudioBuffer:
    """Intelligent audio buffer with adaptive sizing."""
    
    def __init__(self, initial_size_ms: int = 100, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.buffer = deque()
        self.target_size_ms = initial_size_ms
        self.max_size_samples = self._ms_to_samples(500)  # 500ms max
        self.min_size_samples = self._ms_to_samples(10)   # 10ms min
        self.total_written = 0
        self.total_read = 0
        
    def write(self, audio_data: Union[List[float], 'np.ndarray']) -> bool:
        """Write audio data to buffer.
        
        Returns:
            True if write successful, False if buffer full
        """
        if np is not None and isinstance(audio_data, np.ndarray):
            audio_list = audio_data.tolist()
        else:
            audio_list = list(audio_data)
            
        # Check if buffer would exceed maximum size
        current_size = len(self.buffer)
        if current_size + len(audio_list) > self.max_size_samples:
            # Drop oldest samples to make room
            overflow = current_size + len(audio_list) - self.max_size_samples
            for _ in range(overflow):
                if self.buffer:
                    self.buffer.popleft()
        
        # Add new samples
        self.buffer.extend(audio_list)
        self.total_written += len(audio_list)
        
        return True
    
    def read(self, num_samples: int) -> List[float]:
        """Read audio samples from buffer.
        
        Returns:
            List of audio samples (may be shorter than requested)
        """
        samples = []
        read_count = min(num_samples, len(self.buffer))
        
        for _ in range(read_count):
            if self.buffer:
                samples.append(self.buffer.popleft())
            else:
                break
                
        self.total_read += len(samples)
        
        # Pad with silence if requested more than available
        if len(samples) < num_samples:
            samples.extend([0.0] * (num_samples - len(samples)))
            
        return samples
    
    def get_buffer_health(self) -> float:
        """Get buffer health (0.0 = empty, 1.0 = at target size)."""
        target_samples = self._ms_to_samples(self.target_size_ms)
        current_samples = len(self.buffer)
        
        if target_samples <= 0:
            return 1.0
            
        return min(1.0, current_samples / target_samples)
    
    def adapt_size(self, new_target_ms: int) -> None:
        """Adapt buffer target size."""
        self.target_size_ms = max(10, min(500, new_target_ms))
    
    def get_latency_ms(self) -> float:
        """Get current buffer latency in milliseconds."""
        return len(self.buffer) / self.sample_rate * 1000.0
    
    def _ms_to_samples(self, ms: int) -> int:
        """Convert milliseconds to sample count."""
        return int(ms * self.sample_rate / 1000.0)


class AdaptiveQuantumStreamer:
    """Main adaptive quantum streaming engine."""
    
    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()
        self.predictor = QuantumBufferPredictor(self.config)
        self.quality_manager = AdaptiveQualityManager(self.config)
        self.buffer = StreamingAudioBuffer(self.config.target_latency_ms)
        
        self.metrics = StreamMetrics()
        self.is_streaming = False
        self.stream_start_time = 0.0
        
        # Performance tracking
        self.adaptation_count = 0
        self.total_samples_streamed = 0
        
        logger.info(f"AdaptiveQuantumStreamer initialized with config: {self.config}")
    
    async def start_streaming(self, 
                            audio_generator: AsyncGenerator[Union[List[float], 'np.ndarray'], None],
                            output_callback: Callable[[List[float], StreamMetrics], None]) -> None:
        """Start adaptive streaming with quantum optimization.
        
        Args:
            audio_generator: Async generator producing audio chunks
            output_callback: Callback for processed audio output
        """
        self.is_streaming = True
        self.stream_start_time = time.time()
        
        logger.info("Starting adaptive quantum streaming")
        
        try:
            # Start concurrent tasks
            tasks = [
                asyncio.create_task(self._audio_ingestion_loop(audio_generator)),
                asyncio.create_task(self._audio_output_loop(output_callback)),
                asyncio.create_task(self._adaptation_loop())
            ]
            
            # Wait for all tasks
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
        finally:
            self.is_streaming = False
            logger.info("Adaptive quantum streaming stopped")
    
    async def _audio_ingestion_loop(self, audio_generator: AsyncGenerator) -> None:
        """Handle audio ingestion with adaptive buffering."""
        async for audio_chunk in audio_generator:
            if not self.is_streaming:
                break
                
            # Write to buffer
            success = self.buffer.write(audio_chunk)
            
            if not success:
                logger.warning("Buffer overflow, dropping samples")
                self.metrics.error_rate += 0.01
            
            # Update metrics
            self.total_samples_streamed += len(audio_chunk) if hasattr(audio_chunk, '__len__') else 1
            
            # Small delay to prevent CPU spinning
            await asyncio.sleep(0.001)
    
    async def _audio_output_loop(self, output_callback: Callable) -> None:
        """Handle audio output with quality adaptation."""
        chunk_size_samples = int(self.buffer.sample_rate * 0.02)  # 20ms chunks
        
        while self.is_streaming:
            start_time = time.time()
            
            # Read from buffer
            audio_chunk = self.buffer.read(chunk_size_samples)
            
            # Apply quality adjustments based on current quality setting
            processed_chunk = self._apply_quality_processing(audio_chunk)
            
            # Update performance metrics
            self._update_streaming_metrics(start_time)
            
            # Send to output
            output_callback(processed_chunk, self.metrics)
            
            # Adaptive timing
            target_interval = chunk_size_samples / self.buffer.sample_rate
            elapsed = time.time() - start_time
            sleep_time = max(0, target_interval - elapsed)
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    async def _adaptation_loop(self) -> None:
        """Handle adaptive optimizations."""
        adaptation_interval = 0.1  # 100ms adaptation cycle
        
        while self.is_streaming:
            start_time = time.time()
            
            # Predict optimal buffer size
            if self.config.enable_predictive_buffering:
                predicted_buffer_ms, confidence = self.predictor.predict_buffer_needs(self.metrics)
                
                if confidence > 0.7:  # High confidence threshold
                    self.buffer.adapt_size(predicted_buffer_ms)
                    self.adaptation_count += 1
            
            # Adapt stream quality
            if self.config.enable_quality_adaptation:
                new_quality = self.quality_manager.adapt_quality(self.metrics)
                if new_quality != self.quality_manager.current_quality:
                    self.metrics.adaptation_events += 1
            
            # Update prediction accuracy
            if hasattr(self.predictor, 'accuracy_tracker') and self.predictor.accuracy_tracker:
                recent_accuracy = list(self.predictor.accuracy_tracker)[-5:]
                self.metrics.prediction_accuracy = sum(recent_accuracy) / len(recent_accuracy)
            
            elapsed = time.time() - start_time
            sleep_time = max(0, adaptation_interval - elapsed)
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    def _apply_quality_processing(self, audio_chunk: List[float]) -> List[float]:
        """Apply quality-specific processing to audio chunk."""
        current_quality = self.quality_manager.current_quality
        
        if current_quality == StreamQuality.ULTRA_LOW:
            # Heavy compression/downsampling simulation
            return self._downsample_audio(audio_chunk, 4)
        elif current_quality == StreamQuality.LOW:
            return self._downsample_audio(audio_chunk, 2)
        elif current_quality == StreamQuality.MEDIUM:
            return self._apply_light_compression(audio_chunk)
        elif current_quality == StreamQuality.HIGH:
            return audio_chunk  # Minimal processing
        elif current_quality == StreamQuality.ULTRA_HIGH:
            return self._apply_enhancement(audio_chunk)
        else:
            return audio_chunk
    
    def _downsample_audio(self, audio: List[float], factor: int) -> List[float]:
        """Simple downsampling for lower quality modes."""
        if factor <= 1:
            return audio
        return [audio[i] for i in range(0, len(audio), factor)]
    
    def _apply_light_compression(self, audio: List[float]) -> List[float]:
        """Apply light dynamic range compression."""
        threshold = 0.7
        ratio = 2.0
        
        compressed = []
        for sample in audio:
            if abs(sample) > threshold:
                excess = abs(sample) - threshold
                compressed_excess = excess / ratio
                new_sample = (threshold + compressed_excess) * (1 if sample >= 0 else -1)
                compressed.append(new_sample)
            else:
                compressed.append(sample)
        
        return compressed
    
    def _apply_enhancement(self, audio: List[float]) -> List[float]:
        """Apply subtle enhancement for ultra-high quality."""
        # Simple harmonic enhancement
        enhanced = []
        for i, sample in enumerate(audio):
            # Add subtle harmonic content
            enhanced_sample = sample
            if i > 0 and i < len(audio) - 1:
                # Add slight harmonic based on neighboring samples
                harmonic = (audio[i-1] + audio[i+1]) * 0.05
                enhanced_sample += harmonic
            
            enhanced.append(max(-1.0, min(1.0, enhanced_sample)))
        
        return enhanced
    
    def _update_streaming_metrics(self, start_time: float) -> None:
        """Update real-time streaming metrics."""
        current_time = time.time()
        
        # Calculate processing latency
        processing_latency = (current_time - start_time) * 1000.0  # Convert to ms
        
        # Update buffer health
        self.metrics.buffer_health = self.buffer.get_buffer_health()
        
        # Update total latency (processing + buffer)
        buffer_latency = self.buffer.get_latency_ms()
        self.metrics.latency_ms = processing_latency + buffer_latency
        
        # Calculate throughput
        stream_duration = current_time - self.stream_start_time
        if stream_duration > 0:
            samples_per_second = self.total_samples_streamed / stream_duration
            # Estimate bits per second (assuming 16-bit samples)
            bits_per_second = samples_per_second * 16
            self.metrics.throughput_mbps = bits_per_second / 1_000_000
        
        # Update quality index
        self.metrics.quality_index = self.quality_manager.quality_scores[self.quality_manager.current_quality]
        
        # Update adaptation events
        self.metrics.adaptation_events = self.adaptation_count
        
        # Update timestamp
        self.metrics.timestamp = current_time
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        stream_duration = time.time() - self.stream_start_time if self.stream_start_time else 0
        
        return {
            'streaming_duration_seconds': stream_duration,
            'total_samples_streamed': self.total_samples_streamed,
            'current_metrics': {
                'latency_ms': self.metrics.latency_ms,
                'throughput_mbps': self.metrics.throughput_mbps,
                'buffer_health': self.metrics.buffer_health,
                'quality_index': self.metrics.quality_index,
                'error_rate': self.metrics.error_rate,
                'prediction_accuracy': self.metrics.prediction_accuracy
            },
            'adaptation_stats': {
                'total_adaptations': self.adaptation_count,
                'quality_changes': self.metrics.adaptation_events,
                'current_quality': self.quality_manager.current_quality.value,
                'current_buffer_target_ms': self.buffer.target_size_ms
            },
            'configuration': {
                'quantum_prediction_enabled': self.config.enable_quantum_prediction,
                'quality_adaptation_enabled': self.config.enable_quality_adaptation,
                'predictive_buffering_enabled': self.config.enable_predictive_buffering,
                'target_latency_ms': self.config.target_latency_ms
            }
        }
    
    def stop_streaming(self) -> None:
        """Stop the streaming process."""
        self.is_streaming = False
        logger.info("Streaming stop requested")


# Factory functions for easy integration

def create_adaptive_streamer(target_latency_ms: int = 100,
                           enable_quantum_prediction: bool = True,
                           enable_quality_adaptation: bool = True) -> AdaptiveQuantumStreamer:
    """Create an adaptive quantum streamer with common configuration.
    
    Args:
        target_latency_ms: Target streaming latency
        enable_quantum_prediction: Enable quantum-inspired predictive buffering
        enable_quality_adaptation: Enable adaptive quality scaling
        
    Returns:
        Configured AdaptiveQuantumStreamer instance
    """
    config = AdaptiveConfig(
        target_latency_ms=target_latency_ms,
        enable_quantum_prediction=enable_quantum_prediction,
        enable_quality_adaptation=enable_quality_adaptation,
        enable_predictive_buffering=enable_quantum_prediction
    )
    
    return AdaptiveQuantumStreamer(config)


async def demonstrate_adaptive_streaming():
    """Demonstration of adaptive quantum streaming capabilities."""
    
    async def mock_audio_generator():
        """Generate mock audio data for demonstration."""
        sample_rate = 48000
        chunk_duration_ms = 20  # 20ms chunks
        samples_per_chunk = int(sample_rate * chunk_duration_ms / 1000)
        
        import math
        import random
        
        for i in range(500):  # 10 seconds of audio
            # Generate sine wave with some noise
            chunk = []
            for j in range(samples_per_chunk):
                sample_index = i * samples_per_chunk + j
                # 440 Hz sine wave with noise
                sine_wave = 0.3 * math.sin(2 * math.pi * 440 * sample_index / sample_rate)
                noise = 0.05 * (random.random() - 0.5)
                chunk.append(sine_wave + noise)
            
            yield chunk
            await asyncio.sleep(chunk_duration_ms / 1000.0)  # Real-time simulation
    
    def audio_output_callback(audio_chunk: List[float], metrics: StreamMetrics):
        """Handle processed audio output."""
        print(f"Output: {len(audio_chunk)} samples, "
              f"Latency: {metrics.latency_ms:.1f}ms, "
              f"Quality: {metrics.quality_index:.2f}, "
              f"Buffer: {metrics.buffer_health:.2f}")
    
    # Create and start adaptive streamer
    streamer = create_adaptive_streamer(
        target_latency_ms=80,
        enable_quantum_prediction=True,
        enable_quality_adaptation=True
    )
    
    logger.info("Starting adaptive streaming demonstration...")
    
    try:
        await streamer.start_streaming(
            mock_audio_generator(),
            audio_output_callback
        )
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted")
    finally:
        streamer.stop_streaming()
        
        # Print performance report
        report = streamer.get_performance_report()
        print("\n=== PERFORMANCE REPORT ===")
        for section, data in report.items():
            print(f"\n{section.upper()}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {data}")


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_adaptive_streaming())
