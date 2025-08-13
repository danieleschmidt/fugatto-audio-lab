"""Neural Adaptive Enhancement Engine - Generation 1 Enhancement.

Advanced neural-inspired adaptive audio enhancement with real-time learning,
pattern recognition, and contextual audio optimization.
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


class AudioContextType(Enum):
    """Types of audio content for context-aware processing."""
    SPEECH = "speech"
    MUSIC = "music"
    AMBIENT = "ambient"
    NOISE = "noise"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class EnhancementType(Enum):
    """Types of audio enhancements available."""
    NOISE_REDUCTION = "noise_reduction"
    DYNAMIC_RANGE = "dynamic_range"
    FREQUENCY_BALANCE = "frequency_balance"
    SPATIAL_ENHANCEMENT = "spatial_enhancement"
    HARMONIC_ENRICHMENT = "harmonic_enrichment"
    TEMPORAL_SMOOTHING = "temporal_smoothing"
    ADAPTIVE_COMPRESSION = "adaptive_compression"
    INTELLIGIBILITY = "intelligibility"


@dataclass
class AudioFeatures:
    """Comprehensive audio feature representation."""
    spectral_centroid: float = 0.0
    spectral_bandwidth: float = 0.0
    spectral_rolloff: float = 0.0
    zero_crossing_rate: float = 0.0
    energy: float = 0.0
    dynamic_range: float = 0.0
    fundamental_frequency: float = 0.0
    harmonic_ratio: float = 0.0
    noise_floor: float = 0.0
    temporal_stability: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class EnhancementParams:
    """Parameters for audio enhancement processing."""
    noise_reduction_strength: float = 0.5
    dynamic_compression_ratio: float = 2.0
    frequency_boost_low: float = 0.0  # dB
    frequency_boost_mid: float = 0.0  # dB
    frequency_boost_high: float = 0.0  # dB
    spatial_width: float = 1.0
    harmonic_enhancement: float = 0.0
    temporal_smoothing: float = 0.3
    intelligibility_boost: float = 0.0
    adaptation_speed: float = 0.1


class AudioContextAnalyzer:
    """Analyzes audio content to determine context and optimal processing."""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.feature_history = deque(maxlen=50)  # Last 50 analysis frames
        self.context_confidence = 0.0
        self.current_context = AudioContextType.UNKNOWN
        
        # Context classification thresholds (learned/adaptive)
        self.speech_thresholds = {
            'zcr_min': 0.05, 'zcr_max': 0.3,
            'spectral_centroid_min': 500, 'spectral_centroid_max': 4000,
            'fundamental_freq_min': 80, 'fundamental_freq_max': 400
        }
        
        self.music_thresholds = {
            'harmonic_ratio_min': 0.3,
            'dynamic_range_min': 15,  # dB
            'spectral_bandwidth_min': 2000
        }
        
    def analyze_audio_context(self, audio: Union[List[float], 'np.ndarray']) -> Tuple[AudioContextType, float]:
        """Analyze audio to determine context and confidence.
        
        Returns:
            Tuple of (context_type, confidence)
        """
        features = self._extract_features(audio)
        self.feature_history.append(features)
        
        # Classify based on features
        context_scores = self._calculate_context_scores(features)
        
        # Select highest scoring context
        best_context = max(context_scores.items(), key=lambda x: x[1])
        context_type, confidence = best_context
        
        # Apply temporal smoothing
        if len(self.feature_history) >= 3:
            smoothed_context, smoothed_confidence = self._apply_temporal_smoothing(context_type, confidence)
            self.current_context = smoothed_context
            self.context_confidence = smoothed_confidence
        else:
            self.current_context = context_type
            self.context_confidence = confidence
        
        return self.current_context, self.context_confidence
    
    def _extract_features(self, audio: Union[List[float], 'np.ndarray']) -> AudioFeatures:
        """Extract comprehensive audio features."""
        if np is not None and isinstance(audio, np.ndarray):
            audio_array = audio
        else:
            audio_array = np.array(audio) if np else list(audio)
        
        features = AudioFeatures()
        
        if len(audio) == 0:
            return features
        
        # Basic statistics
        if np is not None:
            features.energy = float(np.mean(audio_array ** 2))
            features.dynamic_range = float(20 * np.log10(np.max(np.abs(audio_array)) / (np.sqrt(features.energy) + 1e-10)))
        else:
            features.energy = sum(x**2 for x in audio) / len(audio)
            max_val = max(abs(x) for x in audio)
            rms = (features.energy)**0.5
            features.dynamic_range = 20 * (max_val / (rms + 1e-10))**0.5  # Approximate log
        
        # Zero crossing rate
        features.zero_crossing_rate = self._calculate_zcr(audio)
        
        # Spectral features (simplified without full FFT)
        features.spectral_centroid = self._estimate_spectral_centroid(audio)
        features.spectral_bandwidth = features.spectral_centroid * 0.3  # Rough estimate
        features.spectral_rolloff = features.spectral_centroid * 1.5  # Rough estimate
        
        # Fundamental frequency estimation
        features.fundamental_frequency = self._estimate_fundamental_freq(audio)
        
        # Harmonic ratio estimation
        features.harmonic_ratio = self._estimate_harmonic_ratio(audio)
        
        # Noise floor estimation
        features.noise_floor = self._estimate_noise_floor(audio)
        
        # Temporal stability
        features.temporal_stability = self._calculate_temporal_stability(audio)
        
        return features
    
    def _calculate_zcr(self, audio: Union[List[float], 'np.ndarray']) -> float:
        """Calculate zero crossing rate."""
        if len(audio) < 2:
            return 0.0
        
        zero_crossings = 0
        for i in range(1, len(audio)):
            if (audio[i] >= 0) != (audio[i-1] >= 0):
                zero_crossings += 1
        
        return zero_crossings / (len(audio) - 1)
    
    def _estimate_spectral_centroid(self, audio: Union[List[float], 'np.ndarray']) -> float:
        """Estimate spectral centroid using autocorrelation."""
        if len(audio) < 4:
            return 1000.0  # Default
        
        # Simple autocorrelation-based estimation
        autocorr = []
        max_lag = min(len(audio) // 4, 200)  # Limit lag
        
        for lag in range(1, max_lag):
            corr = 0.0
            count = 0
            for i in range(len(audio) - lag):
                corr += audio[i] * audio[i + lag]
                count += 1
            
            if count > 0:
                autocorr.append(abs(corr / count))
            else:
                autocorr.append(0.0)
        
        if not autocorr:
            return 1000.0
        
        # Find peak in autocorrelation (indicates periodicity)
        max_corr_idx = 0
        max_corr_val = autocorr[0]
        for i, val in enumerate(autocorr):
            if val > max_corr_val:
                max_corr_val = val
                max_corr_idx = i
        
        # Convert lag to frequency
        if max_corr_idx > 0:
            period_samples = max_corr_idx + 1
            estimated_freq = self.sample_rate / period_samples
            # Spectral centroid is typically higher than fundamental
            return min(8000.0, estimated_freq * 2.5)
        
        return 1000.0
    
    def _estimate_fundamental_freq(self, audio: Union[List[float], 'np.ndarray']) -> float:
        """Estimate fundamental frequency using simple autocorrelation."""
        if len(audio) < 8:
            return 0.0
        
        # Autocorrelation for pitch detection
        min_period = int(self.sample_rate / 800)  # 800 Hz max
        max_period = int(self.sample_rate / 50)   # 50 Hz min
        
        max_period = min(max_period, len(audio) // 2)
        
        best_period = 0
        best_correlation = 0.0
        
        for period in range(min_period, max_period):
            correlation = 0.0
            count = 0
            
            for i in range(len(audio) - period):
                correlation += audio[i] * audio[i + period]
                count += 1
            
            if count > 0:
                correlation /= count
                
                if abs(correlation) > abs(best_correlation):
                    best_correlation = correlation
                    best_period = period
        
        if best_period > 0 and abs(best_correlation) > 0.1:
            return self.sample_rate / best_period
        
        return 0.0
    
    def _estimate_harmonic_ratio(self, audio: Union[List[float], 'np.ndarray']) -> float:
        """Estimate harmonic ratio (harmonicity vs noise)."""
        fundamental = self._estimate_fundamental_freq(audio)
        
        if fundamental < 50 or fundamental > 800:
            return 0.0  # No clear fundamental
        
        # Check for harmonics at 2f, 3f, 4f
        harmonic_strength = 0.0
        total_energy = sum(x**2 for x in audio) if audio else 1e-10
        
        for harmonic_num in [2, 3, 4]:
            harmonic_freq = fundamental * harmonic_num
            if harmonic_freq < self.sample_rate / 2:
                # Simple harmonic detection (would be more accurate with FFT)
                harmonic_strength += 0.2  # Placeholder
        
        return min(1.0, harmonic_strength)
    
    def _estimate_noise_floor(self, audio: Union[List[float], 'np.ndarray']) -> float:
        """Estimate noise floor level."""
        if not audio:
            return 0.0
        
        # Sort absolute values and take lower percentile as noise estimate
        abs_values = [abs(x) for x in audio]
        abs_values.sort()
        
        # Take 10th percentile as noise floor estimate
        noise_idx = max(0, int(len(abs_values) * 0.1))
        noise_floor = abs_values[noise_idx] if abs_values else 0.0
        
        return noise_floor
    
    def _calculate_temporal_stability(self, audio: Union[List[float], 'np.ndarray']) -> float:
        """Calculate temporal stability (consistency over time)."""
        if len(audio) < 10:
            return 1.0
        
        # Calculate energy in overlapping windows
        window_size = len(audio) // 5
        if window_size < 2:
            return 1.0
        
        window_energies = []
        for i in range(5):
            start = i * window_size
            end = min(start + window_size, len(audio))
            if end > start:
                window_energy = sum(x**2 for x in audio[start:end]) / (end - start)
                window_energies.append(window_energy)
        
        if len(window_energies) < 2:
            return 1.0
        
        # Calculate coefficient of variation
        mean_energy = sum(window_energies) / len(window_energies)
        if mean_energy == 0:
            return 1.0
        
        variance = sum((e - mean_energy)**2 for e in window_energies) / len(window_energies)
        std_dev = variance**0.5
        cv = std_dev / mean_energy
        
        # Stability is inverse of coefficient of variation
        stability = max(0.0, min(1.0, 1.0 - cv))
        return stability
    
    def _calculate_context_scores(self, features: AudioFeatures) -> Dict[AudioContextType, float]:
        """Calculate confidence scores for each context type."""
        scores = {}
        
        # Speech detection
        speech_score = 0.0
        if (self.speech_thresholds['zcr_min'] <= features.zero_crossing_rate <= self.speech_thresholds['zcr_max'] and
            self.speech_thresholds['spectral_centroid_min'] <= features.spectral_centroid <= self.speech_thresholds['spectral_centroid_max']):
            speech_score += 0.4
        
        if (self.speech_thresholds['fundamental_freq_min'] <= features.fundamental_frequency <= self.speech_thresholds['fundamental_freq_max']):
            speech_score += 0.3
        
        if features.temporal_stability > 0.3:  # Speech has moderate stability
            speech_score += 0.3
        
        scores[AudioContextType.SPEECH] = speech_score
        
        # Music detection
        music_score = 0.0
        if features.harmonic_ratio >= self.music_thresholds['harmonic_ratio_min']:
            music_score += 0.4
        
        if features.dynamic_range >= self.music_thresholds['dynamic_range_min']:
            music_score += 0.3
        
        if features.spectral_bandwidth >= self.music_thresholds['spectral_bandwidth_min']:
            music_score += 0.3
        
        scores[AudioContextType.MUSIC] = music_score
        
        # Ambient/environmental sound detection
        ambient_score = 0.0
        if features.zero_crossing_rate > 0.3 and features.harmonic_ratio < 0.2:
            ambient_score += 0.5
        
        if features.temporal_stability < 0.5:  # Ambient sounds are often variable
            ambient_score += 0.3
        
        if features.spectral_centroid > 2000:  # Often higher frequency content
            ambient_score += 0.2
        
        scores[AudioContextType.AMBIENT] = ambient_score
        
        # Noise detection
        noise_score = 0.0
        if features.harmonic_ratio < 0.1 and features.zero_crossing_rate > 0.4:
            noise_score += 0.6
        
        if features.temporal_stability < 0.3:  # Noise is typically unstable
            noise_score += 0.4
        
        scores[AudioContextType.NOISE] = noise_score
        
        # Mixed content (combination of multiple types)
        mixed_score = 0.0
        high_scores = [s for s in scores.values() if s > 0.3]
        if len(high_scores) >= 2:
            mixed_score = 0.4
        
        scores[AudioContextType.MIXED] = mixed_score
        
        # Unknown (when no clear classification)
        max_score = max(scores.values()) if scores else 0.0
        if max_score < 0.3:
            scores[AudioContextType.UNKNOWN] = 0.5
        else:
            scores[AudioContextType.UNKNOWN] = 0.0
        
        return scores
    
    def _apply_temporal_smoothing(self, current_context: AudioContextType, current_confidence: float) -> Tuple[AudioContextType, float]:
        """Apply temporal smoothing to reduce classification jitter."""
        if len(self.feature_history) < 3:
            return current_context, current_confidence
        
        # Count recent contexts
        recent_contexts = []
        for i in range(max(0, len(self.feature_history) - 5), len(self.feature_history)):
            # Re-analyze recent frames for context consistency
            # (In practice, we'd store the contexts, but for simplicity we'll use current)
            recent_contexts.append(current_context)
        
        # If current context is consistent with recent history, boost confidence
        if recent_contexts.count(current_context) >= len(recent_contexts) * 0.6:
            smoothed_confidence = min(1.0, current_confidence * 1.2)
        else:
            smoothed_confidence = current_confidence * 0.8
        
        return current_context, smoothed_confidence


class AdaptiveEnhancementEngine:
    """Neural-inspired adaptive enhancement engine."""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.context_analyzer = AudioContextAnalyzer(sample_rate)
        
        # Enhancement parameters per context
        self.context_params = {
            AudioContextType.SPEECH: EnhancementParams(
                noise_reduction_strength=0.7,
                dynamic_compression_ratio=3.0,
                frequency_boost_mid=3.0,  # Boost speech frequencies
                intelligibility_boost=0.5,
                temporal_smoothing=0.2
            ),
            AudioContextType.MUSIC: EnhancementParams(
                noise_reduction_strength=0.3,
                dynamic_compression_ratio=1.5,
                frequency_boost_low=1.0,
                frequency_boost_high=2.0,
                harmonic_enhancement=0.3,
                spatial_width=1.2
            ),
            AudioContextType.AMBIENT: EnhancementParams(
                noise_reduction_strength=0.4,
                frequency_boost_high=1.5,
                spatial_width=1.5,
                temporal_smoothing=0.4
            ),
            AudioContextType.NOISE: EnhancementParams(
                noise_reduction_strength=0.9,
                dynamic_compression_ratio=4.0,
                temporal_smoothing=0.6
            ),
            AudioContextType.MIXED: EnhancementParams(
                noise_reduction_strength=0.5,
                dynamic_compression_ratio=2.5,
                frequency_boost_mid=1.5,
                temporal_smoothing=0.3
            ),
            AudioContextType.UNKNOWN: EnhancementParams()  # Conservative defaults
        }
        
        # Adaptive learning
        self.parameter_history = defaultdict(lambda: deque(maxlen=20))
        self.enhancement_effectiveness = defaultdict(float)
        
        # Processing state
        self.previous_audio = deque(maxlen=1024)  # For temporal processing
        self.enhancement_state = {}
        
    def enhance_audio(self, audio: Union[List[float], 'np.ndarray'], 
                     user_preferences: Optional[Dict[str, float]] = None) -> List[float]:
        """Apply adaptive enhancement to audio based on context analysis.
        
        Args:
            audio: Input audio data
            user_preferences: Optional user preference overrides
            
        Returns:
            Enhanced audio data
        """
        if not audio:
            return []
        
        # Analyze audio context
        context, confidence = self.context_analyzer.analyze_audio_context(audio)
        
        # Get appropriate enhancement parameters
        base_params = self.context_params[context]
        
        # Apply user preferences if provided
        if user_preferences:
            enhanced_params = self._apply_user_preferences(base_params, user_preferences)
        else:
            enhanced_params = base_params
        
        # Adapt parameters based on recent performance
        adaptive_params = self._adapt_parameters(enhanced_params, context, confidence)
        
        # Apply enhancements
        enhanced_audio = self._apply_enhancements(audio, adaptive_params, context)
        
        # Learn from results
        self._update_learning(audio, enhanced_audio, adaptive_params, context)
        
        # Update processing state
        if np is not None and isinstance(enhanced_audio, np.ndarray):
            self.previous_audio.extend(enhanced_audio.tolist())
        else:
            self.previous_audio.extend(enhanced_audio)
        
        logger.debug(f"Enhanced audio: context={context.value}, confidence={confidence:.3f}")
        
        return enhanced_audio
    
    def _apply_user_preferences(self, base_params: EnhancementParams, 
                               preferences: Dict[str, float]) -> EnhancementParams:
        """Apply user preferences to base parameters."""
        # Create a copy of base parameters
        enhanced_params = EnhancementParams(
            noise_reduction_strength=base_params.noise_reduction_strength,
            dynamic_compression_ratio=base_params.dynamic_compression_ratio,
            frequency_boost_low=base_params.frequency_boost_low,
            frequency_boost_mid=base_params.frequency_boost_mid,
            frequency_boost_high=base_params.frequency_boost_high,
            spatial_width=base_params.spatial_width,
            harmonic_enhancement=base_params.harmonic_enhancement,
            temporal_smoothing=base_params.temporal_smoothing,
            intelligibility_boost=base_params.intelligibility_boost,
            adaptation_speed=base_params.adaptation_speed
        )
        
        # Apply user overrides
        for pref_name, pref_value in preferences.items():
            if hasattr(enhanced_params, pref_name):
                # Blend user preference with base parameter
                current_value = getattr(enhanced_params, pref_name)
                blended_value = current_value * 0.7 + pref_value * 0.3
                setattr(enhanced_params, pref_name, blended_value)
        
        return enhanced_params
    
    def _adapt_parameters(self, params: EnhancementParams, context: AudioContextType, 
                         confidence: float) -> EnhancementParams:
        """Adapt parameters based on recent performance and context confidence."""
        adapted_params = EnhancementParams(
            noise_reduction_strength=params.noise_reduction_strength,
            dynamic_compression_ratio=params.dynamic_compression_ratio,
            frequency_boost_low=params.frequency_boost_low,
            frequency_boost_mid=params.frequency_boost_mid,
            frequency_boost_high=params.frequency_boost_high,
            spatial_width=params.spatial_width,
            harmonic_enhancement=params.harmonic_enhancement,
            temporal_smoothing=params.temporal_smoothing,
            intelligibility_boost=params.intelligibility_boost,
            adaptation_speed=params.adaptation_speed
        )
        
        # Reduce enhancement strength if confidence is low
        if confidence < 0.7:
            confidence_factor = confidence / 0.7
            adapted_params.noise_reduction_strength *= confidence_factor
            adapted_params.frequency_boost_low *= confidence_factor
            adapted_params.frequency_boost_mid *= confidence_factor
            adapted_params.frequency_boost_high *= confidence_factor
            adapted_params.harmonic_enhancement *= confidence_factor
        
        # Apply learning-based adaptations
        if context in self.enhancement_effectiveness:
            effectiveness = self.enhancement_effectiveness[context]
            if effectiveness < 0.5:  # Poor performance, reduce aggressiveness
                adapted_params.noise_reduction_strength *= 0.8
                adapted_params.dynamic_compression_ratio *= 0.9
            elif effectiveness > 0.8:  # Good performance, slightly increase
                adapted_params.noise_reduction_strength = min(1.0, adapted_params.noise_reduction_strength * 1.1)
        
        return adapted_params
    
    def _apply_enhancements(self, audio: Union[List[float], 'np.ndarray'], 
                           params: EnhancementParams, context: AudioContextType) -> List[float]:
        """Apply the actual audio enhancements."""
        if np is not None and isinstance(audio, np.ndarray):
            enhanced = audio.copy()
        else:
            enhanced = list(audio)
        
        # Apply noise reduction
        if params.noise_reduction_strength > 0:
            enhanced = self._apply_noise_reduction(enhanced, params.noise_reduction_strength)
        
        # Apply dynamic range compression
        if params.dynamic_compression_ratio > 1.0:
            enhanced = self._apply_compression(enhanced, params.dynamic_compression_ratio)
        
        # Apply frequency boosts
        if any([params.frequency_boost_low, params.frequency_boost_mid, params.frequency_boost_high]):
            enhanced = self._apply_frequency_enhancement(enhanced, params)
        
        # Apply harmonic enhancement
        if params.harmonic_enhancement > 0:
            enhanced = self._apply_harmonic_enhancement(enhanced, params.harmonic_enhancement)
        
        # Apply temporal smoothing
        if params.temporal_smoothing > 0:
            enhanced = self._apply_temporal_smoothing(enhanced, params.temporal_smoothing)
        
        # Apply intelligibility boost (for speech)
        if params.intelligibility_boost > 0 and context == AudioContextType.SPEECH:
            enhanced = self._apply_intelligibility_boost(enhanced, params.intelligibility_boost)
        
        # Ensure output is in valid range
        enhanced = self._normalize_output(enhanced)
        
        return enhanced
    
    def _apply_noise_reduction(self, audio: List[float], strength: float) -> List[float]:
        """Apply adaptive noise reduction."""
        if strength <= 0 or not audio:
            return audio
        
        # Simple spectral subtraction approximation
        # Estimate noise floor from quietest 10% of samples
        sorted_abs = sorted([abs(x) for x in audio])
        noise_estimate = sorted_abs[int(len(sorted_abs) * 0.1)] if sorted_abs else 0.0
        
        # Apply noise gate with soft knee
        enhanced = []
        gate_threshold = noise_estimate * (2.0 - strength)  # Higher strength = lower threshold
        
        for sample in audio:
            abs_sample = abs(sample)
            if abs_sample > gate_threshold:
                # Above threshold - minimal processing
                enhanced.append(sample)
            else:
                # Below threshold - apply noise reduction
                reduction_factor = max(0.1, abs_sample / gate_threshold)
                enhanced.append(sample * reduction_factor * (1.0 - strength * 0.8))
        
        return enhanced
    
    def _apply_compression(self, audio: List[float], ratio: float) -> List[float]:
        """Apply dynamic range compression."""
        if ratio <= 1.0 or not audio:
            return audio
        
        # Adaptive threshold based on audio characteristics
        if np is not None:
            rms = float(np.sqrt(np.mean([x**2 for x in audio])))
        else:
            rms = (sum(x**2 for x in audio) / len(audio))**0.5
        
        threshold = rms * 1.5  # Threshold at 1.5x RMS
        
        compressed = []
        for sample in audio:
            abs_sample = abs(sample)
            if abs_sample > threshold:
                # Above threshold - apply compression
                excess = abs_sample - threshold
                compressed_excess = excess / ratio
                new_amplitude = threshold + compressed_excess
                compressed.append((1 if sample >= 0 else -1) * new_amplitude)
            else:
                # Below threshold - no compression
                compressed.append(sample)
        
        return compressed
    
    def _apply_frequency_enhancement(self, audio: List[float], params: EnhancementParams) -> List[float]:
        """Apply frequency-specific enhancements (simplified)."""
        # This is a simplified frequency enhancement
        # In practice, this would use proper filter banks or FFT
        
        enhanced = list(audio)
        
        # Low frequency boost (simulate with gentle filtering)
        if params.frequency_boost_low > 0:
            enhanced = self._apply_low_shelf(enhanced, params.frequency_boost_low)
        
        # Mid frequency boost
        if params.frequency_boost_mid > 0:
            enhanced = self._apply_mid_boost(enhanced, params.frequency_boost_mid)
        
        # High frequency boost
        if params.frequency_boost_high > 0:
            enhanced = self._apply_high_shelf(enhanced, params.frequency_boost_high)
        
        return enhanced
    
    def _apply_low_shelf(self, audio: List[float], gain_db: float) -> List[float]:
        """Apply low-frequency shelf filter (simplified)."""
        if gain_db == 0 or not audio:
            return audio
        
        gain_linear = 10**(gain_db / 20)
        
        # Simple first-order low-pass filter with gain
        filtered = [audio[0]]
        alpha = 0.1  # Low-pass coefficient
        
        for i in range(1, len(audio)):
            # Low-pass filter
            low_passed = alpha * audio[i] + (1 - alpha) * filtered[i-1]
            
            # Apply gain to low frequencies
            enhanced_low = low_passed * gain_linear
            
            # Mix with original
            filtered.append(audio[i] + (enhanced_low - low_passed) * 0.5)
        
        return filtered
    
    def _apply_mid_boost(self, audio: List[float], gain_db: float) -> List[float]:
        """Apply mid-frequency boost (simplified)."""
        if gain_db == 0 or not audio:
            return audio
        
        gain_linear = 10**(gain_db / 20)
        
        # Simple bandpass-like enhancement
        enhanced = []
        window_size = 5
        
        for i in range(len(audio)):
            # Calculate local average (acts like mid-frequency content)
            start = max(0, i - window_size // 2)
            end = min(len(audio), i + window_size // 2 + 1)
            local_avg = sum(audio[start:end]) / (end - start)
            
            # Boost deviation from local average (enhances mid frequencies)
            deviation = audio[i] - local_avg
            enhanced_sample = audio[i] + deviation * (gain_linear - 1) * 0.3
            enhanced.append(enhanced_sample)
        
        return enhanced
    
    def _apply_high_shelf(self, audio: List[float], gain_db: float) -> List[float]:
        """Apply high-frequency shelf filter (simplified)."""
        if gain_db == 0 or not audio:
            return audio
        
        gain_linear = 10**(gain_db / 20)
        
        # Simple high-pass filter with gain
        enhanced = [audio[0]]
        
        for i in range(1, len(audio)):
            # High-pass filter (difference)
            high_passed = audio[i] - audio[i-1]
            
            # Apply gain to high frequencies
            enhanced_high = high_passed * gain_linear
            
            # Mix with original
            enhanced.append(audio[i] + enhanced_high * 0.3)
        
        return enhanced
    
    def _apply_harmonic_enhancement(self, audio: List[float], strength: float) -> List[float]:
        """Apply harmonic enhancement."""
        if strength <= 0 or not audio:
            return audio
        
        enhanced = []
        
        # Add subtle harmonic content based on neighboring samples
        for i in range(len(audio)):
            sample = audio[i]
            
            # Generate harmonics from local context
            harmonic_content = 0.0
            if i > 0 and i < len(audio) - 1:
                # Second harmonic approximation
                harmonic_content += (audio[i-1] + audio[i+1]) * 0.1 * strength
                
                # Third harmonic approximation
                if i > 1 and i < len(audio) - 2:
                    harmonic_content += (audio[i-2] + audio[i+2]) * 0.05 * strength
            
            enhanced_sample = sample + harmonic_content
            enhanced.append(max(-1.0, min(1.0, enhanced_sample)))
        
        return enhanced
    
    def _apply_temporal_smoothing(self, audio: List[float], strength: float) -> List[float]:
        """Apply temporal smoothing to reduce artifacts."""
        if strength <= 0 or not audio:
            return audio
        
        smoothed = [audio[0]]  # First sample unchanged
        
        # Apply smoothing filter
        for i in range(1, len(audio)):
            # Weighted average with previous sample
            smoothed_sample = (1 - strength) * audio[i] + strength * smoothed[i-1]
            smoothed.append(smoothed_sample)
        
        return smoothed
    
    def _apply_intelligibility_boost(self, audio: List[float], strength: float) -> List[float]:
        """Apply intelligibility enhancement for speech."""
        if strength <= 0 or not audio:
            return audio
        
        # Enhance consonant clarity by boosting transients
        enhanced = []
        
        for i in range(len(audio)):
            sample = audio[i]
            
            # Detect transients (rapid changes)
            if i > 0 and i < len(audio) - 1:
                change_rate = abs(audio[i] - audio[i-1]) + abs(audio[i+1] - audio[i])
                
                # Boost samples with high change rate (consonants)
                if change_rate > 0.1:  # Transient threshold
                    boost_factor = 1.0 + strength * 0.3
                    enhanced_sample = sample * boost_factor
                else:
                    enhanced_sample = sample
            else:
                enhanced_sample = sample
            
            enhanced.append(max(-1.0, min(1.0, enhanced_sample)))
        
        return enhanced
    
    def _normalize_output(self, audio: List[float]) -> List[float]:
        """Normalize output to prevent clipping."""
        if not audio:
            return audio
        
        max_abs = max(abs(x) for x in audio)
        
        if max_abs > 1.0:
            # Normalize to 0.95 to leave headroom
            scale_factor = 0.95 / max_abs
            return [x * scale_factor for x in audio]
        
        return audio
    
    def _update_learning(self, original: Union[List[float], 'np.ndarray'], 
                        enhanced: List[float], params: EnhancementParams, 
                        context: AudioContextType) -> None:
        """Update learning parameters based on enhancement results."""
        # Simple effectiveness metric based on energy and dynamic range improvement
        if np is not None:
            orig_array = np.array(original) if not isinstance(original, np.ndarray) else original
            enh_array = np.array(enhanced)
            
            orig_energy = float(np.mean(orig_array ** 2))
            enh_energy = float(np.mean(enh_array ** 2))
            
            orig_dynamic_range = float(np.max(np.abs(orig_array))) / (np.sqrt(orig_energy) + 1e-10)
            enh_dynamic_range = float(np.max(np.abs(enh_array))) / (np.sqrt(enh_energy) + 1e-10)
        else:
            orig_energy = sum(x**2 for x in original) / len(original) if original else 0
            enh_energy = sum(x**2 for x in enhanced) / len(enhanced) if enhanced else 0
            
            orig_max = max(abs(x) for x in original) if original else 0
            enh_max = max(abs(x) for x in enhanced) if enhanced else 0
            
            orig_dynamic_range = orig_max / (orig_energy**0.5 + 1e-10)
            enh_dynamic_range = enh_max / (enh_energy**0.5 + 1e-10)
        
        # Calculate effectiveness score
        energy_improvement = min(2.0, enh_energy / (orig_energy + 1e-10))
        dynamic_improvement = min(2.0, enh_dynamic_range / (orig_dynamic_range + 1e-10))
        
        effectiveness = (energy_improvement + dynamic_improvement) / 4.0  # Normalize to 0-1
        
        # Update running average
        current_effectiveness = self.enhancement_effectiveness.get(context, 0.5)
        learning_rate = params.adaptation_speed
        new_effectiveness = (1 - learning_rate) * current_effectiveness + learning_rate * effectiveness
        
        self.enhancement_effectiveness[context] = max(0.0, min(1.0, new_effectiveness))
        
        # Store parameter history for future adaptation
        param_key = f"{context.value}_effectiveness"
        self.parameter_history[param_key].append(effectiveness)
    
    def get_enhancement_report(self) -> Dict[str, Any]:
        """Generate comprehensive enhancement performance report."""
        report = {
            'context_effectiveness': dict(self.enhancement_effectiveness),
            'current_contexts': {
                'type': self.context_analyzer.current_context.value,
                'confidence': self.context_analyzer.context_confidence
            },
            'enhancement_settings': {},
            'learning_statistics': {}
        }
        
        # Add current enhancement settings for each context
        for context, params in self.context_params.items():
            report['enhancement_settings'][context.value] = {
                'noise_reduction': params.noise_reduction_strength,
                'compression_ratio': params.dynamic_compression_ratio,
                'frequency_boosts': {
                    'low': params.frequency_boost_low,
                    'mid': params.frequency_boost_mid,
                    'high': params.frequency_boost_high
                },
                'harmonic_enhancement': params.harmonic_enhancement,
                'intelligibility_boost': params.intelligibility_boost
            }
        
        # Add learning statistics
        for key, history in self.parameter_history.items():
            if history:
                report['learning_statistics'][key] = {
                    'mean': sum(history) / len(history),
                    'recent_trend': list(history)[-5:] if len(history) >= 5 else list(history),
                    'sample_count': len(history)
                }
        
        return report


# Factory functions for easy integration

def create_neural_enhancer(sample_rate: int = 48000) -> AdaptiveEnhancementEngine:
    """Create a neural adaptive enhancement engine.
    
    Args:
        sample_rate: Audio sample rate
        
    Returns:
        Configured AdaptiveEnhancementEngine instance
    """
    return AdaptiveEnhancementEngine(sample_rate)


def enhance_audio_intelligently(audio: Union[List[float], 'np.ndarray'],
                               sample_rate: int = 48000,
                               user_preferences: Optional[Dict[str, float]] = None) -> List[float]:
    """One-shot intelligent audio enhancement.
    
    Args:
        audio: Input audio data
        sample_rate: Audio sample rate
        user_preferences: Optional user preference overrides
        
    Returns:
        Enhanced audio data
    """
    enhancer = create_neural_enhancer(sample_rate)
    return enhancer.enhance_audio(audio, user_preferences)


if __name__ == "__main__":
    # Demonstration
    import math
    
    # Generate test audio (speech-like)
    sample_rate = 48000
    duration = 2.0  # seconds
    
    test_audio = []
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        # Speech-like signal: fundamental + harmonics + noise
        fundamental = 0.3 * math.sin(2 * math.pi * 200 * t)  # 200 Hz fundamental
        harmonic2 = 0.15 * math.sin(2 * math.pi * 400 * t)   # 2nd harmonic
        harmonic3 = 0.1 * math.sin(2 * math.pi * 600 * t)    # 3rd harmonic
        noise = 0.05 * (2 * (i % 100) / 100 - 1)  # Simple noise
        
        sample = fundamental + harmonic2 + harmonic3 + noise
        test_audio.append(sample)
    
    # Create enhancer and process
    enhancer = create_neural_enhancer(sample_rate)
    
    # Test with different user preferences
    speech_preferences = {
        'noise_reduction_strength': 0.8,
        'intelligibility_boost': 0.7,
        'frequency_boost_mid': 4.0
    }
    
    enhanced_audio = enhancer.enhance_audio(test_audio, speech_preferences)
    
    print(f"Original audio length: {len(test_audio)} samples")
    print(f"Enhanced audio length: {len(enhanced_audio)} samples")
    
    # Generate report
    report = enhancer.get_enhancement_report()
    print("\n=== ENHANCEMENT REPORT ===")
    for section, data in report.items():
        print(f"\n{section.upper()}:")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {data}")
