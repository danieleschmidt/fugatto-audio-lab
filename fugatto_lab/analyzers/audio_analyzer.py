"""Advanced audio signal analysis and feature extraction."""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

try:
    import librosa
    import scipy.signal
    import scipy.stats
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    librosa = None
    scipy = None
    AUDIO_LIBS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """Advanced audio analysis with comprehensive feature extraction."""
    
    def __init__(self, sample_rate: int = 48000):
        """Initialize audio analyzer.
        
        Args:
            sample_rate: Target sample rate for analysis
        """
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.n_fft = 2048
        self.n_mels = 128
        
        # Analysis parameters
        self.frequency_bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'presence': (4000, 6000),
            'brilliance': (6000, 20000)
        }
        
        logger.info(f"AudioAnalyzer initialized for {sample_rate}Hz")
    
    def analyze_comprehensive(self, audio: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive audio analysis.
        
        Args:
            audio: Audio signal to analyze
            
        Returns:
            Dictionary with all analysis results
        """
        results = {}
        
        # Basic statistics
        results['basic'] = self._analyze_basic_stats(audio)
        
        # Spectral analysis
        results['spectral'] = self._analyze_spectral_features(audio)
        
        # Temporal analysis
        results['temporal'] = self._analyze_temporal_features(audio)
        
        # Perceptual analysis
        results['perceptual'] = self._analyze_perceptual_features(audio)
        
        # Content analysis
        results['content'] = self._analyze_content_features(audio)
        
        logger.info(f"Comprehensive analysis completed for {len(audio)/self.sample_rate:.2f}s audio")
        return results
    
    def _analyze_basic_stats(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze basic statistical properties."""
        stats = {
            'duration': len(audio) / self.sample_rate,
            'samples': len(audio),
            'rms': float(np.sqrt(np.mean(audio ** 2))),
            'peak': float(np.max(np.abs(audio))),
            'mean': float(np.mean(audio)),
            'std': float(np.std(audio)),
            'skewness': float(scipy.stats.skew(audio)) if scipy else 0.0,
            'kurtosis': float(scipy.stats.kurtosis(audio)) if scipy else 0.0,
            'zero_crossings': int(np.sum(np.diff(np.sign(audio)) != 0) / 2),
            'energy': float(np.sum(audio ** 2))
        }
        
        # Dynamic range metrics
        if stats['rms'] > 0:
            stats['crest_factor'] = stats['peak'] / stats['rms']
            stats['dynamic_range_db'] = 20 * np.log10(stats['crest_factor'])
        else:
            stats['crest_factor'] = 0.0
            stats['dynamic_range_db'] = -np.inf
            
        return stats
    
    def _analyze_spectral_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze spectral characteristics."""
        spectral = {}
        
        if not AUDIO_LIBS_AVAILABLE:
            logger.warning("Audio analysis libraries not available, using basic FFT analysis")
            return self._basic_spectral_analysis(audio)
        
        try:
            # Spectral features using librosa
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral['centroid_mean'] = float(np.mean(spectral_centroids))
            spectral['centroid_std'] = float(np.std(spectral_centroids))
            spectral['centroid_var'] = float(np.var(spectral_centroids))
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            spectral['rolloff_mean'] = float(np.mean(rolloff))
            spectral['rolloff_std'] = float(np.std(rolloff))
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            spectral['bandwidth_mean'] = float(np.mean(bandwidth))
            spectral['bandwidth_std'] = float(np.std(bandwidth))
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)\n            spectral['contrast_mean'] = float(np.mean(contrast))\n            spectral['contrast_std'] = float(np.std(contrast))\n            \n            # Spectral flatness\n            flatness = librosa.feature.spectral_flatness(y=audio)[0]\n            spectral['flatness_mean'] = float(np.mean(flatness))\n            spectral['flatness_std'] = float(np.std(flatness))\n            \n            # Frequency band energy distribution\n            spectral['band_energy'] = self._analyze_frequency_bands(audio)\n            \n        except Exception as e:\n            logger.warning(f\"Spectral analysis error: {e}\")\n            spectral = self._basic_spectral_analysis(audio)\n            \n        return spectral\n    \n    def _analyze_temporal_features(self, audio: np.ndarray) -> Dict[str, Any]:\n        \"\"\"Analyze temporal characteristics.\"\"\"\n        temporal = {}\n        \n        if not AUDIO_LIBS_AVAILABLE:\n            return self._basic_temporal_analysis(audio)\n        \n        try:\n            # Zero crossing rate\n            zcr = librosa.feature.zero_crossing_rate(audio)[0]\n            temporal['zcr_mean'] = float(np.mean(zcr))\n            temporal['zcr_std'] = float(np.std(zcr))\n            \n            # Tempo and beat tracking\n            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)\n            temporal['tempo'] = float(tempo)\n            temporal['num_beats'] = len(beats)\n            temporal['beat_regularity'] = self._calculate_beat_regularity(beats)\n            \n            # Onset detection\n            onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sample_rate)\n            onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)\n            temporal['num_onsets'] = len(onset_times)\n            temporal['onset_rate'] = len(onset_times) / (len(audio) / self.sample_rate)\n            \n            if len(onset_times) > 1:\n                onset_intervals = np.diff(onset_times)\n                temporal['onset_interval_mean'] = float(np.mean(onset_intervals))\n                temporal['onset_interval_std'] = float(np.std(onset_intervals))\n            \n        except Exception as e:\n            logger.warning(f\"Temporal analysis error: {e}\")\n            temporal = self._basic_temporal_analysis(audio)\n            \n        return temporal\n    \n    def _analyze_perceptual_features(self, audio: np.ndarray) -> Dict[str, Any]:\n        \"\"\"Analyze perceptual audio characteristics.\"\"\"\n        perceptual = {}\n        \n        if not AUDIO_LIBS_AVAILABLE:\n            return {}\n        \n        try:\n            # MFCCs (perceptual features)\n            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)\n            for i in range(min(13, mfccs.shape[0])):\n                perceptual[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))\n                perceptual[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))\n            \n            # Chroma features (pitch class profiles)\n            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)\n            perceptual['chroma_mean'] = float(np.mean(chroma))\n            perceptual['chroma_std'] = float(np.std(chroma))\n            \n            # Dominant pitch classes\n            chroma_mean = np.mean(chroma, axis=1)\n            dominant_pitch_class = np.argmax(chroma_mean)\n            perceptual['dominant_pitch_class'] = int(dominant_pitch_class)\n            perceptual['pitch_class_strength'] = float(chroma_mean[dominant_pitch_class])\n            \n            # Tonnetz (harmonic network coordinates)\n            tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)\n            perceptual['tonnetz_mean'] = float(np.mean(tonnetz))\n            perceptual['tonnetz_std'] = float(np.std(tonnetz))\n            \n        except Exception as e:\n            logger.warning(f\"Perceptual analysis error: {e}\")\n            \n        return perceptual\n    \n    def _analyze_content_features(self, audio: np.ndarray) -> Dict[str, Any]:\n        \"\"\"Analyze content-based features.\"\"\"\n        content = {}\n        \n        try:\n            # Silence detection\n            silence_threshold = 0.001\n            silent_samples = np.sum(np.abs(audio) < silence_threshold)\n            content['silence_ratio'] = float(silent_samples / len(audio))\n            \n            # Voice activity detection (simple energy-based)\n            frame_length = int(0.025 * self.sample_rate)  # 25ms frames\n            hop_length = int(0.01 * self.sample_rate)     # 10ms hop\n            \n            frames = []\n            for i in range(0, len(audio) - frame_length, hop_length):\n                frame = audio[i:i + frame_length]\n                frames.append(frame)\n                \n            if frames:\n                frame_energies = [np.sum(frame ** 2) for frame in frames]\n                energy_threshold = np.percentile(frame_energies, 30)  # Adaptive threshold\n                active_frames = np.sum(np.array(frame_energies) > energy_threshold)\n                content['voice_activity_ratio'] = float(active_frames / len(frames))\n            \n            # Harmonic-percussive separation\n            if AUDIO_LIBS_AVAILABLE:\n                harmonic, percussive = librosa.effects.hpss(audio)\n                harmonic_energy = np.sum(harmonic ** 2)\n                percussive_energy = np.sum(percussive ** 2)\n                total_energy = harmonic_energy + percussive_energy\n                \n                if total_energy > 0:\n                    content['harmonic_ratio'] = float(harmonic_energy / total_energy)\n                    content['percussive_ratio'] = float(percussive_energy / total_energy)\n            \n        except Exception as e:\n            logger.warning(f\"Content analysis error: {e}\")\n            \n        return content\n    \n    def _analyze_frequency_bands(self, audio: np.ndarray) -> Dict[str, float]:\n        \"\"\"Analyze energy distribution across frequency bands.\"\"\"\n        band_energy = {}\n        \n        try:\n            # Compute power spectral density\n            fft = np.fft.fft(audio)\n            magnitude = np.abs(fft[:len(fft)//2])\n            freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)[:len(fft)//2]\n            power = magnitude ** 2\n            \n            total_power = np.sum(power)\n            \n            # Calculate energy in each frequency band\n            for band_name, (low_freq, high_freq) in self.frequency_bands.items():\n                band_mask = (freqs >= low_freq) & (freqs <= high_freq)\n                band_power = np.sum(power[band_mask])\n                band_energy[f'{band_name}_energy'] = float(band_power / (total_power + 1e-10))\n                \n        except Exception as e:\n            logger.warning(f\"Frequency band analysis error: {e}\")\n            \n        return band_energy\n    \n    def _calculate_beat_regularity(self, beats: np.ndarray) -> float:\n        \"\"\"Calculate beat regularity score.\"\"\"\n        if len(beats) < 3:\n            return 0.0\n            \n        intervals = np.diff(beats)\n        if len(intervals) == 0:\n            return 0.0\n            \n        # Coefficient of variation (lower = more regular)\n        cv = np.std(intervals) / (np.mean(intervals) + 1e-10)\n        regularity = 1.0 / (1.0 + cv)  # Convert to 0-1 scale\n        \n        return float(regularity)\n    \n    def _basic_spectral_analysis(self, audio: np.ndarray) -> Dict[str, Any]:\n        \"\"\"Basic spectral analysis using only NumPy.\"\"\"\n        spectral = {}\n        \n        try:\n            fft = np.fft.fft(audio)\n            magnitude = np.abs(fft[:len(fft)//2])\n            freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)[:len(fft)//2]\n            \n            # Spectral centroid\n            spectral['centroid_mean'] = float(np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10))\n            \n            # Spectral rolloff (95% of energy)\n            cumsum_magnitude = np.cumsum(magnitude)\n            rolloff_idx = np.where(cumsum_magnitude >= 0.95 * cumsum_magnitude[-1])[0]\n            if len(rolloff_idx) > 0:\n                spectral['rolloff_mean'] = float(freqs[rolloff_idx[0]])\n            \n            # Frequency band analysis\n            spectral['band_energy'] = self._analyze_frequency_bands(audio)\n            \n        except Exception as e:\n            logger.warning(f\"Basic spectral analysis error: {e}\")\n            \n        return spectral\n    \n    def _basic_temporal_analysis(self, audio: np.ndarray) -> Dict[str, Any]:\n        \"\"\"Basic temporal analysis using only NumPy.\"\"\"\n        temporal = {}\n        \n        try:\n            # Zero crossing rate\n            zero_crossings = np.sum(np.diff(np.sign(audio)) != 0) / 2\n            temporal['zcr_mean'] = float(zero_crossings / len(audio))\n            \n            # Simple energy-based onset detection\n            frame_length = int(0.025 * self.sample_rate)\n            hop_length = int(0.01 * self.sample_rate)\n            \n            energies = []\n            for i in range(0, len(audio) - frame_length, hop_length):\n                frame = audio[i:i + frame_length]\n                energy = np.sum(frame ** 2)\n                energies.append(energy)\n            \n            if energies:\n                energies = np.array(energies)\n                # Detect onsets as peaks in energy\n                diff_energy = np.diff(energies)\n                threshold = np.std(diff_energy) * 2\n                onsets = np.where(diff_energy > threshold)[0]\n                temporal['num_onsets'] = len(onsets)\n                temporal['onset_rate'] = len(onsets) / (len(audio) / self.sample_rate)\n                \n        except Exception as e:\n            logger.warning(f\"Basic temporal analysis error: {e}\")\n            \n        return temporal\n    \n    def compare_audio(self, audio1: np.ndarray, audio2: np.ndarray) -> Dict[str, Any]:\n        \"\"\"Compare two audio signals.\n        \n        Args:\n            audio1: First audio signal\n            audio2: Second audio signal\n            \n        Returns:\n            Comparison metrics\n        \"\"\"\n        comparison = {}\n        \n        try:\n            # Basic similarity metrics\n            min_len = min(len(audio1), len(audio2))\n            a1_trimmed = audio1[:min_len]\n            a2_trimmed = audio2[:min_len]\n            \n            # Cross-correlation\n            if scipy:\n                correlation = scipy.signal.correlate(a1_trimmed, a2_trimmed, mode='valid')\n                max_correlation = float(np.max(correlation) / (np.linalg.norm(a1_trimmed) * np.linalg.norm(a2_trimmed) + 1e-10))\n                comparison['max_correlation'] = max_correlation\n            \n            # Mean squared error\n            mse = float(np.mean((a1_trimmed - a2_trimmed) ** 2))\n            comparison['mse'] = mse\n            \n            # Signal-to-noise ratio\n            signal_power = np.mean(a1_trimmed ** 2)\n            noise_power = np.mean((a1_trimmed - a2_trimmed) ** 2)\n            if noise_power > 0:\n                snr_db = 10 * np.log10(signal_power / noise_power)\n                comparison['snr_db'] = float(snr_db)\n            \n            # Feature-based comparison\n            features1 = self.analyze_comprehensive(audio1)\n            features2 = self.analyze_comprehensive(audio2)\n            \n            # Compare spectral centroids\n            if 'spectral' in features1 and 'spectral' in features2:\n                centroid_diff = abs(features1['spectral'].get('centroid_mean', 0) - \n                                  features2['spectral'].get('centroid_mean', 0))\n                comparison['spectral_centroid_diff'] = float(centroid_diff)\n            \n            # Compare RMS levels\n            rms_diff = abs(features1['basic']['rms'] - features2['basic']['rms'])\n            comparison['rms_diff'] = float(rms_diff)\n            \n        except Exception as e:\n            logger.error(f\"Audio comparison error: {e}\")\n            \n        return comparison