"""Advanced audio analysis and feature extraction capabilities."""

import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import librosa
    import scipy.signal
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    librosa = None
    scipy = None
    AUDIO_LIBS_AVAILABLE = False
    warnings.warn("Advanced audio analysis libraries not available. Using basic analysis only.")

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """Advanced audio analysis and feature extraction."""
    
    def __init__(self, sample_rate: int = 48000):
        """Initialize audio analyzer.
        
        Args:
            sample_rate: Sample rate for analysis
        """
        self.sample_rate = sample_rate
        self.frequency_bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_midrange': (250, 500),
            'midrange': (500, 2000),
            'upper_midrange': (2000, 4000),
            'presence': (4000, 8000),
            'brilliance': (8000, 20000)
        }
        
        logger.info(f"AudioAnalyzer initialized: {sample_rate}Hz")
    
    def analyze_comprehensive(self, audio: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive audio analysis.
        
        Args:
            audio: Audio signal to analyze
            
        Returns:
            Comprehensive analysis results
        """
        analysis = {
            'basic': self._analyze_basic_features(audio),
            'spectral': self._analyze_spectral_features(audio),
            'temporal': self._analyze_temporal_features(audio),
            'perceptual': self._analyze_perceptual_features(audio),
            'content': self._analyze_content_features(audio)
        }
        
        # Add metadata
        analysis['metadata'] = {
            'sample_rate': self.sample_rate,
            'duration_seconds': len(audio) / self.sample_rate,
            'num_samples': len(audio),
            'analysis_libs_available': AUDIO_LIBS_AVAILABLE
        }
        
        logger.debug(f"Comprehensive analysis completed for {len(audio)} samples")
        return analysis
    
    def _analyze_basic_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze basic audio features."""
        basic = {}
        
        try:
            # Statistical measures
            basic['rms'] = float(np.sqrt(np.mean(audio ** 2)))
            basic['peak'] = float(np.max(np.abs(audio)))
            basic['mean'] = float(np.mean(audio))
            basic['std'] = float(np.std(audio))
            basic['skewness'] = float(self._calculate_skewness(audio))
            basic['kurtosis'] = float(self._calculate_kurtosis(audio))
            
            # Dynamic range
            if basic['rms'] > 0:
                basic['crest_factor_db'] = 20 * np.log10(basic['peak'] / basic['rms'])
            else:
                basic['crest_factor_db'] = 0.0
            
            # Zero crossings
            zero_crossings = np.sum(np.diff(np.signbit(audio))) / 2
            basic['zero_crossing_rate'] = float(zero_crossings / len(audio))
            
            # Energy measures
            basic['energy'] = float(np.sum(audio ** 2))
            basic['power'] = basic['energy'] / len(audio)
            
        except Exception as e:
            logger.warning(f"Basic feature analysis error: {e}")
            
        return basic
    
    def _analyze_spectral_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze spectral characteristics."""
        spectral = {}
        
        if not AUDIO_LIBS_AVAILABLE:
            return self._basic_spectral_analysis(audio)
        
        try:
            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral['centroid_mean'] = float(np.mean(centroid))
            spectral['centroid_std'] = float(np.std(centroid))
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            spectral['rolloff_mean'] = float(np.mean(rolloff))
            spectral['rolloff_std'] = float(np.std(rolloff))
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            spectral['bandwidth_mean'] = float(np.mean(bandwidth))
            spectral['bandwidth_std'] = float(np.std(bandwidth))
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            spectral['contrast_mean'] = float(np.mean(contrast))
            spectral['contrast_std'] = float(np.std(contrast))
            
            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(y=audio)[0]
            spectral['flatness_mean'] = float(np.mean(flatness))
            spectral['flatness_std'] = float(np.std(flatness))
            
            # Frequency band energy distribution
            spectral['band_energy'] = self._analyze_frequency_bands(audio)
            
        except Exception as e:
            logger.warning(f"Spectral analysis error: {e}")
            spectral = self._basic_spectral_analysis(audio)
            
        return spectral
    
    def _analyze_temporal_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal characteristics."""
        temporal = {}
        
        if not AUDIO_LIBS_AVAILABLE:
            return self._basic_temporal_analysis(audio)
        
        try:
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            temporal['zcr_mean'] = float(np.mean(zcr))
            temporal['zcr_std'] = float(np.std(zcr))
            
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            temporal['tempo'] = float(tempo)
            temporal['num_beats'] = len(beats)
            temporal['beat_regularity'] = self._calculate_beat_regularity(beats)
            
            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sample_rate)
            onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)
            temporal['num_onsets'] = len(onset_times)
            temporal['onset_rate'] = len(onset_times) / (len(audio) / self.sample_rate)
            
            if len(onset_times) > 1:
                onset_intervals = np.diff(onset_times)
                temporal['onset_interval_mean'] = float(np.mean(onset_intervals))
                temporal['onset_interval_std'] = float(np.std(onset_intervals))
            
        except Exception as e:
            logger.warning(f"Temporal analysis error: {e}")
            temporal = self._basic_temporal_analysis(audio)
            
        return temporal
    
    def _analyze_perceptual_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze perceptual audio characteristics."""
        perceptual = {}
        
        if not AUDIO_LIBS_AVAILABLE:
            return {}
        
        try:
            # MFCCs (perceptual features)
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            for i in range(min(13, mfccs.shape[0])):
                perceptual[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
                perceptual[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
            
            # Chroma features (pitch class profiles)
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            perceptual['chroma_mean'] = float(np.mean(chroma))
            perceptual['chroma_std'] = float(np.std(chroma))
            
            # Dominant pitch classes
            chroma_mean = np.mean(chroma, axis=1)
            dominant_pitch_class = np.argmax(chroma_mean)
            perceptual['dominant_pitch_class'] = int(dominant_pitch_class)
            perceptual['pitch_class_strength'] = float(chroma_mean[dominant_pitch_class])
            
            # Tonnetz (harmonic network coordinates)
            tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)
            perceptual['tonnetz_mean'] = float(np.mean(tonnetz))
            perceptual['tonnetz_std'] = float(np.std(tonnetz))
            
        except Exception as e:
            logger.warning(f"Perceptual analysis error: {e}")
            
        return perceptual
    
    def _analyze_content_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze content-based features."""
        content = {}
        
        try:
            # Silence detection
            silence_threshold = 0.001
            silent_samples = np.sum(np.abs(audio) < silence_threshold)
            content['silence_ratio'] = float(silent_samples / len(audio))
            
            # Voice activity detection (simple energy-based)
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.01 * self.sample_rate)     # 10ms hop
            
            frames = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                frames.append(frame)
                
            if frames:
                frame_energies = [np.sum(frame ** 2) for frame in frames]
                energy_threshold = np.percentile(frame_energies, 30)  # Adaptive threshold
                active_frames = np.sum(np.array(frame_energies) > energy_threshold)
                content['voice_activity_ratio'] = float(active_frames / len(frames))
            
            # Harmonic-percussive separation
            if AUDIO_LIBS_AVAILABLE:
                harmonic, percussive = librosa.effects.hpss(audio)
                harmonic_energy = np.sum(harmonic ** 2)
                percussive_energy = np.sum(percussive ** 2)
                total_energy = harmonic_energy + percussive_energy
                
                if total_energy > 0:
                    content['harmonic_ratio'] = float(harmonic_energy / total_energy)
                    content['percussive_ratio'] = float(percussive_energy / total_energy)
            
        except Exception as e:
            logger.warning(f"Content analysis error: {e}")
            
        return content
    
    def _analyze_frequency_bands(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze energy distribution across frequency bands."""
        band_energy = {}
        
        try:
            # Compute power spectral density
            fft = np.fft.fft(audio)
            magnitude = np.abs(fft[:len(fft)//2])
            freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)[:len(fft)//2]
            power = magnitude ** 2
            
            total_power = np.sum(power)
            
            # Calculate energy in each frequency band
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.sum(power[band_mask])
                band_energy[f'{band_name}_energy'] = float(band_power / (total_power + 1e-10))
                
        except Exception as e:
            logger.warning(f"Frequency band analysis error: {e}")
            
        return band_energy
    
    def _calculate_skewness(self, audio: np.ndarray) -> float:
        """Calculate skewness of audio signal."""
        mean = np.mean(audio)
        std = np.std(audio)
        if std == 0:
            return 0.0
        return np.mean(((audio - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, audio: np.ndarray) -> float:
        """Calculate kurtosis of audio signal."""
        mean = np.mean(audio)
        std = np.std(audio)
        if std == 0:
            return 0.0
        return np.mean(((audio - mean) / std) ** 4) - 3.0
    
    def _calculate_beat_regularity(self, beats: np.ndarray) -> float:
        """Calculate beat regularity score."""
        if len(beats) < 3:
            return 0.0
            
        intervals = np.diff(beats)
        if len(intervals) == 0:
            return 0.0
            
        # Coefficient of variation (lower = more regular)
        cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
        regularity = 1.0 / (1.0 + cv)  # Convert to 0-1 scale
        
        return float(regularity)
    
    def _basic_spectral_analysis(self, audio: np.ndarray) -> Dict[str, Any]:
        """Basic spectral analysis using only NumPy."""
        spectral = {}
        
        try:
            fft = np.fft.fft(audio)
            magnitude = np.abs(fft[:len(fft)//2])
            freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)[:len(fft)//2]
            
            # Spectral centroid
            spectral['centroid_mean'] = float(np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10))
            
            # Spectral rolloff (95% of energy)
            cumsum_magnitude = np.cumsum(magnitude)
            rolloff_idx = np.where(cumsum_magnitude >= 0.95 * cumsum_magnitude[-1])[0]
            if len(rolloff_idx) > 0:
                spectral['rolloff_mean'] = float(freqs[rolloff_idx[0]])
            
            # Frequency band analysis
            spectral['band_energy'] = self._analyze_frequency_bands(audio)
            
        except Exception as e:
            logger.warning(f"Basic spectral analysis error: {e}")
            
        return spectral
    
    def _basic_temporal_analysis(self, audio: np.ndarray) -> Dict[str, Any]:
        """Basic temporal analysis using only NumPy."""
        temporal = {}
        
        try:
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(audio)) != 0) / 2
            temporal['zcr_mean'] = float(zero_crossings / len(audio))
            
            # Simple energy-based onset detection
            frame_length = int(0.025 * self.sample_rate)
            hop_length = int(0.01 * self.sample_rate)
            
            energies = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                energy = np.sum(frame ** 2)
                energies.append(energy)
            
            if energies:
                energies = np.array(energies)
                # Detect onsets as peaks in energy
                diff_energy = np.diff(energies)
                threshold = np.std(diff_energy) * 2
                onsets = np.where(diff_energy > threshold)[0]
                temporal['num_onsets'] = len(onsets)
                temporal['onset_rate'] = len(onsets) / (len(audio) / self.sample_rate)
                
        except Exception as e:
            logger.warning(f"Basic temporal analysis error: {e}")
            
        return temporal
    
    def compare_audio(self, audio1: np.ndarray, audio2: np.ndarray) -> Dict[str, Any]:
        """Compare two audio signals.
        
        Args:
            audio1: First audio signal
            audio2: Second audio signal
            
        Returns:
            Comparison metrics
        """
        comparison = {}
        
        try:
            # Basic similarity metrics
            min_len = min(len(audio1), len(audio2))
            a1_trimmed = audio1[:min_len]
            a2_trimmed = audio2[:min_len]
            
            # Cross-correlation
            if scipy:
                correlation = scipy.signal.correlate(a1_trimmed, a2_trimmed, mode='valid')
                max_correlation = float(np.max(correlation) / (np.linalg.norm(a1_trimmed) * np.linalg.norm(a2_trimmed) + 1e-10))
                comparison['max_correlation'] = max_correlation
            
            # Mean squared error
            mse = float(np.mean((a1_trimmed - a2_trimmed) ** 2))
            comparison['mse'] = mse
            
            # Signal-to-noise ratio
            signal_power = np.mean(a1_trimmed ** 2)
            noise_power = np.mean((a1_trimmed - a2_trimmed) ** 2)
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
                comparison['snr_db'] = float(snr_db)
            
            # Feature-based comparison
            features1 = self.analyze_comprehensive(audio1)
            features2 = self.analyze_comprehensive(audio2)
            
            # Compare spectral centroids
            if 'spectral' in features1 and 'spectral' in features2:
                centroid_diff = abs(features1['spectral'].get('centroid_mean', 0) - 
                                  features2['spectral'].get('centroid_mean', 0))
                comparison['spectral_centroid_diff'] = float(centroid_diff)
            
            # Compare RMS levels
            rms_diff = abs(features1['basic']['rms'] - features2['basic']['rms'])
            comparison['rms_diff'] = float(rms_diff)
            
        except Exception as e:
            logger.error(f"Audio comparison error: {e}")
            
        return comparison
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            'analyzer': 'AudioAnalyzer',
            'sample_rate': self.sample_rate,
            'frequency_bands': self.frequency_bands,
            'audio_libs_available': AUDIO_LIBS_AVAILABLE,
            'supported_features': [
                'basic', 'spectral', 'temporal', 'perceptual', 'content'
            ]
        }