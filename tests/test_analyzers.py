"""Tests for audio analysis modules."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fugatto_lab.analyzers import AudioAnalyzer
from fugatto_lab.analyzers.audio_analyzer import AudioAnalyzer as AudioAnalyzerClass


class TestAudioAnalyzer:
    """Test cases for AudioAnalyzer."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = AudioAnalyzer(sample_rate=44100)
        assert analyzer.sample_rate == 44100
        assert hasattr(analyzer, 'frequency_bands')
        assert len(analyzer.frequency_bands) > 0
    
    def test_basic_audio_analysis(self):
        """Test basic audio analysis functionality."""
        analyzer = AudioAnalyzer()
        
        # Generate test audio (1 second of sine wave)
        sample_rate = 48000
        t = np.linspace(0, 1, sample_rate)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Perform analysis
        results = analyzer.analyze_comprehensive(audio)
        
        # Check structure
        assert 'basic' in results
        assert 'spectral' in results
        assert 'temporal' in results
        assert 'perceptual' in results
        assert 'content' in results
        
        # Check basic statistics
        basic = results['basic']
        assert basic['duration'] == pytest.approx(1.0, rel=1e-2)
        assert basic['samples'] == len(audio)
        assert basic['rms'] > 0
        assert basic['peak'] > 0
    
    def test_spectral_analysis(self):
        """Test spectral feature extraction."""
        analyzer = AudioAnalyzer()
        
        # Generate complex test signal
        sample_rate = 48000
        t = np.linspace(0, 2, sample_rate * 2)
        audio = (np.sin(2 * np.pi * 440 * t) + 
                0.5 * np.sin(2 * np.pi * 880 * t) +
                0.2 * np.random.normal(0, 1, len(t))).astype(np.float32)
        
        spectral = analyzer._analyze_spectral_features(audio)
        
        # Check for expected features
        expected_features = ['centroid_mean', 'rolloff_mean', 'bandwidth_mean']
        for feature in expected_features:
            assert feature in spectral
            assert isinstance(spectral[feature], (int, float))
            assert spectral[feature] > 0
    
    def test_temporal_analysis(self):
        """Test temporal feature extraction."""
        analyzer = AudioAnalyzer()
        
        # Generate rhythmic audio
        sample_rate = 48000
        duration = 4
        t = np.linspace(0, duration, sample_rate * duration)
        
        # Create beats at 120 BPM
        beat_freq = 2.0  # 120 BPM = 2 Hz
        envelope = np.abs(np.sin(2 * np.pi * beat_freq * t))
        audio = (envelope * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        temporal = analyzer._analyze_temporal_features(audio)
        
        # Check temporal features
        assert 'zcr_mean' in temporal
        assert 'tempo' in temporal
        assert 'num_onsets' in temporal
        
        # Tempo should be around 120 BPM
        if 'tempo' in temporal and temporal['tempo'] > 0:
            assert 100 < temporal['tempo'] < 140  # Allow some tolerance
    
    def test_perceptual_analysis(self):
        """Test perceptual feature extraction."""
        analyzer = AudioAnalyzer()
        
        # Generate musical audio
        sample_rate = 48000
        t = np.linspace(0, 3, sample_rate * 3)
        
        # C major chord (C, E, G)
        audio = (np.sin(2 * np.pi * 261.63 * t) +  # C4
                np.sin(2 * np.pi * 329.63 * t) +   # E4
                np.sin(2 * np.pi * 392.00 * t)).astype(np.float32)  # G4
        
        perceptual = analyzer._analyze_perceptual_features(audio)
        
        # Check MFCC features
        mfcc_features = [k for k in perceptual.keys() if k.startswith('mfcc_')]
        assert len(mfcc_features) > 0
        
        # Check chroma features
        assert 'chroma_mean' in perceptual
        assert 'dominant_pitch_class' in perceptual
    
    def test_content_analysis(self):
        """Test content-based feature extraction."""
        analyzer = AudioAnalyzer()
        
        # Generate mixed content audio
        sample_rate = 48000
        t = np.linspace(0, 2, sample_rate * 2)
        
        # Mix harmonic and percussive elements
        harmonic = np.sin(2 * np.pi * 440 * t)
        percussive = np.random.normal(0, 0.1, len(t)) * np.exp(-t * 2)
        silence = np.zeros(sample_rate // 2)  # 0.5 seconds of silence
        
        audio = np.concatenate([harmonic[:sample_rate//2], 
                               percussive[:sample_rate//2],
                               silence,
                               harmonic[sample_rate//2:]]).astype(np.float32)
        
        content = analyzer._analyze_content_features(audio)
        
        # Check content features
        assert 'silence_ratio' in content
        assert 'voice_activity_ratio' in content
        assert 0 <= content['silence_ratio'] <= 1
        assert 0 <= content['voice_activity_ratio'] <= 1
        
        # Should detect some silence
        assert content['silence_ratio'] > 0.1
    
    def test_frequency_band_analysis(self):
        """Test frequency band energy analysis."""
        analyzer = AudioAnalyzer()
        
        # Generate multi-band audio
        sample_rate = 48000
        t = np.linspace(0, 1, sample_rate)
        
        # Different frequency components
        bass = np.sin(2 * np.pi * 100 * t)      # Bass
        mid = np.sin(2 * np.pi * 1000 * t)      # Mid
        high = np.sin(2 * np.pi * 5000 * t)     # High
        
        audio = (bass + mid + high).astype(np.float32)
        
        band_energy = analyzer._analyze_frequency_bands(audio)
        
        # Check that all bands are present
        expected_bands = ['bass_energy', 'mid_energy', 'presence_energy']
        for band in expected_bands:
            assert band in band_energy
            assert isinstance(band_energy[band], (int, float))
            assert 0 <= band_energy[band] <= 1
        
        # Energy distribution should sum to approximately 1
        total_energy = sum(band_energy.values())
        assert 0.8 <= total_energy <= 1.2  # Allow some tolerance
    
    def test_audio_comparison(self):
        """Test audio comparison functionality."""
        analyzer = AudioAnalyzer()
        
        # Generate similar and different audio signals
        sample_rate = 48000
        t = np.linspace(0, 1, sample_rate)
        
        audio1 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        audio2 = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.9  # Similar
        audio3 = np.sin(2 * np.pi * 880 * t).astype(np.float32)  # Different frequency
        
        # Compare similar audio
        similar_comparison = analyzer.compare_audio(audio1, audio2)
        assert 'mse' in similar_comparison
        assert 'overall_similarity' in similar_comparison
        
        # Compare different audio
        different_comparison = analyzer.compare_audio(audio1, audio3)
        
        # Similar audio should have lower MSE
        assert similar_comparison['mse'] < different_comparison['mse']
    
    def test_fallback_analysis_without_libraries(self):
        """Test analysis fallback when audio libraries are not available."""
        with patch('fugatto_lab.analyzers.audio_analyzer.AUDIO_LIBS_AVAILABLE', False):
            analyzer = AudioAnalyzer()
            
            # Generate test audio
            sample_rate = 48000
            t = np.linspace(0, 1, sample_rate)
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            # Should still return basic analysis
            results = analyzer.analyze_comprehensive(audio)
            
            assert 'basic' in results
            assert 'spectral' in results  # Should use fallback
            assert 'temporal' in results  # Should use fallback
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        analyzer = AudioAnalyzer()
        
        # Empty audio
        empty_audio = np.array([], dtype=np.float32)
        results = analyzer.analyze_comprehensive(empty_audio)
        assert 'basic' in results
        
        # Very short audio
        short_audio = np.array([0.1, -0.1, 0.05], dtype=np.float32)
        results = analyzer.analyze_comprehensive(short_audio)
        assert results['basic']['samples'] == 3
        
        # Silence
        silence = np.zeros(48000, dtype=np.float32)
        results = analyzer.analyze_comprehensive(silence)
        assert results['basic']['rms'] == 0
        assert results['basic']['peak'] == 0
        
        # Very loud audio (clipping)
        loud_audio = np.ones(48000, dtype=np.float32) * 2.0
        results = analyzer.analyze_comprehensive(loud_audio)
        assert results['basic']['peak'] >= 1.0
    
    @pytest.mark.slow
    def test_performance_with_long_audio(self):
        """Test performance with longer audio files."""
        analyzer = AudioAnalyzer()
        
        # Generate 30 seconds of audio
        sample_rate = 48000
        duration = 30
        t = np.linspace(0, duration, sample_rate * duration)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        import time
        start_time = time.time()
        results = analyzer.analyze_comprehensive(audio)
        analysis_time = time.time() - start_time
        
        # Should complete within reasonable time (less than 10 seconds)
        assert analysis_time < 10.0
        assert results['basic']['duration'] == pytest.approx(duration, rel=1e-2)
    
    def test_beat_regularity_calculation(self):
        """Test beat regularity calculation."""
        analyzer = AudioAnalyzer()
        
        # Regular beats
        regular_beats = np.array([0, 1, 2, 3, 4])
        regularity = analyzer._calculate_beat_regularity(regular_beats)
        assert regularity > 0.8  # Should be highly regular
        
        # Irregular beats
        irregular_beats = np.array([0, 0.8, 2.3, 2.9, 4.1])
        irregularity = analyzer._calculate_beat_regularity(irregular_beats)
        assert irregularity < regularity  # Should be less regular
        
        # Too few beats
        few_beats = np.array([0, 1])
        result = analyzer._calculate_beat_regularity(few_beats)
        assert result == 0.0
    
    def test_analyzer_with_different_sample_rates(self):
        """Test analyzer with different sample rates."""
        # Test with common sample rates
        sample_rates = [22050, 44100, 48000, 96000]
        
        for sr in sample_rates:
            analyzer = AudioAnalyzer(sample_rate=sr)
            assert analyzer.sample_rate == sr
            
            # Generate appropriate test audio
            t = np.linspace(0, 1, sr)
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            results = analyzer.analyze_comprehensive(audio)
            assert results['basic']['duration'] == pytest.approx(1.0, rel=1e-2)


@pytest.mark.integration
class TestAudioAnalyzerIntegration:
    """Integration tests for AudioAnalyzer."""
    
    def test_analyzer_with_real_audio_features(self):
        """Test analyzer with more realistic audio features."""
        analyzer = AudioAnalyzer()
        
        # Generate more realistic audio signal
        sample_rate = 48000
        duration = 5
        t = np.linspace(0, duration, sample_rate * duration)
        
        # Music-like signal with multiple components
        fundamental = 440  # A4
        harmonics = [1, 2, 3, 4, 5]
        amplitudes = [1.0, 0.5, 0.3, 0.2, 0.1]
        
        audio = np.zeros_like(t)
        for harmonic, amplitude in zip(harmonics, amplitudes):
            audio += amplitude * np.sin(2 * np.pi * fundamental * harmonic * t)
        
        # Add some rhythmic variation
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz modulation
        audio *= envelope
        
        # Add realistic noise
        audio += 0.05 * np.random.normal(0, 1, len(audio))
        audio = audio.astype(np.float32)
        
        # Perform comprehensive analysis
        results = analyzer.analyze_comprehensive(audio)
        
        # Validate realistic results
        assert results['basic']['duration'] == pytest.approx(duration, rel=1e-2)
        assert 0.1 < results['basic']['rms'] < 1.0
        assert 0.2 < results['basic']['peak'] <= 1.0
        
        # Should detect harmonic content
        if 'harmonic_ratio' in results['content']:
            assert results['content']['harmonic_ratio'] > 0.5
        
        # Should have reasonable spectral characteristics
        if 'centroid_mean' in results['spectral']:
            assert 200 < results['spectral']['centroid_mean'] < 8000