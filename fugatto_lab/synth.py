"""
Audio synthesis engine using scipy.signal.
Converts AudioCondition parameters into waveforms.
"""
from __future__ import annotations
import numpy as np
import wave
import io
from pathlib import Path
from typing import Optional, Union
from scipy import signal as sp_signal

from fugatto_lab.conditioners import AudioCondition


class AudioSynthEngine:
    """
    scipy.signal-based synthesis engine.
    Supports additive synthesis, envelopes, reverb simulation.
    """
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def synthesize(self, condition: AudioCondition) -> np.ndarray:
        """Synthesize audio from an AudioCondition. Returns float32 array [-1, 1]."""
        sr = self.sample_rate
        duration = max(condition.duration_s, 0.01)
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        waveform = np.zeros(n_samples, dtype=np.float64)

        # Additive synthesis: fundamental + harmonics
        f0 = max(condition.pitch_hz, 20.0)
        for h in range(1, condition.harmonics + 1):
            freq = f0 * h
            if freq > sr / 2:
                break
            # Harmonic amplitude decays with timbre mix
            amp = (1.0 / h) * (1.0 - condition.timbre) + (1.0 / (h ** 0.5)) * condition.timbre
            waveform += amp * np.sin(2 * np.pi * freq * t)

        # Normalize before envelope
        peak = np.abs(waveform).max()
        if peak > 0:
            waveform /= peak

        # Apply ADSR-like envelope
        waveform *= self._make_envelope(n_samples, condition.attack, condition.release, sr)

        # Noise floor
        waveform += np.random.randn(n_samples) * condition.noise_floor

        # Reverb: simple FIR comb filter
        if condition.reverb > 0:
            waveform = self._apply_reverb(waveform, sr, mix=condition.reverb)

        # Final normalization
        peak = np.abs(waveform).max()
        if peak > 0:
            waveform = (waveform / peak * 0.9).astype(np.float32)
        else:
            waveform = waveform.astype(np.float32)

        return waveform

    def _make_envelope(self, n_samples: int, attack: float, release: float, sr: int) -> np.ndarray:
        env = np.ones(n_samples)
        attack_samples = min(int(attack * sr), n_samples // 4)
        release_samples = min(int(release * sr), n_samples // 4)
        if attack_samples > 0:
            env[:attack_samples] = np.linspace(0, 1, attack_samples)
        if release_samples > 0:
            env[-release_samples:] = np.linspace(1, 0, release_samples)
        return env

    def _apply_reverb(self, audio: np.ndarray, sr: int, mix: float = 0.3) -> np.ndarray:
        """Simple reverb via delayed copies (Schroeder-style comb filter)."""
        delays_ms = [30, 45, 60, 80]
        result = audio.copy()
        for d_ms in delays_ms:
            delay_samples = int(sr * d_ms / 1000)
            if delay_samples >= len(audio):
                continue
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples] * (mix / len(delays_ms))
            result = result + delayed
        return result


class AudioSynthPipeline:
    """
    End-to-end pipeline: text prompt OR audio → AudioCondition → waveform → WAV.
    """
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        from fugatto_lab.conditioners import TextConditioner, AudioConditioner as AudioCond
        self.text_conditioner = TextConditioner()
        self.audio_conditioner = AudioCond(sample_rate)
        self.engine = AudioSynthEngine(sample_rate)

    def from_text(self, prompt: str) -> np.ndarray:
        """Generate audio from text prompt."""
        condition = self.text_conditioner.condition(prompt)
        return self.engine.synthesize(condition)

    def from_audio(self, audio: np.ndarray, prompt_override: Optional[str] = None) -> np.ndarray:
        """Generate audio conditioned on input audio (style transfer)."""
        condition = self.audio_conditioner.condition(audio, self.sample_rate)
        if prompt_override:
            text_cond = self.text_conditioner.condition(prompt_override)
            # Blend conditions
            condition.timbre = (condition.timbre + text_cond.timbre) / 2
            condition.tempo_bpm = text_cond.tempo_bpm
        return self.engine.synthesize(condition)

    def save_wav(self, audio: np.ndarray, path: Union[str, Path]) -> Path:
        """Save audio array to WAV file (16-bit PCM)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = (audio * 32767).astype(np.int16)
        with wave.open(str(path), 'w') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self.sample_rate)
            f.writeframes(samples.tobytes())
        return path

    def to_bytes(self, audio: np.ndarray) -> bytes:
        """Encode audio to WAV bytes."""
        buf = io.BytesIO()
        samples = (audio * 32767).astype(np.int16)
        with wave.open(buf, 'w') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self.sample_rate)
            f.writeframes(samples.tobytes())
        return buf.getvalue()
