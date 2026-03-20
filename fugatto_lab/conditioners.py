"""
Text and Audio Conditioners for generative audio synthesis.
Parses prompt features and analyzes input audio for conditioning parameters.
"""
from __future__ import annotations
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


INSTRUMENT_KEYWORDS = {
    "piano": {"timbre": 0.3, "harmonics": 8, "attack": 0.02},
    "guitar": {"timbre": 0.5, "harmonics": 6, "attack": 0.01},
    "violin": {"timbre": 0.7, "harmonics": 12, "attack": 0.05},
    "drums": {"timbre": 0.1, "harmonics": 2, "attack": 0.001},
    "flute": {"timbre": 0.8, "harmonics": 4, "attack": 0.03},
    "bass": {"timbre": 0.2, "harmonics": 5, "attack": 0.02},
    "synth": {"timbre": 0.6, "harmonics": 10, "attack": 0.01},
    "trumpet": {"timbre": 0.6, "harmonics": 9, "attack": 0.02},
}

MOOD_KEYWORDS = {
    "happy": {"pitch_shift": 2, "tempo_scale": 1.1},
    "sad": {"pitch_shift": -2, "tempo_scale": 0.85},
    "energetic": {"pitch_shift": 0, "tempo_scale": 1.3},
    "calm": {"pitch_shift": -1, "tempo_scale": 0.75},
    "dark": {"pitch_shift": -5, "tempo_scale": 0.9},
    "bright": {"pitch_shift": 3, "tempo_scale": 1.05},
}


@dataclass
class AudioCondition:
    """Conditioning parameters derived from text or audio analysis."""
    pitch_hz: float = 440.0          # base pitch in Hz
    tempo_bpm: float = 120.0         # tempo
    duration_s: float = 2.0          # output duration in seconds
    instrument: str = "synth"        # instrument preset
    timbre: float = 0.5              # 0=pure sine, 1=rich harmonics
    harmonics: int = 6               # number of harmonics
    attack: float = 0.02             # attack time in seconds
    release: float = 0.1             # release time in seconds
    reverb: float = 0.2              # reverb mix [0,1]
    noise_floor: float = 0.001       # background noise level
    pitch_shift_semitones: float = 0.0
    tempo_scale: float = 1.0
    extra: dict = field(default_factory=dict)


class TextConditioner:
    """
    Parses text prompts to extract audio conditioning parameters.
    Supports pitch, tempo, instrument, mood keywords.
    """
    def __init__(self):
        self._tempo_re = re.compile(r"(\d+)\s*bpm", re.IGNORECASE)
        self._pitch_re = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.IGNORECASE)
        self._duration_re = re.compile(r"(\d+(?:\.\d+)?)\s*(?:seconds?|s\b)", re.IGNORECASE)

    def condition(self, prompt: str) -> AudioCondition:
        """Parse a text prompt and return AudioCondition."""
        cond = AudioCondition()
        prompt_lower = prompt.lower()

        # Extract tempo
        m = self._tempo_re.search(prompt)
        if m:
            cond.tempo_bpm = float(m.group(1))

        # Extract pitch
        m = self._pitch_re.search(prompt)
        if m:
            cond.pitch_hz = float(m.group(1))

        # Extract duration
        m = self._duration_re.search(prompt)
        if m:
            cond.duration_s = min(float(m.group(1)), 30.0)

        # Instrument detection
        for inst, params in INSTRUMENT_KEYWORDS.items():
            if inst in prompt_lower:
                cond.instrument = inst
                cond.timbre = params["timbre"]
                cond.harmonics = params["harmonics"]
                cond.attack = params["attack"]
                break

        # Mood detection
        for mood, params in MOOD_KEYWORDS.items():
            if mood in prompt_lower:
                cond.pitch_shift_semitones += params["pitch_shift"]
                cond.tempo_scale *= params["tempo_scale"]
                break

        # Reverb keywords
        if any(w in prompt_lower for w in ["reverb", "hall", "room", "echo"]):
            cond.reverb = 0.6

        # Apply pitch shift
        cond.pitch_hz *= 2 ** (cond.pitch_shift_semitones / 12.0)
        cond.tempo_bpm *= cond.tempo_scale

        return cond

    def batch_condition(self, prompts: list[str]) -> list[AudioCondition]:
        return [self.condition(p) for p in prompts]


class AudioConditioner:
    """
    Analyzes input audio to extract style transfer parameters.
    Returns an AudioCondition based on the audio's spectral and temporal properties.
    """
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def condition(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> AudioCondition:
        """Analyze audio and return conditioning parameters."""
        sr = sample_rate or self.sample_rate
        cond = AudioCondition()
        cond.duration_s = len(audio) / sr

        if len(audio) == 0:
            return cond

        # Estimate fundamental pitch via zero-crossing rate → approximate F0
        zero_crossings = np.where(np.diff(np.sign(audio)))[0]
        if len(zero_crossings) > 1:
            avg_period_samples = 2 * len(audio) / max(len(zero_crossings), 1)
            estimated_f0 = sr / max(avg_period_samples, 1)
            cond.pitch_hz = float(np.clip(estimated_f0, 20, 4000))

        # Estimate RMS energy → noise floor
        rms = float(np.sqrt(np.mean(audio ** 2)))
        cond.noise_floor = max(rms * 0.01, 1e-6)

        # Spectral centroid → timbre proxy
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
        if fft.sum() > 0:
            centroid = float(np.sum(freqs * fft) / fft.sum())
            # Normalize to [0, 1] (0=bass, 1=treble)
            cond.timbre = float(np.clip(centroid / (sr / 2), 0, 1))

        # Estimate tempo via autocorrelation of energy envelope
        hop = sr // 100
        energy = np.array([np.mean(audio[i:i+hop]**2) for i in range(0, len(audio)-hop, hop)])
        if len(energy) > 10:
            corr = np.correlate(energy, energy, mode='full')[len(energy)-1:]
            # Look for first peak beyond minimum lag (0.25s)
            min_lag = int(0.25 * 100)
            max_lag = int(2.0 * 100)
            if max_lag < len(corr):
                peak = np.argmax(corr[min_lag:max_lag]) + min_lag
                beat_period_s = peak / 100.0
                if beat_period_s > 0:
                    cond.tempo_bpm = float(np.clip(60.0 / beat_period_s, 40, 240))

        return cond
