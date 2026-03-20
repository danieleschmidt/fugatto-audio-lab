import numpy as np
import pytest
from fugatto_lab.conditioners import TextConditioner, AudioConditioner, AudioCondition

def test_text_conditioner_default():
    tc = TextConditioner()
    cond = tc.condition("play a sound")
    assert isinstance(cond, AudioCondition)
    assert cond.tempo_bpm > 0

def test_text_conditioner_tempo():
    tc = TextConditioner()
    cond = tc.condition("play at 140 bpm")
    assert cond.tempo_bpm == pytest.approx(140.0, rel=0.05)

def test_text_conditioner_pitch():
    tc = TextConditioner()
    cond = tc.condition("a tone at 880 hz")
    assert cond.pitch_hz == pytest.approx(880.0, rel=0.01)

def test_text_conditioner_duration():
    tc = TextConditioner()
    cond = tc.condition("5 seconds of music")
    assert cond.duration_s == pytest.approx(5.0, rel=0.01)

def test_text_conditioner_instrument_piano():
    tc = TextConditioner()
    cond = tc.condition("play a piano melody")
    assert cond.instrument == "piano"

def test_text_conditioner_instrument_violin():
    tc = TextConditioner()
    cond = tc.condition("violin strings")
    assert cond.instrument == "violin"

def test_text_conditioner_mood_happy():
    tc = TextConditioner()
    cond = tc.condition("happy upbeat music")
    # Happy shifts pitch up +2 semitones and tempo up
    assert cond.tempo_bpm > 120.0

def test_text_conditioner_reverb():
    tc = TextConditioner()
    cond = tc.condition("with reverb in a hall")
    assert cond.reverb > 0.5

def test_text_conditioner_batch():
    tc = TextConditioner()
    results = tc.batch_condition(["piano at 120 bpm", "violin 2 seconds"])
    assert len(results) == 2

def test_audio_conditioner_basic():
    sr = 22050
    # Simple sine wave
    t = np.linspace(0, 1.0, sr)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    ac = AudioConditioner(sr)
    cond = ac.condition(audio, sr)
    assert cond.duration_s == pytest.approx(1.0, rel=0.01)
    assert cond.pitch_hz > 0
    assert 0 <= cond.timbre <= 1

def test_audio_conditioner_empty():
    ac = AudioConditioner(22050)
    cond = ac.condition(np.array([]), 22050)
    assert cond.pitch_hz == 440.0  # default

def test_audio_conditioner_noise_floor():
    sr = 22050
    audio = np.random.randn(sr).astype(np.float32) * 0.5
    ac = AudioConditioner(sr)
    cond = ac.condition(audio, sr)
    assert cond.noise_floor > 0
