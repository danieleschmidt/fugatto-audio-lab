import numpy as np
import pytest
import tempfile
from pathlib import Path
from fugatto_lab.conditioners import AudioCondition
from fugatto_lab.synth import AudioSynthEngine, AudioSynthPipeline

def test_synth_engine_basic():
    engine = AudioSynthEngine(sample_rate=22050)
    cond = AudioCondition(pitch_hz=440.0, duration_s=0.5)
    audio = engine.synthesize(cond)
    assert audio.dtype == np.float32
    assert len(audio) == pytest.approx(22050 * 0.5, rel=0.01)

def test_synth_engine_amplitude():
    engine = AudioSynthEngine(22050)
    cond = AudioCondition(pitch_hz=440.0, duration_s=0.5, noise_floor=0.0)
    audio = engine.synthesize(cond)
    assert np.abs(audio).max() <= 1.0

def test_synth_engine_no_silence():
    engine = AudioSynthEngine(22050)
    cond = AudioCondition(pitch_hz=440.0, duration_s=0.5, noise_floor=0.001)
    audio = engine.synthesize(cond)
    assert np.abs(audio).max() > 0.0

def test_pipeline_from_text():
    pipe = AudioSynthPipeline(sample_rate=8000)
    audio = pipe.from_text("piano at 440 hz for 1 seconds")
    assert len(audio) > 0
    assert audio.dtype == np.float32

def test_pipeline_from_audio():
    pipe = AudioSynthPipeline(sample_rate=8000)
    sr = 8000
    input_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
    audio = pipe.from_audio(input_audio)
    assert len(audio) > 0

def test_pipeline_save_wav(tmp_path):
    pipe = AudioSynthPipeline(sample_rate=8000)
    audio = pipe.from_text("synth 1 seconds")
    wav_path = tmp_path / "test.wav"
    result = pipe.save_wav(audio, wav_path)
    assert result.exists()
    assert result.stat().st_size > 44  # WAV header is 44 bytes

def test_pipeline_to_bytes():
    pipe = AudioSynthPipeline(sample_rate=8000)
    audio = pipe.from_text("drums 1 seconds")
    data = pipe.to_bytes(audio)
    assert len(data) > 44
    assert data[:4] == b"RIFF"

def test_pipeline_reverb():
    pipe = AudioSynthPipeline(sample_rate=8000)
    audio = pipe.from_text("guitar with reverb 1 seconds")
    assert len(audio) > 0

def test_synth_different_instruments():
    engine = AudioSynthEngine(8000)
    for inst in ["piano", "guitar", "violin"]:
        from fugatto_lab.conditioners import TextConditioner
        cond = TextConditioner().condition(f"{inst} 0.5 seconds")
        audio = engine.synthesize(cond)
        assert len(audio) > 0
