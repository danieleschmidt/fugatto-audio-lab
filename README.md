# fugatto-audio-lab

Generative audio playground inspired by NVIDIA Fugatto's text+audio multi-conditioning approach.

## Overview

Implements the full conditioning pipeline without requiring the actual Fugatto model:

- **TextConditioner**: Parse text prompts for pitch (Hz), tempo (BPM), instrument, mood → `AudioCondition`
- **AudioConditioner**: Analyze input audio (spectral centroid, ZCR, energy) → `AudioCondition`
- **AudioSynthEngine**: scipy.signal-based additive synthesis with harmonics, ADSR envelope, reverb
- **AudioSynthPipeline**: End-to-end text→audio and audio→audio (style transfer) with WAV export

## Example

```python
from fugatto_lab.synth import AudioSynthPipeline

pipe = AudioSynthPipeline(sample_rate=22050)

# Text to audio
audio = pipe.from_text("piano melody at 440 hz, 120 bpm, 3 seconds with reverb")
pipe.save_wav(audio, "output.wav")

# Audio style transfer
import numpy as np
input_audio = np.sin(2 * np.pi * 220 * np.linspace(0, 2, 44100))
audio = pipe.from_audio(input_audio.astype(np.float32), prompt_override="violin")
pipe.save_wav(audio, "styled.wav")
```

## Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```
