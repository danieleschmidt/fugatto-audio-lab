# Fugatto Audio Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.0+](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Demo](https://img.shields.io/badge/Demo-Live-brightgreen.svg)](https://fugatto-lab.demo)
[![Paper](https://img.shields.io/badge/Paper-NVIDIA%20Fugatto-red.svg)](https://research.nvidia.com/labs/adlr/fugatto/)

Plug-and-play generative audio playground with live "prompt → sound" preview. The missing toolkit for NVIDIA's Fugatto transformer with text+audio multi-conditioning.

## 🎵 Overview

NVIDIA's Fugatto (Full-spectrum Generative Audio Transformer) demonstrated unprecedented control over audio generation but lacks accessible fine-tuning and evaluation tools. This lab provides:

- **Live Generation UI** with real-time waveform preview
- **Weight Converters** for EnCodec → Fugatto migration
- **Loudness-Aware Loss** for perceptually balanced training
- **MOS Evaluation** via crowd-sourcing scripts
- **Multi-Conditioning** with text, audio, and attribute controls

## 🎧 Live Demo

Try it at [fugatto-lab.demo](https://fugatto-lab.demo) or run locally:

```bash
# Quick demo server
python -m fugatto_lab.demo --port 7860
```

<p align="center">
  <img src="docs/images/fugatto_demo.gif" width="800" alt="Fugatto Audio Lab Demo">
</p>

## ✨ Key Features

- **Text → Audio**: "A cat meowing with reverb in a cathedral"
- **Audio → Audio**: Transform any sound with text instructions
- **Voice Cloning**: Zero-shot voice synthesis from 3-second samples
- **Music Generation**: Create stems, full tracks, or sound effects
- **Fine Control**: Adjust timbre, pitch, tempo, and spatial attributes

## 📋 Requirements

```bash
# Core dependencies
python>=3.10
torch>=2.3.0
torchaudio>=2.3.0
transformers>=4.40.0
accelerate>=0.30.0
einops>=0.7.0
librosa>=0.10.0
soundfile>=0.12.0

# Audio processing
encodec>=0.1.1
audiocraft>=1.2.0
pesq>=0.0.4
pystoi>=0.3.3
torchmetrics[audio]>=1.0.0

# UI components
gradio>=4.37.0
streamlit>=1.35.0
plotly>=5.20.0
wavesurfer-js>=7.7.0

# Evaluation tools
crowdkit>=1.2.0
mturk-crowd-beta-client>=1.0.0
```

## 🛠️ Installation

### Option 1: Quick Install

```bash
# Install from PyPI
pip install fugatto-audio-lab

# Download pretrained weights
fugatto-lab download-weights --model base

# Launch playground
fugatto-lab launch
```

### Option 2: Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/fugatto-audio-lab.git
cd fugatto-audio-lab

# Create environment
conda create -n fugatto python=3.10
conda activate fugatto

# Install dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

## 🚀 Quick Start

### Basic Generation

```python
from fugatto_lab import FugattoModel, AudioProcessor

# Load model
model = FugattoModel.from_pretrained("nvidia/fugatto-base")
processor = AudioProcessor()

# Text to audio
audio = model.generate(
    prompt="Ocean waves crashing on a beach with seagulls",
    duration_seconds=10,
    temperature=0.8
)

# Save result
processor.save_audio(audio, "ocean_waves.wav", sample_rate=48000)
```

### Audio Transformation

```python
# Load input audio
input_audio = processor.load_audio("speech.wav")

# Transform with text conditioning
transformed = model.transform(
    audio=input_audio,
    prompt="Add robot voice effect with metallic echo",
    strength=0.7  # How much to modify original
)

# Compare before/after
processor.plot_comparison(input_audio, transformed)
```

### Multi-Conditioning

```python
# Complex generation with multiple controls
audio = model.generate_multi(
    text_prompt="Jazz piano solo",
    style_audio="reference_jazz.wav",
    attributes={
        "tempo": 120,
        "key": "C major",
        "dynamics": "crescendo",
        "reverb": 0.3
    },
    duration_seconds=30
)
```

## 🎛️ Web Interface

### Gradio Playground

```python
import gradio as gr
from fugatto_lab.ui import create_playground

# Launch interactive playground
app = create_playground(
    model_name="nvidia/fugatto-base",
    enable_voice_clone=True,
    enable_music_gen=True,
    max_duration=30
)

app.launch(share=True)
```

### Streamlit Dashboard

```bash
# Run analysis dashboard
streamlit run fugatto_lab/dashboard.py

# Features:
# - Batch processing
# - A/B testing
# - Quality metrics
# - Export management
```

## 🔄 Weight Conversion

### From EnCodec Models

```python
from fugatto_lab.converters import EnCodecConverter

# Convert Facebook EnCodec to Fugatto format
converter = EnCodecConverter()

fugatto_weights = converter.convert(
    encodec_checkpoint="facebook/encodec_48khz",
    target_format="fugatto",
    optimize_for_inference=True
)

# Verify conversion
converter.validate_conversion(fugatto_weights)
```

### From AudioCraft

```python
from fugatto_lab.converters import AudioCraftConverter

# Migrate MusicGen/AudioGen models
converter = AudioCraftConverter()

converter.convert_musicgen(
    "facebook/musicgen-large",
    output_path="fugatto_musicgen_large.pt"
)
```

## 🎯 Fine-Tuning

### Prepare Dataset

```python
from fugatto_lab.data import AudioDataset, DatasetPreprocessor

# Prepare custom dataset
preprocessor = DatasetPreprocessor(
    sample_rate=48000,
    normalize_loudness=True,
    target_lufs=-14.0
)

dataset = preprocessor.prepare_dataset(
    audio_dir="data/my_sounds/",
    captions_file="data/captions.json",
    augment=True
)
```

### Loudness-Aware Training

```python
from fugatto_lab.training import FugattoTrainer, LoudnessAwareLoss

# Initialize trainer with perceptual loss
trainer = FugattoTrainer(
    model=model,
    loss_fn=LoudnessAwareLoss(
        mse_weight=0.7,
        loudness_weight=0.2,
        spectral_weight=0.1
    ),
    learning_rate=1e-4
)

# Fine-tune on your data
trainer.train(
    train_dataset=dataset,
    epochs=10,
    batch_size=16,
    gradient_accumulation_steps=4,
    save_steps=500
)
```

## 📊 Evaluation Tools

### Automated Metrics

```python
from fugatto_lab.evaluation import AudioQualityMetrics

metrics = AudioQualityMetrics()

# Compute objective metrics
results = metrics.evaluate_batch(
    generated_files=["gen1.wav", "gen2.wav"],
    reference_files=["ref1.wav", "ref2.wav"],
    metrics=["pesq", "stoi", "fad", "kl_divergence"]
)

print(f"Average PESQ: {results['pesq_mean']:.3f}")
print(f"Average STOI: {results['stoi_mean']:.3f}")
```

### Crowd-Sourced MOS

```python
from fugatto_lab.evaluation import MOSEvaluator

# Setup MOS evaluation
evaluator = MOSEvaluator(
    platform="mturk",  # or "prolific", "local"
    num_raters_per_sample=5,
    payment_per_hit=0.50
)

# Launch evaluation
mos_scores = evaluator.run_evaluation(
    audio_files=generated_samples,
    reference_files=ground_truth_samples,
    questions=[
        "Rate the overall quality",
        "Rate the naturalness",
        "Does it match the text description?"
    ]
)

# Aggregate results
report = evaluator.generate_report(mos_scores)
```

## 🎼 Advanced Examples

### Voice Cloning

```python
from fugatto_lab.voice import VoiceCloner

cloner = VoiceCloner(model)

# Clone voice from reference
cloned_speech = cloner.clone(
    reference_audio="speaker_reference.wav",
    text="Hello, this is a cloned voice speaking new text",
    prosody_transfer=True,
    denoise_reference=True
)
```

### Music Stem Generation

```python
from fugatto_lab.music import StemGenerator

stem_gen = StemGenerator(model)

# Generate individual stems
stems = stem_gen.generate_stems(
    style="electronic dance",
    bpm=128,
    key="G minor",
    stems=["drums", "bass", "lead", "pads"],
    duration_seconds=16
)

# Mix stems
full_track = stem_gen.mix_stems(stems, normalize=True)
```

### Sound Effect Design

```python
from fugatto_lab.sfx import SoundEffectStudio

sfx_studio = SoundEffectStudio(model)

# Layer multiple effects
explosion = sfx_studio.design_layered(
    base_prompt="explosion",
    layers=[
        {"prompt": "deep rumble", "delay_ms": 0, "gain": 1.0},
        {"prompt": "glass shattering", "delay_ms": 50, "gain": 0.7},
        {"prompt": "debris falling", "delay_ms": 200, "gain": 0.5}
    ],
    master_reverb=0.3
)
```

## 🔧 Configuration

### Model Settings

```yaml
# config/fugatto_config.yaml
model:
  name: nvidia/fugatto-base
  precision: fp16
  max_length: 1500  # ~30 seconds at 48kHz
  
audio:
  sample_rate: 48000
  channels: 1
  codec_bitrate: 6.0  # kbps
  
generation:
  temperature: 0.8
  top_p: 0.95
  cfg_scale: 3.0
  
training:
  batch_size: 8
  learning_rate: 1e-4
  warmup_steps: 1000
```

## 🐳 Docker Deployment

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Install Fugatto Lab
RUN pip install fugatto-audio-lab

# Copy configs
COPY config/ /app/config/

# Expose ports
EXPOSE 7860 8501

# Launch services
CMD ["fugatto-lab", "serve", "--all-services"]
```

```bash
# Run with GPU support
docker run --gpus all -p 7860:7860 fugatto-audio-lab
```

## 📈 Performance Optimization

### Inference Speed

```python
from fugatto_lab.optimize import OptimizationPipeline

# Optimize model for deployment
optimizer = OptimizationPipeline()

fast_model = optimizer.optimize(
    model,
    techniques=[
        "torch_compile",
        "flash_attention",
        "kv_cache",
        "batch_processing"
    ],
    target_latency_ms=100
)

# Benchmark
optimizer.benchmark(fast_model, num_runs=100)
```

### Memory Efficiency

```python
# Enable memory-efficient generation
model.enable_memory_efficient_attention()
model.enable_gradient_checkpointing()

# Stream long audio
for chunk in model.stream_generate(prompt, chunk_size_seconds=2):
    # Process chunk immediately
    play_audio(chunk)
```

## 🤝 Contributing

We welcome contributions! Priority areas:
- Additional audio effects and transformations
- Improved evaluation metrics
- Multi-GPU training optimizations
- Real-time streaming support
- Community model zoo

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 Citation

```bibtex
@software{fugatto_audio_lab,
  title={Fugatto Audio Lab: Toolkit for Controllable Audio Generation},
  author={Daniel Schmidt},
  year={2025},
  url={https://github.com/danieleschmidt/fugatto-audio-lab}
}

@article{nvidia_fugatto_2024,
  title={Fugatto: Full-spectrum Generative Audio Transformer},
  author={NVIDIA Research},
  journal={arXiv preprint},
  year={2024}
}
```

## 🏆 Acknowledgments

- NVIDIA Research for the Fugatto model
- The AudioCraft team for EnCodec
- Contributors to the open-source audio ML community

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

## 🔗 Resources

- [Documentation](https://fugatto-lab.readthedocs.io)
- [Model Zoo](https://huggingface.co/collections/fugatto-lab/models)
- [Colab Notebooks](https://github.com/yourusername/fugatto-audio-lab/tree/main/notebooks)
- [Discord Community](https://discord.gg/fugatto-lab)
- [YouTube Tutorials](https://youtube.com/@fugatto-lab)

## 📧 Contact

- **GitHub Issues**: Bug reports and features
- **Email**: fugatto-lab@yourdomain.com
- **Twitter**: [@FugattoLab](https://twitter.com/fugattolab)
