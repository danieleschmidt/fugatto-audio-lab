#!/bin/bash
# Development environment setup script for Fugatto Audio Lab

set -e

echo "🎵 Setting up Fugatto Audio Lab development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    htop \
    tree \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    pulseaudio \
    alsa-utils

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

if [ -f requirements-dev.txt ]; then
    pip install -r requirements-dev.txt
fi

# Install project in development mode
echo "🔧 Installing project in development mode..."
pip install -e ".[dev]"

# Setup pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "✅ Pre-commit hooks installed"
else
    echo "⚠️  pre-commit not found, skipping hooks setup"
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p models
mkdir -p data/raw
mkdir -p data/processed
mkdir -p outputs
mkdir -p logs
mkdir -p .cache

# Set up Git configuration
echo "🔐 Setting up Git configuration..."
git config --global --add safe.directory /workspace
git config pull.rebase false

# Download sample data if not exists
echo "📊 Setting up sample data..."
if [ ! -f data/sample_audio.wav ]; then
    echo "Creating sample audio file..."
    python -c "
import numpy as np
import scipy.io.wavfile as wav
sample_rate = 48000
duration = 5
t = np.linspace(0, duration, int(sample_rate * duration))
audio = 0.3 * np.sin(2 * np.pi * 440 * t) * np.exp(-t/2)
wav.write('data/sample_audio.wav', sample_rate, (audio * 32767).astype(np.int16))
print('✅ Sample audio created')
"
fi

# Test installation
echo "🧪 Testing installation..."
python -c "
import fugatto_lab
from fugatto_lab.core import FugattoModel, AudioProcessor
print('✅ Core modules import successfully')

# Test health check
from fugatto_lab.monitoring import HealthChecker
checker = HealthChecker()
print('✅ Health checker initialized')

# Test basic functionality
processor = AudioProcessor()
print('✅ Audio processor ready')
"

echo "🎉 Development environment setup complete!"
echo ""
echo "🚀 Quick start commands:"
echo "  fugatto-lab health          # Check system health"
echo "  fugatto-lab serve           # Start web interface"
echo "  pytest                     # Run tests"
echo "  python -m fugatto_lab.demo  # Run demo"
echo ""
echo "📚 Useful directories:"
echo "  /workspace/models    # Model storage"
echo "  /workspace/data      # Dataset storage"
echo "  /workspace/outputs   # Generated audio"
echo "  /workspace/logs      # Application logs"