#!/usr/bin/env python3
"""Setup script for Fugatto Audio Lab development environment."""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, check=True, shell=False):
    """Run a shell command with error handling."""
    print(f"Running: {cmd}")
    try:
        if shell:
            result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd.split(), check=check, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        raise


def check_prerequisites():
    """Check if required tools are installed."""
    print("Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10+ is required")
        sys.exit(1)
    
    # Check if CUDA is available
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch not installed yet, will install during setup")
    
    # Check if git is available
    try:
        run_command("git --version")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Git is required but not found")
        sys.exit(1)


def setup_environment():
    """Set up the development environment."""
    print("Setting up development environment...")
    
    # Create necessary directories
    dirs = ["logs", "outputs", "models", "cache", "configs", "reports"]
    for directory in dirs:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Install development dependencies
    print("Installing development dependencies...")
    run_command("pip install -e .[dev]")
    
    # Install pre-commit hooks
    print("Installing pre-commit hooks...")
    run_command("pre-commit install")
    
    # Generate initial configuration files
    create_config_files()


def create_config_files():
    """Create initial configuration files."""
    print("Creating configuration files...")
    
    # Create .env.example
    env_example = """# Fugatto Audio Lab Environment Variables
# Copy this file to .env and customize for your setup

# Model configuration
FUGATTO_MODEL_NAME=nvidia/fugatto-base
FUGATTO_MODEL_CACHE=./cache
FUGATTO_OUTPUT_DIR=./outputs

# GPU configuration
CUDA_VISIBLE_DEVICES=0

# API configuration
FUGATTO_API_HOST=0.0.0.0
FUGATTO_API_PORT=7860

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/fugatto.log

# Development
DEBUG=false
RELOAD=false
"""
    
    with open(".env.example", "w") as f:
        f.write(env_example)
    
    # Create basic configuration
    config_yaml = """# Fugatto Audio Lab Configuration
model:
  name: nvidia/fugatto-base
  precision: fp16
  max_length: 1500
  cache_dir: ./cache

audio:
  sample_rate: 48000
  channels: 1
  codec_bitrate: 6.0

generation:
  temperature: 0.8
  top_p: 0.95
  cfg_scale: 3.0

training:
  batch_size: 8
  learning_rate: 1e-4
  warmup_steps: 1000
  max_steps: 10000

paths:
  output_dir: ./outputs
  log_dir: ./logs
  model_dir: ./models
"""
    
    Path("configs").mkdir(exist_ok=True)
    with open("configs/default.yaml", "w") as f:
        f.write(config_yaml)


def run_initial_tests():
    """Run initial tests to verify setup."""
    print("Running initial tests...")
    try:
        run_command("python -c 'import fugatto_lab; print(\"Import successful\")'")
        run_command("pytest tests/ -v --tb=short", check=False)  # Don't fail if tests fail
    except subprocess.CalledProcessError:
        print("Some tests failed, but setup completed successfully")


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎵 Fugatto Audio Lab Setup Complete! 🎵")
    print("="*60)
    print("\nNext steps:")
    print("1. Copy .env.example to .env and customize settings")
    print("2. Download model weights: fugatto-lab download-weights")
    print("3. Run the playground: fugatto-lab launch")
    print("4. Start developing: make test")
    print("\nUseful commands:")
    print("  make help           - Show all available commands")
    print("  make test           - Run test suite")
    print("  make quality        - Run code quality checks")
    print("  make docs           - Build documentation")
    print("  make docker-build   - Build Docker image")
    print("\nFor more information, see docs/DEVELOPMENT.md")
    print("="*60)


def main():
    """Main setup function."""
    print("Fugatto Audio Lab Development Setup")
    print("="*40)
    
    try:
        check_prerequisites()
        setup_environment()
        run_initial_tests()
        print_next_steps()
    except Exception as e:
        print(f"\nSetup failed with error: {e}")
        print("Please check the error messages above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()