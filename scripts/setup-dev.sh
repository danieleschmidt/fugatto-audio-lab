#!/bin/bash
# Development environment setup script for Fugatto Audio Lab

set -e

echo "🎵 Setting up Fugatto Audio Lab development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running in virtual environment
check_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "Not running in a virtual environment"
        print_status "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_success "Already in virtual environment: $VIRTUAL_ENV"
    fi
}

# Check Python version
check_python() {
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    required_version="3.10"
    
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
        print_success "Python $python_version is compatible"
    else
        print_error "Python $required_version or higher is required, but $python_version is installed"
        exit 1
    fi
}

# Check for GPU support
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        print_success "GPU detected: $gpu_info"
    else
        print_warning "No NVIDIA GPU detected. The application will run on CPU only."
    fi
}

# Install system dependencies (Ubuntu/Debian)
install_system_deps() {
    if command -v apt-get &> /dev/null; then
        print_status "Installing system dependencies..."
        sudo apt-get update
        sudo apt-get install -y \
            python3-dev \
            build-essential \
            ffmpeg \
            libsndfile1 \
            portaudio19-dev \
            libasound2-dev \
            libavcodec-dev \
            libavformat-dev \
            libavutil-dev \
            libswresample-dev \
            pkg-config
        print_success "System dependencies installed"
    elif command -v brew &> /dev/null; then
        print_status "Installing system dependencies with Homebrew..."
        brew install ffmpeg portaudio libsndfile pkg-config
        print_success "System dependencies installed"
    else
        print_warning "Could not detect package manager. Please install system dependencies manually."
    fi
}

# Install Python dependencies
install_python_deps() {
    print_status "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    print_status "Installing development dependencies..."
    pip install -e ".[dev]"
    
    print_success "Python dependencies installed"
}

# Set up pre-commit hooks
setup_precommit() {
    print_status "Setting up pre-commit hooks..."
    pre-commit install
    pre-commit install --hook-type commit-msg
    print_success "Pre-commit hooks installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating project directories..."
    mkdir -p models data logs temp
    print_success "Project directories created"
}

# Download sample models (optional)
download_sample_models() {
    read -p "Download sample models for testing? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Downloading sample models..."
        python -c "
import os
# Create placeholder model files for testing
os.makedirs('models/test', exist_ok=True)
with open('models/test/sample_model.pt', 'w') as f:
    f.write('# Sample model file for testing')
print('Sample model files created')
"
        print_success "Sample models downloaded"
    fi
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Test imports
    python -c "
import torch
import torchaudio
import librosa
import soundfile
print('✓ Core audio libraries imported successfully')

if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('ℹ CUDA not available (CPU-only mode)')

print('✓ Installation verification complete')
"
    
    # Test project structure
    for dir in models data logs temp; do
        if [[ -d "$dir" ]]; then
            print_success "Directory '$dir' exists"
        else
            print_error "Directory '$dir' missing"
        fi
    done
}

# Generate development configuration
create_dev_config() {
    print_status "Creating development configuration..."
    
    cat > .env.dev << EOF
# Development environment configuration
LOG_LEVEL=DEBUG
DEBUG=true
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Model configuration
MODEL_CACHE_DIR=./models
DEFAULT_MODEL=test/sample_model

# Development settings
ENABLE_RELOAD=true
ENABLE_METRICS=true
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Database (optional)
# DATABASE_URL=sqlite:///./fugatto_lab.db
# REDIS_URL=redis://localhost:6379/0
EOF
    
    print_success "Development configuration created (.env.dev)"
}

# Create useful development scripts
create_dev_scripts() {
    print_status "Creating development scripts..."
    
    # Test script
    cat > scripts/test.sh << 'EOF'
#!/bin/bash
echo "🧪 Running tests..."
pytest tests/ -v --cov=fugatto_lab --cov-report=html --cov-report=term
echo "📊 Coverage report generated in htmlcov/"
EOF
    
    # Lint script
    cat > scripts/lint.sh << 'EOF'
#!/bin/bash
echo "🔍 Running linters..."
echo "Running ruff..."
ruff check fugatto_lab tests
echo "Running black..."
black --check fugatto_lab tests
echo "Running isort..."
isort --check-only fugatto_lab tests
echo "Running mypy..."
mypy fugatto_lab
echo "✅ Linting complete"
EOF
    
    # Format script
    cat > scripts/format.sh << 'EOF'
#!/bin/bash
echo "🎨 Formatting code..."
black fugatto_lab tests
isort fugatto_lab tests
ruff --fix fugatto_lab tests
echo "✅ Code formatted"
EOF
    
    # Make scripts executable
    chmod +x scripts/*.sh
    
    print_success "Development scripts created in scripts/"
}

# Show next steps
show_next_steps() {
    echo
    echo "🎉 Development environment setup complete!"
    echo
    echo "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Start the development server: python -m fugatto_lab.server --debug"
    echo "3. Run tests: ./scripts/test.sh"
    echo "4. Format code: ./scripts/format.sh"
    echo "5. Check code quality: ./scripts/lint.sh"
    echo
    echo "Useful commands:"
    echo "- pytest tests/                    # Run tests"
    echo "- pre-commit run --all-files      # Run pre-commit hooks"
    echo "- docker-compose up               # Start with Docker"
    echo "- code .                          # Open in VS Code"
    echo
    echo "📚 Documentation: docs/"
    echo "🐛 Issues: https://github.com/yourusername/fugatto-audio-lab/issues"
    echo
}

# Main execution
main() {
    print_status "Starting Fugatto Audio Lab development setup..."
    
    check_python
    check_venv
    check_gpu
    install_system_deps
    install_python_deps
    setup_precommit
    create_directories
    create_dev_config
    create_dev_scripts
    download_sample_models
    verify_installation
    show_next_steps
}

# Run main function
main "$@"