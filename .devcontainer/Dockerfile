# Development container for Fugatto Audio Lab
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    curl \
    wget \
    build-essential \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    libasound2-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    pkg-config \
    sudo \
    zsh \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set up Python environment
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install development tools
RUN pip install \
    black \
    isort \
    ruff \
    mypy \
    pytest \
    pytest-cov \
    pytest-xdist \
    pre-commit \
    bandit[toml] \
    safety

# Install PyTorch with CUDA support
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER $USERNAME

# Set up zsh and oh-my-zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Set default shell to zsh
ENV SHELL=/bin/zsh

# Add helpful aliases
RUN echo 'alias ll="ls -la"' >> ~/.zshrc && \
    echo 'alias la="ls -la"' >> ~/.zshrc && \
    echo 'alias pytest="python -m pytest"' >> ~/.zshrc && \
    echo 'alias fugatto="python -m fugatto_lab.server"' >> ~/.zshrc

# Set environment variables for development
ENV PYTHONPATH=/workspace
ENV LOG_LEVEL=DEBUG
ENV MODEL_CACHE_DIR=/workspace/models

# Expose ports
EXPOSE 8000 8001

# Default command
CMD ["zsh"]