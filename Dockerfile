# Fugatto Audio Lab Docker Image
# Optimized for NVIDIA GPU acceleration and production deployment

FROM nvidia/cuda:12.2-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash fugatto
USER fugatto
WORKDIR /home/fugatto

# Copy requirements first for better caching
COPY --chown=fugatto:fugatto requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --user --no-warn-script-location -r requirements.txt

# Copy application code
COPY --chown=fugatto:fugatto . .

# Install the package
RUN pip install --user --no-warn-script-location -e .

# Add user's pip bin to PATH
ENV PATH="/home/fugatto/.local/bin:${PATH}"

# Create necessary directories
RUN mkdir -p logs outputs models cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import fugatto_lab; print('OK')" || exit 1

# Expose ports
EXPOSE 7860 8501

# Default command
CMD ["fugatto-lab", "serve", "--host", "0.0.0.0", "--port", "7860"]