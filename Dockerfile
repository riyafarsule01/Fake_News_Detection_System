# Production-Ready Dockerfile for Fake News Detection
# Combines build-time model training with runtime flexibility

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p models data/processed

# Download NLTK data (if needed)
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)" || true

# Check if data exists and run ML pipeline during build
RUN if [ -f "data/raw_data/True.csv" ] && [ -f "data/raw_data/Fake.csv" ]; then \
    echo " Running preprocessing..." && \
    python src/preprocess.py && \
    echo " Training models..." && \
    python src/train.py && \
    echo " Models trained during build!"; \
    else \
    echo " Raw data not found. Models will be trained at runtime."; \
    fi

# Create a health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:3000/health')" || exit 1

# Expose port
EXPOSE 3000

# Start the application
CMD ["python", "app/main.py"]