# MM-RAG Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY mmrag/requirements.txt /app/mmrag/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r mmrag/requirements.txt

# Copy the rest of the application
COPY . /app/

# Install MM-RAG in development mode
RUN pip install -e .

# Create data directory
RUN mkdir -p /app/mmrag/data

# Expose FastAPI port
EXPOSE 8000

# Default command: run FastAPI server
CMD ["uvicorn", "mmrag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
