# Lightweight Python 3.11 image
FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install minimal system dependencies for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and model weights
COPY . .

# Expose default port
EXPOSE 8000

# Run Uvicorn with dynamic port binding for cloud platforms (Render, Railway, etc.)
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
