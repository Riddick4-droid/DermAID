# Dockerfile
# Build: docker build -t dermaid .
# Run:   docker run -p 8000:8000 --env-file .env dermaid

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (OpenCV etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY configs/ configs/
COPY src/ src/

# Volume mount points (created at runtime if needed)
RUN mkdir -p chroma_db sessions uploads

EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]