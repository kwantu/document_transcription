# -------------------------
# Base image
# -------------------------
FROM python:3.10-slim

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Working directory
# -------------------------
WORKDIR /api

# -------------------------
# Python dependencies
# -------------------------
COPY requirements.txt .

# IMPORTANT:
# torch + ultralytics require numpy < 2
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy<2.0" && \
    pip install --no-cache-dir -r requirements.txt

# -------------------------
# Copy application code
# -------------------------
COPY app app
COPY scripts scripts
COPY samples samples

# Copy YOLO model (when available)
# If not present, this line is harmless
COPY *.pt . || true

# -------------------------
# Environment
# -------------------------
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/api

# -------------------------
# Expose API port
# -------------------------
EXPOSE 5005

# -------------------------
# Start FastAPI
# -------------------------
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5005"]
