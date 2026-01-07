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
# App directory
# -------------------------
WORKDIR /api

# -------------------------
# Python dependencies
# -------------------------
COPY requirements.txt .

# numpy<2.0 is REQUIRED for torch + ultralytics compatibility
RUN pip install --upgrade pip && \
    pip install "numpy<2.0" && \
    pip install -r requirements.txt

# -------------------------
# Application code
# -------------------------
COPY app app
COPY scripts scripts
COPY samples samples

# -------------------------
# YOLO model
# -------------------------
COPY yolo11s.pt /api/yolo11s.pt

# -------------------------
# Environment
# -------------------------
ENV PYTHONPATH=/api
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# -------------------------
# Network
# -------------------------
EXPOSE 5005

# -------------------------
# Start API
# -------------------------
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5005"]
