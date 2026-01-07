FROM python:3.10-slim

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Set workdir
# -------------------------
WORKDIR /api

# -------------------------
# Python deps
# -------------------------
COPY requirements.txt .

# IMPORTANT: numpy<2 for torch compatibility
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy<2.0" && \
    pip install --no-cache-dir -r requirements.txt

# -------------------------
# Copy application
# -------------------------
COPY app app
COPY yolo11s.pt yolo11s.pt

# -------------------------
# Environment
# -------------------------
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# -------------------------
# Expose port
# -------------------------
EXPOSE 5005

# -------------------------
# Run API
# -------------------------
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5005"]
