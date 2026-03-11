FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libzbar0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /api

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Fix Ultralytics config warning
RUN mkdir -p /root/.config/Ultralytics && chmod -R 777 /root/.config/Ultralytics

# Copy application
COPY app app
COPY scripts scripts

# Copy model weights
COPY app/core/model_weights /api/app/core/model_weights

ENV PYTHONPATH=/api
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 5005

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5005"]