FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /api

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install "numpy<2.0" && \
    pip install -r requirements.txt

COPY app app
COPY scripts scripts
COPY samples samples

ENV PYTHONPATH=/api

EXPOSE 5005

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5005"]
