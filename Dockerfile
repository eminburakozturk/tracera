FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    CUDA_VISIBLE_DEVICES=""

WORKDIR /app

# OpenCV ve sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# space/requirements.txt (CPU torch içerir)
COPY space/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# space/ paketinin içeriğini /app dizinine kopyala
COPY space/ /app/

EXPOSE 7860

CMD ["python", "app.py"]
