# Backend (FastAPI) — posture/eye-blink detection API + websocket
FROM python:3.10-slim

WORKDIR /app

# dlib is compiled from source here, which needs a C++ toolchain + cmake; the rest
# are runtime libs opencv-python and mediapipe need on a headless Debian base.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install deps first so this layer is cached unless requirements.txt changes.
COPY requirements.txt setup.py ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/ holds the SQLite database — mounted as a volume in docker-compose so it
# survives container rebuilds instead of living only inside the image.
RUN mkdir -p /app/data

EXPOSE 8000

# Binding 0.0.0.0 (not the app's own default of "localhost") is required for the
# port to be reachable from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
