# FARAS Backend - Render Deployment
# Builds dlib directly from source via git clone (bypasses pip wheel issues)

FROM python:3.11-bullseye

# Install ALL build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libboost-python-dev \
    libboost-thread-dev \
    libboost-all-dev \
    libx11-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

# Install numpy first
RUN pip install numpy==1.26.4

# Build dlib from source directly (most reliable method)
RUN git clone --depth 1 https://github.com/davisking/dlib.git /tmp/dlib && \
    cd /tmp/dlib && \
    python setup.py install && \
    rm -rf /tmp/dlib

# Install face_recognition_models
RUN pip install git+https://github.com/ageitgey/face_recognition_models

# Install face_recognition (no dlib install, already done above)
RUN pip install face-recognition==1.3.0 --no-deps
RUN pip install Pillow

# Install remaining packages
RUN pip install \
    flask==3.0.3 \
    flask-cors==4.0.1 \
    opencv-python-headless==4.10.0.84 \
    requests==2.32.3 \
    gunicorn==22.0.0

COPY . .
RUN mkdir -p known_faces

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1", "app:app"]
