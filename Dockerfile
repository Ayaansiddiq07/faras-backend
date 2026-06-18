# FARAS Backend - Render Deployment
# Uses debian bullseye-slim which has better cmake/dlib support

FROM python:3.11-bullseye

# Install ALL required build tools for dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install numpy first - dlib needs it during build
RUN pip install numpy==1.26.4

# Install dlib from source - bullseye has all needed libs
RUN pip install dlib==19.24.2

# Install face recognition models
RUN pip install git+https://github.com/ageitgey/face_recognition_models

# Install face_recognition
RUN pip install face-recognition==1.3.0

# Install rest of packages
RUN pip install \
    flask==3.0.3 \
    flask-cors==4.0.1 \
    opencv-python-headless==4.10.0.84 \
    requests==2.32.3 \
    gunicorn==22.0.0

# Copy project files
COPY . .

# Ensure known_faces folder exists
RUN mkdir -p known_faces

EXPOSE 5000

# Single worker - face recognition is CPU heavy
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1", "app:app"]
