# FARAS Backend - Dockerfile for Render
# Uses Python 3.11 slim + builds dlib from source

FROM python:3.11-slim

# Install system dependencies required by dlib and opencv
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libboost-thread-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (Docker cache optimization)
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip
RUN pip install numpy==1.26.4
RUN pip install dlib==19.24.2
RUN pip install face-recognition==1.3.0
RUN pip install git+https://github.com/ageitgey/face_recognition_models
RUN pip install flask==3.0.3 flask-cors==4.0.1 opencv-python-headless==4.10.0.84 requests==2.32.3 gunicorn==22.0.0

# Copy app code
COPY . .

# Create known_faces folder
RUN mkdir -p known_faces

# Expose port
EXPOSE 5000

# Start gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1", "app:app"]
