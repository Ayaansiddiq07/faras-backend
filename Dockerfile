FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install numpy==1.26.4
RUN pip install opencv-python-headless==4.10.0.84
RUN pip install tf-keras==2.16.0
RUN pip install deepface==0.0.93
RUN pip install flask==3.0.3 flask-cors==4.0.1 requests==2.32.3 gunicorn==22.0.0

COPY . .
RUN mkdir -p known_faces

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1", "app:app"]
