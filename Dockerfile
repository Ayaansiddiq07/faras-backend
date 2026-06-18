# FARAS Backend - Render Deployment
# Uses prebuilt dlib-bin wheel instead of compiling dlib from source.
# Render's free tier gives only 512MB RAM, which is not enough to compile
# dlib's C++ (a single translation unit like object_detection.cpp can need
# 1GB+ of RAM for the compiler alone, regardless of build parallelism).
# dlib-bin ships a prebuilt manylinux wheel for the same dlib version, so
# installing it is just unpacking a .whl - no compilation, no OOM risk.

FROM python:3.11-bullseye

WORKDIR /app

# Minimal runtime libs needed by opencv-python-headless / dlib-bin at
# import time (no compilers, no -dev packages, no cmake needed anymore).
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# setuptools>=81 removed the pkg_resources module. face_recognition_models'
# __init__.py does "from pkg_resources import resource_filename" at import
# time, so a too-new setuptools makes face_recognition_models *install*
# successfully but crash on import - which is exactly what surfaced at
# runtime ("please install face_recognition_models") even though it really
# was installed. Pinning below 81 keeps pkg_resources available.
RUN pip install --upgrade pip "setuptools<81" wheel

# Install numpy first (face_recognition_models / dlib-bin expect it present)
RUN pip install numpy==1.26.4

# Prebuilt dlib wheel - matches the dlib version face-recognition==1.3.0
# expects, but installs in seconds with no compilation.
RUN pip install dlib-bin==19.24.2.post1

# face_recognition_models (pure Python/data, no compilation needed)
RUN pip install git+https://github.com/ageitgey/face_recognition_models

# face-recognition itself, without letting it pull the real "dlib" package
# as a dependency (which would trigger a source build and undo everything
# above). dlib-bin already provides the "dlib" importable module.
RUN pip install face-recognition==1.3.0 --no-deps
RUN pip install Pillow Click

# Remaining application dependencies
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
