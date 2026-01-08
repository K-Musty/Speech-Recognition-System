FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

# Install system dependencies first (they change less often)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app

CMD ["uwsgi", "app.ini"]
