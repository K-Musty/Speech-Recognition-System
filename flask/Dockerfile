FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app

# Install audio system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

CMD ["uwsgi", "app.ini"]
