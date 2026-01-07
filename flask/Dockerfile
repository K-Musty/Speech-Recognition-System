FROM tensorflow/tensorflow:2.15.0
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uwsgi", "app.ini"]
