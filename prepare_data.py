import librosa
from librosa.feature import mfcc
import os
import json

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")  # Add any extensions you support

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):

    # Create data dictionary
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # loop through all sub dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Ensure we are not at root level
        if dirpath != dataset_path:
            # Update mappings
            category = dirpath.split("/")[-1]  # dataset/down -> [dataset, down]
            data["mappings"].append(category)

            # Loop through filenames and extract MFCCs
            for f in filenames:

                # Get file path
                file_path = os.path.join(dirpath, f)

                # Check if the file is an audio file
                if f.endswith(AUDIO_EXTENSIONS):
                    try:
                        # load audio file
                        signal, sr = librosa.load(file_path)

                        # Ensure the audio file is at least 1 sec
                        if len(signal) >= SAMPLES_TO_CONSIDER:
                            # Set/enforce 1 sec long signal
                            signal = signal[:SAMPLES_TO_CONSIDER]

                            # Extract the MFCC (keyword arguments only)
                            MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

                            # store data
                            data["labels"].append(i-1)
                            data["MFCCs"].append(MFCCs.T.tolist())
                            data["files"].append(file_path)

                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

    # Save the data to JSON
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)
