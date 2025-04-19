import librosa
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

SAVED_MODEL_PATH = "model.h5"
SAMPLES_TO_CONSIDER = 22050

class _Keyword_Spotting_Service:
    model = None
    _mappings = [
        "down",
        "up",
        "yes",
        "no",
        "right",
        "left"
    ]
    _instance = None

    def predict(self, file_path):
        # Extract MFCCs
        MFCCs = self.preprocess(file_path) # ( # segment, # coefficients)

        # Convert 2d MFCCs array into 4d array -> (# samples, # segments, # coefficients, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # Make prediction
        predictions = self.model.predict(MFCCs) # [ {0.1, 0.6, 0.1, ...} ]
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        """Extract MFCCs from audio file.

        :param file_path (str): Path of audio file
        :param num_mfcc (int): # of coefficients to extract
        :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples

        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        """

        # Load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # Ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # Extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        return MFCCs.T

def Keyword_Spotting_Service():

    # Ensure that we Only have one instance
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance