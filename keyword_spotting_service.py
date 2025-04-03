import librosa
import tensorflow as tf
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


def Keyword_Spotting_Service():

    # Ensure that we Only have one instance
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
