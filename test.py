import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf

path = r"C:\Users\lhari\Downloads\Music\Lut Gaye.mp3"

# Load the model
model = tf.keras.models.load_model(r"valid_model.h5")
audio, sr = librosa.load(path, sr=44100)