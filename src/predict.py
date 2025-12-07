import tensorflow as tf
import numpy as np
from .feature_extraction import extract_all_features
import joblib

def predict_genre(audio_file, model_path, label_encoder_path):
    """Predict genre for a single audio file"""
    # Load model and label encoder
    model = tf.keras.models.load_model(model_path)
    label_encoder = joblib.load(label_encoder_path)
    
    # Load and process audio
    audio, sr = librosa.load(audio_file, duration=30)
    features = extract_all_features(audio, sr).reshape(1, -1)
    
    # Predict
    predictions = model.predict(features)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Decode label
    predicted_genre = label_encoder.inverse_transform([predicted_class])[0]
    
    return predicted_genre, confidence
