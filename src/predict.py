import tensorflow as tf
import numpy as np
import librosa
import joblib
import os

def extract_features_for_prediction(audio_file, duration=30):
    """Extract the same features used during training"""
    try:
        audio, sr = librosa.load(audio_file, duration=duration)
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Combine all features
        features = np.concatenate([
            mfcc_mean,
            [np.mean(spectral_centroids), np.mean(spectral_rolloff), np.mean(spectral_bandwidth)],
            chroma_mean
        ])
        
        return features.reshape(1, -1)  # Reshape for single prediction
    except Exception as e:
        print(f"Error extracting features from {audio_file}: {e}")
        return None

def predict_genre(audio_file, model_path='models/music_genre_model.h5', 
                 label_encoder_path='models/label_encoder.pkl'):
    """Predict genre for a single audio file"""
    
    # Check if model files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")
    
    # Load model and label encoder
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading label encoder from {label_encoder_path}...")
    label_encoder = joblib.load(label_encoder_path)
    
    # Extract features
    print(f"Extracting features from {audio_file}...")
    features = extract_features_for_prediction(audio_file)
    
    if features is None:
        raise ValueError("Could not extract features from audio file")
    
    # Make prediction
    print("Making prediction...")
    predictions = model.predict(features)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Decode prediction
    predicted_genre = label_encoder.inverse_transform([predicted_class_idx])[0]
    
    print(f"Prediction completed: {predicted_genre} (confidence: {confidence:.3f})")
    return predicted_genre, confidence
