import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import librosa
import joblib
import os

def extract_features_from_file(filename, duration=30):
    """Extract features from a single audio file"""
    try:
        audio, sr = librosa.load(filename, duration=duration)
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
        
        return features
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def train_model(data_path, model_save_path):
    """Train the music genre classification model"""
    print(f"Loading dataset from {data_path}...")
    
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Extract features for all files
    features = []
    labels = []
    
    for idx, row in df.iterrows():
        feature_vector = extract_features_from_file(row['filename'])
        if feature_vector is not None:
            features.append(feature_vector)
            labels.append(row['genre'])
        
        if len(features) % 50 == 0:
            print(f"Processed {len(features)} files...")
    
    if len(features) == 0:
        raise ValueError("No valid audio files found for training")
    
    X = np.array(features)
    y = np.array(labels)
    
    print(f"Extracted features for {len(X)} samples")
    print(f"Feature vector shape: {X.shape}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Create model
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test Accuracy: {test_accuracy:.3f}")
    
    # Save model and encoder
    model.save(model_save_path)
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    
    print(f"Model saved to {model_save_path}")
    print(f"Label encoder saved to models/label_encoder.pkl")
    
    return model, history
