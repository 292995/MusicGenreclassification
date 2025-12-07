import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .model import create_model
from .feature_extraction import extract_all_features
import joblib

def train_model(data_path, model_save_path):
    """Train the music genre classification model"""
    # Load dataset (assumes CSV with 'filename' and 'genre' columns)
    df = pd.read_csv(data_path)
    
    # Extract features
    features = []
    labels = []
    
    for idx, row in df.iterrows():
        audio, sr = librosa.load(row['filename'], duration=30)
        if audio is not None:
            feature_vector = extract_all_features(audio, sr)
            features.append(feature_vector)
            labels.append(row['genre'])
    
    features = np.array(features)
    labels = np.array(labels)
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = create_model(features.shape[1], len(label_encoder.classes_))
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.3f}")
    
    # Save model and label encoder
    model.save(model_save_path)
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    
    return model, history
