"""
CNN-based Music Genre Classification Model

This module implements a Convolutional Neural Network for music genre classification
using spectrogram representations of audio data. The CNN architecture is designed
to capture both temporal and frequency patterns in music signals.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

class CNNMusicClassifier:
    """
    Convolutional Neural Network for music genre classification.
    Uses mel-spectrogram features for improved performance.
    """
    
    def __init__(self, input_shape=(128, 1294, 1), num_classes=10):
        """
        Initialize the CNN model.
        
        Args:
            input_shape (tuple): Shape of input spectrograms (freq_bins, time_steps, channels)
            num_classes (int): Number of music genres to classify
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self._build_model()
        logger.info(f"CNN Music Classifier initialized with input shape {input_shape}, {num_classes} classes")
    
    def _build_model(self):
        """Build the CNN architecture."""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        self.model = model
        logger.info("CNN model architecture built successfully")
    
    def extract_spectrogram_features(self, audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
        """
        Extract mel-spectrogram features from audio.
        
        Args:
            audio (array): Audio signal array
            sr (int): Sample rate
            n_mels (int): Number of mel frequency bins
            n_fft (int): FFT window size
            hop_length (int): Hop length for STFT
        
        Returns:
            array: Mel-spectrogram as 3D tensor (freq_bins, time_steps, 1)
        """
        try:
            # Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr, 
                n_mels=n_mels, 
                n_fft=n_fft, 
                hop_length=hop_length
            )
            
            # Convert to log scale (dB)
            log_mel_spec = librosa.power_to_db(mel_spec)
            
            # Normalize
            log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
            
            # Add channel dimension
            log_mel_spec = np.expand_dims(log_mel_spec, axis=-1)
            
            return log_mel_spec
        except Exception as e:
            logger.error(f"Error extracting spectrogram features: {e}")
            raise
    
    def extract_mfcc_spectrogram_features(self, audio, sr=22050, n_mfcc=13, n_mels=128):
        """
        Extract combined MFCC and spectrogram features.
        
        Args:
            audio (array): Audio signal array
            sr (int): Sample rate
            n_mfcc (int): Number of MFCC coefficients
            n_mels (int): Number of mel frequency bins
        
        Returns:
            tuple: (mfcc_features, mel_spectrogram)
        """
        try:
            # Extract MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # Extract mel-spectrogram
            mel_spec = self.extract_spectrogram_features(audio, sr, n_mels)
            
            return mfcc_mean, mel_spec
        except Exception as e:
            logger.error(f"Error extracting combined features: {e}")
            raise
    
    def preprocess_audio_for_cnn(self, audio_file, duration=30, target_shape=None):
        """
        Preprocess audio file for CNN input.
        
        Args:
            audio_file (str): Path to audio file
            duration (int): Duration to load in seconds
            target_shape (tuple): Target shape for spectrogram (if None, uses model input shape)
        
        Returns:
            array: Preprocessed spectrogram
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, duration=duration, sr=22050)
            
            # Extract spectrogram
            spectrogram = self.extract_spectrogram_features(audio, sr)
            
            # Pad or trim to target shape if specified
            if target_shape:
                target_freq, target_time = target_shape[0], target_shape[1]
                current_freq, current_time = spectrogram.shape[0], spectrogram.shape[1]
                
                # Pad or trim frequency dimension
                if current_freq < target_freq:
                    pad_width = target_freq - current_freq
                    spectrogram = np.pad(spectrogram, ((0, pad_width), (0, 0), (0, 0)), mode='constant')
                elif current_freq > target_freq:
                    spectrogram = spectrogram[:target_freq, :, :]
                
                # Pad or trim time dimension
                if current_time < target_time:
                    pad_width = target_time - current_time
                    spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
                elif current_time > target_time:
                    spectrogram = spectrogram[:, :target_time, :]
            
            return spectrogram
        except Exception as e:
            logger.error(f"Error preprocessing audio for CNN: {e}")
            raise
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1):
        """
        Train the CNN model.
        
        Args:
            X_train (array): Training features (spectrograms)
            y_train (array): Training labels
            X_val (array): Validation features
            y_val (array): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            verbose (int): Verbosity level
        
        Returns:
            History: Training history object
        """
        try:
            logger.info(f"Starting CNN training with {len(X_train)} samples")
            
            # Define callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=15,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'models/cnn_best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=False
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=verbose
            )
            
            logger.info(f"Training completed with final validation accuracy: {max(history.history['val_accuracy']):.4f}")
            return history
            
        except Exception as e:
            logger.error(f"Error during CNN training: {e}")
            raise
    
    def predict(self, X):
        """
        Make predictions on input data.
        
        Args:
            X (array): Input spectrograms
        
        Returns:
            array: Prediction probabilities
        """
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            logger.error(f"Error during CNN prediction: {e}")
            raise
    
    def predict_single(self, audio_file):
        """
        Predict genre for a single audio file.
        
        Args:
            audio_file (str): Path to audio file
        
        Returns:
            tuple: (predicted_class, confidence, all_probabilities)
        """
        try:
            # Preprocess audio
            spectrogram = self.preprocess_audio_for_cnn(audio_file)
            spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
            
            # Make prediction
            probabilities = self.predict(spectrogram)[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            return predicted_class, confidence, probabilities
        except Exception as e:
            logger.error(f"Error predicting single audio file: {e}")
            raise
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test (array): Test features
            y_test (array): Test labels
        
        Returns:
            dict: Evaluation metrics
        """
        try:
            results = self.model.evaluate(X_test, y_test, verbose=0)
            metrics = {
                'test_loss': results[0],
                'test_accuracy': results[1],
                'test_top_3_accuracy': results[2] if len(results) > 2 else None
            }
            logger.info(f"Model evaluation completed: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        try:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath):
        """
        Load a pre-trained model.
        
        Args:
            filepath (str): Path to load the model from
        """
        try:
            self.model = tf.keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def create_cnn_ensemble_model(input_shape=(128, 1294, 1), num_classes=10):
    """
    Create an ensemble CNN model with multiple parallel branches.
    
    Args:
        input_shape (tuple): Input shape for spectrograms
        num_classes (int): Number of classes
    
    Returns:
        Model: Ensemble CNN model
    """
    input_layer = layers.Input(shape=input_shape)
    
    # Branch 1: Standard CNN
    branch1 = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    branch1 = layers.BatchNormalization()(branch1)
    branch1 = layers.MaxPooling2D((2, 2))(branch1)
    branch1 = layers.Dropout(0.25)(branch1)
    
    branch1 = layers.Conv2D(64, (3, 3), activation='relu')(branch1)
    branch1 = layers.BatchNormalization()(branch1)
    branch1 = layers.MaxPooling2D((2, 2))(branch1)
    branch1 = layers.Dropout(0.25)(branch1)
    
    # Branch 2: Wider kernels
    branch2 = layers.Conv2D(32, (5, 5), activation='relu')(input_layer)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.MaxPooling2D((2, 2))(branch2)
    branch2 = layers.Dropout(0.25)(branch2)
    
    branch2 = layers.Conv2D(64, (5, 5), activation='relu')(branch2)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.MaxPooling2D((2, 2))(branch2)
    branch2 = layers.Dropout(0.25)(branch2)
    
    # Merge branches
    merged = layers.concatenate([branch1, branch2])
    
    # Common dense layers
    merged = layers.Flatten()(merged)
    merged = layers.Dense(512, activation='relu')(merged)
    merged = layers.BatchNormalization()(merged)
    merged = layers.Dropout(0.5)(merged)
    merged = layers.Dense(256, activation='relu')(merged)
    merged = layers.Dropout(0.5)(merged)
    output = layers.Dense(num_classes, activation='softmax')(merged)
    
    model = models.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']
    )
    
    return model

# Example usage and testing
if __name__ == "__main__":
    print("Testing CNN Music Classifier...")
    
    # Initialize classifier
    cnn_classifier = CNNMusicClassifier(input_shape=(128, 1294, 1), num_classes=10)
    
    # Print model summary
    cnn_classifier.model.summary()
    
    print("CNN Music Classifier initialized successfully!")
    print("Features: Spectrogram-based CNN with batch normalization and dropout")
    print("Architecture: 4 conv blocks + dense layers")
    print("Expected input: Mel-spectrogram (128 freq bins, variable time steps, 1 channel)")
