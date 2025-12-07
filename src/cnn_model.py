"""
CNN-based Music Genre Classification Model

This module implements a Convolutional Neural Network for music genre classification
using spectrogram representations of audio data. The CNN architecture is designed
to capture both temporal and frequency patterns in music signals.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import librosa
import logging
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cnn_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CNNMusicClassifier:
    """
    Convolutional Neural Network for music genre classification.
    Uses mel-spectrogram features for improved performance.
    """
    
    def __init__(self, input_shape=(128, 1294, 1), num_classes=10, learning_rate=0.001):
        """
        Initialize the CNN model.
        
        Args:
            input_shape (tuple): Shape of input spectrograms (freq_bins, time_steps, channels)
            num_classes (int): Number of music genres to classify
            learning_rate (float): Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None
        self.label_encoder = None
        self.history = None
        self._build_model()
        logger.info(f"CNN Music Classifier initialized with input shape {input_shape}, {num_classes} classes")
    
    def _build_model(self):
        """Build the CNN architecture."""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling instead of Flatten to reduce parameters
            layers.GlobalAveragePooling2D(),
            
            # Dense layers with batch normalization
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        self.model = model
        logger.info("CNN model architecture built successfully")
    
    def extract_spectrogram_features(self, audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512, duration=30):
        """
        Extract mel-spectrogram features from audio.
        
        Args:
            audio (array): Audio signal array
            sr (int): Sample rate
            n_mels (int): Number of mel frequency bins
            n_fft (int): FFT window size
            hop_length (int): Hop length for STFT
            duration (int): Expected duration in seconds
        
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
    
    def load_and_preprocess_dataset(self, df, max_samples=None):
        """
        Load and preprocess entire dataset for CNN training.
        
        Args:
            df (pd.DataFrame): DataFrame with 'filename' and 'genre' columns
            max_samples (int): Maximum number of samples to process (for testing)
        
        Returns:
            tuple: (X, y) where X is spectrograms and y is encoded labels
        """
        try:
            logger.info(f"Loading and preprocessing dataset with {len(df)} samples")
            
            if max_samples:
                df = df.sample(min(max_samples, len(df)))
                logger.info(f"Limited to {len(df)} samples for processing")
            
            # Initialize label encoder if not already done
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                all_genres = df['genre'].unique()
                self.label_encoder.fit(all_genres)
            
            X = []
            y = []
            
            for idx, row in df.iterrows():
                try:
                    # Load and preprocess audio
                    spectrogram = self.preprocess_audio_for_cnn(row['filename'])
                    X.append(spectrogram)
                    
                    # Encode label
                    encoded_label = self.label_encoder.transform([row['genre']])[0]
                    y.append(encoded_label)
                    
                    if len(X) % 50 == 0:
                        logger.info(f"Processed {len(X)}/{len(df)} samples")
                        
                except Exception as e:
                    logger.warning(f"Error processing {row['filename']}: {e}")
                    continue
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Dataset preprocessing completed: {X.shape} spectrograms, {y.shape} labels")
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading and preprocessing dataset: {e}")
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
                callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=20,
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=1
                ),
                callbacks.ModelCheckpoint(
                    'models/cnn_best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
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
            
            self.history = history
            logger.info(f"Training completed with final validation accuracy: {max(history.history['val_accuracy']):.4f}")
            return history
            
        except Exception as e:
            logger.error(f"Error during CNN training: {e}")
            raise
    
    def train_from_dataframe(self, df, test_size=0.2, val_size=0.2, epochs=100, batch_size=32, max_samples=None):
        """
        Complete training pipeline from DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with 'filename' and 'genre' columns
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of training data for validation
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            max_samples (int): Maximum samples to use (for testing)
        
        Returns:
            History: Training history object
        """
        try:
            # Preprocess dataset
            X, y = self.load_and_preprocess_dataset(df, max_samples)
            
            # Split data
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
            )
            
            logger.info(f"Data split: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")
            
            # Train model
            history = self.train(X_train, y_train, X_val, y_val, epochs, batch_size)
            
            # Evaluate on test set
            test_results = self.evaluate(X_test, y_test)
            logger.info(f"Test results: {test_results}")
            
            return history
            
        except Exception as e:
            logger.error(f"Error in complete training pipeline: {e}")
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
            if len(X.shape) == 3:
                X = np.expand_dims(X, axis=0)  # Add batch dimension if single sample
            
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
            tuple: (predicted_genre, confidence, all_probabilities, genre_probabilities_dict)
        """
        try:
            if self.label_encoder is None:
                raise ValueError("Model must be trained first to have label encoder")
            
            # Preprocess audio
            spectrogram = self.preprocess_audio_for_cnn(audio_file)
            spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
            
            # Make prediction
            probabilities = self.predict(spectrogram)[0]
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]
            
            # Decode label
            predicted_genre = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            # Create genre-probability mapping
            genre_probs = {
                genre: float(prob) 
                for genre, prob in zip(self.label_encoder.classes_, probabilities)
            }
            
            return predicted_genre, confidence, probabilities, genre_probs
            
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
                'test_loss': float(results[0]),
                'test_accuracy': float(results[1]),
                'test_top_3_accuracy': float(results[2]) if len(results) > 2 else None
            }
            logger.info(f"Model evaluation completed: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def get_model_summary(self):
        """
        Get model architecture summary.
        
        Returns:
            str: Model summary as string
        """
        try:
            import io
            stream = io.StringIO()
            self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
            summary_string = stream.getvalue()
            return summary_string
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            raise
    
    def save_model(self, model_path, label_encoder_path=None):
        """
        Save the trained model and label encoder.
        
        Args:
            model_path (str): Path to save the model
            label_encoder_path (str): Path to save the label encoder (optional)
        """
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save label encoder if available
            if self.label_encoder is not None and label_encoder_path:
                os.makedirs(os.path.dirname(label_encoder_path), exist_ok=True)
                with open(label_encoder_path, 'wb') as f:
                    pickle.dump(self.label_encoder, f)
                logger.info(f"Label encoder saved to {label_encoder_path}")
            
            # Save training history if available
            if self.history is not None:
                history_path = model_path.replace('.h5', '_history.pkl')
                with open(history_path, 'wb') as f:
                    pickle.dump(self.history.history, f)
                logger.info(f"Training history saved to {history_path}")
                
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    @classmethod
    def load_model(cls, model_path, label_encoder_path=None, input_shape=(128, 1294, 1), num_classes=10):
        """
        Load a pre-trained model.
        
        Args:
            model_path (str): Path to load the model from
            label_encoder_path (str): Path to load the label encoder from (optional)
            input_shape (tuple): Input shape of the model
            num_classes (int): Number of classes in the model
        
        Returns:
            CNNMusicClassifier: Loaded model instance
        """
        try:
            # Create new instance
            instance = cls(input_shape=input_shape, num_classes=num_classes)
            
            # Load model
            instance.model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            # Load label encoder if provided
            if label_encoder_path and os.path.exists(label_encoder_path):
                with open(label_encoder_path, 'rb') as f:
                    instance.label_encoder = pickle.load(f)
                logger.info(f"Label encoder loaded from {label_encoder_path}")
            
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def create_advanced_cnn_ensemble_model(input_shape=(128, 1294, 1), num_classes=10):
    """
    Create an advanced ensemble CNN model with multiple parallel branches.
    
    Args:
        input_shape (tuple): Input shape for spectrograms
        num_classes (int): Number of classes
    
    Returns:
        Model: Advanced ensemble CNN model
    """
    input_layer = layers.Input(shape=input_shape)
    
    # Branch 1: Standard CNN with residual connections
    branch1_input = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    branch1 = layers.BatchNormalization()(branch1_input)
    branch1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(branch1)
    branch1 = layers.BatchNormalization()(branch1)
    branch1 = layers.Add()([branch1_input, branch1])  # Residual connection
    branch1 = layers.MaxPooling2D((2, 2))(branch1)
    branch1 = layers.Dropout(0.25)(branch1)
    
    # Branch 2: Wider kernels
    branch2 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(branch2)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.MaxPooling2D((2, 2))(branch2)
    branch2 = layers.Dropout(0.25)(branch2)
    
    # Branch 3: Dilated convolutions
    branch3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=2)(input_layer)
    branch3 = layers.BatchNormalization()(branch3)
    branch3 = layers.MaxPooling2D((2, 2))(branch3)
    branch3 = layers.Dropout(0.25)(branch3)
    
    # Merge branches
    merged = layers.concatenate([branch1, branch2, branch3])
    
    # Additional conv layers on merged features
    merged = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merged)
    merged = layers.BatchNormalization()(merged)
    merged = layers.MaxPooling2D((2, 2))(merged)
    merged = layers.Dropout(0.25)(merged)
    
    merged = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(merged)
    merged = layers.BatchNormalization()(merged)
    merged = layers.MaxPooling2D((2, 2))(merged)
    merged = layers.Dropout(0.25)(merged)
    
    # Global average pooling and dense layers
    merged = layers.GlobalAveragePooling2D()(merged)
    merged = layers.Dense(512, activation='relu')(merged)
    merged = layers.BatchNormalization()(merged)
    merged = layers.Dropout(0.5)(merged)
    merged = layers.Dense(256, activation='relu')(merged)
    merged = layers.BatchNormalization()(merged)
    merged = layers.Dropout(0.5)(merged)
    merged = layers.Dense(128, activation='relu')(merged)
    merged = layers.Dropout(0.3)(merged)
    output = layers.Dense(num_classes, activation='softmax')(merged)
    
    model = models.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']
    )
    
    return model

def create_transfer_learning_model(input_shape=(128, 1294, 1), num_classes=10, base_model_name='ResNet50'):
    """
    Create a transfer learning model using pre-trained CNN.
    
    Args:
        input_shape (tuple): Input shape for spectrograms
        num_classes (int): Number of classes
        base_model_name (str): Name of base model (currently ResNet50 adapted)
    
    Returns:
        Model: Transfer learning model
    """
    # Note: For audio, we'll adapt ResNet by reshaping input
    # In practice, you might want to use models specifically designed for audio
    
    input_layer = layers.Input(shape=input_shape)
    
    # First, adapt input for ResNet-like architecture
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # ResNet-style blocks
    def residual_block(x, filters, stride=1):
        shortcut = x
        x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3, 3), padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)
        
        if stride != 1:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, activation=None)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x
    
    # Apply residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    
    # Global average pooling and final layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']
    )
    
    return model

def main():
    """Main function for testing the CNN model."""
    print("Testing CNN Music Classifier...")
    
    # Initialize classifier
    cnn_classifier = CNNMusicClassifier(input_shape=(128, 1294, 1), num_classes=10)
    
    # Print model summary
    print("\nModel Architecture:")
    print(cnn_classifier.get_model_summary())
    
    # Example of creating advanced models
    print("\nCreating advanced ensemble model...")
    advanced_model = create_advanced_cnn_ensemble_model()
    print("Advanced model created successfully!")
    
    print("\nCNN Music Classifier initialized successfully!")
    print("Features: Spectrogram-based CNN with batch normalization and dropout")
    print("Architecture: 4 conv blocks + global average pooling + dense layers")
    print("Expected input: Mel-spectrogram (128 freq bins, variable time steps, 1 channel)")

if __name__ == "__main__":
    main()
