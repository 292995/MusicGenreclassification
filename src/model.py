import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape, num_classes=10):
    """Create neural network model for genre classification"""
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
