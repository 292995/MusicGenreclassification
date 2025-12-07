from src.train import train_model
from src.predict import predict_genre

def main():
    # Training mode
    # train_model('data/processed/features.csv', 'models/music_genre_model.h5')
    
    # Prediction mode
    # genre, confidence = predict_genre('path/to/audio.mp3', 'models/music_genre_model.h5', 'models/label_encoder.pkl')
    # print(f"Predicted genre: {genre} (confidence: {confidence:.3f})")
    
    print("Music Genre Classification Project")
    print("Run training: python -m src.train")
    print("Make prediction: python -m src.predict --audio_file path")

if __name__ == "__main__":
    main()
