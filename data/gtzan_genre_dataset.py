"""
Data preparation script for GTZAN dataset
This script creates the required CSV file for training
"""
import os
import pandas as pd

def create_dataset_csv(dataset_path, output_path):
    """
    Create CSV file with filename and genre columns
    Assumes dataset structure: dataset_path/genre/*.au
    """
    data = []
    
    for genre in os.listdir(dataset_path):
        genre_path = os.path.join(dataset_path, genre)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                if file.endswith(('.au', '.wav', '.mp3')):
                    full_path = os.path.join(genre_path, file)
                    data.append({
                        'filename': full_path,
                        'genre': genre
                    })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Created dataset CSV with {len(df)} samples")
    print(f"Genres: {df['genre'].unique()}")

if __name__ == "__main__":
    # Example usage (uncomment with actual dataset path)
    # create_dataset_csv('/path/to/gtzan', 'data/processed/features.csv')
    pass
