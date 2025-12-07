# Data Directory

## Data Structure
- `raw/` - Original audio files (not in repo due to size)
- `processed/` - Processed features (not in repo)

## Dataset Requirements
- Audio files in .mp3, .wav format
- Organized by genre folders
- Each file should be 30 seconds long

## Data Sources
- GTZAN dataset
- Free Music Archive (FMA)
- Million Song Dataset (subset)

## Feature Extraction
Features are extracted using:
- MFCC (13 coefficients)
- Spectral features (centroid, rolloff, bandwidth)
- Chroma features (12 dimensions)
