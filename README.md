
# Music Genre Classification

## Overview
A Python-based machine learning project designed to identify music genres using neural network models. The project achieves 80% accuracy through various preprocessing tasks and algorithm optimization.

## Features
- Audio feature extraction (MFCC, spectral features, etc.)
- Neural network model for genre classification
- Preprocessing pipeline for audio data
- Achieves 80% accuracy on test dataset

## Requirements
- Python 3.7+
- See `requirements.txt` for complete list

## Setup
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Download dataset (see data/README.md)
4. Run training: `python main.py`

## Usage
1. Train model: `python -m src.train`
2. Make predictions: `python -m src.predict --audio_file path/to/audio.mp3`

## Project Structure
- `src/` - Source code modules
- `notebooks/` - Jupyter notebooks for exploration
- `data/` - Raw and processed data (not in repo)
- `models/` - Saved model files (not in repo)

## Results
- Achieved 80% accuracy on test dataset
- Classifies 10 music genres
- Detailed results in `results/` folder
