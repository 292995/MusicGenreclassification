import librosa
import numpy as np
import pandas as pd

def load_audio_file(file_path, duration=30):
    """Load audio file with specified duration"""
    try:
        audio, sr = librosa.load(file_path, duration=duration, sr=22050)
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def normalize_audio(audio):
    """Normalize audio to [-1, 1] range"""
    return librosa.util.normalize(audio)

def trim_silence(audio, top_db=20):
    """Remove silence from beginning and end"""
    return librosa.effects.trim(audio, top_db=top_db)[0]
