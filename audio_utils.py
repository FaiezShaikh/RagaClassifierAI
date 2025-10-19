
import librosa
import numpy as np

def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    tempo, _ = librosa.beat.beat_track(y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))
    return {
        "tempo": round(tempo, 2),
        "pitch": round(pitch, 2),
        "mfcc_mean": round(np.mean(mfccs), 2)
    }
