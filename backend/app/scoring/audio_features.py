import librosa, numpy as np, torch

def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return torch.tensor(mfcc.mean(axis=1), dtype=torch.float32)
