import librosa
import numpy as np
import torch


def extract_features(audio: np.ndarray, sr: int) -> torch.Tensor:
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    return torch.tensor(mfcc_mean, dtype=torch.float32)
