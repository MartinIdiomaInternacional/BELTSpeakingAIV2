import librosa, numpy as np, torch
from app.config import settings
from app.scoring.audio_features import extract_features

MODEL = None

def evaluate_audio(path):
    audio, sr = librosa.load(path, sr=16000)
    seconds = len(audio) / 16000.0

    if settings.USE_DUMMY_SCORING or MODEL is None:
        norm = np.clip(seconds/10,0,1)
        if norm>0.8:
            lvl=("C1","Highly fluent.","Improve nuance.")
        elif norm>0.6:
            lvl=("B2","Generally fluent.","Work on precision.")
        elif norm>0.4:
            lvl=("B1","Basic communication.","Strengthen structure.")
        else:
            lvl=("A2","Simple communication.","Practice basics.")
        return {"level":lvl[0],"explanation":lvl[1],"recommendations":lvl[2],"seconds":seconds}

    # model path placeholder
    feats = extract_features(audio,sr)
    pred = torch.sigmoid(feats.mean()).item()
    level = "C1" if pred>0.8 else "B2" if pred>0.6 else "B1" if pred>0.4 else "A2"
    return {"level":level,"explanation":"Model placeholder.","recommendations":"N/A","seconds":seconds}
