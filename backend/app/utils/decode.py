import base64, io, subprocess, tempfile, os
import soundfile as sf

class AudioParseError(Exception):
    pass

def _read_wav_bytes_to_mono_float(b: bytes):
    buf = io.BytesIO(b)
    audio, sr = sf.read(buf, dtype="float32", always_2d=False)
    if hasattr(audio, 'ndim') and audio.ndim > 1:
        import numpy as _np
        audio = _np.mean(audio, axis=1)
    import numpy as _np
    peak = float(max(1e-8, _np.max(_np.abs(audio))))
    audio = audio / peak
    return audio, sr

def decode_base64_maybe_webm_to_wav_mono_float(b64: str):
    raw = base64.b64decode(b64)

    # Try WAV first
    try:
        return _read_wav_bytes_to_mono_float(raw)
    except Exception:
        pass

    # Fallback: ffmpeg decode from webm/opus -> wav
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as fin:
            fin.write(raw)
            fin.flush()
            in_path = fin.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fout:
            out_path = fout.name

        cmd = [
            "ffmpeg", "-y",
            "-i", in_path,
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            out_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(out_path, "rb") as f:
            wav_bytes = f.read()
        audio, sr = _read_wav_bytes_to_mono_float(wav_bytes)
        return audio, sr
    except Exception as e:
        raise AudioParseError(f"Failed to decode audio: {e}")
    finally:
        try:
            if 'in_path' in locals() and os.path.exists(in_path): os.remove(in_path)
            if 'out_path' in locals() and os.path.exists(out_path): os.remove(out_path)
        except Exception:
            pass
