from typing import Optional

from .openai_client import get_client


def transcribe_audio(path: str, language: str = "en") -> Optional[str]:
    """
    Use OpenAI's Whisper-based API to transcribe the audio at *path*.
    Returns the transcript text, or None if something goes wrong.
    """
    client = get_client()

    # NOTE:
    # - You can use "whisper-1" or one of the gpt-4o-mini-transcribe models,
    #   depending on your account and pricing/preferences.
    model_name = "whisper-1"

    try:
        with open(path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model=model_name,
                file=f,
                response_format="text",
                language=language,  # "en" for English
            )
    except Exception as e:
        # In production you might want structured logging instead of print
        print(f"[transcribe_audio] Error during transcription: {e}")
        return None

    # For response_format="text" the API returns a string
    if isinstance(transcription, str):
        return transcription

    # Fallback if API shape changes
    return getattr(transcription, "text", None) or str(transcription)
