import os
from typing import Optional

from openai import OpenAI

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """
    Lazily create and cache a single OpenAI client using the OPENAI_API_KEY
    environment variable. Raises a clear error if the key is missing.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it in your backend Render service before enabling ASR/NLP scoring."
            )
        _client = OpenAI(api_key=api_key)
    return _client
