
import re
def basic_text_metrics(text: str) -> dict:
    if not text:
        return {"wpm_proxy": 0.0, "ttr": 0.0, "tokens": 0}
    words = re.findall(r"[\w'-]+", text.lower())
    tokens = len(words)
    types = len(set(words))
    ttr = types / max(1, tokens)
    return {"tokens": tokens, "unique": types, "ttr": ttr}
