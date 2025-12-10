import json
from typing import Any, Dict

from app.nlp.openai_client import get_client


JSONDict = Dict[str, Any]


def score_transcript_cefr(
    transcript: str,
    task_instruction: str,
    target_language: str = "English",
    model_name: str = "gpt-4o-mini",
) -> JSONDict:
    """
    Use an LLM to evaluate the *transcript* according to CEFR speaking descriptors.

    Returns a JSON-like dict, for example:
    {
      "overall_level": "B2",
      "dimensions": {
        "fluency": "B2",
        "grammatical_range": "B1+",
        "grammatical_accuracy": "B1",
        "lexical_range": "B2",
        "lexical_control": "B1+",
        "pronunciation": "B2-",
        "coherence": "B2"
      },
      "overall_comment": "...",
      "improvement_advice": "...",
      "transcript": "..."
    }
    """
    client = get_client()

    system_prompt = (
        "You are an experienced Cambridge / CEFR speaking examiner. "
        "You evaluate candidates on the CEFR scale (A1, A2, B1, B1+, B2, B2+, C1, C2). "
        "You are strict but fair and you provide concise, practical feedback."
    )

    user_prompt = f"""
You will receive:
1) The task instruction given to the candidate.
2) The candidate's transcript (automatic transcription, may contain minor errors).

Task instruction:
\"\"\"{task_instruction}\"\"\"

Transcript:
\"\"\"{transcript}\"\"\"

Please evaluate the candidate's speaking performance in {target_language} ONLY, ignoring transcription typos if the intended meaning is still clear.

Return a JSON object with the following structure (and nothing else):

{{
  "overall_level": "CEFR level string, e.g. B1+, B2, C1",
  "dimensions": {{
    "fluency": "CEFR level",
    "grammatical_range": "CEFR level",
    "grammatical_accuracy": "CEFR level",
    "lexical_range": "CEFR level",
    "lexical_control": "CEFR level",
    "pronunciation": "CEFR level (perceived from text + hints)",
    "coherence": "CEFR level"
  }},
  "overall_comment": "2–4 sentences summarizing the candidate's strengths and weaknesses.",
  "improvement_advice": "3–6 bullet-style suggestions in plain text separated by line breaks.",
  "transcript": "Cleaned-up transcript if needed, otherwise copy the original."
}}
"""

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    content = completion.choices[0].message.content
    try:
        data: JSONDict = json.loads(content)
    except Exception:
        # If parsing fails, wrap raw content
        data = {
            "overall_level": "N/A",
            "dimensions": {},
            "overall_comment": "Automatic text evaluation failed to parse JSON.",
            "improvement_advice": "",
            "transcript": transcript,
            "raw_response": content,
        }
    return data
