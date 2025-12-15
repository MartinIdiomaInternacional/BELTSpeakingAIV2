TASK_TIPS = {
    1: "For introductions, examiners look for clear structure (present, past, future) and natural pacing.",
    2: "For situations, strong answers follow a clear story: context → problem → action → result.",
    3: "For opinions, high scores require a clear position, supporting reasons, and logical connectors."
}

DIMENSION_FEEDBACK = {
    "fluency": {
        "high": "Your speech was generally continuous with minimal hesitation.",
        "mid": "Your speech was understandable, but pauses or hesitations affected flow.",
        "low": "Frequent pauses or long hesitations interrupted your speech."
    },
    "coherence": {
        "high": "Ideas were logically connected and easy to follow.",
        "mid": "Your message was clear overall, but transitions between ideas could be smoother.",
        "low": "Ideas were difficult to follow due to weak organization."
    },
    "pronunciation": {
        "high": "Pronunciation was clear and rarely interfered with understanding.",
        "mid": "Some pronunciation issues were noticeable but did not block communication.",
        "low": "Pronunciation problems made understanding difficult at times."
    },
    "lexical_range": {
        "high": "You used a good range of vocabulary appropriate to the task.",
        "mid": "Vocabulary was adequate, though somewhat repetitive.",
        "low": "Limited vocabulary reduced precision of meaning."
    },
    "grammatical_accuracy": {
        "high": "Grammar errors were infrequent and did not affect clarity.",
        "mid": "Grammar errors were noticeable but meaning was usually clear.",
        "low": "Frequent grammar errors made understanding more difficult."
    }
}


def band(score: float):
    if score >= 4.5:
        return "high"
    if score >= 2.5:
        return "mid"
    return "low"


def build_feedback(task_id: int, dimensions: dict):
    strengths = []
    priorities = []
    dimension_feedback = {}

    for dim, payload in dimensions.items():
        score = payload.get("score")
        if score is None:
            continue

        level_band = band(score)
        text = DIMENSION_FEEDBACK.get(dim, {}).get(level_band)
        if text:
            dimension_feedback[dim] = text

        if score >= 4.0:
            strengths.append(dim.replace("_", " "))
        elif score < 3.0:
            priorities.append(dim.replace("_", " "))

    summary = (
        "Your performance shows developing speaking ability with clear strengths "
        "and some areas that need further practice."
        if priorities else
        "Your performance shows solid control across most speaking dimensions."
    )

    return {
        "summary": summary,
        "strengths": strengths,
        "priorities": priorities,
        "dimension_feedback": dimension_feedback,
        "task_tip": TASK_TIPS.get(task_id)
    }
