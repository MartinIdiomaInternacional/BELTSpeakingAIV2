
from typing import Dict
TEMPLATES = {
    "en": {"title":"Speaking Evaluation Report","intro":"This report summarizes your speaking performance aligned to the CEFR.","strengths":"Strengths","areas":"Areas to Improve","next":"Next Steps"},
    "es": {"title":"Reporte de Evaluación de Expresión Oral","intro":"Este reporte resume tu desempeño en expresión oral alineado al MCER.","strengths":"Fortalezas","areas":"Áreas de Mejora","next":"Próximos Pasos"},
}
BULLETS_BY_LEVEL = {
    "A1": {"strengths":["Can produce very short phrases.","Basic personal information."],"areas":["Expand sentence length.","Improve basic verb tenses."],"next":["Practice 3×30s daily monologues."]},
    "B1": {"strengths":["Can narrate experiences.","Connect ideas with simple linkers."],"areas":["Reduce long pauses.","Increase vocabulary range."],"next":["Record weekly 2-min talks; focus on connectors."]},
    "B2": {"strengths":["Clear, extended speech.","Good argumentation."],"areas":["Refine accuracy under pressure.","Control filler words."],"next":["Simulate Q&A to improve spontaneity."]},
    "C1": {"strengths":["Flexible discourse.","Nuanced vocabulary."],"areas":["Pronunciation fine-tuning.","Precision of idioms."],"next":["Targeted drilling on weak phonemes."]},
    "C2": {"strengths":["Near-native control."],"areas":["Style adaptation per audience."],"next":["Practice rhetorical strategies."]},
}
def build_feedback(level: str, lang: str) -> Dict:
    tpl = TEMPLATES.get(lang, TEMPLATES["en"])
    bullets = BULLETS_BY_LEVEL.get(level, BULLETS_BY_LEVEL["B1"])
    return {
        "title": tpl["title"],
        "intro": tpl["intro"],
        "sections": [
            {"title": tpl["strengths"], "items": bullets["strengths"]},
            {"title": tpl["areas"], "items": bullets["areas"]},
            {"title": tpl["next"], "items": bullets["next"]},
        ]
    }
