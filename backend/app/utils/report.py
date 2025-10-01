from typing import Dict
from .language_feedback import build_feedback

HTML_TMPL = '''<!doctype html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>body{font-family:system-ui,Segoe UI,Arial;margin:24px;}h1{font-size:22px}h2{font-size:18px}ul{line-height:1.5}</style>
</head><body>
<h1>{title}</h1>
<p><strong>Session:</strong> {session_id} · <strong>Level:</strong> {level} · <strong>Score:</strong> {score:.2f} · <strong>Confidence:</strong> {conf:.2f}</p>
<p>{intro}</p>
{sections}
</body></html>'''

def render_html(session_id: str, level: str, score: float, conf: float, lang: str) -> Dict:
    fb = build_feedback(level, lang)
    sect_html = "".join([
        f"<h2>{s['title']}</h2><ul>" + "".join([f"<li>{item}</li>" for item in s['items']]) + "</ul>"
        for s in fb["sections"]
    ])
    html = HTML_TMPL.format(
        title=fb["title"],
        session_id=session_id,
        level=level,
        score=score,
        conf=conf,
        intro=fb["intro"],
        sections=sect_html,
    )
    return {"feedback": fb, "html": html}
