
from typing import Dict, List

PROMPTS: Dict[str, List[dict]] = {
    "A1": [
        {"id":"A1-1","text":"Describe your daily routine in 30 seconds."},
        {"id":"A1-2","text":"Talk about your family and where you live."},
        {"id":"A1-3","text":"What do you usually eat for breakfast, lunch, and dinner?"},
    ],
    "A2": [
        {"id":"A2-1","text":"Explain what you did last weekend and what you will do next weekend."},
        {"id":"A2-2","text":"Describe a favorite place in your city and why you like it."},
        {"id":"A2-3","text":"Tell us about a simple recipe you know and how to prepare it."},
    ],
    "B1": [
        {"id":"B1-1","text":"Describe a challenge at work and how you solved it."},
        {"id":"B1-2","text":"Explain a problem you had while traveling and what you did."},
        {"id":"B1-3","text":"Tell us about a time you learned a new skill and how you practiced."},
    ],
    "B1+": [
        {"id":"B1p-1","text":"Compare two ways your team could improve customer response times."},
        {"id":"B1p-2","text":"Describe two productivity methods you’ve tried and their trade-offs."},
        {"id":"B1p-3","text":"Explain how you would organize a small event with limited budget."},
    ],
    "B2": [
        {"id":"B2-1","text":"Present a proposal to optimize a process, including pros and cons."},
        {"id":"B2-2","text":"Evaluate two tools for your team and recommend one with justification."},
        {"id":"B2-3","text":"Discuss a recent industry change and its impact on your role."},
    ],
    "B2+": [
        {"id":"B2p-1","text":"Argue for or against remote work policies in your company context."},
        {"id":"B2p-2","text":"Debate the risks and benefits of adopting a new technology this year."},
        {"id":"B2p-3","text":"Propose criteria to measure success in a cross‑functional project."},
    ],
    "C1": [
        {"id":"C1-1","text":"Summarize a complex project you led and reflect on trade-offs."},
        {"id":"C1-2","text":"Explain a controversial decision at work, addressing counterarguments."},
        {"id":"C1-3","text":"Outline a change-management plan and anticipated risks."},
    ],
    "C2": [
        {"id":"C2-1","text":"Critically evaluate a market strategy and propose an alternative."},
        {"id":"C2-2","text":"Deliver a persuasive pitch that reframes an entrenched viewpoint."},
        {"id":"C2-3","text":"Synthesize insights from conflicting reports to advise executives."},
    ],
}
