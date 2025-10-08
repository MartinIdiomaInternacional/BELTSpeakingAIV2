
from typing import Dict, List

PROMPTS: Dict[str, List[dict]] = {
    "A1": [{"id":"A1-1","text":"Describe your daily routine in 30 seconds."}],
    "A2": [{"id":"A2-1","text":"Explain what you did last weekend and what you will do next weekend."}],
    "B1": [{"id":"B1-1","text":"Describe a challenge at work and how you solved it."}],
    "B1+": [{"id":"B1p-1","text":"Compare two ways your team could improve customer response times."}],
    "B2": [{"id":"B2-1","text":"Present a proposal to optimize a process, including pros and cons."}],
    "B2+": [{"id":"B2p-1","text":"Argue for or against remote work policies in your company context."}],
    "C1": [{"id":"C1-1","text":"Summarize a complex project you led and reflect on trade-offs."}],
    "C2": [{"id":"C2-1","text":"Critically evaluate a market strategy and propose an alternative."}],
}
