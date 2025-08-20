import os, json, tempfile, traceback, re, math
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple, List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import numpy as np
import librosa

# -------------------- Meta --------------------
APP_VERSION = "BELT Speaking Test API v1.1 (CEFR 5-dim + richer fluency/pron + band confidence)"
SAFE_MODE   = os.getenv("BELT_SAFE_MODE", "0") == "1"
TEEN_TONE   = os.getenv("TEEN_TONE", "0") == "1"

# -------------------- Weights (0..1) --------------------
W_REL  = float(os.getenv("W_REL",  "0.20"))     # relevance/coherence to question
W_FLU  = float(os.getenv("W_FLU",  "0.30"))     # fluency
W_GRA  = float(os.getenv("W_GRA",  "0.20"))     # grammar
W_PRO  = float(os.getenv("W_PRO",  "0.15"))     # pronunciation (proxies)
W_VOC  = float(os.getenv("W_VOC",  "0.15"))     # vocabulary range/variety

# -------------------- CEFR thresholds (0..1) --------------------
T_A2 = float(os.getenv("T_A2", "0.25"))
T_B1 = float(os.getenv("T_B1", "0.40"))
T_B2 = float(os.getenv("T_B2", "0.60"))
T_C1 = float(os.getenv("T_C1", "0.75"))

# -------------------- ASR / Embedding / Feedback config --------------------
USE_OPENAI_ASR       = os.getenv("USE_OPENAI_ASR", "0") == "1"
OPENAI_ASR_MODEL     = os.getenv("OPENAI_ASR_MODEL", "whisper-1")

USE_OPENAI_EMBED     = os.getenv("USE_OPENAI_EMBED", "1") == "1"
OPENAI_EMBED_MODEL   = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

USE_OPENAI_GRAMMAR   = os.getenv("USE_OPENAI_GRAMMAR", "0") == "1"    # optional LLM grammar rater (0..1)
OPENAI_GRAMMAR_MODEL = os.getenv("OPENAI_GRAMMAR_MODEL", "gpt-4o-mini")

USE_LLM_FEEDBACK     = os.getenv("USE_LLM_FEEDBACK", "0") == "1"
OPENAI_FEEDBACK_MODEL= os.getenv("OPENAI_FEEDBACK_MODEL", "gpt-4o-mini")

PROMPTS_PATH         = Path(os.getenv("PROMPTS_PATH", "prompts.json"))

# -------------------- App --------------------
app = FastAPI(title="BELT Speaking Test API", version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# -------------------- Prompts --------------------
@lru_cache(maxsize=1)
def load_prompts():
    if not PROMPTS_PATH.exists(): return {}
    try:
        data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
        return {str(k).strip().upper(): v for k, v in data.items()}
    except Exception:
        return {}

@app.get("/prompts/{level}")
async def get_prompts(level: str):
    prompts = load_prompts()
    lvl = level.strip().upper()
    if lvl not in prompts:
        raise HTTPException(status_code=404, detail=f"No prompts found for level '{level}'.")
    return {"level": lvl, "count": len(prompts[lvl]), "questions": prompts[lvl]}

# -------------------- Audio utils --------------------
def load_audio_to_16k_mono(path: str):
    y, sr = librosa.load(path, sr=16000, mono=True)
    if y.size == 0: return np.zeros((1,), dtype=np.float32), 16000
    return y.astype(np.float32), 16000

def voice_intervals(y: np.ndarray, top_db: float = 30.0) -> List[Tuple[int,int]]:
    # Voiced segments in samples
    return librosa.effects.split(y, top_db=top_db)

def speech_activity_metrics(y: np.ndarray, sr: int = 16000, intervals: Optional[List[Tuple[int,int]]]=None):
    duration = len(y) / sr if len(y) else 0.0
    if intervals is None:
        intervals = voice_intervals(y)
    if len(y) == 0:
        return {"duration_sec": 0.0, "voiced_ratio": 0.0, "num_segments": 0,
                "avg_segment_sec": 0.0, "long_pauses": 0, "silence_pct": 1.0}, intervals
    voiced = sum((e - s) for s, e in intervals) / len(y)
    seg_durs = [ (e - s) / sr for s, e in intervals ] if len(intervals) else []
    avg_seg = float(np.mean(seg_durs)) if seg_durs else 0.0
    long_pauses = 0
    if len(intervals) > 1:
        for i in range(1, len(intervals)):
            gap = (intervals[i][0] - intervals[i-1][1]) / sr
            if gap >= 0.7: long_pauses += 1
    silence_pct = float(1.0 - voiced)
    return {"duration_sec": float(duration), "voiced_ratio": float(voiced), "num_segments": int(len(intervals)),
            "avg_segment_sec": avg_seg, "long_pauses": int(long_pauses), "silence_pct": silence_pct}, intervals

# -------------------- Optional SSL features (not used directly in v1.1 score) --------------------
_SSL_MODEL = None
_SSL_MODEL_NAME = None
def get_ssl_embedding(y_16k: np.ndarray):
    if SAFE_MODE:
        return None, None, "SAFE_MODE_DISABLED_SSL"
    import torch, torchaudio
    global _SSL_MODEL, _SSL_MODEL_NAME
    if _SSL_MODEL is None:
        try:
            b = torchaudio.pipelines.WAV2VEC2_BASE
            _SSL_MODEL = b.get_model().eval(); _SSL_MODEL_NAME = "WAV2VEC2_BASE"
        except Exception:
            b = torchaudio.pipelines.HUBERT_BASE
            _SSL_MODEL = b.get_model().eval(); _SSL_MODEL_NAME = "HUBERT_BASE"
    wav = __import__("torch").tensor(y_16k, dtype=__import__("torch").float32).unsqueeze(0)
    with __import__("torch").no_grad():
        try:
            feats = _SSL_MODEL.extract_features(wav)[0][-1]
            if feats.dim() == 3: feats = feats.squeeze(0)
        except Exception:
            out = _SSL_MODEL(wav); feats = (out[0] if isinstance(out,(tuple,list)) else out).squeeze(0)
    var = feats.var(dim=0).mean().item()
    emb = feats.mean(dim=0).cpu().numpy()
    return emb, float(var), _SSL_MODEL_NAME

# -------------------- ASR --------------------
def _get_asr_mode():
    if USE_OPENAI_ASR and os.getenv("OPENAI_API_KEY"): return "openai"
    if not SAFE_MODE: return "local"
    return "none"

_ASR_MODEL = None
_ASR_BUNDLE = None
def _local_asr(y_16k: np.ndarray):
    # Returns (transcript, mean_max_prob) where mean_max_prob approximates ASR confidence for pronunciation proxy
    try:
        import torch, torchaudio
        global _ASR_MODEL, _ASR_BUNDLE
        if _ASR_MODEL is None:
            _ASR_BUNDLE = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            _ASR_MODEL = _ASR_BUNDLE.get_model().eval()
        wav = torch.tensor(y_16k, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emissions, _ = _ASR_MODEL(wav)          # (B,T,V)
            probs = emissions.softmax(dim=-1).max(dim=-1).values.mean().item()  # pronunciation proxy
            idx = emissions[0].argmax(dim=-1).tolist()
        labels = _ASR_BUNDLE.get_labels()
        blank = 0
        tokens, last = [], None
        for i in idx:
            if i != blank and i != last: tokens.append(labels[i])
            last = i
        text = "".join(tokens).replace("|"," ").strip().lower() or None
        return text, float(probs)
    except Exception:
        return None, None

def _openai_whisper(tmp_path: str):
    # Returns (transcript, None) — Whisper API does not expose token confidences
    try:
        from openai import OpenAI
        client = OpenAI()
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(model=OPENAI_ASR_MODEL, file=f)
        text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
        return (str(text).strip().lower()) if text else None, None
    except Exception:
        return None, None

def get_transcript(y_16k: np.ndarray, tmp_path: str):
    mode = _get_asr_mode()
    if mode == "openai":
        t, conf = _openai_whisper(tmp_path)
        if t: return t, "openai", conf
        if not SAFE_MODE:
            t, conf = _local_asr(y_16k)
            if t: return t, "local-fallback", conf
        return None, "none", None
    if mode == "local":
        t, conf = _local_asr(y_16k)
        return (t, "local", conf) if t else (None, "none", None)
    return None, "none", None

# -------------------- Text helpers & features --------------------
STOP = {"the","a","an","to","of","in","on","for","with","and","or","but","is","are","was","were","be","am","as","at","by",
        "it","this","that","i","you","he","she","they","we","my","your","our","me","him","her","them","us"}
FILLERS = {"um","uh","er","eh","hmm","like","you","know","kind","of","sort","of"}
SMALL_3SG = {"go","do","have","want","like","need","think","say","work","live","study","make","take","know","talk","use"}

def _normalize(v, lo, hi): return max(0.0, min(1.0, (v - lo) / (hi - lo))) if hi > lo else 0.0
def _tokens(t: str): return [w for w in re.findall(r"[a-zA-Z']+", (t or "").lower())]
def _content_tokens(t: str): return [w for w in _tokens(t) if w not in STOP]

# --- Syllable approximation & extra fluency features ---
_vowel_groups = re.compile(r"[aeiouy]+", re.I)
def approx_syllables(text: str) -> int:
    if not text: return 0
    words = re.findall(r"[a-zA-Z']+", text.lower())
    syl = 0
    for w in words:
        groups = _vowel_groups.findall(w)
        syl += max(1, len(groups))
    return syl

def articulation_rate_sps(transcript: str, voiced_seconds: float) -> float:
    syl = approx_syllables(transcript)
    return float(syl / max(0.5, voiced_seconds))  # syllables per second

def mean_length_of_run(intervals: List[Tuple[int,int]], sr: int) -> float:
    if not intervals: return 0.0
    durs = [(e - s) / sr for s, e in intervals]
    return float(np.mean(durs)) if durs else 0.0

def pitch_stats(y: np.ndarray, sr: int):
    try:
        f0, vflag, _ = librosa.pyin(y, fmin=80, fmax=400, sr=sr)
        f0 = f0[~np.isnan(f0)]
        if f0.size == 0: return {"f0_med": 0.0, "f0_iqr": 0.0}
        q1, med, q3 = np.quantile(f0, [0.25, 0.5, 0.75])
        return {"f0_med": float(med), "f0_iqr": float(q3 - q1)}
    except Exception:
        return {"f0_med": 0.0, "f0_iqr": 0.0}

# ---- Text features for fluency ----
def text_features(transcript: str, duration_sec: float):
    if not transcript or duration_sec <= 0:
        return {"words":0,"wpm":0.0,"lexical_diversity":0.0,"filler_ratio":0.0}
    words = _tokens(transcript)
    total = len(words); uniq = len(set(words))
    fillers = sum(1 for w in words if w in FILLERS)
    minutes = max(1e-6, duration_sec / 60.0)
    return {
        "words": total,
        "wpm": float(total / minutes),
        "lexical_diversity": float((uniq/total) if total else 0.0),
        "filler_ratio": float((fillers/total) if total else 0.0)
    }

def audio_fluency_score(m):
    vr_n   = _normalize(m["voiced_ratio"], 0.20, 0.80)
    seg_n  = _normalize(m["avg_segment_sec"], 0.30, 2.00)
    pause_n= 1.0 - _normalize(m["long_pauses"], 1, 6)
    return 0.45*vr_n + 0.35*seg_n + 0.20*pause_n

def text_fluency_score(tf):
    wpm_n = _normalize(tf["wpm"], 70, 170)
    lex_n = _normalize(tf["lexical_diversity"], 0.30, 0.65)
    filler_penalty = 1.0 - _normalize(tf["filler_ratio"], 0.02, 0.12)
    return 0.55*wpm_n + 0.30*lex_n + 0.15*filler_penalty

def fluency_extra_features(transcript: Optional[str], m: dict, intervals: List[Tuple[int,int]], sr: int):
    voiced_sec = sum((e - s)/sr for s, e in intervals) if intervals else max(0.2, m["voiced_ratio"]*m["duration_sec"])
    art_rate = articulation_rate_sps(transcript or "", voiced_sec)  # syl/sec
    mlr_sec  = mean_length_of_run(intervals, sr)                    # sec
    sil_pct  = float(m.get("silence_pct", 1.0 - m.get("voiced_ratio",0.0)))
    # Normalize (tune ranges with pilot data)
    art_n = _normalize(art_rate, 3.0, 5.2)           # good band ~3.5–5.0 syl/s
    mlr_n = _normalize(mlr_sec, 0.8, 2.2)            # mean run length 0.8–2.2s
    sil_n = 1.0 - _normalize(sil_pct, 0.20, 0.50)    # prefer ~20–35% silence
    return float(0.35*art_n + 0.35*mlr_n + 0.30*sil_n), {
        "articulation_sps": float(art_rate),
        "mlr_sec": float(mlr_sec),
        "silence_pct": float(sil_pct)
    }

def fluency_score(m, transcript, duration_sec, intervals, sr):
    tf = text_features(transcript, duration_sec) if transcript else {"words":0,"wpm":0,"lexical_diversity":0,"filler_ratio":0}
    a = audio_fluency_score(m)
    t = text_fluency_score(tf)
    base = float(0.5*a + 0.5*t)
    extra, extra_stats = fluency_extra_features(transcript, m, intervals, sr)
    return float(0.6*base + 0.4*extra), tf, extra_stats

# ---- Relevance (semantic) ----
def relevance_heuristic(q: str, a: str) -> float:
    qset = set(_content_tokens(q)); aset = set(_content_tokens(a))
    if not qset or not aset: return 0.0
    inter = len(qset & aset); union = len(qset | aset)
    jacc = inter/union; recall = inter/len(qset)
    return float(0.6*jacc + 0.4*recall)

def relevance_openai(q: str, a: str) -> float:
    try:
        from openai import OpenAI
        client = OpenAI()
        m = OPENAI_EMBED_MODEL
        qv = client.embeddings.create(model=m, input=q).data[0].embedding
        av = client.embeddings.create(model=m, input=a).data[0].embedding
        dot = sum(x*y for x,y in zip(qv,av))
        nq = math.sqrt(sum(x*x for x in qv)); na = math.sqrt(sum(x*x for x in av))
        return float(dot/(nq*na + 1e-9))
    except Exception:
        return relevance_heuristic(q, a)

def relevance_score(question: Optional[str], transcript: Optional[str]) -> float:
    if not question or not transcript: return 0.0
    if USE_OPENAI_EMBED and os.getenv("OPENAI_API_KEY"):
        return relevance_openai(question, transcript)
    return relevance_heuristic(question, transcript)

# ---- Grammar (heuristic + optional LLM) ----
def grammar_analysis(transcript: str) -> Dict[str, int]:
    txt = transcript or ""
    errors = {
        "repetition": 0,
        "run_on": 0,
        "sv_agreement": 0,
        "aux_mismatch": 0,
        "irregular_past": 0,
        "article_mismatch": 0
    }
    if not txt.strip():
        return errors

    # repeated words
    errors["repetition"] += len(re.findall(r"\b(\w+)\s+\1\b", txt, flags=re.IGNORECASE))

    # run-on / missing punctuation if long w/out .,?! 
    words = _tokens(txt)
    if len(words) > 40 and not re.search(r"[.!?]", txt): errors["run_on"] += 1

    # simple subject-verb agreement
    errors["sv_agreement"] += len(re.findall(
        r"\b(he|she|it)\s+(go|do|have|want|like|need|think|say|work|live|study|make|take|know|talk|use)\b", txt, flags=re.IGNORECASE))
    errors["aux_mismatch"] += len(re.findall(r"\b(I|you|we|they)\s+was\b", txt, flags=re.IGNORECASE))
    errors["aux_mismatch"] += len(re.findall(r"\b(he|she|it)\s+were\b", txt, flags=re.IGNORECASE))

    # very common irregular past mistakes
    errors["irregular_past"] += len(re.findall(r"\bgoed\b|\bbuyed\b|\brunned\b", txt, flags=re.IGNORECASE))

    # article a/an mismatch (very rough)
    errors["article_mismatch"] += len(re.findall(r"\ba\s+[aeiouAEIOU]\w+", txt))
    errors["article_mismatch"] += len(re.findall(r"\ban\s+[^aeiouAEIOU\W]\w+", txt))

    return errors

def grammar_heuristic(transcript: str) -> Tuple[float, Dict[str, int], float]:
    words = _tokens(transcript)
    n = max(1, len(words))
    errs = grammar_analysis(transcript)
    total_errs = sum(errs.values())
    # error density per 100 words
    density = (total_errs / n) * 100.0
    # fewer errors per length => higher score; tune range with pilot data
    score = max(0.0, min(1.0, 1.0 - _normalize(density, 4.0, 18.0)))
    return float(score), errs, float(density)

def grammar_openai_score(transcript: str) -> Optional[float]:
    if not (USE_OPENAI_GRAMMAR and os.getenv("OPENAI_API_KEY") and transcript): return None
    try:
        from openai import OpenAI
        client = OpenAI()
        prompt = ("Rate the grammatical accuracy of the following spoken English transcript on a 0..1 scale "
                  "(0=very inaccurate, 1=native-like). Only return the number.\n\n" + transcript)
        resp = client.chat.completions.create(
            model=OPENAI_GRAMMAR_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0, max_tokens=5
        )
        text = resp.choices[0].message.content.strip()
        m = re.findall(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", text)
        if not m: return None
        val = float(m[0])
        return max(0.0, min(1.0, val))
    except Exception:
        return None

def grammar_score(transcript: str) -> Tuple[float, Dict[str,int], float]:
    llm = grammar_openai_score(transcript)
    base, errs, density = grammar_heuristic(transcript)
    return (llm if llm is not None else base), errs, density

# ---- Pronunciation proxies ----
def pronunciation_from_asr_conf(conf: Optional[float]) -> Optional[float]:
    if conf is None: return None
    return float(_normalize(conf, 0.35, 0.92))

def pronunciation_from_audio(y: np.ndarray, sr: int):
    try:
        flat = librosa.feature.spectral_flatness(y=y)[0]
        inv_flat = 1.0 - np.nanmean(flat)
        inv_n = _normalize(inv_flat, 0.05, 0.65)
        f0stats = pitch_stats(y, sr)
        voiced_proxy = _normalize(f0stats["f0_iqr"], 20, 120)  # more range usually clearer prosody
        return float(0.55*voiced_proxy + 0.45*inv_n), f0stats
    except Exception:
        return None, {"f0_med":0.0,"f0_iqr":0.0}

def pronunciation_score(y: np.ndarray, sr: int, m, asr_conf: Optional[float]):
    s_asr = pronunciation_from_asr_conf(asr_conf)
    s_audio, f0stats = pronunciation_from_audio(y, sr) if y is not None and y.size else (None, {"f0_med":0.0,"f0_iqr":0.0})
    if s_asr is not None and s_audio is not None:
        s = float(0.6*s_asr + 0.4*s_audio)
    elif s_asr is not None:
        s = s_asr
    elif s_audio is not None:
        s = s_audio
    else:
        s = float(_normalize(m["voiced_ratio"], 0.25, 0.85))
    # small nudge with pitch range (helps when ASR conf flatlines)
    pron_extra = 0.25 * _normalize(f0stats.get("f0_iqr",0.0), 20, 120)
    s = float(0.9*s + 0.1*pron_extra)
    return s, f0stats

# ---- Vocabulary (variety + sophistication + light POS diversity + bigram variety) ----
def pos_diversity_heuristic(tokens: List[str]) -> Dict[str, float]:
    # ultra-light POS guesser (no heavy models): adjectives by common suffixes; adverbs by -ly; verbs by small list/patterns
    if not tokens:
        return {"noun_like":0,"verb_like":0,"adj_like":0,"adv_like":0,"diversity":0.0}
    verbs_common = {"be","is","are","was","were","am","have","has","had","do","did","does","make","go","say","see","get","know","think","take","give","find","tell","work","call","try","ask","need","feel","become","leave","put","keep","let","begin","seem","help","talk","turn","start","show","hear","play","run","move","live","believe","bring","happen","write","provide","sit","stand","lose","pay","meet","include","continue","set","learn","change","lead","watch","follow","stop","create","speak","read","allow","add","spend","grow","open","walk","win","offer","remember","love","consider"}
    adj_suffix = ("ous","ful","able","ible","al","ive","ic","ish","less","y","ary","ory","ant","ent","ate","ed","ing")
    adv_suffix = ("ly",)

    noun_like=verb_like=adj_like=adv_like=0
    for w in tokens:
        if len(w)<=2: continue
        if w.endswith(adv_suffix): adv_like+=1; continue
        if w in verbs_common or w=="to": verb_like+=1; continue
        if w.endswith(adj_suffix): adj_like+=1; continue
        noun_like+=1

    total = noun_like+verb_like+adj_like+adv_like
    types_used = sum(1 for x in [noun_like,verb_like,adj_like,adv_like] if x>0)
    diversity = types_used/4.0
    return {"noun_like":noun_like/total if total else 0.0,
            "verb_like":verb_like/total if total else 0.0,
            "adj_like":adj_like/total if total else 0.0,
            "adv_like":adv_like/total if total else 0.0,
            "diversity":diversity}

def vocabulary_score(transcript: str):
    if not transcript: 
        return 0.0, {"ttr":0.0,"herdan_c":0.0,"adv_ratio":0.0,"pos_diversity":0.0,"bigram_variety":0.0}
    toks = _content_tokens(transcript)
    n = len(toks); v = len(set(toks))
    # Type-token & Herdan's C (robust at different lengths)
    ttr = (v/n) if n else 0.0
    herdan_c = (math.log(max(1,v))/math.log(max(2,n))) if n>1 else 0.0

    # wordfreq Zipf to estimate advanced vocab ratio
    adv_ratio = 0.0
    try:
        from wordfreq import zipf_frequency
        zs = [zipf_frequency(w, "en") for w in toks]
        adv = [z for z in zs if z > 0 and z < 3.5]  # lower Zipf = rarer
        adv_ratio = (len(adv)/len(zs)) if zs else 0.0
    except Exception:
        pass

    # POS diversity (lightweight)
    posd = pos_diversity_heuristic(toks)
    pos_div = posd["diversity"]

    # Bigram variety (collocation originality proxy)
    bigrams = list(zip(toks, toks[1:])) if n>1 else []
    uniq_bi = len(set(bigrams))
    bigram_variety = (uniq_bi / max(1,len(bigrams))) if bigrams else 0.0

    # Normalize to 0..1
    ttr_n = _normalize(ttr, 0.30, 0.70)
    herdan_n = _normalize(herdan_c, 0.65, 1.10)
    adv_n = _normalize(adv_ratio, 0.05, 0.30)
    pos_n = _normalize(pos_div, 0.35, 0.85)
    bigram_n = _normalize(bigram_variety, 0.60, 0.90)

    score = float(0.35*ttr_n + 0.20*herdan_n + 0.25*adv_n + 0.10*pos_n + 0.10*bigram_n)
    return score, {"ttr":float(ttr), "herdan_c":float(herdan_c), "adv_ratio":float(adv_ratio),
                   "pos_diversity":float(pos_div), "bigram_variety":float(bigram_variety)}

# ---- CEFR mapping & confidence ----
BANDS = ["A1-A2","A2-B1","B1-B2","B2-C1","C1-C2"]
def band_centers_and_edges():
    edges = [0.0, T_A2, T_B1, T_B2, T_C1, 1.0]
    centers = [ (edges[i]+edges[i+1])/2.0 for i in range(5) ]
    return centers, edges

def band_probabilities(score: float, sigma: float = 0.12) -> Dict[str, float]:
    centers, _ = band_centers_and_edges()
    # Gaussian kernel around band centers
    exps = [math.exp(-((score - c)**2)/(2*sigma*sigma)) for c in centers]
    s = sum(exps) + 1e-9
    probs = [e/s for e in exps]
    return {BANDS[i]: float(probs[i]) for i in range(5)}

def map_score_to_level(score):
    if score < T_A2:   return "A1-A2 (provisional)"
    if score < T_B1:   return "A2-B1 (provisional)"
    if score < T_B2:   return "B1-B2 (provisional)"
    if score < T_C1:   return "B2-C1 (provisional)"
    return "C1-C2 (provisional)"

# ---- Feedback ----
def feedback_from_dims(dim, tf, vstats):
    strengths, areas, tips = [], [], []

    if TEEN_TONE:
        # ---- Teen-friendly wording ----
        # Fluency
        if dim["fluency"] >= 0.65:
            strengths.append("Nice flow — easy to follow. 🙌")
        if dim["fluency"] < 0.45:
            areas.append("Flow got a bit choppy with long pauses.")
            tips.append("Plan your first sentence, then link ideas (because / for example / however).")
        if tf.get("filler_ratio", 0) > 0.08:
            areas.append("Lots of fillers (um/uh/like).")
            tips.append("Quick silent pauses beat fillers — breathe, then continue.")

        # Grammar
        if dim["grammar"] >= 0.70:
            strengths.append("Grammar mostly on point. ✅")
        if dim["grammar"] < 0.50:
            areas.append("Some grammar slips (tense/articles/agreement).")
            tips.append("Watch 3rd-person -s and tricky past forms (go→went, buy→bought).")

        # Pronunciation
        if dim["pronunciation"] >= 0.70:
            strengths.append("Pronunciation was clear overall.")
        if dim["pronunciation"] < 0.50:
            areas.append("Sometimes hard to catch words.")
            tips.append("Practice stress & rhythm; record yourself and mimic a model answer.")

        # Vocabulary
        if dim["vocabulary"] >= 0.65:
            strengths.append("Good word variety — nice choices.")
        if dim["vocabulary"] < 0.45:
            areas.append("Words repeated a lot.")
            tips.append("Swap repeats for synonyms and add one strong example per idea.")

        # Relevance
        if dim["relevance"] >= 0.70:
            strengths.append("You answered the actual question clearly. 🎯")
        if dim["relevance"] < 0.50:
            areas.append("Answer drifted off-topic.")
            tips.append("Start by restating the question in your first sentence.")

        return {"strengths": strengths[:5], "areas": areas[:6], "tips": tips[:6]}

    # ---- Neutral tone ----
    if dim["fluency"] >= 0.65: strengths.append("Fluency: steady pace with few disruptive pauses.")
    if dim["fluency"] < 0.45:
        areas.append("Fluency: long pauses or choppy delivery reduced flow.")
        tips.append("Before speaking, plan a topic sentence; link ideas with connectors (because, for example, however).")
    if tf.get("filler_ratio",0) > 0.08:
        areas.append("High filler-word rate (um/uh/like).")
        tips.append("Use short silent pauses instead of fillers and slow down slightly.")

    if dim["grammar"] >= 0.70: strengths.append("Grammar: mostly accurate forms for the level.")
    if dim["grammar"] < 0.50:
        areas.append("Grammar: noticeable agreement/tense/article issues.")
        tips.append("Review present simple 3rd-person 's' and past irregulars (go→went, buy→bought).")

    if dim["pronunciation"] >= 0.70: strengths.append("Pronunciation: generally clear and intelligible.")
    if dim["pronunciation"] < 0.50:
        areas.append("Pronunciation: clarity/voicing inconsistent.")
        tips.append("Practice minimal pairs and sentence stress; record yourself and mimic native rhythm.")

    if dim["vocabulary"] >= 0.65: strengths.append("Vocabulary: good variety and some less frequent words.")
    if dim["vocabulary"] < 0.45:
        areas.append("Vocabulary: limited range / repetition.")
        tips.append("Paraphrase repeated words and add a concrete example per idea.")
    if vstats.get("adv_ratio",0) > 0.30 and dim["grammar"] < 0.55:
        tips.append("Use advanced words only where accurate; prioritize clear grammar and collocations.")

    if dim["relevance"] >= 0.70: strengths.append("Relevance: answer addressed the question directly.")
    if dim["relevance"] < 0.50:
        areas.append("Relevance: response drifted from the question.")
        tips.append("Start by restating the question in your first sentence to stay on-topic.")

    return {"strengths": strengths[:5], "areas": areas[:6], "tips": tips[:6]}

def llm_feedback(question, transcript, dim) -> Optional[dict]:
    if not (USE_LLM_FEEDBACK and os.getenv("OPENAI_API_KEY") and transcript):
        return None
    try:
        from openai import OpenAI
        client = OpenAI()
        if TEEN_TONE:
            sys = (
                "You are a friendly CEFR speaking coach for teenagers. "
                "Give short, encouraging, actionable bullets. Avoid complicated words. "
                "Focus on: relevance to the question, fluency, grammar, pronunciation, vocabulary. "
                "Suggest 2–3 concrete tips. Keep it under 120 words. Use a positive, supportive tone."
            )
        else:
            sys = "You are a CEFR speaking assessor. Provide concise, actionable feedback in bullets. Avoid praise fluff."
        user = {"question":question, "transcript":transcript, "dimensions":dim}
        msg = client.chat.completions.create(
            model=OPENAI_FEEDBACK_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":json.dumps(user)}],
            temperature=0.2, max_tokens=300
        )
        return {"summary": msg.choices[0].message.content.strip()}
    except Exception:
        return None

# -------------------- Core evaluation --------------------
class EvalTextIn(BaseModel):
    question: str
    transcript: str
    duration_sec: Optional[float] = None

def evaluate_core(tmp_path: Optional[str]=None, raw_bytes: Optional[bytes]=None, question: Optional[str]=None):
    # Load audio if present
    y = None; sr = 16000
    intervals = []
    if tmp_path:
        y, sr = load_audio_to_16k_mono(tmp_path)
        intervals = voice_intervals(y)

    # Activity metrics
    m, intervals = speech_activity_metrics(y if y is not None else np.array([],dtype=np.float32), sr, intervals)

    # Transcript + ASR confidence (if audio)
    transcript, asr_provider, asr_conf = (None, "text", None)
    if y is not None and tmp_path is not None:
        transcript, asr_provider, asr_conf = get_transcript(y, tmp_path)

    # Fluency (base + extra)
    flu, tf, flu_extra = fluency_score(m, transcript, m["duration_sec"], intervals, sr)

    # Relevance
    rel = relevance_score(question, transcript) if transcript and question else 0.0

    # Grammar
    gra, gram_errs, gram_density = grammar_score(transcript or "")

    # Pronunciation
    pro, f0stats = pronunciation_score(y if y is not None else np.array([],dtype=np.float32), sr, m, asr_conf)

    # Vocabulary
    voc, vstats = vocabulary_score(transcript or "")

    # Weighted overall
    wsum = W_REL + W_FLU + W_GRA + W_PRO + W_VOC
    overall = (W_REL*rel + W_FLU*flu + W_GRA*gra + W_PRO*pro + W_VOC*voc) / (wsum if wsum>0 else 1.0)
    level = map_score_to_level(overall)

    # Confidence by band probabilities
    band_probs = band_probabilities(overall)
    top_band = max(band_probs.items(), key=lambda kv: kv[1])[0]
    top_p = band_probs[top_band]
    sorted_probs = sorted(band_probs.items(), key=lambda kv: kv[1], reverse=True)
    margin = float(sorted_probs[0][1] - (sorted_probs[1][1] if len(sorted_probs)>1 else 0.0))
    should_stop = bool(top_p >= 0.85 and margin >= 0.15)

    # Feedback
    dim = {"relevance":rel, "fluency":flu, "grammar":gra, "pronunciation":pro, "vocabulary":voc}
    fb = feedback_from_dims(dim, tf, vstats)
    llm_fb = llm_feedback(question, transcript, dim)
    if llm_fb: fb = {**fb, **llm_fb}

    return {
        "ok": True,
        "version": APP_VERSION,
        "mode": "SAFE" if SAFE_MODE else "FULL",
        "asr_provider": asr_provider,
        "question_echo": question,
        "transcript": transcript,
        "metrics": {
            "duration_sec": m["duration_sec"],
            "voiced_ratio": m["voiced_ratio"],
            "silence_pct": m["silence_pct"],
            "segments": m["num_segments"],
            "avg_segment_sec": m["avg_segment_sec"],
            "long_pauses": m["long_pauses"],
            "fluency_extras": flu_extra,     # articulation_sps, mlr_sec, silence_pct
            "pitch": f0stats                  # f0_med, f0_iqr
        },
        "scores": {
            "overall": overall,
            "relevance": rel,
            "fluency": flu,
            "grammar": gra,
            "pronunciation": pro,
            "vocabulary": voc,
            "wpm": tf.get("wpm",0.0),
            "lexical_diversity": tf.get("lexical_diversity",0.0),
            "filler_ratio": tf.get("filler_ratio",0.0),
            "vocab_stats": vstats,
            "grammar_errors": gram_errs,
            "grammar_error_density_per100": gram_density
        },
        "provisional_level": level,
        "confidence": {
            "band_probs": band_probs,
            "top_band": top_band,
            "top_p": float(top_p),
            "margin": float(margin),
            "early_stop_suggested": should_stop
        },
        "feedback": fb
    }

# -------------------- Routes --------------------
@app.get("/health")
async def health():
    prompts = load_prompts()
    return {
        "status":"ok","version":APP_VERSION,
        "mode":"SAFE" if SAFE_MODE else "FULL",
        "prompt_levels": sorted(list(prompts.keys())),
        "weights": {"rel":W_REL,"flu":W_FLU,"gra":W_GRA,"pro":W_PRO,"voc":W_VOC},
        "thresholds":{"A2":T_A2,"B1":T_B1,"B2":T_B2,"C1":T_C1},
        "teen_tone": TEEN_TONE
    }

@app.get("/config")
async def config():
    return {
        "mode":"SAFE" if SAFE_MODE else "FULL",
        "weights":{"relevance":W_REL,"fluency":W_FLU,"grammar":W_GRA,"pronunciation":W_PRO,"vocabulary":W_VOC},
        "thresholds":{"A2":T_A2,"B1":T_B1,"B2":T_B2,"C1":T_C1},
        "asr_mode": _get_asr_mode(),
        "embed_mode": "openai" if (USE_OPENAI_EMBED and os.getenv("OPENAI_API_KEY")) else "heuristic",
        "grammar_mode": "openai" if (USE_OPENAI_GRAMMAR and os.getenv("OPENAI_API_KEY")) else "heuristic",
        "feedback_mode": "openai" if (USE_LLM_FEEDBACK and os.getenv("OPENAI_API_KEY")) else "rule",
        "teen_tone": TEEN_TONE
    }

@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...), question: Optional[str] = Form(None)):
    if not file.filename.lower().endswith((".wav",".mp3",".m4a",".flac",".ogg",".webm")):
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload wav/mp3/m4a/flac/ogg/webm.")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
            tmp.write(await file.read()); tmp_path = tmp.name
        result = evaluate_core(tmp_path=tmp_path, question=question)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}\n{traceback.format_exc()}")
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path): os.unlink(tmp_path)
        except Exception: pass

@app.post("/evaluate-bytes")
async def evaluate_bytes(request: Request):
    try:
        data = await request.body()
        if not data: raise HTTPException(status_code=400, detail="Empty request body")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            tmp.write(data); tmp_path = tmp.name
        question = request.headers.get("x-question")
        result = evaluate_core(tmp_path=tmp_path, question=question)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}\n{traceback.format_exc()}")
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path): os.unlink(tmp_path)
        except Exception: pass

# Text-only endpoint for ChatGPT Actions / typed answers
@app.post("/evaluate-text")
async def evaluate_text(body: EvalTextIn):
    q = (body.question or "").strip()
    t = (body.transcript or "").strip()
    # estimate duration if missing
    words = len(_tokens(t)); est_wpm=150.0
    dur = body.duration_sec if (body.duration_sec and body.duration_sec>0) else max(8.0, min(90.0, (words*60.0)/est_wpm))
    # Build pseudo-metrics to reuse fluency/pron/vocab logic (no audio)
    num_segs = max(1, t.count(".")+t.count("!")+t.count("?"))
    avg_seg_sec = max(0.5, min(3.0, (max(1,words)/num_segs)*60.0/est_wpm))
    m = {"duration_sec": float(dur), "voiced_ratio":0.60, "num_segments": num_segs,
         "avg_segment_sec": avg_seg_sec, "long_pauses": t.count("..."), "silence_pct": 0.40}
    intervals = []  # no audio; synthesize extras
    sr = 16000

    # Fluency
    flu, tf, flu_extra = fluency_score(m, t, m["duration_sec"], intervals, sr)
    # Relevance
    rel = relevance_score(q, t) if q and t else 0.0
    # Grammar
    gra, gram_errs, gram_density = grammar_score(t)
    # Pronunciation (no audio): neutral proxy from voiced_ratio fallback
    pro, f0stats = ( _normalize(m["voiced_ratio"], 0.25, 0.85), {"f0_med":0.0,"f0_iqr":0.0} )
    # Vocabulary
    voc, vstats = vocabulary_score(t)

    # Overall
    wsum = W_REL + W_FLU + W_GRA + W_PRO + W_VOC
    overall = (W_REL*rel + W_FLU*flu + W_GRA*gra + W_PRO*pro + W_VOC*voc) / (wsum if wsum>0 else 1.0)
    level = map_score_to_level(overall)
    band_probs = band_probabilities(overall)
    top_band = max(band_probs.items(), key=lambda kv: kv[1])[0]
    top_p = band_probs[top_band]
    sorted_probs = sorted(band_probs.items(), key=lambda kv: kv[1], reverse=True)
    margin = float(sorted_probs[0][1] - (sorted_probs[1][1] if len(sorted_probs)>1 else 0.0))
    should_stop = bool(top_p >= 0.85 and margin >= 0.15)

    dim = {"relevance":rel,"fluency":flu,"grammar":gra,"pronunciation":pro,"vocabulary":voc}
    fb = feedback_from_dims(dim, tf, vstats)
    llm_fb = llm_feedback(q, t, dim)
    if llm_fb: fb = {**fb, **llm_fb}
    return {
        "ok": True, "mode": "TEXT", "version": APP_VERSION,
        "question_echo": q, "transcript": t,
        "metrics": {
            "duration_sec": m["duration_sec"],
            "voiced_ratio": m["voiced_ratio"],
            "silence_pct": m["silence_pct"],
            "segments": m["num_segments"],
            "avg_segment_sec": m["avg_segment_sec"],
            "long_pauses": m["long_pauses"],
            "fluency_extras": flu_extra,
            "pitch": f0stats
        },
        "scores":{"overall":overall, **dim, "wpm":tf["wpm"], "lexical_diversity":tf["lexical_diversity"],
                  "filler_ratio":tf["filler_ratio"], "vocab_stats":vstats,
                  "grammar_errors": gram_errs, "grammar_error_density_per100": gram_density},
        "provisional_level": level,
        "confidence": {"band_probs": band_probs, "top_band": top_band, "top_p": float(top_p), "margin": float(margin),
                       "early_stop_suggested": should_stop},
        "feedback": fb
    }

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT","8000")), reload=False)
