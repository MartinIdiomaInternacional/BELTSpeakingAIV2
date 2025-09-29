/* BELT Voice Evaluator - Main JS
   v5: Adds Mic Check warm-up with live meter and /warmup-check, then adaptive test.
*/

const API_BASE = "";
const CONFIG_ENDPOINT = `${API_BASE}/config`;

const SEL = {
  // warm-up
  btnWarmup:"#btnWarmup",
  liveLevel:"#liveLevel",
  liveDb:"#liveDb",
  warmupResult:"#warmupResult",
  nativeLang:"#nativeLang",
  btnStartSession:"#btnStartSession",

  // test controls
  btnStart:"#btnStart",
  btnStopSend:"#btnStopSend",
  btnRetry:"#btnRetry",
  promptText:"#promptText",
  levelBadge:"#levelBadge",
  attemptBadge:"#attemptBadge",

  // prep
  prepInstructions:"#prepInstructions",
  prepBar:"#prepBar",
  prepTimer:"#prepTimer",

  // recording
  timerText:"#timerText",
  progressBar:"#progressBar",

  // results
  scoresContainer:"#scoresContainer",
  metrics:"#metrics",
  recsContainer:"#recommendations",
  messages:"#messages",
  finalReport:"#finalReport",
  audioPreview:"#audioPreview"
};

const STR = {
  defaultPrompt:"Speak for ~60 seconds.",
  starting:"Starting recorder…",
  recording:"Recording…",
  processing:"Scoring your answer…",
  stopped:"Stopped. Sending audio…",
  ready:"Ready.",
  error:"Something went wrong.",
  prepMsg:"You have up to <strong>30 seconds</strong> to think. Recording will start automatically."
};

const SCORE_COLORS = { good:"#16a34a", mid:"#f59e0b", low:"#dc2626" };

// Config/state
let SESSION_ID=null, CURRENT_LEVEL="A1", CURRENT_PROMPT="", CURRENT_PROMPT_ID=null, ATTEMPT=0;
let RECORD_SECONDS=60, PASS_AVG_THRESHOLD=0.7, PASS_MIN_THRESHOLD=0.6, DEBUG_RETURN_TRANSCRIPT=false;
let PREP_SECONDS=30;
let SELECTED_NATIVE_LANG="Spanish"; // default

// Recorder state
let mediaStream=null, mediaRecorder=null, recordedChunks=[], isRecording=false;
let timerInterval=null, startTimestamp=null;

// Prep countdown state
let prepInterval=null, prepRemaining=0, prepActive=false;

// Warm-up live meter
let meterCtx=null, meterAnalyser=null, meterSource=null, meterRAF=null;

// ---------- DOM helpers ----------
function $(s){ return document.querySelector(s); }
function setText(s,t){ const el=$(s); if(el) el.textContent=t; }
function setHTML(s,h){ const el=$(s); if(el) el.innerHTML=h; }
function setDisabled(s,d){ const el=$(s); if(el) el.disabled=d; }
function appendMessage(msg,type="info"){ const el=$(SEL.messages); if(!el)return; const d=document.createElement("div"); d.className=`msg ${type}`; d.innerHTML=msg; el.prepend(d); }

function safeColor(seconds){
  if(seconds <= 5) return "#dc2626";
  if(seconds <= 15) return "#f59e0b";
  return "#16a34a";
}

// ---------- Rendering ----------
function renderPrompt(p,l){
  setText(SEL.levelBadge,l||"");
  setHTML(SEL.promptText,p||STR.defaultPrompt);
  setHTML(SEL.prepInstructions, STR.prepMsg + ` <span class="muted">Feedback will be provided in <strong>${SELECTED_NATIVE_LANG}</strong>.</span>`);
}
function renderAttempt(n){ setText(SEL.attemptBadge, n?`Attempt ${n}`:""); }

function chip(label,val){
  const v=Math.round((val??0)*100);
  let color=SCORE_COLORS.low;
  if(v>=PASS_MIN_THRESHOLD*100 && v<PASS_AVG_THRESHOLD*100) color=SCORE_COLORS.mid;
  if(v>=PASS_AVG_THRESHOLD*100) color=SCORE_COLORS.good;
  return `<div class="score-chip" style="border:1px solid #e5e7eb;border-radius:999px;padding:6px 10px;display:inline-flex;align-items:center;gap:8px;margin:4px;"><span style="font-weight:600">${label}</span><span style="background:${color};color:#fff;border-radius:999px;padding:2px 8px;">${v}</span></div>`;
}
function renderScores(r){
  const b=$(SEL.scoresContainer); if(!b) return;
  if(!r||!r.scores){ setHTML(SEL.scoresContainer,""); return; }
  const s=r.scores, avg=r.average??0;
  const html=`<div>${chip("Fluency",s.fluency)}${chip("Grammar",s.grammar)}${chip("Vocabulary",s.vocabulary)}${chip("Pronunciation",s.pronunciation)}${chip("Coherence",s.coherence)}</div>
    <div style="margin-top:6px"><strong>Average:</strong> ${(avg*100).toFixed(0)}
      ${r.decision?`<span style="margin-left:8px;padding:2px 8px;border-radius:6px;background:${r.decision==='advance'?'#d1fae5':'#fee2e2'};color:${r.decision==='advance'?'#064e3b':'#7f1d1d'}">${r.decision}</span>`:""}
      ${r.confidence!=null?`<span class="muted" style="margin-left:8px;">Confidence: ${(r.confidence*100).toFixed(0)}%</span>`:""}
    </div>`;
  setHTML(SEL.scoresContainer,html);
  const m=$(SEL.metrics);
  if(m){
    const sr = r.speech_ratio==null ? "n/a" : r.speech_ratio.toFixed(2);
    const snr = r.snr==null ? "n/a" : r.snr.toFixed(1);
    m.textContent = `Speech ratio: ${sr}   SNR: ${snr} dB`;
  }
}
function renderRecommendations(recs){
  if(!recs||!recs.length){ setHTML(SEL.recsContainer,""); return; }
  setHTML(SEL.recsContainer, `<ul>${recs.map(r=>`<li>${r}</li>`).join("")}</ul>`);
}
function renderFinalReportLink(id){
  const el=$(SEL.finalReport); if(!el) return;
  el.innerHTML=`<a href="${API_BASE}/report/${encodeURIComponent(id)}" target="_blank" rel="noopener">Open Final Report</a>`;
}
function resetScoresUI(){ setHTML(SEL.scoresContainer,""); setHTML(SEL.recsContainer,""); setHTML(SEL.metrics,""); }

// ---------- Warm-up mic level ----------
async function ensureMic(){
  if(mediaStream) return mediaStream;
  return await navigator.mediaDevices.getUserMedia({ audio:true });
}
function startLiveMeter(stream){
  try{
    meterCtx = new (window.AudioContext||window.webkitAudioContext)();
    meterSource = meterCtx.createMediaStreamSource(stream);
    meterAnalyser = meterCtx.createAnalyser();
    meterAnalyser.fftSize = 2048;
    meterSource.connect(meterAnalyser);
    const data = new Uint8Array(meterAnalyser.fftSize);
    const bar = $(SEL.liveLevel), label=$(SEL.liveDb);
    const loop = ()=>{
      meterRAF = requestAnimationFrame(loop);
      meterAnalyser.getByteTimeDomainData(data);
      // compute RMS
      let sum=0;
      for(let i=0;i<data.length;i++){
        const v=(data[i]-128)/128; sum+=v*v;
      }
      const rms=Math.sqrt(sum/data.length); // 0..~1
      const pct=Math.min(100, Math.max(0, Math.floor(rms*160)));
      if(bar) bar.style.width = `${pct}%`;
      if(label) label.textContent = `Level: ${pct}%`;
    };
    loop();
  }catch(e){
    appendMessage(`Live meter unavailable: ${e.message}`,"error");
  }
}
function stopLiveMeter(){
  if(meterRAF) cancelAnimationFrame(meterRAF);
  meterRAF=null;
  try{ meterCtx && meterCtx.close(); }catch(_){}
  meterCtx=null; meterAnalyser=null; meterSource=null;
  const bar=$(SEL.liveLevel); if(bar) bar.style.width="0%";
  setText(SEL.liveDb, "Level: 0%");
}

async function runWarmup(){
  setDisabled(SEL.btnWarmup,true);
  setDisabled(SEL.btnStartSession,true);

  appendMessage("Starting mic check…");
  let stream=null;
  try{
    stream = await ensureMic();
  }catch(e){
    setHTML(SEL.warmupResult, `<div class="msg warn">Mic permission denied. Please allow microphone access.</div>`);
    setDisabled(SEL.btnWarmup,false);
    return;
  }

  startLiveMeter(stream);

  // Record ~5 seconds
  const durationMs = 5000;
  const chunks=[];
  let rec=null;
  try{
    rec = new MediaRecorder(stream, { mimeType:"audio/webm;codecs=opus" });
  }catch(_){
    rec = new MediaRecorder(stream);
  }
  rec.ondataavailable=(e)=>{ if(e.data && e.data.size>0) chunks.push(e.data); };
  rec.onstop=async ()=>{
    stopLiveMeter();
    // stop tracks
    try{ stream.getTracks().forEach(t=>t.stop()); }catch(_){}
    const blob=new Blob(chunks,{type:"audio/webm"});
    // send to /warmup-check
    const fd=new FormData(); fd.append("file", blob, "warmup.webm");
    let ok=false, sr=null, snr=null, note=null, thresholds=null;
    try{
      const resp=await fetch(`${API_BASE}/warmup-check`,{method:"POST", body:fd});
      const data=await resp.json();
      ok = !!data.ok;
      sr = data.speech_ratio ?? null;
      snr = data.snr ?? null;
      note = data.note ?? null;
      thresholds = data.thresholds || null;
    }catch(e){
      appendMessage(`Warm-up check failed: ${e.message}`,"error");
      setDisabled(SEL.btnWarmup,false);
      return;
    }

    const resEl=$(SEL.warmupResult);
    const srTxt = sr==null ? "n/a" : sr.toFixed(2);
    const snrTxt = snr==null ? "n/a" : snr.toFixed(1);

    if(ok){
      resEl.className = "msg ok";
      resEl.innerHTML = `Mic check passed. Speech ratio: ${srTxt}, SNR: ${snrTxt} dB ${note?`(${note})`:""}. You can start the test.`;
      setDisabled(SEL.btnStartSession,false);
    }else{
      const minSR = thresholds?.min_speech_ratio ?? 0.5;
      const minSNR = thresholds?.min_snr_db ?? 15.0;
      resEl.className = "msg warn";
      resEl.innerHTML = `Mic check failed. Speech ratio: ${srTxt} (need >= ${minSR}), SNR: ${snrTxt} dB (need >= ${minSNR} dB). 
      Try moving closer to the mic, reducing background noise, or increasing input volume.`;
      setDisabled(SEL.btnStartSession,true);
    }
    setDisabled(SEL.btnWarmup,false);
  };

  rec.start();
  setTimeout(()=>{ try{ rec.stop(); }catch(_){ } }, durationMs);
}

// ---------- Prep countdown ----------
function resetPrepUI(){
  const bar=$(SEL.prepBar), t=$(SEL.prepTimer);
  if(bar) bar.style.width="0%";
  if(t) { t.textContent=`Prep: ${PREP_SECONDS}s`; t.style.color = safeColor(PREP_SECONDS); }
}
function stopPrep(){ if(prepInterval){ clearInterval(prepInterval); prepInterval=null; } prepActive=false; }
function startPrepCountdown(){
  stopPrep();
  let elapsed=0;
  prepActive=true;
  resetPrepUI();
  setDisabled(SEL.btnStart, false);
  setDisabled(SEL.btnStopSend, true);
  const bar=$(SEL.prepBar), t=$(SEL.prepTimer);
  prepInterval = setInterval(()=>{
    if(!prepActive){ clearInterval(prepInterval); return; }
    elapsed += 1;
    const remaining = Math.max(0, PREP_SECONDS - elapsed);
    if(bar){
      const pct = Math.min(100, Math.floor((elapsed / PREP_SECONDS) * 100));
      bar.style.width = `${pct}%`;
      bar.style.background = safeColor(remaining);
    }
    if(t){
      t.textContent = `Prep: ${remaining}s`;
      t.style.color = safeColor(remaining);
    }
    if(remaining <= 0){
      stopPrep();
      startRecording();
    }
  }, 1000);
}

// ---------- Recording progress ----------
function resetProgressUI(){
  const bar=$(SEL.progressBar), t=$(SEL.timerText);
  if(bar) bar.style.width="0%";
  if(t) t.textContent=`0/${RECORD_SECONDS}s`;
}
function startTimer(){
  resetProgressUI();
  startTimestamp=Date.now();
  timerInterval=setInterval(()=>{
    const elapsed=Math.floor((Date.now()-startTimestamp)/1000);
    const pct=Math.min(100,Math.floor((elapsed/RECORD_SECONDS)*100));
    const bar=$(SEL.progressBar), t=$(SEL.timerText);
    if(bar) bar.style.width=`${pct}%`;
    if(t) t.textContent=`${Math.min(elapsed,RECORD_SECONDS)}/${RECORD_SECONDS}s`;
    if(elapsed>=RECORD_SECONDS) stopRecordingAndSend();
  },250);
}
function stopTimer(){ clearInterval(timerInterval); timerInterval=null; }

// ---------- Recorder ----------
async function startRecording(){
  if(prepActive) stopPrep();
  if(isRecording) return;
  appendMessage(STR.starting);
  try{
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  }catch(e){
    appendMessage("Mic permission denied or unavailable.","error");
    return;
  }
  recordedChunks=[];
  try{ mediaRecorder=new MediaRecorder(mediaStream,{mimeType:"audio/webm;codecs=opus"}); }
  catch(e){ mediaRecorder=new MediaRecorder(mediaStream); }
  mediaRecorder.ondataavailable=(e)=>{ if(e.data && e.data.size>0) recordedChunks.push(e.data); };
  mediaRecorder.onstop=handleRecorderStop;

  mediaRecorder.start();
  isRecording=true;
  setDisabled(SEL.btnStart,true);
  setDisabled(SEL.btnStopSend,false);
  appendMessage(STR.recording);
  startTimer();
}
function handleRecorderStop(){
  stopTimer();
  if(mediaStream){ mediaStream.getTracks().forEach(t=>t.stop()); }
  isRecording=false;
  setDisabled(SEL.btnStart,false);
  setDisabled(SEL.btnStopSend,true);

  const blob=new Blob(recordedChunks,{type:"audio/webm"});
  const kb=(blob.size/1024).toFixed(1);
  appendMessage(`Captured audio blob: ${kb} KB`);
  const audioEl=$(SEL.audioPreview); if(audioEl){ audioEl.src=URL.createObjectURL(blob); }
  submitRecording(blob).catch(e=>appendMessage(`${STR.error} ${e?.message||e}`,"error"));
}
function stopRecordingAndSend(){ if(!isRecording||!mediaRecorder) return; appendMessage(STR.stopped); mediaRecorder.stop(); }

// ---------- Backend integration ----------
async function loadConfig(){
  try{
    const resp=await fetch(CONFIG_ENDPOINT); const cfg=await resp.json();
    RECORD_SECONDS=cfg.RECORD_SECONDS??RECORD_SECONDS;
    PASS_AVG_THRESHOLD=parseFloat(cfg.PASS_AVG_THRESHOLD??PASS_AVG_THRESHOLD);
    PASS_MIN_THRESHOLD=parseFloat(cfg.PASS_MIN_THRESHOLD??PASS_MIN_THRESHOLD);
    DEBUG_RETURN_TRANSCRIPT=!!cfg.DEBUG_RETURN_TRANSCRIPT;
  }catch(e){}
}

async function startSession(){
  try{
    SELECTED_NATIVE_LANG = ($(SEL.nativeLang)?.value || "Spanish").trim() || "Spanish";

    const fd=new FormData();
    fd.append("level",CURRENT_LEVEL);
    fd.append("native_language", SELECTED_NATIVE_LANG);

    const resp=await fetch(`${API_BASE}/start-session`,{method:"POST",body:fd});
    const data=await resp.json();
    SESSION_ID=data.session_id; CURRENT_LEVEL=data.level;
    CURRENT_PROMPT=data.prompt||STR.defaultPrompt;
    CURRENT_PROMPT_ID=typeof data.prompt_id==="number"?data.prompt_id:null;
    ATTEMPT=1;

    renderPrompt(CURRENT_PROMPT,CURRENT_LEVEL);
    renderAttempt(ATTEMPT);
    resetScoresUI(); resetProgressUI();

    appendMessage(`Session started. Prep time begins. Feedback language: <strong>${SELECTED_NATIVE_LANG}</strong>.`);
    startPrepCountdown();

    setDisabled(SEL.btnRetry,false);
    setDisabled(SEL.btnStart,false);
    setDisabled(SEL.btnStopSend,true);
  }catch(e){ appendMessage(`${STR.error} ${e?.message||e}`,"error"); }
}

async function fetchPromptForLevel(level){
  const url=`${API_BASE}/prompts/${encodeURIComponent(level)}?session_id=${encodeURIComponent(SESSION_ID)}`;
  const resp=await fetch(url); const data=await resp.json();
  CURRENT_PROMPT=data.instructions||STR.defaultPrompt;
  CURRENT_PROMPT_ID=typeof data.prompt_id==="number"?data.prompt_id:null;
  renderPrompt(CURRENT_PROMPT,level);
  resetScoresUI(); resetProgressUI();
  appendMessage("New prompt. Prep time begins.");
  startPrepCountdown();
  return CURRENT_PROMPT;
}

async function submitRecording(blob){
  try{
    appendMessage(STR.processing);
    const fd=new FormData();
    fd.append("session_id",SESSION_ID);
    fd.append("file",blob,"answer.webm");
    if(CURRENT_PROMPT_ID!==null){ fd.append("prompt_id",String(CURRENT_PROMPT_ID)); }
    else { fd.append("question",CURRENT_PROMPT); }

    const resp=await fetch(`${API_BASE}/submit-response`,{method:"POST",body:fd});
    const result=await resp.json();
    if(!resp.ok){ appendMessage(result?.detail||"Submit failed.","error"); return; }

    if(DEBUG_RETURN_TRANSCRIPT && result.transcript!==undefined){
      const snippet=(result.transcript||"").slice(0,160);
      appendMessage(`Transcript: ${snippet}${result.transcript && result.transcript.length>160 ? "..." : ""}`);
    } else if(result.average===0){
      appendMessage("No speech detected or transcription disabled. Check mic & OPENAI_API_KEY.", "error");
    }

    renderScores(result);
    renderRecommendations(result.recommendations);
    renderAttempt(result.attempt);
    ATTEMPT=result.attempt;

    if(result.decision==="advance" && result.next_level){
      CURRENT_LEVEL=result.next_level;
      CURRENT_PROMPT=result.next_prompt;
      CURRENT_PROMPT_ID=result.next_prompt_id;
      renderPrompt(CURRENT_PROMPT,CURRENT_LEVEL);
      renderAttempt(1);
      resetScoresUI(); resetProgressUI();
      appendMessage(`Advanced to ${CURRENT_LEVEL}. Prep time begins.`);
      startPrepCountdown();
    } else if(result.decision==="stop"){
      appendMessage("Evaluation stopped.");
      renderFinalReportLink(SESSION_ID);
      if(result.session_recommendations){ renderRecommendations(result.session_recommendations); }
      stopPrep();
    }
  }catch(e){ appendMessage(`${STR.error} ${e?.message||e}`,"error"); }
}

async function retrySameLevel(){
  if(!SESSION_ID) return;
  await fetchPromptForLevel(CURRENT_LEVEL);
  ATTEMPT+=1; renderAttempt(ATTEMPT);
  appendMessage(`New question for ${CURRENT_LEVEL}.`);
}

// ---------- Wire-up ----------
function bindUI(){
  const w=$(SEL.btnWarmup), s=$(SEL.btnStartSession), r=$(SEL.btnStart), x=$(SEL.btnStopSend), q=$(SEL.btnRetry);
  if(w) w.addEventListener("click", runWarmup);
  if(s) s.addEventListener("click", startSession);
  if(r) r.addEventListener("click", startRecording);
  if(x) x.addEventListener("click", stopRecordingAndSend);
  if(q) q.addEventListener("click", retrySameLevel);

  const nl=$(SEL.nativeLang);
  if(nl) nl.addEventListener("change", ()=>{ SELECTED_NATIVE_LANG = nl.value; });
}

async function init(){
  bindUI();
  await loadConfig();
  SELECTED_NATIVE_LANG = ($(SEL.nativeLang)?.value || "Spanish");
  renderPrompt(STR.defaultPrompt,CURRENT_LEVEL);
  renderAttempt(0);
  resetPrepUI();
  resetProgressUI();
  setDisabled(SEL.btnRetry,true); setDisabled(SEL.btnStart,true); setDisabled(SEL.btnStopSend,true);
  appendMessage(STR.ready);
  window.addEventListener("error",(e)=>appendMessage(`JS Error: ${e.message}`,"error"));
}
document.addEventListener("DOMContentLoaded", init);
