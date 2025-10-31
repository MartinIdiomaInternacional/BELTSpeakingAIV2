import { useEffect, useRef, useState } from 'react'

// ==== CONFIG (env-overridable) ====
const MAX_TURNS  = Number(import.meta.env.VITE_MAX_TURNS || 6)
const BUDGET_MIN = Number(import.meta.env.VITE_BUDGET_MIN || 7)          // total recording minutes
const THINK_SEC  = Number(import.meta.env.VITE_THINK_SECONDS || 30)      // per-turn thinking (not counted)

// ---- API helpers ----
async function api(path, body){
  const res = await fetch(`/api${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {})
  })
  if(!res.ok){
    let text = ''
    try { text = await res.text() } catch {}
    throw new Error(text || res.statusText)
  }
  return res.json()
}
async function startSession(candidate_id, native_language){
  return api('/start', { candidate_id, native_language })
}
function stripDataUrl(b64){
  if (!b64) return b64
  const i = b64.indexOf('base64,')
  return i >= 0 ? b64.slice(i + 'base64,'.length) : b64
}
async function sendAudio(session_id, rawBase64){
  const clean = stripDataUrl(rawBase64)
  if (!clean || clean.length < 5000) {
    throw new Error('Audio seems empty/too short. Please try again.')
  }
  // multiple aliases for backend compatibility
  return api('/evaluate-bytes', {
    session_id,
    wav_base64: clean,
    audio_base64: clean,
    webm_base64: clean,
  })
}

// ---- Small UI bits ----
function HR(){ return <div style={{height:1, background:'#ccc', margin:'12px 0'}}/> }

function ProgressBar({ value, max, height=8, label }){
  const p = Math.max(0, Math.min(1, max ? value / max : 0))
  return (
    <div style={{width:'100%', gap:8}}>
      {label && <div className="small" style={{marginBottom:6}}>{label}</div>}
      <div style={{width:'100%', height, background:'#eee', borderRadius:6, overflow:'hidden'}}>
        <div style={{width:`${(p*100).toFixed(1)}%`, height:'100%', background:'#0ea5e9'}}/>
      </div>
    </div>
  )
}

function Ding(){
  // a tiny sine “ding” using WebAudio
  function play(){
    try{
      const ctx = new (window.AudioContext||window.webkitAudioContext)()
      const o = ctx.createOscillator()
      const g = ctx.createGain()
      o.type = 'sine'
      o.frequency.value = 880
      o.connect(g); g.connect(ctx.destination)
      const now = ctx.currentTime
      g.gain.setValueAtTime(0.0001, now)
      g.gain.exponentialRampToValueAtTime(0.4, now+0.02)
      g.gain.exponentialRampToValueAtTime(0.0001, now+0.25)
      o.start(now); o.stop(now+0.27)
      o.onended = ()=>ctx.close()
    }catch{}
  }
  return <button style={{display:'none'}} aria-hidden onClick={play} id="ding-btn">ding</button>
}

function BudgetStrip({ totalRecordingSec, turnsSoFar }){
  const mm = Math.floor(totalRecordingSec/60)
  const ss = Math.floor(totalRecordingSec - mm*60).toString().padStart(2,'0')
  return (
    <div style={{display:'flex', alignItems:'center', gap:16, justifyContent:'flex-end'}}>
      <div>Turn <b>{Math.min(turnsSoFar + 1, MAX_TURNS)}</b> / {MAX_TURNS}</div>
      <div>Recording time: <b>{mm}:{ss}</b> / {BUDGET_MIN}m</div>
    </div>
  )
}

// ---- Recorder (embedded here to keep file count small) ----
function WaveformCanvas({ stream }){
  const ref = useRef(null)
  const raf = useRef(0)
  const ctxRef = useRef(null)
  const analyserRef = useRef(null)
  useEffect(()=>{
    cancelAnimationFrame(raf.current)
    if(!stream){ return }
    const ac = new (window.AudioContext||window.webkitAudioContext)()
    const src = ac.createMediaStreamSource(stream)
    const analyser = ac.createAnalyser()
    analyser.fftSize = 1024
    src.connect(analyser)
    ctxRef.current = ac
    analyserRef.current = analyser
    const draw = ()=>{
      raf.current = requestAnimationFrame(draw)
      const el = ref.current
      if(!el) return
      const c = el.getContext('2d')
      el.width  = el.clientWidth
      el.height = 90
      const w = el.width, h = el.height
      c.clearRect(0,0,w,h)
      c.fillStyle = '#f8fafc'; c.fillRect(0,0,w,h)
      const buf = new Float32Array(analyser.fftSize)
      try{ analyser.getFloatTimeDomainData(buf) }catch{ return }
      c.strokeStyle = '#0ea5e9'; c.lineWidth = 2; c.beginPath()
      for(let i=0;i<buf.length;i++){
        const x = (i/(buf.length-1))*w
        const y = h/2 + buf[i]*h*0.45
        if(i===0) c.moveTo(x,y); else c.lineTo(x,y)
      }
      c.stroke()
    }
    draw()
    return ()=>{ cancelAnimationFrame(raf.current); try{ ac.close() }catch{} }
  },[stream])
  return <canvas ref={ref} style={{width:'100%', height:90, background:'#f8fafc', border:'1px solid #e5e7eb', borderRadius:6}}/>
}

function Recorder({
  triggerStart = 0,
  onComplete,
  minSpeakSeconds = 15,
  silenceStopSeconds = 3,
  silenceThreshold = 0.012,
  monitorFps = 20,
  chunkMs = 250,
  onError,
}){
  const [stream, setStream] = useState(null)
  const [recording, setRecording] = useState(false)
  const [err, setErr] = useState('')
  const recRef = useRef(null)
  const chunksRef = useRef([])
  const audioCtxRef = useRef(null)
  const analyserRef = useRef(null)
  const tickTimerRef = useRef(null)
  const voicedAccumRef = useRef(0)
  const silenceAccumRef = useRef(0)
  const lastTickRef = useRef(0)
  const prevTriggerRef = useRef(triggerStart)

  useEffect(()=>{
    if (triggerStart !== prevTriggerRef.current){
      prevTriggerRef.current = triggerStart
      start().catch(()=>{})
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [triggerStart])

  useEffect(()=>{
    return ()=>{ stopMonitor(); try{ stream?.getTracks().forEach(t=>t.stop()) }catch{} }
  },[stream])

  function reportError(msg){
    setErr(msg)
    onError?.(msg)
  }

  async function ensureStream(){
    try{
      const s = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation:true, noiseSuppression:true }
      })
      setStream(s)
      const ac = new (window.AudioContext||window.webkitAudioContext)()
      const src = ac.createMediaStreamSource(s)
      const an = ac.createAnalyser()
      an.fftSize = 1024
      src.connect(an)
      audioCtxRef.current = ac
      analyserRef.current = an
      return s
    }catch(e){
      console.error(e)
      reportError('Microphone permission denied or unavailable. Please allow mic access and try again.')
      throw e
    }
  }

  function pickMime(){
    const prefs = [
      'audio/webm;codecs=opus','audio/webm',
      'audio/ogg;codecs=opus','audio/ogg','audio/mp4'
    ]
    for(const t of prefs){
      try{
        if(window.MediaRecorder?.isTypeSupported?.(t)) return t
      }catch{}
    }
    return ''
  }

  function blobToBase64(blob){
    return new Promise((resolve)=>{
      const r = new FileReader()
      r.onload = ()=> resolve((r.result||'').toString())
      r.readAsDataURL(blob)
    })
  }

  async function start(){
    if (recording) return
    setErr('')
    const s = await ensureStream()
    const mimeType = pickMime()
    if(!mimeType && !window.MediaRecorder){
      reportError('MediaRecorder not supported in this browser.')
      return
    }
    const mr = new MediaRecorder(s, mimeType ? { mimeType } : {})
    recRef.current = mr
    chunksRef.current = []
    voicedAccumRef.current = 0
    silenceAccumRef.current = 0

    mr.ondataavailable = (e)=>{ if(e.data && e.data.size>0) chunksRef.current.push(e.data) }
    mr.onerror = (e)=>{ console.error('MediaRecorder error', e); reportError('Recording error. Please try again.') }
    mr.onstop = async ()=>{
      stopMonitor()
      try{
        const blob = new Blob(chunksRef.current, { type: mimeType || 'audio/webm' })
        chunksRef.current = []
        const base64 = await blobToBase64(blob)
        try{ s.getTracks().forEach(t=>t.stop()) }catch{}
        onComplete?.({ base64 })
      }catch(err2){
        console.error(err2)
        reportError('Could not finalize audio. Please try again.')
      }
    }

    lastTickRef.current = performance.now()
    mr.start(chunkMs)
    setRecording(true)
    startMonitor()
  }

  function stop(){
    const mr = recRef.current
    if(!mr || mr.state !== 'recording') return
    try{ mr.requestData() }catch{}
    try{ mr.stop() }catch(e){ console.error(e) }
    setRecording(false)
  }

  function startMonitor(){
    const analyser = analyserRef.current
    if(!analyser) return
    const buf = new Float32Array(analyser.fftSize)
    const intervalMs = 1000/monitorFps
    const tick = ()=>{
      tickTimerRef.current = setTimeout(()=>{
        try{ analyser.getFloatTimeDomainData(buf) }catch{ return }
        let sum = 0; for(let i=0;i<buf.length;i++){ sum += buf[i]*buf[i] }
        const rms = Math.sqrt(sum/buf.length)
        const now = performance.now()
        const dt = Math.max(0,(now - lastTickRef.current)/1000)
        lastTickRef.current = now
        if(rms > silenceThreshold){
          voicedAccumRef.current += dt
          silenceAccumRef.current = 0
        }else{
          if(voicedAccumRef.current >= minSpeakSeconds){
            silenceAccumRef.current += dt
            if(silenceAccumRef.current >= silenceStopSeconds){ stop(); return }
          }
        }
        tick()
      }, intervalMs)
    }
    tick()
  }
  function stopMonitor(){ if(tickTimerRef.current){ clearTimeout(tickTimerRef.current); tickTimerRef.current=null } }

  return (
    <div style={{border:'1px solid #e5e7eb', borderRadius:8, padding:12}}>
      <div style={{fontWeight:600, marginBottom:8}}>Recording</div>
      <WaveformCanvas stream={stream}/>
      <div style={{display:'flex', gap:8, marginTop:8, alignItems:'center'}}>
        <button className="btn" onClick={start} disabled={recording}>Start</button>
        <button className="btn" onClick={stop} disabled={!recording}>Stop</button>
        <div className="small" style={{marginLeft:'auto'}}>Auto-stop after {minSpeakSeconds}s spoken + {silenceStopSeconds}s silence</div>
      </div>
      {err && <div className="small" style={{color:'#b91c1c', marginTop:8}}>{err}</div>}
    </div>
  )
}

// ---- MAIN APP ----
export default function App(){
  const [sessionId, setSessionId] = useState(null)
  const [candidateId, setCandidateId] = useState('demo-user')
  const [nativeLang, setNativeLang] = useState('en')

  const [prompt, setPrompt] = useState('')
  const [currentLevel, setCurrentLevel] = useState('A1')
  const [turnsSoFar, setTurnsSoFar] = useState(0)

  const [totalRecordingSec, setTotalRecordingSec] = useState(0)
  const [phase, setPhase] = useState('idle') // idle | prep | analyzing | finished
  const [error, setError] = useState('')
  const [final, setFinal] = useState(null)

  // thinking timer
  const [thinkLeft, setThinkLeft] = useState(THINK_SEC)
  const thinkTimerRef = useRef(null)
  const [startTrigger, setStartTrigger] = useState(0)

  function clearThinkTimer(){ if(thinkTimerRef.current){ clearInterval(thinkTimerRef.current); thinkTimerRef.current=null } }
  function beginThinkCountdown(){
    clearThinkTimer()
    setThinkLeft(THINK_SEC)
    thinkTimerRef.current = setInterval(()=>{
      setThinkLeft(prev=>{
        const next = Math.max(0, prev - 1)
        if(next===0){
          clearThinkTimer()
          document.getElementById('ding-btn')?.click()
          setStartTrigger(t=>t+1) // auto-start recording
        }
        return next
      })
    }, 1000)
  }

  async function onStart(){
    try{
      setError('')
      const res = await startSession(candidateId, nativeLang)
      setSessionId(res.session_id)
      setPrompt(res.prompt)
      setCurrentLevel(res.current_level)
      setTurnsSoFar(0)
      setTotalRecordingSec(0)
      setFinal(null)
      setPhase('prep')
      beginThinkCountdown()
    }catch(e){
      console.error(e)
      setError('Could not start session. Please try again.')
    }
  }

  function handleStartNow(){
    clearThinkTimer()
    setStartTrigger(t=>t+1)
  }

  function handleRecError(msg){ setError(msg) }

  async function onRecordingComplete({ base64 }){
    if(!sessionId) return
    try{
      setPhase('analyzing')
      const res = await sendAudio(sessionId, base64)
      if (typeof res.total_recording_sec === 'number'){
        setTotalRecordingSec(res.total_recording_sec)
      } else if (res.turn?.duration_sec){
        setTotalRecordingSec(t=>t + (res.turn.duration_sec||0))
      }

      if(res.finished){
        setPhase('finished')
        if(res.final_level){
          setFinal({
            level: res.final_level,
            score: res.final_score_0_8,
            confidence: res.final_confidence,
            feedback: res.feedback,
          })
        }
        return
      }

      // prepare next turn
      setTurnsSoFar(t=>t+1)
      if(res.next_level) setCurrentLevel(res.next_level)
      if(res.next_prompt) setPrompt(res.next_prompt)
      setPhase('prep')
      beginThinkCountdown()
    }catch(e){
      console.error(e)
      setError('Upload/evaluation failed. Please try again.')
      setPhase('prep')
      beginThinkCountdown()
    }
  }

  useEffect(()=>()=>clearThinkTimer(),[])

  const recordingBudgetSec = BUDGET_MIN*60

  return (
    <div style={{maxWidth:960, margin:'24px auto', padding:'0 16px', fontFamily:'system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial'}}>
      <h1>BELT Speaking AI — Adaptive</h1>

      {sessionId && <BudgetStrip totalRecordingSec={totalRecordingSec} turnsSoFar={turnsSoFar} />}

      <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:12, alignItems:'end', marginTop:12}}>
        <div>
          <label className="small">Candidate ID</label>
          <input value={candidateId} onChange={e=>setCandidateId(e.target.value)} style={{width:'100%'}}/>
        </div>
        <div>
          <label className="small">Native Language</label>
          <input value={nativeLang} onChange={e=>setNativeLang(e.target.value)} style={{width:'100%'}}/>
        </div>
      </div>
      <div style={{marginTop:12, display:'flex', gap:8}}>
        <button className="btn" onClick={onStart} disabled={!!sessionId && phase!=='finished'}>Start Session</button>
        {error && <div className="small" style={{color:'#b91c1c', alignSelf:'center'}}>{error}</div>}
      </div>

      {sessionId && phase !== 'finished' && (
        <>
          <HR/>
          <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', gap:16}}>
            <div>
              <div className="small">Level: <b>{currentLevel}</b></div>
              <div style={{marginTop:6}}>{prompt}</div>
            </div>
            <div style={{minWidth:260}}>
              <ProgressBar
                value={totalRecordingSec}
                max={recordingBudgetSec}
                label={`Recording budget used`}
              />
            </div>
          </div>

          {phase === 'prep' && (
            <div style={{marginTop:12, display:'grid', gridTemplateColumns:'1fr auto', gap:12, alignItems:'center'}}>
              <ProgressBar value={THINK_SEC - thinkLeft} max={THINK_SEC} label={`Thinking time (${thinkLeft}s left)`}/>
              <button className="btn" onClick={handleStartNow}>Start recording now</button>
            </div>
          )}

          <div style={{marginTop:12}}>
            {/* KEY here forces a clean remount every turn, fixing the “Start does nothing” issue */}
            <Recorder
              key={`${sessionId}-${turnsSoFar}`}
              triggerStart={startTrigger}
              onComplete={onRecordingComplete}
              onError={handleRecError}
              minSpeakSeconds={15}
              silenceStopSeconds={3}
              silenceThreshold={0.012}
              monitorFps={20}
              chunkMs={250}
            />
          </div>
        </>
      )}

      {phase === 'analyzing' && <div style={{marginTop:12}}>Analyzing…</div>}

      {phase === 'finished' && final && (
        <>
          <HR/>
          <h3>Final result</h3>
          <div>Level: <b>{final.level}</b></div>
          {final.score != null && <div>Score (0–8): <b>{final.score.toFixed ? final.score.toFixed(2) : final.score}</b></div>}
          {final.confidence != null && <div>Confidence: <b>{(final.confidence*100).toFixed(0)}%</b></div>}
          {final.feedback && <div style={{marginTop:8}}>{final.feedback}</div>}
        </>
      )}

      <Ding/>
    </div>
  )
}
