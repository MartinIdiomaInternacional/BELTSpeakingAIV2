import { useEffect, useState } from 'react'
import Recorder from './components/Recorder'

const MAX_TURNS = Number(import.meta.env.VITE_MAX_TURNS || 6)
const BUDGET_MIN = Number(import.meta.env.VITE_BUDGET_MIN || 7)

async function api(path, body){
  const res = await fetch(`/api${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {})
  })
  if(!res.ok){
    const text = await res.text()
    throw new Error(text || res.statusText)
  }
  return res.json()
}

async function startSession(candidate_id, native_language){
  return api('/start', { candidate_id, native_language })
}

// --- NEW: robustness helpers for audio payload + aliasing for backward-compat
function stripDataUrl(b64){
  if (!b64) return b64
  const i = b64.indexOf('base64,')
  return i >= 0 ? b64.slice(i + 'base64,'.length) : b64
}

async function sendAudio(session_id, rawBase64){
  const clean = stripDataUrl(rawBase64)

  // lightweight sanity check to avoid empty / too-short uploads
  if (!clean || clean.length < 5000) {
    throw new Error('Audio capture seems empty or too short. Please try again.')
  }

  // send multiple aliases so it works with older backends too
  return api('/evaluate-bytes', {
    session_id,
    wav_base64: clean,
    audio_base64: clean,
    webm_base64: clean,
  })
}

function BudgetStrip({ startedAt, turnsSoFar }){
  const [now, setNow] = useState(Date.now())
  useEffect(()=>{
    const id = setInterval(()=> setNow(Date.now()), 1000)
    return ()=> clearInterval(id)
  },[])
  const elapsedMinFloat = startedAt ? (now - startedAt)/60000 : 0
  const mm = Math.floor(elapsedMinFloat)
  const ss = Math.floor((elapsedMinFloat - mm)*60).toString().padStart(2, '0')
  return (
    <div className="card" style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
      <div>Turn <b>{Math.min(turnsSoFar + 1, MAX_TURNS)}</b> / {MAX_TURNS}</div>
      <div>Time: <b>{mm}:{ss}</b> / {BUDGET_MIN}m</div>
    </div>
  )
}

export default function App(){
  const [sessionId, setSessionId] = useState(null)
  const [candidateId, setCandidateId] = useState('demo-user')
  const [nativeLang, setNativeLang] = useState('en')
  const [prompt, setPrompt] = useState('')
  const [currentLevel, setCurrentLevel] = useState('A1')
  const [turnsSoFar, setTurnsSoFar] = useState(0)
  const [startedAt, setStartedAt] = useState(null)
  const [phase, setPhase] = useState('idle') // idle | prep | recording | analyzing | finished
  const [final, setFinal] = useState(null)
  const [error, setError] = useState('')

  async function onStart(){
    try{
      setError('')
      const res = await startSession(candidateId, nativeLang)
      setSessionId(res.session_id)
      setPrompt(res.prompt)
      setCurrentLevel(res.current_level)
      setTurnsSoFar(0)
      setFinal(null)
      setStartedAt(Date.now())
      setPhase('prep')
    }catch(e){
      console.error(e)
      setError('Could not start session. Please try again.')
    }
  }

  async function onRecordingComplete({ base64 }){
    if(!sessionId) return
    try{
      setPhase('analyzing')
      const res = await sendAudio(sessionId, base64)

      if(res.finished){
        if(res.final_level){
          setFinal({
            level: res.final_level,
            score: res.final_score_0_8,
            confidence: res.final_confidence,
            feedback: res.feedback,
          })
        }
        setPhase('finished')
        return
      }
      // Not finished: prepare next turn
      setTurnsSoFar(t => t + 1)
      if(res.next_level) setCurrentLevel(res.next_level)
      if(res.next_prompt) setPrompt(res.next_prompt)
      setPhase('prep')
    }catch(e){
      console.error(e)
      setError('Upload failed. Please try again.')
      setPhase('prep')
    }
  }

  return (
    <div className="container">
      <h1>BELT Speaking AI — Adaptive</h1>
      {sessionId && <BudgetStrip startedAt={startedAt} turnsSoFar={turnsSoFar} />}

      <div className="card">
        <label>Candidate ID</label>
        <input value={candidateId} onChange={e=>setCandidateId(e.target.value)} />
        <label>Native Language</label>
        <input value={nativeLang} onChange={e=>setNativeLang(e.target.value)} placeholder="en, es, pt..." />
        <button className="btn" onClick={onStart} disabled={!!sessionId && phase!=='finished'}>Start Session</button>
        {error && <div className="small" style={{color:'var(--danger)', marginTop:8}}>{error}</div>}
      </div>

      {sessionId && phase !== 'finished' && (
        <div className="card">
          <div className="small">Level: <b>{currentLevel}</b></div>
          <div style={{marginTop:8}}>{prompt}</div>
        </div>
      )}

      {sessionId && phase !== 'finished' && (
        <Recorder
          autoStart={false}
          onComplete={onRecordingComplete}
          minSpeakSeconds={15}
          silenceStopSeconds={3}
          silenceThreshold={0.012}
          monitorFps={20}
          chunkMs={250}
        />
      )}

      {phase === 'analyzing' && (
        <div className="card">Analyzing…</div>
      )}

      {phase === 'finished' && final && (
        <div className="card">
          <h3>Final result</h3>
          <div>Level: <b>{final.level}</b></div>
          {final.score != null && <div>Score (0–8): <b>{final.score.toFixed ? final.score.toFixed(2) : final.score}</b></div>}
          {final.confidence != null && <div>Confidence: <b>{(final.confidence*100).toFixed(0)}%</b></div>}
        </div>
      )}
    </div>
  )
}
