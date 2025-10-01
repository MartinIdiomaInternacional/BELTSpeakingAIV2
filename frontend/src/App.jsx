import { useState } from 'react'
import Countdown from './components/Countdown'
import Recorder from './components/Recorder'
import LanguageSelect from './components/LanguageSelect'
import { startSession, getPrompts, evaluateBytes, getReport } from './lib/api'
import './styles.css'

export default function App(){
  const [candidateId, setCandidateId] = useState('demo-123')
  const [nativeLang, setNativeLang] = useState('es')
  const [targetLevel, setTargetLevel] = useState('B1')

  const [session, setSession] = useState(null)
  const [prompts, setPrompts] = useState([])
  const [currentPrompt, setCurrentPrompt] = useState(null)
  const [phase, setPhase] = useState('setup') // setup -> prep -> record -> result / probe
  const [autoRecord, setAutoRecord] = useState(false)
  const [lastEval, setLastEval] = useState(null)
  const [reportHtml, setReportHtml] = useState('')

  async function onStart(){
    const res = await startSession(candidateId, nativeLang, targetLevel)
    setSession(res.session_id)
    const ps = await getPrompts(targetLevel)
    setPrompts(ps)
    setCurrentPrompt(ps[0])
    setPhase('prep')
  }

  function onPrepDone(){ setAutoRecord(true); setPhase('record') }

  async function onRecordingComplete({ base64, sampleRate }){
    const r = await evaluateBytes(session, sampleRate, base64, currentPrompt?.id)
    setLastEval(r)
    if(r.needs_probe){
      setPhase('probe')
    } else {
      setPhase('result')
      const rep = await getReport(session, nativeLang)
      setReportHtml(rep.html)
    }
  }

  function startProbe(){
    setAutoRecord(false)
    setPhase('prep')
    setCurrentPrompt({ id:'probe', text: lastEval?.probe_prompt || 'Please expand your previous answer.' })
  }

  return (
    <div style={{maxWidth:860, margin:'0 auto', display:'grid', gap:12}}>
      <h1>Working Speaking AI Eval 2.0</h1>

      {phase==='setup' && (
        <div className="card">
          <div style={{display:'grid', gap:8}}>
            <label>Candidate ID
              <input value={candidateId} onChange={e=>setCandidateId(e.target.value)} />
            </label>
            <label>Target Level
              <select value={targetLevel} onChange={(e)=>setTargetLevel(e.target.value)}>
                {['A1','A2','B1','B1+','B2','B2+','C1','C2'].map(l=>(<option key={l} value={l}>{l}</option>))}
              </select>
            </label>
            <LanguageSelect onChange={setNativeLang} />
            <button className="btn" onClick={onStart}>Start</button>
          </div>
        </div>
      )}

      {phase!=='setup' && currentPrompt && (
        <div className="card">
          <h3>Prompt</h3>
          <p>{currentPrompt.text}</p>
        </div>
      )}

      {phase==='prep' && (
        <Countdown seconds={30} onDone={onPrepDone} />
      )}

      {phase==='record' && (
        <Recorder autoStart={autoRecord} onComplete={onRecordingComplete} />
      )}

      {phase==='probe' && (
        <div className="card">
          <p>We need a short follow-up sample to confirm your level.</p>
          <button className="btn" onClick={startProbe}>Start Probe</button>
        </div>
      )}

      {phase==='result' && (
        <div className="card">
          <h3>Results</h3>
          {lastEval && (
            <div>
              <div>Level: <strong>{lastEval.base.level}</strong></div>
              <div>Score: {lastEval.base.score_0_8.toFixed(2)} / 8</div>
              <div>Confidence: {lastEval.base.confidence.toFixed(2)}</div>
            </div>
          )}
          <div style={{marginTop:12}}>
            <h3>Report</h3>
            <div dangerouslySetInnerHTML={{__html: reportHtml}} />
          </div>
        </div>
      )}
    </div>
  )
}
