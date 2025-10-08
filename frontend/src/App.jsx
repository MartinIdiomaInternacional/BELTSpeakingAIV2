import { useState } from 'react'
import Countdown from './components/Countdown'
import Recorder from './components/Recorder'
import LanguageSelect from './components/LanguageSelect'
import { startSession, evaluateBytes, getReport } from './lib/api'
import './styles.css'

export default function App() {
  const [candidateId, setCandidateId] = useState('demo-123')
  const [nativeLang, setNativeLang] = useState('es')

  const [session, setSession] = useState(null)
  const [prompt, setPrompt] = useState(null)

  // setup -> prep -> record -> result
  const [phase, setPhase] = useState('setup')

  // autoRecord only matters on entering 'record'
  const [autoRecord, setAutoRecord] = useState(false)

  const [lastTurn, setLastTurn] = useState(null)
  const [reportHtml, setReportHtml] = useState('')

  async function onStart() {
    const res = await startSession(candidateId, nativeLang)
    setSession(res.session_id)
    setPrompt(res.prompt)              // ensure prompt is ready first
    setAutoRecord(false)
    setPhase('prep')                   // then move to prep
  }

  function enterRecording() {
    // Called by countdown expiry OR “Start recording now”
    setAutoRecord(true)
    setPhase('record')
  }

  async function onRecordingComplete({ base64, sampleRate }) {
    // Block UI from staying in record mode while we wait
    setPhase('prep')                   // move to prep view while we compute
    setAutoRecord(false)

    const r = await evaluateBytes(session, sampleRate, base64)
    setLastTurn(r.turn)

    if (!r.finished) {
      // Make sure the next prompt is visible BEFORE any recording can start again
      if (r.next_prompt) setPrompt(r.next_prompt)
      // Stay in 'prep' until user starts or countdown ends
      setAutoRecord(false)
      setPhase('prep')
    } else {
      const rep = await getReport(session, nativeLang)
      setReportHtml(rep.html)
      setPhase('result')
    }
  }

  return (
    <div style={{ maxWidth: 860, margin: '0 auto', display: 'grid', gap: 12 }}>
      <h1>Working Speaking AI Eval 2.2 — Adaptive Pro</h1>

      {phase === 'setup' && (
        <div className="card">
          <div style={{ display: 'grid', gap: 8 }}>
            <label>Candidate ID
              <input value={candidateId} onChange={e => setCandidateId(e.target.value)} />
            </label>
            <LanguageSelect onChange={setNativeLang} />
            <button className="btn" onClick={onStart}>Start</button>
          </div>
        </div>
      )}

      {phase !== 'setup' && prompt && (
        <div className="card">
          <h3>Prompt</h3>
          <p style={{ marginBottom: 0 }}>{prompt}</p>
        </div>
      )}

      {phase === 'prep' && (
        <Countdown
          seconds={30}
          onDone={enterRecording}
          onStartNow={enterRecording}
          showStartNow
        />
      )}

      {phase === 'record' && (
        <Recorder autoStart={autoRecord} onComplete={onRecordingComplete} />
      )}

      {lastTurn && (
        <div className="card">
          <h3>Last Turn</h3>
          <div>Asked: <strong>{lastTurn.asked_level}</strong></div>
          {!lastTurn.quality_ok && (
            <div style={{ color: 'var(--danger)' }}>Quality: {lastTurn.quality_reason}</div>
          )}
          {lastTurn.quality_ok && (
            <div>
              {lastTurn.inferred_level && <div>Inferred: <strong>{lastTurn.inferred_level}</strong></div>}
              {lastTurn.score_0_8 != null && <div>Score: {lastTurn.score_0_8.toFixed(2)} / 8</div>}
              {lastTurn.confidence != null && <div>Confidence: {lastTurn.confidence.toFixed(2)}</div>}
              {lastTurn.transcription && (
                <div style={{ marginTop: 8 }}>
                  <div className="label">Auto transcription (optional)</div>
                  <div style={{ whiteSpace: 'pre-wrap' }}>{lastTurn.transcription}</div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {phase === 'result' && (
        <div className="card">
          <h3>Final Report</h3>
          <div dangerouslySetInnerHTML={{ __html: reportHtml }} />
        </div>
      )}
    </div>
  )
}
