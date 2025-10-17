import { useEffect, useRef, useState } from 'react'
import WaveformCanvas from './WaveformCanvas'

function blobToBase64(blob){
  return new Promise((resolve)=>{
    const reader = new FileReader()
    reader.onload = ()=> resolve(reader.result.split(',')[1])
    reader.readAsDataURL(blob)
  })
}

function pickMime(){
  const prefs = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/ogg',
    'audio/mp4',            // Safari 17+
  ]
  for (const t of prefs){
    if (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) return t
  }
  return '' // let browser choose
}

export default function Recorder({
  autoStart = false,
  onComplete,
  minSpeakSeconds = 15,
  silenceStopSeconds = 3,
  silenceThreshold = 0.012,
  monitorFps = 20,
  chunkMs = 250,            // <= IMPORTANT: periodic chunks
}){
  const chunksRef = useRef([])
  const [stream, setStream] = useState(null)
  const recRef = useRef(null)
  const [recording, setRecording] = useState(false)
  const [err, setErr] = useState('')

  // analysis
  const audioCtxRef = useRef(null)
  const analyserRef = useRef(null)
  const rafRef = useRef(null)
  const voicedAccumRef = useRef(0)
  const silenceAccumRef = useRef(0)
  const lastTickRef = useRef(0)
  const doneRef = useRef(false)       // ensure onComplete fires once

  // Build analyser when stream exists
  useEffect(()=>{
    if(!stream) return
    const ctx = new (window.AudioContext || window.webkitAudioContext)()
    const src = ctx.createMediaStreamSource(stream)
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 1024
    src.connect(analyser)
    audioCtxRef.current = ctx
    analyserRef.current = analyser
    return ()=>{ try{ ctx.close() }catch{} }
  }, [stream])

  // Auto-start if requested
  useEffect(()=>{
    if(autoStart){
      start()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoStart])

  async function ensureStream(){
    if (stream) return stream
    try{
      const s = await navigator.mediaDevices.getUserMedia({ audio: true })
      setStream(s)
      return s
    }catch(e){
      console.error(e)
      setErr('Microphone permission denied or unavailable. Please allow mic access and try again.')
      throw e
    }
  }

  async function start(){
    try{
      const s = await ensureStream()
      setErr('')

      // (re)create MediaRecorder every start to avoid stale handlers
      const mimeType = pickMime()
      const mr = new MediaRecorder(s, mimeType ? { mimeType } : {})
      recRef.current = mr
      chunksRef.current = []
      doneRef.current = false

      mr.ondataavailable = (e)=>{
        if(e.data && e.data.size > 0){
          chunksRef.current.push(e.data)
        }
      }

      mr.onerror = (e)=> {
        console.error('MediaRecorder error', e)
        setErr('Recording error. Please try again.')
      }

      mr.onstop = async ()=>{
        stopMonitor()
        try{
          const blob = new Blob(chunksRef.current, { type: mimeType || 'audio/webm' })
          chunksRef.current = []
          if (!doneRef.current){
            doneRef.current = true
            const base64 = await blobToBase64(blob)
            // close tracks after we have the blob
            try{
              s.getTracks().forEach(t => t.stop())
            }catch{}
            onComplete?.({ base64, sampleRate: 48000 })
          }
        }catch(err2){
          console.error(err2)
          setErr('Could not finalize audio. Please try again.')
        }
      }

      // reset counters & start
      voicedAccumRef.current = 0
      silenceAccumRef.current = 0
      lastTickRef.current = performance.now()
      mr.start(chunkMs)                 // <= periodic chunks
      setRecording(true)
      startMonitor()
    }catch(_){}
  }

  function stop(){
    const mr = recRef.current
    if(!mr || mr.state !== 'recording') return
    try{
      // flush last chunk then stop
      mr.requestData()
    }catch{}
    try{
      mr.stop()
    }catch(e){
      console.error(e)
    }
    setRecording(false)
  }

  function startMonitor(){
    const analyser = analyserRef.current
    if(!analyser) return
    const buf = new Float32Array(analyser.fftSize)
    const intervalMs = 1000/monitorFps

    const tick = ()=>{
      rafRef.current = setTimeout(()=>{
        try{
          analyser.getFloatTimeDomainData(buf)
        }catch{
          // If stream closed while stopping
          return
        }
        let sum = 0
        for(let i=0;i<buf.length;i++){ const v = buf[i]; sum += v*v }
        const rms = Math.sqrt(sum / buf.length)

        const now = performance.now()
        const dt = Math.max(0, (now - lastTickRef.current)/1000)
        lastTickRef.current = now

        if(rms > silenceThreshold){
          voicedAccumRef.current += dt
          silenceAccumRef.current = 0
        } else {
          if(voicedAccumRef.current >= minSpeakSeconds){
            silenceAccumRef.current += dt
            if(silenceAccumRef.current >= silenceStopSeconds){
              stop()
              return
            }
          }
        }
        tick()
      }, intervalMs)
    }
    lastTickRef.current = performance.now()
    tick()
  }

  function stopMonitor(){
    if(rafRef.current){ clearTimeout(rafRef.current); rafRef.current = null }
  }

  return (
    <div className="card">
      <h3>Recording</h3>
      <WaveformCanvas stream={stream} />
      <div style={{display:'flex', gap:8, marginTop:8}}>
        <button className="btn" onClick={start} disabled={recording}>Start</button>
        <button className="btn" onClick={stop} disabled={!recording}>Stop</button>
        <div className="small" style={{marginLeft:'auto'}}>
          Auto-stop after {minSpeakSeconds}s spoken + {silenceStopSeconds}s silence
        </div>
      </div>
      {err && <div className="small" style={{color:'var(--danger)', marginTop:8}}>{err}</div>}
    </div>
  )
}
