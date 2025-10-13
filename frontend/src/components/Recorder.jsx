import { useEffect, useRef, useState } from 'react'
import WaveformCanvas from './WaveformCanvas'

function blobToBase64(blob){
  return new Promise((resolve)=>{
    const reader = new FileReader();
    reader.onload = ()=> resolve(reader.result.split(',')[1])
    reader.readAsDataURL(blob)
  })
}

export default function Recorder({
  autoStart = false,
  onComplete,
  minSpeakSeconds = 15,
  silenceStopSeconds = 3,
  silenceThreshold = 0.012,
  monitorFps = 20
}){
  const chunksRef = useRef([])
  const [stream, setStream] = useState(null)
  const [rec, setRec] = useState(null)
  const [recording, setRecording] = useState(false)
  const [err, setErr] = useState('')

  // Audio monitoring
  const audioCtxRef = useRef(null)
  const analyserRef = useRef(null)
  const rafRef = useRef(null)
  const voicedAccumRef = useRef(0)
  const silenceAccumRef = useRef(0)
  const lastTickRef = useRef(0)

  // Build a MediaRecorder only after we have a stream
  useEffect(()=>{
    if(!stream) return
    setErr('')
    const mr = new MediaRecorder(stream, { mimeType: 'audio/webm' })
    mr.ondataavailable = (e)=>{ if(e.data?.size>0) chunksRef.current.push(e.data) }
    mr.onstop = async ()=>{
      stopMonitor()
      const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
      chunksRef.current = []
      const base64 = await blobToBase64(blob)
      onComplete?.({ base64, sampleRate: 48000 })
    }
    setRec(mr)

    const ctx = new (window.AudioContext || window.webkitAudioContext)()
    const src = ctx.createMediaStreamSource(stream)
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 1024
    src.connect(analyser)
    audioCtxRef.current = ctx
    analyserRef.current = analyser
    return ()=>{ try{ ctx.close() }catch{} }
  },[stream])

  // If autoStart is requested and we can, start when ready (after stream exists)
  useEffect(()=>{
    if(autoStart){
      if(rec && !recording){ start() }
      else if(!stream){ requestStreamAndMaybeStart(true) }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoStart, rec, stream])

  async function requestStreamAndMaybeStart(startAfter=false){
    try{
      const s = await navigator.mediaDevices.getUserMedia({ audio: true })
      setStream(s)
      if(startAfter && rec){ start() } // if MediaRecorder already exists
      if(startAfter && !rec){
        // Wait a tick for rec to be created by the useEffect above
        setTimeout(()=> start(), 0)
      }
    }catch(e){
      console.error(e)
      setErr('Microphone permission denied or unavailable. Please allow mic access and try again.')
    }
  }

  function start(){
    if(!rec){
      // We don't have a stream yet: request it and start when ready
      requestStreamAndMaybeStart(true)
      return
    }
    if(rec.state==='recording') return
    voicedAccumRef.current = 0
    silenceAccumRef.current = 0
    lastTickRef.current = performance.now()
    rec.start()
    setRecording(true)
    startMonitor()
  }

  function stop(){
    if(rec && rec.state==='recording'){
      rec.stop()
      setRecording(false)
    }
  }

  function startMonitor(){
    const analyser = analyserRef.current
    if(!analyser) return
    const buf = new Float32Array(analyser.fftSize)
    const intervalMs = 1000/monitorFps

    function tick(){
      rafRef.current = setTimeout(()=>{
        analyser.getFloatTimeDomainData(buf)
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
