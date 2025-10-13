import { useEffect, useRef, useState } from 'react'
import WaveformCanvas from './WaveformCanvas'

function blobToBase64(blob){
  return new Promise((resolve)=>{
    const reader = new FileReader();
    reader.onload = ()=> resolve(reader.result.split(',')[1])
    reader.readAsDataURL(blob)
  })
}

/**
 * Auto-stop logic:
 * - Accumulate "voiced" time when frame RMS > silenceThreshold
 * - If voiced >= minSpeakSeconds AND consecutive silence >= silenceStopSeconds => stop()
 */
export default function Recorder({
  autoStart = false,
  onComplete,
  minSpeakSeconds = 15,
  silenceStopSeconds = 3,
  silenceThreshold = 0.012,     // ~ -38 dBFS-ish; tweak if needed
  monitorFps = 20               // how often we sample audio for silence detection
}){
  const chunksRef = useRef([])
  const [stream, setStream] = useState(null)
  const [rec, setRec] = useState(null)
  const [recording, setRecording] = useState(false)

  // Audio monitoring
  const audioCtxRef = useRef(null)
  const sourceRef = useRef(null)
  const analyserRef = useRef(null)
  const rafRef = useRef(null)
  const voicedAccumRef = useRef(0)         // seconds of voiced speech (RMS > threshold)
  const silenceAccumRef = useRef(0)        // seconds of continuous silence after minSpeakSeconds reached
  const lastTickRef = useRef(0)

  useEffect(()=>{
    navigator.mediaDevices.getUserMedia({audio:true}).then(setStream)
  },[])

  useEffect(()=>{
    if(!stream) return
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
    // build monitor graph
    const ctx = new (window.AudioContext || window.webkitAudioContext)()
    const src = ctx.createMediaStreamSource(stream)
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 1024
    src.connect(analyser)
    audioCtxRef.current = ctx
    sourceRef.current = src
    analyserRef.current = analyser
    return ()=>{ try{ ctx.close() }catch{} }
  },[stream])

  useEffect(()=>{
    if(autoStart && rec && !recording){ start() }
  }, [autoStart, rec])

  function start(){
    if(!rec || rec.state==='recording') return
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
        // RMS
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
        <button className="btn" onClick={start} disabled={!rec || recording}>Start</button>
        <button className="btn" onClick={stop} disabled={!rec || !recording}>Stop</button>
        <div className="small" style={{marginLeft:'auto'}}>
          Auto-stop after {minSpeakSeconds}s spoken + {silenceStopSeconds}s silence
        </div>
      </div>
    </div>
  )
}
