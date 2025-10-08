
import { useEffect, useRef, useState } from 'react'
import WaveformCanvas from './WaveformCanvas'

function blobToBase64(blob){
  return new Promise((resolve)=>{
    const reader = new FileReader();
    reader.onload = ()=> resolve(reader.result.split(',')[1])
    reader.readAsDataURL(blob)
  })
}

export default function Recorder({ autoStart=false, onComplete }){
  const chunksRef = useRef([])
  const [stream, setStream] = useState(null)
  const [rec, setRec] = useState(null)
  const [recording, setRecording] = useState(false)

  useEffect(()=>{ navigator.mediaDevices.getUserMedia({audio:true}).then(setStream) },[])

  useEffect(()=>{
    if(!stream) return
    const mr = new MediaRecorder(stream, { mimeType: 'audio/webm' })
    mr.ondataavailable = (e)=>{ if(e.data?.size>0) chunksRef.current.push(e.data) }
    mr.onstop = async ()=>{
      const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
      chunksRef.current = []
      const base64 = await blobToBase64(blob)
      onComplete?.({ base64, sampleRate: 48000 })
    }
    setRec(mr)
  },[stream])

  useEffect(()=>{ if(autoStart && rec && !recording){ start() } }, [autoStart, rec])

  function start(){ if(rec && rec.state!=='recording'){ rec.start(); setRecording(true) } }
  function stop(){ if(rec && rec.state==='recording'){ rec.stop(); setRecording(false) } }

  return (
    <div className="card">
      <h3>Recording</h3>
      <WaveformCanvas stream={stream} />
      <div style={{display:'flex', gap:8, marginTop:8}}>
        <button className="btn" onClick={start} disabled={!rec || recording}>Start</button>
        <button className="btn" onClick={stop} disabled={!rec || !recording}>Stop</button>
      </div>
    </div>
  )
}
