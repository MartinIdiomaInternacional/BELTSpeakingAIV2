
import { useEffect, useRef } from 'react'
export default function WaveformCanvas({ stream }){
  const ref = useRef(null)
  useEffect(()=>{
    if(!stream) return
    const audioCtx = new (window.AudioContext||window.webkitAudioContext)()
    const src = audioCtx.createMediaStreamSource(stream)
    const analyser = audioCtx.createAnalyser(); analyser.fftSize=1024
    src.connect(analyser)
    const data = new Uint8Array(analyser.fftSize)
    const canvas = ref.current
    const ctx = canvas.getContext('2d')
    function draw(){
      requestAnimationFrame(draw)
      analyser.getByteTimeDomainData(data)
      ctx.clearRect(0,0,canvas.width,canvas.height)
      ctx.beginPath()
      const h=canvas.height,w=canvas.width
      for(let i=0;i<data.length;i++){
        const x = w * i / data.length
        const y = (data[i]/255.0) * h
        if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y)
      }
      ctx.stroke()
    }
    draw()
    return ()=>audioCtx.close()
  },[stream])
  return <canvas className="wave" ref={ref} width={600} height={80} />
}
