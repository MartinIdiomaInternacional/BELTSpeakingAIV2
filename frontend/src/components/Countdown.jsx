
import { useEffect, useState } from 'react'
export default function Countdown({ seconds=30, onDone }) {
  const [left, setLeft] = useState(seconds)
  useEffect(()=>{
    const t = setInterval(()=>setLeft((s)=>{
      if(s<=1){ clearInterval(t); onDone?.(); return 0 }
      return s-1
    }), 1000)
    return ()=>clearInterval(t)
  },[])
  const pct = ((seconds-left)/seconds)*100
  let color = 'var(--ok)'
  if(left<=15 && left>5) color = 'var(--warn)'
  if(left<=5) color = 'var(--danger)'
  return (
    <div className="card">
      <div style={{display:'flex', justifyContent:'space-between', alignItems:'baseline'}}>
        <h3>Preparation time</h3>
        <div style={{fontSize:28, color}}>{left}s</div>
      </div>
      <div className="progress" aria-label="countdown">
        <div style={{ width: pct+'%', background: color }} />
      </div>
      <p className="small">Recording starts automatically.</p>
    </div>
  )
}
