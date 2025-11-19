import { useEffect, useRef, useState } from 'react'

export default function Countdown({
  seconds = 30,
  onDone,            // called when countdown hits zero
  onStartNow,        // called when user clicks Start now
  showStartNow = true
}) {
  const [left, setLeft] = useState(seconds)
  const timerRef = useRef(null)

  useEffect(() => {
    timerRef.current = setInterval(() => {
      setLeft((s) => {
        if (s <= 1) {
          clearInterval(timerRef.current)
          onDone?.()
          return 0
        }
        return s - 1
      })
    }, 1000)
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [])

  const pct = ((seconds - left) / seconds) * 100
  let color = 'var(--ok)'
  if (left <= 15 && left > 5) color = 'var(--warn)'
  if (left <= 5) color = 'var(--danger)'

  function handleStartNow() {
    if (timerRef.current) clearInterval(timerRef.current)
    onStartNow?.()
  }

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <h3>Preparation time</h3>
        <div style={{ fontSize: 28, color }}>{left}s</div>
      </div>
      <div className="progress" aria-label="countdown">
        <div style={{ width: pct + '%', background: color }} />
      </div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 8 }}>
        <p className="small" style={{ margin: 0 }}>Recording will start automatically.</p>
        {showStartNow && (
          <button className="btn" onClick={handleStartNow}>
            Start recording now
          </button>
        )}
      </div>
    </div>
  )
}
