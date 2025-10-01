import { useState } from 'react'

export default function LanguageSelect({ onChange }) {
  const [lang, setLang] = useState('es')
  return (
    <div className="card" style={{marginBottom:16}}>
      <div className="label">Native language for feedback</div>
      <select value={lang} onChange={(e)=>{ setLang(e.target.value); onChange?.(e.target.value); }}>
        <option value="en">English</option>
        <option value="es">Español</option>
        <option value="pt">Português</option>
        <option value="fr">Français</option>
      </select>
    </div>
  )
}
