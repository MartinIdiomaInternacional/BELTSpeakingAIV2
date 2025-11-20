import React from "react";

export default function LanguageSelect({ value, onChange }) {
  return (
    <div className="language-select">
      <label>
        Test language:
        <select value={value} onChange={(e) => onChange(e.target.value)}>
          <option value="en">English</option>
        </select>
      </label>
    </div>
  );
}
