import DimensionRadar from "./components/DimensionRadar";
import React, { useState, useMemo } from "react";
import Recorder from "./components/Recorder";
import LanguageSelect from "./components/LanguageSelect";

const TASKS = [
  {
    id: 1,
    title: "Task 1 – Personal Introduction",
    text: "Tell me about your background, where you live, what you do, and one interesting fact about yourself.",
    maxSeconds: 45,
  },
  {
    id: 2,
    title: "Task 2 – Describe a Situation",
    text: "Describe a challenging situation you faced recently and how you handled it.",
    maxSeconds: 60,
  },
  {
    id: 3,
    title: "Task 3 – Opinion Question",
    text: "Do you think technology has improved communication? Why or why not?",
    maxSeconds: 60,
  },
];

// Dimensions returned by the backend
const DIMENSIONS = [
  "fluency",
  "grammatical_range",
  "grammatical_accuracy",
  "lexical_range",
  "lexical_control",
  "pronunciation",
  "coherence",
];

// Numeric (0–6) → CEFR label
function numericToLevel(score) {
  if (score == null || isNaN(score)) return "N/A";
  if (score < 0.5) return "A1";
  if (score < 1.5) return "A2";
  if (score < 2.5) return "B1";
  if (score < 3.5) return "B1+";
  if (score < 4.5) return "B2";
  if (score < 5.5) return "B2+";
  return "C1/C2";
}

export default function App() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [language, setLanguage] = useState("en");
  const [results, setResults] = useState({}); // { taskId: evaluationResult }

  const currentTask = TASKS[currentIndex];

  const handleFinished = (taskId, data) => {
    setResults((prev) => ({ ...prev, [taskId]: data }));
  };

  const nextTask = () => {
    if (currentIndex < TASKS.length - 1) {
      setCurrentIndex((i) => i + 1);
    }
  };

  const isLastTask = currentIndex === TASKS.length - 1;

  // ------------------------------------------------------------
  // GLOBAL HYBRID SUMMARY (Option C)
  // ------------------------------------------------------------
  const globalSummary = useMemo(() => {
    const completedResults = Object.values(results).filter(Boolean);
    if (completedResults.length !== TASKS.length) return null;

    let dimScores = [];
    let perDim = {};
    DIMENSIONS.forEach((d) => (perDim[d] = { sum: 0, count: 0 }));

    let taskTotals = [];

    completedResults.forEach((r) => {
      if (typeof r.text_total_score === "number") {
        taskTotals.push(r.text_total_score);
      }

      const dims = r.text_dimensions || {};
      DIMENSIONS.forEach((d) => {
        const info = dims[d];
        if (info && typeof info.score === "number") {
          dimScores.push(info.score);
          perDim[d].sum += info.score;
          perDim[d].count += 1;
        }
      });
    });

    if (!dimScores.length || !taskTotals.length) return null;

    const avgDimScore =
      dimScores.reduce((a, b) => a + b, 0) / dimScores.length;

    const avgTaskTotal =
      taskTotals.reduce((a, b) => a + b, 0) / taskTotals.length;

    const hybridScore = 0.6 * avgDimScore + 0.4 * avgTaskTotal;

    let dimensionAverages = {};
    DIMENSIONS.forEach((d) => {
      if (perDim[d].count > 0) {
        dimensionAverages[d] = perDim[d].sum / perDim[d].count;
      }
    });

    return {
      avgDimScore,
      avgTaskTotal,
      hybridScore,
      globalLevel: numericToLevel(hybridScore),
      dimensionAverages,
    };
  }, [results]);

  // ------------------------------------------------------------
  // RENDER
  // ------------------------------------------------------------
  return (
    <div className="app">
      <header className="app-header">
        <h1>BELT Speaking AI 2.0</h1>
        <p className="subtitle">
          Prototype – automatic speaking evaluation with CEFR-like levels.
        </p>
      </header>

      <LanguageSelect value={language} onChange={setLanguage} />

      <Recorder
        key={currentTask.id}
        taskId={currentTask.id}
        task={currentTask}
        onFinished={handleFinished}
      />

      <div className="task-navigation">
        <button
          onClick={nextTask}
          disabled={isLastTask}
          className="btn tertiary"
        >
          {isLastTask ? "No more tasks" : "Next task"}
        </button>
        <p>
          Task {currentIndex + 1} of {TASKS.length}
        </p>
      </div>

      {/* ------------------------------------------------------ */}
      {/* GLOBAL RESULTS (shown only after all 3 tasks) */}
      {/* ------------------------------------------------------ */}
      {globalSummary && (
        <section className="global-results">
          <h2>Overall Speaking Result</h2>
<h3>Dimension Profile</h3>
<DimensionRadar
  dimensionAverages={globalSummary.dimensionAverages}
/>

          <p className="global-level">
            <strong>Global CEFR Level:</strong>{" "}
            <span className="level-badge">
              {globalSummary.globalLevel}
            </span>
          </p>

          <p className="score-details">
            Hybrid score (0–6):{" "}
            <strong>{globalSummary.hybridScore.toFixed(2)}</strong>{" "}
            <br />
            <small>
              (60% dimension averages + 40% task totals)
            </small>
          </p>

          <h3>Dimension Averages</h3>
          <table className="dimension-table">
            <thead>
              <tr>
                <th>Dimension</th>
                <th>Average Score (0–6)</th>
                <th>CEFR</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(globalSummary.dimensionAverages).map(
                ([dim, score]) => (
                  <tr key={dim}>
                    <td>{dim.replace(/_/g, " ")}</td>
                    <td>{score.toFixed(2)}</td>
                    <td>{numericToLevel(score)}</td>
                  </tr>
                )
              )}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
