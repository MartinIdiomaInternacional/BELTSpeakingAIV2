import React, { useState, useMemo } from "react";
import Recorder from "./components/Recorder";
import LanguageSelect from "./components/LanguageSelect";
import DimensionRadar from "./components/DimensionRadar";

const TASKS = [
  {
    id: 1,
    title: "Task 1 – Personal Introduction",
    maxSeconds: 45,
    prompts: [
      "Tell me about your background, where you live, what you do, and one interesting fact about yourself.",
      "Introduce yourself: where you’re from, what you do, and what you enjoy doing in your free time.",
      "Tell me about your daily routine and one thing you’re currently working on or learning.",
      "Describe your hometown or neighborhood and what you like (or don’t like) about it.",
    ],
  },
  {
    id: 2,
    title: "Task 2 – Describe a Situation",
    maxSeconds: 60,
    prompts: [
      "Describe a challenging situation you faced recently and how you handled it.",
      "Talk about a time something didn’t go as planned. What happened and what did you do?",
      "Describe a problem you had at work or school and how it was resolved.",
      "Tell me about a difficult decision you had to make. What did you choose and why?",
    ],
  },
  {
    id: 3,
    title: "Task 3 – Opinion Question",
    maxSeconds: 60,
    prompts: [
      "Do you think technology has improved communication? Why or why not?",
      "Should people work from home more often? Why or why not?",
      "Do social media platforms do more harm than good? Explain your opinion.",
      "Is it better to specialize in one skill or be a generalist? Why?",
    ],
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

// Dimension weights (sum to 1.00)
const DIMENSION_WEIGHTS = {
  fluency: 0.25,
  pronunciation: 0.2,
  coherence: 0.15,
  grammatical_range: 0.15,
  grammatical_accuracy: 0.1,
  lexical_range: 0.1,
  lexical_control: 0.05,
};

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
  // global = 0.6 * weighted_dim_avg + 0.4 * avg_task_total
  // ------------------------------------------------------------
  const globalSummary = useMemo(() => {
    const completedResults = Object.values(results).filter(Boolean);
    if (completedResults.length !== TASKS.length) return null;

    // Build per-dimension accumulators across all tasks
    let perDim = {};
    DIMENSIONS.forEach((d) => (perDim[d] = { sum: 0, count: 0 }));

    // Collect task total scores (0–6) from GPT results
    let taskTotals = [];

    completedResults.forEach((r) => {
      if (typeof r.text_total_score === "number") {
        taskTotals.push(r.text_total_score);
      }

      const dims = r.text_dimensions || {};
      DIMENSIONS.forEach((d) => {
        const info = dims[d];
        if (info && typeof info.score === "number") {
          perDim[d].sum += info.score;
          perDim[d].count += 1;
        }
      });
    });

    if (!taskTotals.length) return null;

    // Average of task total scores (0–6)
    const avgTaskTotal =
      taskTotals.reduce((a, b) => a + b, 0) / taskTotals.length;

    // Dimension averages (0–6) + weighted dimension score
    let dimensionAverages = {};
    let weightedSum = 0;
    let weightTotal = 0;

    DIMENSIONS.forEach((d) => {
      if (perDim[d].count > 0) {
        const avg = perDim[d].sum / perDim[d].count;
        dimensionAverages[d] = avg;

        const w = DIMENSION_WEIGHTS[d] || 0;
        weightedSum += avg * w;
        weightTotal += w;
      }
    });

    const avgDimScore = weightTotal > 0 ? weightedSum / weightTotal : null;
    if (avgDimScore == null) return null;

    // Hybrid score (0–6)
    const hybridScore = 0.6 * avgDimScore + 0.4 * avgTaskTotal;

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
  showFeedback={isLastTask}   // ✅ only show feedback on Task 3
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

          <p className="global-level">
            <strong>Global CEFR Level:</strong>{" "}
            <span className="level-badge">{globalSummary.globalLevel}</span>
          </p>

          <p className="score-details">
            Hybrid score (0–6):{" "}
            <strong>{globalSummary.hybridScore.toFixed(2)}</strong>
            <br />
            <small>
              (60% weighted dimension score + 40% average task total)
            </small>
          </p>

          {/* ✅ Add the Radar here */}
          <h3>Dimension Profile</h3>
          <DimensionRadar
            dimensionAverages={globalSummary.dimensionAverages}
          />

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
