import React, { useState } from "react";
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

export default function App() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [language, setLanguage] = useState("en");
  const [results, setResults] = useState({});

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
    </div>
  );
}
