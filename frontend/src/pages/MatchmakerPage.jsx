import { useState } from "react";

import MatchResults from "../components/matchmaker/MatchResults.jsx";
import QuizQuestionCard from "../components/matchmaker/QuizQuestionCard.jsx";
import QuizStepper from "../components/matchmaker/QuizStepper.jsx";
import ErrorState from "../components/shared/ErrorState.jsx";
import LoadingState from "../components/shared/LoadingState.jsx";
import { recommendMatches } from "../services/matchmakerService.js";

const questions = [
  {
    key: "size",
    kicker: "Country size",
    title: "What kind of country scale feels right?",
    options: [
      { value: "small", label: "Small", description: "Compact and easy to explore" },
      { value: "medium", label: "Medium", description: "Balanced national scale" },
      { value: "large", label: "Large", description: "Big landscapes and variety" },
      { value: "any", label: "No preference", description: "Surprise me" },
    ],
  },
  {
    key: "population",
    kicker: "Population style",
    title: "Which population profile sounds appealing?",
    options: [
      { value: "less", label: "Less populated" },
      { value: "moderate", label: "Moderately populated" },
      { value: "high", label: "Highly populated" },
      { value: "any", label: "No preference" },
    ],
  },
  {
    key: "geography",
    kicker: "Geographic character",
    title: "Pick a landscape personality.",
    options: [
      { value: "island", label: "Island nation" },
      { value: "coastal", label: "Coastal country" },
      { value: "landlocked", label: "Landlocked country" },
      { value: "any", label: "Open to anywhere" },
    ],
  },
  {
    key: "government",
    kicker: "Public profile",
    title: "Which civic profile should we look for?",
    options: [
      { value: "republic", label: "Republic" },
      { value: "monarchy", label: "Monarchy" },
      { value: "parliamentary", label: "Parliamentary style" },
      { value: "any", label: "No preference" },
    ],
  },
  {
    key: "language_profile",
    kicker: "Language profile",
    title: "What language setting do you prefer?",
    options: [
      { value: "single", label: "One dominant official language" },
      { value: "multilingual", label: "Multiple official languages" },
      { value: "any", label: "No preference" },
    ],
  },
  {
    key: "include_surprising_matches",
    kicker: "Discovery mode",
    title: "Should we include unexpected matches?",
    options: [
      { value: true, label: "Yes", description: "Add some surprise" },
      { value: false, label: "No", description: "Keep it focused" },
    ],
  },
];

export default function MatchmakerPage() {
  const [started, setStarted] = useState(false);
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState({ limit: 8 });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const currentQuestion = questions[step];

  const answer = (key, value) => {
    setAnswers((current) => ({ ...current, [key]: value }));
  };

  const submit = async () => {
    setLoading(true);
    setError("");
    try {
      setResult(await recommendMatches(answers));
    } catch (err) {
      setError(err.message || "Could not create recommendations.");
    } finally {
      setLoading(false);
    }
  };

  if (!started) {
    return (
      <section className="hero-panel">
        <p className="eyebrow">CountryScope Matchmaker</p>
        <h1>What kind of country feels like your match?</h1>
        <p>Answer a few friendly questions and discover countries that fit your profile.</p>
        <button className="primary-button" type="button" onClick={() => setStarted(true)}>
          Start matching
        </button>
      </section>
    );
  }

  if (loading) return <LoadingState message="Finding your country matches..." />;

  return (
    <div className="page-stack">
      {!result && (
        <>
          <QuizStepper current={step} total={questions.length} />
          <QuizQuestionCard
            question={currentQuestion}
            value={answers[currentQuestion.key]}
            onAnswer={answer}
          />
          <div className="button-row">
            <button className="ghost-button" type="button" disabled={step === 0} onClick={() => setStep(step - 1)}>
              Back
            </button>
            {step < questions.length - 1 ? (
              <button className="primary-button" type="button" onClick={() => setStep(step + 1)}>
                Next
              </button>
            ) : (
              <button className="primary-button" type="button" onClick={submit}>
                Show my matches
              </button>
            )}
          </div>
        </>
      )}
      {error && <ErrorState message={error} onRetry={submit} />}
      {result && <MatchResults result={result} />}
    </div>
  );
}
