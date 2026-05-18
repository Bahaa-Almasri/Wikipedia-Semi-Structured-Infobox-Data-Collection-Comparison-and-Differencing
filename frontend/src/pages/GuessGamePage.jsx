import { useState } from "react";

import GameRound from "../components/game/GameRound.jsx";
import RoundResultModal from "../components/game/RoundResultModal.jsx";
import ScoreBoard from "../components/game/ScoreBoard.jsx";
import ErrorState from "../components/shared/ErrorState.jsx";
import LoadingState from "../components/shared/LoadingState.jsx";
import { getGuessRound, submitGuessAnswer } from "../services/gameService.js";

const TOTAL_ROUNDS = 5;

export default function GuessGamePage() {
  const [started, setStarted] = useState(false);
  const [roundNumber, setRoundNumber] = useState(1);
  const [score, setScore] = useState(0);
  const [roundData, setRoundData] = useState(null);
  const [selected, setSelected] = useState("");
  const [result, setResult] = useState(null);
  const [finished, setFinished] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadRound = async (nextRound = roundNumber) => {
    setLoading(true);
    setError("");
    setResult(null);
    setSelected("");
    try {
      setRoundData(await getGuessRound());
      setRoundNumber(nextRound);
    } catch (err) {
      setError(err.message || "Could not load a game round.");
    } finally {
      setLoading(false);
    }
  };

  const start = async () => {
    setStarted(true);
    setFinished(false);
    setScore(0);
    await loadRound(1);
  };

  const submit = async () => {
    setLoading(true);
    setError("");
    try {
      const payload = await submitGuessAnswer(roundData.round_id, selected);
      setResult(payload);
      if (payload.correct) setScore((current) => current + 1);
    } catch (err) {
      setError(err.message || "Could not submit your answer.");
    } finally {
      setLoading(false);
    }
  };

  const next = async () => {
    if (roundNumber >= TOTAL_ROUNDS) {
      setFinished(true);
      return;
    }
    await loadRound(roundNumber + 1);
  };

  if (!started) {
    return (
      <section className="hero-panel">
        <p className="eyebrow">Guess the Country</p>
        <h1>Use similarity clues to identify the hidden country.</h1>
        <p>Each clue is a nearby country profile. Pick the mystery country before the round ends.</p>
        <button className="primary-button" type="button" onClick={start}>
          Start game
        </button>
      </section>
    );
  }

  if (finished) {
    return (
      <section className="hero-panel">
        <p className="eyebrow">Final score</p>
        <h1>{score} out of {TOTAL_ROUNDS}</h1>
        <p>You solved {Math.round((score / TOTAL_ROUNDS) * 100)}% of the country neighbor clues.</p>
        <button className="primary-button" type="button" onClick={start}>
          Play again
        </button>
      </section>
    );
  }

  return (
    <div className="page-stack">
      <ScoreBoard score={score} round={roundNumber} totalRounds={TOTAL_ROUNDS} />
      {loading && !roundData && <LoadingState message="Preparing a mystery country..." />}
      {error && <ErrorState message={error} onRetry={() => loadRound(roundNumber)} />}
      {roundData && (
        <GameRound
          roundData={roundData}
          selected={selected}
          onSelect={setSelected}
          onSubmit={submit}
          disabled={loading || Boolean(result)}
        />
      )}
      <RoundResultModal result={result} onNext={next} isFinal={roundNumber >= TOTAL_ROUNDS} />
    </div>
  );
}
