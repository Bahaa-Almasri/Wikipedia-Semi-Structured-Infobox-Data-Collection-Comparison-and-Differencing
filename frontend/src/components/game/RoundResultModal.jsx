export default function RoundResultModal({ result, onNext, isFinal }) {
  if (!result) return null;

  return (
    <section className={result.correct ? "result-card correct" : "result-card incorrect"}>
      <h2>{result.correct ? "Correct" : "Not quite"}</h2>
      <p>
        The country was <strong>{result.correct_display_name}</strong>.
      </p>
      <p>{result.explanation}</p>
      <button className="primary-button" type="button" onClick={onNext}>
        {isFinal ? "See final score" : "Next round"}
      </button>
    </section>
  );
}
