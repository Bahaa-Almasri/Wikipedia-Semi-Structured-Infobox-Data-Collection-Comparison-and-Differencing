export default function QuizStepper({ current, total }) {
  const percent = Math.round(((current + 1) / total) * 100);
  return (
    <div className="quiz-stepper">
      <span>
        Step {current + 1} of {total}
      </span>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}
