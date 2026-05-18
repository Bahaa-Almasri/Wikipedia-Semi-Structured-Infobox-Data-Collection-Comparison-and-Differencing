export default function QuizQuestionCard({ question, value, onAnswer }) {
  return (
    <section className="quiz-card">
      <p className="eyebrow">{question.kicker}</p>
      <h2>{question.title}</h2>
      <div className="answer-grid">
        {question.options.map((option) => (
          <button
            key={option.value}
            type="button"
            className={value === option.value ? "answer-card selected" : "answer-card"}
            onClick={() => onAnswer(question.key, option.value)}
          >
            <strong>{option.label}</strong>
            {option.description && <span>{option.description}</span>}
          </button>
        ))}
      </div>
    </section>
  );
}
