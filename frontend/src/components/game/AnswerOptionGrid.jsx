export default function AnswerOptionGrid({ options = [], selected, disabled, onSelect }) {
  return (
    <div className="answer-grid">
      {options.map((option) => (
        <button
          key={option.country}
          type="button"
          disabled={disabled}
          className={selected === option.country ? "answer-card selected" : "answer-card"}
          onClick={() => onSelect(option.country)}
        >
          <strong>{option.display_name}</strong>
        </button>
      ))}
    </div>
  );
}
