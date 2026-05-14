import AnswerOptionGrid from "./AnswerOptionGrid.jsx";
import ClueCountryCard from "./ClueCountryCard.jsx";

export default function GameRound({ roundData, selected, onSelect, onSubmit, disabled }) {
  return (
    <section className="game-round">
      <div className="section-heading">
        <p className="eyebrow">Guess from neighbors</p>
        <h2>This mystery country is closely related to...</h2>
      </div>
      <div className="card-grid compact-grid">
        {(roundData?.clues || []).map((clue) => (
          <ClueCountryCard key={clue.country} clue={clue} />
        ))}
      </div>
      <div className="section-heading">
        <p className="eyebrow">Your answer</p>
        <h2>Which country is hidden?</h2>
      </div>
      <AnswerOptionGrid
        options={roundData?.options || []}
        selected={selected}
        disabled={disabled}
        onSelect={onSelect}
      />
      <button className="primary-button" type="button" disabled={!selected || disabled} onClick={onSubmit}>
        Submit answer
      </button>
    </section>
  );
}
