export default function ScoreBoard({ score, round, totalRounds }) {
  return (
    <div className="scoreboard">
      <span>Round {round} / {totalRounds}</span>
      <strong>Score {score}</strong>
    </div>
  );
}
