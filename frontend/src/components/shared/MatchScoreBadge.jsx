import { scoreLabel, scorePercent } from "../../utils/scoreFormatters.js";

export default function MatchScoreBadge({ score }) {
  return (
    <div className="score-badge" title={`${scorePercent(score)} match score`}>
      <strong>{scorePercent(score)}</strong>
      <span>{scoreLabel(score)}</span>
    </div>
  );
}
