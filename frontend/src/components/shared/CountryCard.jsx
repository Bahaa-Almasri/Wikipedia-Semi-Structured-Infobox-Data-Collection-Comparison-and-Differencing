import MatchScoreBadge from "./MatchScoreBadge.jsx";
import ReasonTag from "./ReasonTag.jsx";

export default function CountryCard({
  country,
  title,
  score,
  reasons = [],
  onClick,
  actionLabel = "Explore",
  featured = false,
}) {
  const displayName = title || country?.display_name || country?.country || country?.slug;

  return (
    <article className={featured ? "country-card country-card-featured" : "country-card"}>
      <div>
        <p className="eyebrow">Country profile</p>
        <h3>{displayName}</h3>
      </div>
      {typeof score === "number" && <MatchScoreBadge score={score} />}
      {reasons.length > 0 && (
        <div className="reason-row">
          {reasons.slice(0, 3).map((reason) => (
            <ReasonTag key={reason}>{reason}</ReasonTag>
          ))}
        </div>
      )}
      {onClick && (
        <button className="ghost-button" type="button" onClick={onClick}>
          {actionLabel}
        </button>
      )}
    </article>
  );
}
