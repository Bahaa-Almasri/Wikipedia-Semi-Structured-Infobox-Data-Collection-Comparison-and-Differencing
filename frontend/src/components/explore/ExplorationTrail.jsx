import { displayCountry } from "../../utils/countryFormatters.js";

export default function ExplorationTrail({ trail = [], countries = [], onSelect }) {
  if (trail.length === 0) {
    return null;
  }

  return (
    <div className="trail">
      <span>Trail</span>
      {trail.map((slug, index) => (
        <button key={`${slug}-${index}`} type="button" onClick={() => onSelect(slug)}>
          {displayCountry(slug, countries)}
        </button>
      ))}
    </div>
  );
}
