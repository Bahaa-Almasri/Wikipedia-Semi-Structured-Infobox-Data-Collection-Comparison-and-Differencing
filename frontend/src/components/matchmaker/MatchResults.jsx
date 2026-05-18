import { useNavigate } from "react-router-dom";

import CountryCard from "../shared/CountryCard.jsx";

export default function MatchResults({ result }) {
  const navigate = useNavigate();
  if (!result?.top_match) return null;

  return (
    <section className="page-stack">
      <CountryCard
        featured
        country={result.top_match}
        title={`Your strongest match: ${result.top_match.display_name}`}
        score={result.top_match.score}
        reasons={result.top_match.reasons}
        onClick={() => navigate(`/explore?country=${result.top_match.country}`)}
        actionLabel="Explore this match"
      />
      <div className="section-heading">
        <p className="eyebrow">More places to discover</p>
        <h2>Recommended countries</h2>
      </div>
      <div className="card-grid">
        {(result.recommendations || []).slice(1).map((item) => (
          <CountryCard
            key={item.country}
            country={item}
            score={item.score}
            reasons={item.reasons}
            onClick={() => navigate(`/explore?country=${item.country}`)}
            actionLabel="Explore"
          />
        ))}
      </div>
    </section>
  );
}
