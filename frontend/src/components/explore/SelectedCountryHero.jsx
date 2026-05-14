import MatchScoreBadge from "../shared/MatchScoreBadge.jsx";

export default function SelectedCountryHero({ country, topScore }) {
  if (!country) {
    return (
      <section className="hero-panel compact">
        <p className="eyebrow">Start exploring</p>
        <h1>Pick a country to discover its closest neighbors.</h1>
      </section>
    );
  }

  return (
    <section className="hero-panel compact">
      <div>
        <p className="eyebrow">Now exploring</p>
        <h1>{country.display_name}</h1>
        <p>Browse countries that appear closest in the project similarity space.</p>
      </div>
      {typeof topScore === "number" && <MatchScoreBadge score={topScore} />}
    </section>
  );
}
