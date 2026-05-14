import CountryCard from "../shared/CountryCard.jsx";

export default function SimilarCountryGrid({ results = [], countriesBySlug = {}, onExplore }) {
  return (
    <div className="card-grid">
      {results.map((result) => {
        const slug = result.country;
        const country = countriesBySlug[slug] || { slug, display_name: slug };
        return (
          <CountryCard
            key={slug}
            country={country}
            score={result.score}
            reasons={result.reasons || ["A close country profile", "Worth exploring next"]}
            onClick={() => onExplore(slug)}
            actionLabel="Explore this"
          />
        );
      })}
    </div>
  );
}
