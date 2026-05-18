import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import ExplorationTrail from "../components/explore/ExplorationTrail.jsx";
import SelectedCountryHero from "../components/explore/SelectedCountryHero.jsx";
import SimilarCountryGrid from "../components/explore/SimilarCountryGrid.jsx";
import WorldMap from "../components/map/WorldMap.jsx";
import CountrySearch from "../components/shared/CountrySearch.jsx";
import EmptyState from "../components/shared/EmptyState.jsx";
import ErrorState from "../components/shared/ErrorState.jsx";
import LoadingState from "../components/shared/LoadingState.jsx";
import useCountries from "../hooks/useCountries.js";
import { getVsmSimilarCountries } from "../services/countryService.js";
import { countryBySlug } from "../utils/countryFormatters.js";

export default function ExplorePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialCountry = searchParams.get("country") || "";
  const { countries, loading, error, reload } = useCountries();
  const [selected, setSelected] = useState(initialCountry);
  const [results, setResults] = useState([]);
  const [trail, setTrail] = useState([]);
  const [resultError, setResultError] = useState("");
  const [resultsLoading, setResultsLoading] = useState(false);
  const bySlug = useMemo(() => countryBySlug(countries), [countries]);

  const explore = (slug) => {
    if (!slug) return;
    setSelected(slug);
    setSearchParams({ country: slug });
    setTrail((current) => [...current.filter((item) => item !== slug), slug].slice(-6));
  };

  useEffect(() => {
    if (!selected) return;
    let active = true;
    setResultsLoading(true);
    setResultError("");
    getVsmSimilarCountries(selected, 8)
      .then((payload) => {
        if (!active) return;
        if (payload?.status === "insufficient_terms") {
          setResults([]);
          setResultError(
            payload.message ||
              "Not enough meaningful terms after filtering. Try selecting broader features.",
          );
          return;
        }
        setResults(payload.results || []);
      })
      .catch((err) => {
        if (active) setResultError(err.message || "Could not load similar countries.");
      })
      .finally(() => {
        if (active) setResultsLoading(false);
      });
    return () => {
      active = false;
    };
  }, [selected]);

  if (loading) return <LoadingState message="Loading countries..." />;
  if (error) return <ErrorState message={error} onRetry={reload} />;

  const selectedCountry = bySlug[selected];
  const highlighted = results.map((result) => result.country);

  return (
    <div className="page-stack">
      <section className="split-layout">
        <aside className="side-panel">
          <CountrySearch countries={countries} value={selected} onChange={explore} />
          <button
            className="ghost-button full-width"
            type="button"
            onClick={() => explore(countries[Math.floor(Math.random() * countries.length)]?.slug)}
          >
            Surprise me
          </button>
          <ExplorationTrail trail={trail} countries={countries} onSelect={explore} />
        </aside>
        <div className="page-stack">
          <SelectedCountryHero country={selectedCountry} topScore={results[0]?.score} />
          {resultsLoading && <LoadingState message="Finding nearby country profiles..." />}
          {resultError && <ErrorState message={resultError} />}
          {!selected && (
            <EmptyState
              title="Choose a country"
              message="Start with any country and CountryScope will show nearby profiles."
            />
          )}
          {selected && !resultsLoading && results.length > 0 && (
            <SimilarCountryGrid results={results} countriesBySlug={bySlug} onExplore={explore} />
          )}
        </div>
      </section>
      <WorldMap
        countries={countries}
        highlighted={highlighted}
        selected={selected}
        title="Countries you may also find interesting"
        onSelect={explore}
      />
    </div>
  );
}
