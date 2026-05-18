import CountryHighlightLayer from "./CountryHighlightLayer.jsx";

export default function WorldMap({
  countries = [],
  highlighted = [],
  selected,
  title = "Discovery map",
  onSelect,
}) {
  const highlightedSet = new Set(highlighted);
  const visible = countries
    .filter((country) => country.slug === selected || highlightedSet.has(country.slug))
    .slice(0, 24);

  return (
    <section className="map-panel">
      <div className="section-heading">
        <p className="eyebrow">Map view</p>
        <h2>{title}</h2>
      </div>
      <div className="world-map-grid" role="list">
        {visible.length === 0 ? (
          <p className="muted">Countries will appear here after a selection.</p>
        ) : (
          <CountryHighlightLayer countries={visible} selected={selected} onSelect={onSelect} />
        )}
      </div>
    </section>
  );
}
