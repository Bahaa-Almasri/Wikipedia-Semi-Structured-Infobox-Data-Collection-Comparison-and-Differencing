export default function CountryHighlightLayer({ countries = [], selected, onSelect }) {
  return (
    <>
      {countries.map((country) => (
        <button
          key={country.slug}
          type="button"
          className={country.slug === selected ? "map-chip selected" : "map-chip"}
          title={country.display_name}
          onClick={() => onSelect?.(country.slug)}
        >
          {country.display_name}
        </button>
      ))}
    </>
  );
}
