export default function MapTooltip({ country }) {
  if (!country) return null;
  return <span className="map-tooltip">{country.display_name}</span>;
}
