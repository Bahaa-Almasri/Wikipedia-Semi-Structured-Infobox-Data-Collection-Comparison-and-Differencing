export default function CountrySearch({ countries = [], value, onChange, label = "Choose a country" }) {
  return (
    <label className="field-label">
      <span>{label}</span>
      <select value={value || ""} onChange={(event) => onChange(event.target.value)}>
        <option value="">Select a country</option>
        {countries.map((country) => (
          <option key={country.slug} value={country.slug}>
            {country.display_name}
          </option>
        ))}
      </select>
    </label>
  );
}
