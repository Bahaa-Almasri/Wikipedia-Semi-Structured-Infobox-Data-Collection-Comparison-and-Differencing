export function displayCountry(countryOrSlug, countries = []) {
  if (!countryOrSlug) return "";
  if (typeof countryOrSlug === "object") {
    return countryOrSlug.display_name || countryOrSlug.country || countryOrSlug.slug || "";
  }
  const match = countries.find((country) => country.slug === countryOrSlug);
  return match?.display_name || countryOrSlug.replaceAll("_", " ");
}

export function countryBySlug(countries = []) {
  return Object.fromEntries(countries.map((country) => [country.slug, country]));
}
