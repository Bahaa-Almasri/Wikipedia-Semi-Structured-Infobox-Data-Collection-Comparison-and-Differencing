import { useEffect, useState } from "react";

import { listCountries } from "../services/countryService.js";

export default function useCountries() {
  const [countries, setCountries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      setCountries(await listCountries());
    } catch (err) {
      setError(err.message || "Could not load countries.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return { countries, loading, error, reload: load };
}
