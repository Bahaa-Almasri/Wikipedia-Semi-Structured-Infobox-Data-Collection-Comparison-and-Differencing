import { get, post } from "./apiClient.js";

export function listCountries() {
  return get("/countries");
}

export function getCountryJson(slug) {
  return get(`/countries/${slug}/json`);
}

export function getTedSimilarCountries(country, topK = 8) {
  return post("/similarity-ranking", {
    country,
    top_k: topK,
  });
}

export function getVsmSimilarCountries(country, topK = 8, features = null) {
  const payload = {
    country,
    top_k: topK,
    metric: "cosine",
    mode: "field",
  };
  if (features?.length) {
    payload.features = features;
  }
  return post("/vsm/similarity-ranking", payload);
}
