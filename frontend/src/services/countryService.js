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

export function getVsmSimilarCountries(country, topK = 8) {
  return post("/vsm/similarity-ranking", {
    country,
    top_k: topK,
    metric: "cosine",
    mode: "field",
  });
}
