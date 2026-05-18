import { post } from "./apiClient.js";

export function recommendMatches(preferences) {
  return post("/matchmaker/recommend", preferences);
}
