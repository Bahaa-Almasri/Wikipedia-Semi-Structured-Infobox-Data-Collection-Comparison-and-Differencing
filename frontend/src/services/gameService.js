import { get, post } from "./apiClient.js";

export function getGuessRound() {
  return get("/game/guess-round");
}

export function submitGuessAnswer(roundId, selectedCountry) {
  return post("/game/submit-answer", {
    round_id: roundId,
    selected_country: selectedCountry,
  });
}
