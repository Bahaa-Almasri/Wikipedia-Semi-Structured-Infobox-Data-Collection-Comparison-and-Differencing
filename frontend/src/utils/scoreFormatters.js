export function scorePercent(score) {
  if (typeof score !== "number" || Number.isNaN(score)) {
    return "0%";
  }
  return `${Math.round(Math.max(0, Math.min(score, 1)) * 100)}%`;
}

export function scoreLabel(score) {
  if (score >= 0.85) return "Excellent match";
  if (score >= 0.7) return "Strong match";
  if (score >= 0.5) return "Interesting match";
  return "Curious match";
}
