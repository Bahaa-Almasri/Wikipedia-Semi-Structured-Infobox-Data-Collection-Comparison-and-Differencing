// When VITE_API_URL is unset, requests use same-origin paths and Vite proxies to the API.
const API_URL = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
const WIKI_PREFIX = API_URL ? `${API_URL}/wikiinfobox` : "/wikiinfobox";

async function request(path, options = {}) {
  const response = await fetch(`${WIKI_PREFIX}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      // Keep the generic detail when the response is not JSON.
    }
    throw new Error(detail);
  }

  if (response.status === 204) {
    return null;
  }
  return response.json();
}

export function get(path) {
  return request(path);
}

export function post(path, body = {}) {
  return request(path, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
