import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In Docker Compose, set VITE_PROXY_TARGET=http://wikinfobox-api:80 on the frontend service.
const apiProxyTarget = process.env.VITE_PROXY_TARGET || "http://localhost:8970";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/wikiinfobox": {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      "/health": {
        target: apiProxyTarget,
        changeOrigin: true,
      },
    },
  },
  preview: {
    host: "0.0.0.0",
    port: 4173,
    proxy: {
      "/wikiinfobox": {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      "/health": {
        target: apiProxyTarget,
        changeOrigin: true,
      },
    },
  },
});
