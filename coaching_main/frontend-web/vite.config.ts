import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The FastAPI backend runs on :8000. Everything the dashboard talks to is
// proxied in dev so the browser only ever sees one origin - that keeps the
// native WebSocket on the same host and avoids CORS entirely.
const BACKEND = "http://localhost:8000";

export default defineConfig(({ isSsrBuild }) => ({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: BACKEND,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      "/ws": {
        target: BACKEND.replace("http", "ws"),
        ws: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      output: {
        // Recharts is by far the largest dependency and changes rarely;
        // keeping it in its own chunk means app edits do not invalidate
        // it in the browser cache. Not applicable to the SSR build, where
        // these packages are external.
        manualChunks: isSsrBuild
          ? undefined
          : { react: ["react", "react-dom"], charts: ["recharts"] },
      },
    },
  },
}));
