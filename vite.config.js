import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "app/static/dist",
    emptyOutDir: true,
    lib: {
      entry: "frontend/realtime-controls.jsx",
      name: "CompanionRealtimeControls",
      formats: ["iife"],
      fileName: () => "realtime-controls.iife.js",
    },
  },
});
