import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Porta 5173 combina com o default de BACKEND_CORS_ORIGINS no backend.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
  },
});
