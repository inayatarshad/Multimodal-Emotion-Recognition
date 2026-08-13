import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The API's CORS allowlist names this origin explicitly, and the proxy keeps the
    // WebSocket on the same origin as the page so no separate CORS dance is needed.
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/health': { target: 'http://localhost:8000', changeOrigin: true },
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        // Recharts and framer-motion dominate the bundle; splitting them keeps the
        // initial payload small enough for the Lighthouse performance budget.
        manualChunks: {
          charts: ['recharts'],
          motion: ['framer-motion'],
        },
      },
    },
  },
});
