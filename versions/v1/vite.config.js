import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

// Shared geometry/viewer code lives in ../../frontend/src (alias @shared). Dev: proxy /api to the FastAPI backend.
export default defineConfig({
  plugins: [react()],
  resolve: {
    // shared source sits outside this package, so pin bare imports to THIS package's node_modules (needed in Docker/Render)
    alias: {
      '@shared': path.resolve(__dirname, '../../frontend/src'),
      three: path.resolve(__dirname, 'node_modules/three'),
      react: path.resolve(__dirname, 'node_modules/react'),
      'react-dom': path.resolve(__dirname, 'node_modules/react-dom'),
    },
    dedupe: ['react', 'react-dom'],
  },
  server: { port: 5174, fs: { allow: ['../..'] }, proxy: { '/api': process.env.VITE_API_PROXY || 'http://localhost:8000' } },
})
