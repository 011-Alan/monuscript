import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'
import { initAnonymousSession } from './services/supabase'

/**
 * Bootstrap sequence:
 * 1. Initialise (or restore) a Supabase anonymous session.
 * 2. Only then render the React app, so every API call has a valid JWT ready.
 */
async function bootstrap() {
  try {
    await initAnonymousSession();
  } catch (err) {
    // Non-fatal: log and continue so the UI still loads.
    console.error('[MonuScript] Anonymous auth init failed:', err);
  }

  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  )
}

bootstrap();
