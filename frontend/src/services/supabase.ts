/**
 * MonuScript - Supabase Anonymous Authentication Service
 *
 * Provides a singleton Supabase client and an `initAnonymousSession` helper
 * that guarantees every visitor has a unique, persistent anonymous identity
 * without a login screen.
 *
 * Design constraints:
 *  - Only one anonymous account is ever created per browser (session is
 *    stored in localStorage by @supabase/supabase-js automatically).
 *  - Initialization is asynchronous; the app must await `initAnonymousSession`
 *    before firing any authenticated API requests.
 *  - `getAuthToken` returns the current JWT access token so that `api.ts`
 *    can attach it as a Bearer header.
 */

import { createClient, SupabaseClient, Session } from '@supabase/supabase-js';

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL as string;
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY as string;

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  console.warn(
    '[MonuScript] VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY is not set. ' +
    'Authentication will not work until these environment variables are provided.'
  );
}

// Singleton client – created once at module load time.
export const supabase: SupabaseClient = createClient(
  SUPABASE_URL || 'http://placeholder.supabase.co',
  SUPABASE_ANON_KEY || 'placeholder-key',
  {
    auth: {
      persistSession: true,
      autoRefreshToken: true,
      storageKey: 'monuscript-auth',
    },
  }
);

/** In-memory reference to the active session, updated after init. */
let _session: Session | null = null;

/**
 * Initialise (or restore) an anonymous Supabase session.
 *
 * Call this ONCE before rendering the app. Subsequent calls from anywhere in
 * the app are safe – they will resolve immediately using the cached session.
 *
 * @returns The active Supabase Session.
 */
export async function initAnonymousSession(): Promise<Session | null> {
  // 1. Check if a valid session already exists in localStorage.
  const { data: existingData } = await supabase.auth.getSession();
  if (existingData?.session) {
    _session = existingData.session;
    return _session;
  }

  // 2. No session found – create a new anonymous one.
  const { data, error } = await supabase.auth.signInAnonymously();
  if (error) {
    console.error('[MonuScript] Failed to create anonymous session:', error.message);
    // Non-fatal: the app will still render but API calls will receive 401.
    return null;
  }

  _session = data.session;
  return _session;
}

/**
 * Returns the current JWT access token, refreshing the session if needed.
 * Returns null if no session is active (e.g., Supabase not configured).
 */
export async function getAuthToken(): Promise<string | null> {
  const { data } = await supabase.auth.getSession();
  return data?.session?.access_token ?? null;
}
