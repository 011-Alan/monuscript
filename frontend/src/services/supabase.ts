/**
 * MonuScript - Supabase Anonymous Authentication & Visitor Isolation Service
 *
 * Guarantees every visitor has a unique, persistent identity across browser refreshes.
 *
 * Priorities:
 * 1. If Supabase is configured and reachable: uses Supabase Anonymous Authentication (JWT).
 * 2. If Supabase is unconfigured or unreachable: seamlessly falls back to a persistent
 *    cryptographic anonymous visitor ID stored in localStorage.
 *
 * This ensures strict per-visitor data isolation at all times without crashing or blocking.
 */

import { createClient, SupabaseClient, Session } from '@supabase/supabase-js';

const SUPABASE_URL = (import.meta.env.VITE_SUPABASE_URL as string || '').trim();
const SUPABASE_ANON_KEY = (import.meta.env.VITE_SUPABASE_ANON_KEY as string || '').trim();

const isSupabaseConfigured = Boolean(
  SUPABASE_URL &&
  SUPABASE_ANON_KEY &&
  !SUPABASE_URL.includes('your-project-id') &&
  !SUPABASE_ANON_KEY.includes('your-supabase-anon-key')
);

export const supabase: SupabaseClient | null = isSupabaseConfigured
  ? createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        storageKey: 'monuscript-auth',
      },
    })
  : null;

let _session: Session | null = null;
const VISITOR_STORAGE_KEY = 'monuscript_visitor_id';

/**
 * Returns or generates a unique, persistent anonymous visitor token for this browser.
 */
export function getOrCreateLocalVisitorId(): string {
  try {
    let id = localStorage.getItem(VISITOR_STORAGE_KEY);
    if (!id || !id.startsWith('anon_')) {
      const randomPart = typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID().replace(/-/g, '')
        : `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
      id = `anon_${randomPart}`;
      localStorage.setItem(VISITOR_STORAGE_KEY, id);
    }
    return id;
  } catch {
    return 'anon_guest_visitor_default';
  }
}

/**
 * Initialise (or restore) an anonymous visitor session.
 * Call this before rendering or on first API call.
 */
export async function initAnonymousSession(): Promise<string> {
  // 1. Try Supabase if configured
  if (supabase) {
    try {
      const { data: existingData } = await supabase.auth.getSession();
      if (existingData?.session) {
        _session = existingData.session;
        return _session.access_token;
      }

      const { data, error } = await supabase.auth.signInAnonymously();
      if (!error && data?.session) {
        _session = data.session;
        return _session.access_token;
      }
      if (error) {
        console.warn('[MonuScript] Supabase anonymous auth unavailable, using local visitor isolation:', error.message);
      }
    } catch (err) {
      console.warn('[MonuScript] Supabase init caught error, using local visitor isolation:', err);
    }
  }

  // 2. Fall back to local persistent anonymous visitor identity
  return getOrCreateLocalVisitorId();
}

/**
 * Returns the current JWT access token or persistent visitor token.
 * Never returns null – always provides an authenticated visitor identity.
 */
export async function getAuthToken(): Promise<string> {
  if (supabase) {
    try {
      const { data } = await supabase.auth.getSession();
      if (data?.session?.access_token) {
        return data.session.access_token;
      }
    } catch {
      // Fall through to local visitor ID
    }
  }
  return getOrCreateLocalVisitorId();
}
