/**
 * MonuScript - API Service Layer
 *
 * Communicates with the Flask backend API.
 * Automatically attaches the visitor's authentication token to every request.
 */

import { getAuthToken } from './supabase';
import { Conversation, Message, MonumentMetadata, RecognitionResponse, SystemHealth } from '../types';

const ENV_API = import.meta.env.VITE_API_BASE_URL as string | undefined;

/**
 * Robustly resolve API_BASE URL:
 * Handles trailing slashes, missing /api/v1, or empty values.
 */
export function getApiBase(): string {
  if (!ENV_API || !ENV_API.trim()) {
    return '/api/v1';
  }
  const cleaned = ENV_API.trim().replace(/\/+$/, '');
  if (cleaned.endsWith('/api/v1')) {
    return cleaned;
  }
  if (cleaned.endsWith('/api')) {
    return `${cleaned}/v1`;
  }
  return `${cleaned}/api/v1`;
}

export const API_BASE = getApiBase();

/**
 * Format relative image URLs into absolute URLs when backend is on a different domain.
 */
export function formatImageUrl(url: string | null | undefined): string | undefined {
  if (!url) return undefined;
  if (
    url.startsWith('http://') ||
    url.startsWith('https://') ||
    url.startsWith('blob:') ||
    url.startsWith('data:')
  ) {
    return url;
  }

  // Derive origin from API_BASE if it's an absolute URL
  if (API_BASE.startsWith('http://') || API_BASE.startsWith('https://')) {
    const origin = API_BASE.replace(/\/api(\/v1)?\/?$/, '');
    return `${origin}${url.startsWith('/') ? '' : '/'}${url}`;
  }

  return url;
}

/** Build common headers, always including the visitor's authenticated identity. */
async function authHeaders(extra: Record<string, string> = {}): Promise<HeadersInit> {
  const token = await getAuthToken();
  const headers: Record<string, string> = { ...extra };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
    headers['X-Visitor-Id'] = token;
  }
  return headers;
}

export async function fetchHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_BASE}/health`, {
    headers: await authHeaders(),
  });
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchMonuments(): Promise<MonumentMetadata[]> {
  const res = await fetch(`${API_BASE}/monuments`, {
    headers: await authHeaders(),
  });
  if (!res.ok) throw new Error(`Failed to load monuments: ${res.statusText}`);
  const data = await res.json();
  return data.monuments || [];
}

export async function fetchConversations(): Promise<Conversation[]> {
  const res = await fetch(`${API_BASE}/conversations`, {
    headers: await authHeaders(),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error(data?.error?.message || `Failed to fetch conversations (${res.status})`);
  }
  const data = await res.json();
  return data.conversations || [];
}

export async function createConversation(title?: string): Promise<Conversation> {
  const res = await fetch(`${API_BASE}/conversations`, {
    method: 'POST',
    headers: await authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ title: title || 'New Monument Exploration' }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error(data?.error?.message || `Failed to create conversation (${res.status})`);
  }
  const data = await res.json();
  return data.conversation;
}

export async function fetchConversationDetails(
  convId: string
): Promise<{ conversation: Conversation; messages: Message[] }> {
  const res = await fetch(`${API_BASE}/conversations/${convId}`, {
    headers: await authHeaders(),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error(data?.error?.message || `Failed to fetch conversation details (${res.status})`);
  }
  return res.json();
}

export async function renameConversation(convId: string, title: string): Promise<Conversation> {
  const res = await fetch(`${API_BASE}/conversations/${convId}`, {
    method: 'PATCH',
    headers: await authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ title }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error(data?.error?.message || `Failed to rename conversation (${res.status})`);
  }
  const data = await res.json();
  return data.conversation;
}

export async function deleteConversation(convId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/conversations/${convId}`, {
    method: 'DELETE',
    headers: await authHeaders(),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error(data?.error?.message || `Failed to delete conversation (${res.status})`);
  }
}

export async function recognizeImage(
  file: File,
  conversationId?: string,
  mode: 'single' | 'multiple' = 'single'
): Promise<RecognitionResponse> {
  const formData = new FormData();
  formData.append('image', file);
  if (conversationId) {
    formData.append('conversation_id', conversationId);
  }

  const endpoint =
    mode === 'multiple' ? `${API_BASE}/recognize/multiple` : `${API_BASE}/recognize`;

  const headers = await authHeaders();

  let res: Response;
  try {
    res = await fetch(endpoint, {
      method: 'POST',
      headers,
      body: formData,
    });
  } catch (netErr: any) {
    throw new Error(
      `Cannot connect to backend server (${netErr.message || 'Network error'}). If you are using Render, the free instance might be waking up (30s) — please try again.`
    );
  }

  let data: any;
  try {
    data = await res.json();
  } catch {
    throw new Error(`Server returned status ${res.status} with non-JSON response.`);
  }

  if (!res.ok || data.success === false) {
    const errorMsg =
      data?.error?.message ||
      data?.message ||
      `Recognition failed with status ${res.status}`;
    throw new Error(errorMsg);
  }

  return data;
}
