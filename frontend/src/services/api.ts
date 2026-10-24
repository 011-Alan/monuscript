/**
 * MonuScript - API Service Layer
 *
 * Every function that calls the Flask backend automatically attaches the
 * Supabase anonymous session JWT as an Authorization: Bearer header.
 *
 * Public (unauthenticated) endpoints – /api/v1/health and /api/v1/monuments –
 * do not strictly need the header but include it anyway for consistency.
 */

import { getAuthToken } from './supabase';
import { Conversation, Message, MonumentMetadata, RecognitionResponse, SystemHealth } from '../types';

const ENV_API = import.meta.env.VITE_API_BASE_URL;
const API_BASE = ENV_API ? ENV_API.replace(/\/$/, '') : '/api/v1';

/** Build common headers, including the JWT when one is available. */
async function authHeaders(extra: Record<string, string> = {}): Promise<HeadersInit> {
  const token = await getAuthToken();
  const headers: Record<string, string> = { ...extra };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
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
  return data.monuments;
}

export async function fetchConversations(): Promise<Conversation[]> {
  const res = await fetch(`${API_BASE}/conversations`, {
    headers: await authHeaders(),
  });
  if (!res.ok) throw new Error(`Failed to fetch conversations: ${res.statusText}`);
  const data = await res.json();
  return data.conversations;
}

export async function createConversation(title?: string): Promise<Conversation> {
  const res = await fetch(`${API_BASE}/conversations`, {
    method: 'POST',
    headers: await authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ title: title || 'New Monument Exploration' }),
  });
  if (!res.ok) throw new Error(`Failed to create conversation: ${res.statusText}`);
  const data = await res.json();
  return data.conversation;
}

export async function fetchConversationDetails(
  convId: string
): Promise<{ conversation: Conversation; messages: Message[] }> {
  const res = await fetch(`${API_BASE}/conversations/${convId}`, {
    headers: await authHeaders(),
  });
  if (!res.ok) throw new Error(`Failed to fetch conversation details: ${res.statusText}`);
  return res.json();
}

export async function renameConversation(convId: string, title: string): Promise<Conversation> {
  const res = await fetch(`${API_BASE}/conversations/${convId}`, {
    method: 'PATCH',
    headers: await authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error(`Failed to rename conversation: ${res.statusText}`);
  const data = await res.json();
  return data.conversation;
}

export async function deleteConversation(convId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/conversations/${convId}`, {
    method: 'DELETE',
    headers: await authHeaders(),
  });
  if (!res.ok) throw new Error(`Failed to delete conversation: ${res.statusText}`);
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

  // Note: do NOT set Content-Type here – the browser sets it automatically
  // with the correct multipart boundary when using FormData.
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: await authHeaders(),
    body: formData,
  });

  const data = await res.json();
  if (!res.ok && !data.error) {
    throw new Error(`Upload failed with status ${res.status}`);
  }
  return data;
}
