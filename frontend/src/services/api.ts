import { Conversation, Message, MonumentMetadata, RecognitionResponse, SystemHealth } from '../types';

const API_BASE = '/api/v1';

export async function fetchHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchMonuments(): Promise<MonumentMetadata[]> {
  const res = await fetch(`${API_BASE}/monuments`);
  if (!res.ok) throw new Error(`Failed to load monuments: ${res.statusText}`);
  const data = await res.json();
  return data.monuments;
}

export async function fetchConversations(): Promise<Conversation[]> {
  const res = await fetch(`${API_BASE}/conversations`);
  if (!res.ok) throw new Error(`Failed to fetch conversations: ${res.statusText}`);
  const data = await res.json();
  return data.conversations;
}

export async function createConversation(title?: string): Promise<Conversation> {
  const res = await fetch(`${API_BASE}/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: title || 'New Monument Exploration' })
  });
  if (!res.ok) throw new Error(`Failed to create conversation: ${res.statusText}`);
  const data = await res.json();
  return data.conversation;
}

export async function fetchConversationDetails(convId: string): Promise<{ conversation: Conversation; messages: Message[] }> {
  const res = await fetch(`${API_BASE}/conversations/${convId}`);
  if (!res.ok) throw new Error(`Failed to fetch conversation details: ${res.statusText}`);
  return res.json();
}

export async function renameConversation(convId: string, title: string): Promise<Conversation> {
  const res = await fetch(`${API_BASE}/conversations/${convId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title })
  });
  if (!res.ok) throw new Error(`Failed to rename conversation: ${res.statusText}`);
  const data = await res.json();
  return data.conversation;
}

export async function deleteConversation(convId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/conversations/${convId}`, {
    method: 'DELETE'
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

  const endpoint = mode === 'multiple' ? `${API_BASE}/recognize/multiple` : `${API_BASE}/recognize`;
  const res = await fetch(endpoint, {
    method: 'POST',
    body: formData
  });

  const data = await res.json();
  if (!res.ok && !data.error) {
    throw new Error(`Upload failed with status ${res.status}`);
  }
  return data;
}
