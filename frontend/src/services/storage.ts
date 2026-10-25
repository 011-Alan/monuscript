/**
 * MonuScript - Shared Browser Memory & Persistent Storage Service
 *
 * Provides robust browser-isolated, multi-tab shared memory storage using localStorage.
 *
 * Characteristics:
 * 1. "For each browser it stores separately":
 *    Different browsers (Chrome, Firefox, Safari, Edge) each have their own separate
 *    localStorage sandbox and unique visitor identity.
 * 2. "Shared memory within the same browser":
 *    Across page refreshes, browser restarts, and across multiple tabs within the same browser,
 *    all conversations, messages, and the active conversation ID are persistently shared.
 * 3. Offline & Cold-Start Resilient:
 *    If the backend server is waking up (e.g. Render free tier cold start) or restarts its
 *    ephemeral container, the user's chats are NEVER lost.
 */

import { Conversation, Message, StoredResult } from '../types';
import { getOrCreateLocalVisitorId } from './supabase';

const PREFIX = 'monuscript_';

function getConversationsKey(visitorId: string): string {
  return `${PREFIX}convs_${visitorId}`;
}

function getMessagesKey(visitorId: string, convId: string): string {
  return `${PREFIX}msgs_${visitorId}_${convId}`;
}

function getActiveConvKey(visitorId: string): string {
  return `${PREFIX}active_conv_${visitorId}`;
}

/**
 * Returns all conversations stored in this browser's shared memory for the current visitor.
 */
export function getLocalConversations(): Conversation[] {
  try {
    const visitorId = getOrCreateLocalVisitorId();
    const raw = localStorage.getItem(getConversationsKey(visitorId));
    if (!raw) return [];
    const parsed: Conversation[] = JSON.parse(raw);
    return Array.isArray(parsed)
      ? parsed.sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
      : [];
  } catch (err) {
    console.warn('[MonuScript Storage] Error reading local conversations:', err);
    return [];
  }
}

/**
 * Persists conversations to this browser's shared memory.
 */
export function saveLocalConversations(conversations: Conversation[]): void {
  try {
    const visitorId = getOrCreateLocalVisitorId();
    localStorage.setItem(getConversationsKey(visitorId), JSON.stringify(conversations));
  } catch (err) {
    console.warn('[MonuScript Storage] Error saving local conversations:', err);
  }
}

/**
 * Upserts a single conversation in browser shared memory.
 */
export function upsertLocalConversation(conv: Conversation): void {
  try {
    const current = getLocalConversations();
    const index = current.findIndex(c => c.id === conv.id);
    let updated: Conversation[];
    if (index >= 0) {
      updated = [...current];
      updated[index] = { ...updated[index], ...conv, updated_at: conv.updated_at || new Date().toISOString() };
    } else {
      updated = [conv, ...current];
    }
    saveLocalConversations(updated);
  } catch (err) {
    console.warn('[MonuScript Storage] Error upserting local conversation:', err);
  }
}

/**
 * Renames a conversation in browser shared memory.
 */
export function renameLocalConversation(convId: string, newTitle: string): void {
  try {
    const current = getLocalConversations();
    const updated = current.map(c =>
      c.id === convId ? { ...c, title: newTitle, updated_at: new Date().toISOString() } : c
    );
    saveLocalConversations(updated);
  } catch (err) {
    console.warn('[MonuScript Storage] Error renaming local conversation:', err);
  }
}

/**
 * Deletes a conversation and its messages from browser shared memory.
 */
export function deleteLocalConversation(convId: string): void {
  try {
    const visitorId = getOrCreateLocalVisitorId();
    const current = getLocalConversations();
    const remaining = current.filter(c => c.id !== convId);
    saveLocalConversations(remaining);

    localStorage.removeItem(getMessagesKey(visitorId, convId));

    if (getLocalActiveConvId() === convId) {
      setLocalActiveConvId(remaining.length > 0 ? remaining[0].id : null);
    }
  } catch (err) {
    console.warn('[MonuScript Storage] Error deleting local conversation:', err);
  }
}

/**
 * Returns all messages for a specific conversation from browser shared memory.
 */
export function getLocalMessages(convId: string): Message[] {
  try {
    if (!convId) return [];
    const visitorId = getOrCreateLocalVisitorId();
    const raw = localStorage.getItem(getMessagesKey(visitorId, convId));
    if (!raw) return [];
    const parsed: Message[] = JSON.parse(raw);
    return Array.isArray(parsed)
      ? parsed.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
      : [];
  } catch (err) {
    console.warn(`[MonuScript Storage] Error reading messages for ${convId}:`, err);
    return [];
  }
}

/**
 * Persists messages for a specific conversation in browser shared memory.
 */
export function saveLocalMessages(convId: string, messages: Message[]): void {
  try {
    if (!convId) return;
    const visitorId = getOrCreateLocalVisitorId();
    localStorage.setItem(getMessagesKey(visitorId, convId), JSON.stringify(messages));
  } catch (err) {
    console.warn(`[MonuScript Storage] Error saving messages for ${convId}:`, err);
  }
}

/**
 * Appends a message to browser shared memory and bumps the parent conversation's timestamp.
 */
export function appendLocalMessage(convId: string, message: Message): void {
  try {
    if (!convId) return;
    const existing = getLocalMessages(convId);
    // Avoid exact duplicate message IDs
    if (!existing.some(m => m.id === message.id)) {
      const updated = [...existing, message];
      saveLocalMessages(convId, updated);
    }

    // Bump conversation updated_at in local list
    const convs = getLocalConversations();
    const convIndex = convs.findIndex(c => c.id === convId);
    if (convIndex >= 0) {
      const updatedConvs = [...convs];
      updatedConvs[convIndex] = {
        ...updatedConvs[convIndex],
        updated_at: new Date().toISOString()
      };
      saveLocalConversations(updatedConvs);
    }
  } catch (err) {
    console.warn(`[MonuScript Storage] Error appending message for ${convId}:`, err);
  }
}

/**
 * Gets the active conversation ID stored in browser shared memory.
 */
export function getLocalActiveConvId(): string | null {
  try {
    const visitorId = getOrCreateLocalVisitorId();
    return localStorage.getItem(getActiveConvKey(visitorId));
  } catch {
    return null;
  }
}

/**
 * Sets the active conversation ID in browser shared memory.
 */
export function setLocalActiveConvId(convId: string | null): void {
  try {
    const visitorId = getOrCreateLocalVisitorId();
    const key = getActiveConvKey(visitorId);
    if (convId) {
      localStorage.setItem(key, convId);
    } else {
      localStorage.removeItem(key);
    }
  } catch (err) {
    console.warn('[MonuScript Storage] Error setting active conv id:', err);
  }
}

/**
 * Merges local conversations with remote backend conversations:
 * Union by id, keeping the version with the newest updated_at or non-empty content.
 */
export function mergeConversations(local: Conversation[], remote: Conversation[]): Conversation[] {
  const map = new Map<string, Conversation>();

  // 1. Add all local
  for (const c of local) {
    map.set(c.id, c);
  }

  // 2. Merge or add remote
  for (const r of remote) {
    if (!map.has(r.id)) {
      map.set(r.id, r);
    } else {
      const loc = map.get(r.id)!;
      const locTime = new Date(loc.updated_at).getTime();
      const remTime = new Date(r.updated_at).getTime();
      // Keep remote if newer, unless local was renamed to something more descriptive
      if (remTime > locTime && r.title !== 'New Monument Exploration') {
        map.set(r.id, r);
      }
    }
  }

  return Array.from(map.values()).sort(
    (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
  );
}

/**
 * Merges local messages with remote backend messages:
 * Combines distinct messages by id, preserving results and image attachments.
 */
export function mergeMessages(local: Message[], remote: Message[]): Message[] {
  const map = new Map<string, Message>();

  for (const m of local) {
    map.set(m.id, m);
  }

  for (const r of remote) {
    if (!map.has(r.id)) {
      map.set(r.id, r);
    } else {
      const loc = map.get(r.id)!;
      // Merge results if remote has them and local didn't, or vice-versa
      const mergedResults = (r.results && r.results.length > 0)
        ? r.results
        : (loc.results || []);
      map.set(r.id, {
        ...loc,
        ...r,
        results: mergedResults
      });
    }
  }

  return Array.from(map.values()).sort(
    (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
  );
}
