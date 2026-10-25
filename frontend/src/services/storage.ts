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
 *    ephemeral container, the user's chats and uploaded photographs are NEVER lost.
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

function getThumbCacheKey(key: string): string {
  // Strip any query parameters or domains so keys normalize cleanly
  const cleanKey = key.split('?')[0].replace(/^https?:\/\/[^/]+/, '');
  return `${PREFIX}thumb_${cleanKey}`;
}

/**
 * Caches a lightweight image thumbnail data URL in browser storage.
 */
export function saveThumbnailCache(key: string, dataUrl: string): void {
  try {
    if (!key || !dataUrl || !dataUrl.startsWith('data:')) return;
    localStorage.setItem(getThumbCacheKey(key), dataUrl);
  } catch {
    // Quota reached or private browsing mode - fail gracefully
  }
}

/**
 * Retrieves a cached image thumbnail data URL if available.
 */
export function getThumbnailCache(key: string): string | null {
  try {
    if (!key) return null;
    return localStorage.getItem(getThumbCacheKey(key));
  } catch {
    return null;
  }
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

    // Also clean up any cached thumbnails for these messages
    const msgs = getLocalMessages(convId);
    for (const m of msgs) {
      if (m.id) localStorage.removeItem(getThumbCacheKey(m.id));
      if (m.image_url) localStorage.removeItem(getThumbCacheKey(m.image_url));
    }

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
 * Automatically recovers thumbnails from cache if image_url was a remote path.
 */
export function getLocalMessages(convId: string): Message[] {
  try {
    if (!convId) return [];
    const visitorId = getOrCreateLocalVisitorId();
    const raw = localStorage.getItem(getMessagesKey(visitorId, convId));
    if (!raw) return [];
    const parsed: Message[] = JSON.parse(raw);
    const msgs = Array.isArray(parsed)
      ? parsed.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
      : [];

    return msgs.map(m => {
      // Auto-heal: If image_url is a remote path or missing, check if cached thumbnail exists
      if (m.image_url && !m.image_url.startsWith('data:')) {
        const cached = getThumbnailCache(m.id) || getThumbnailCache(m.image_url);
        if (cached) {
          return { ...m, image_url: cached };
        }
      } else if (!m.image_url && m.role === 'user') {
        const cached = getThumbnailCache(m.id);
        if (cached) {
          return { ...m, image_url: cached };
        }
      }
      return m;
    });
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

    // Cache any data URL thumbnails for resilient offline & refresh display
    for (const m of messages) {
      if (m.image_url && m.image_url.startsWith('data:')) {
        saveThumbnailCache(m.id, m.image_url);
      }
    }

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

    // Cache thumbnail data URL
    if (message.image_url && message.image_url.startsWith('data:')) {
      saveThumbnailCache(message.id, message.image_url);
    }

    // Avoid exact duplicate message IDs or redundant identical messages sent in immediate succession
    const isDuplicate = existing.some(m => {
      if (m.id === message.id) return true;
      if (m.role === message.role && m.content === message.content) {
        const diff = Math.abs(new Date(m.created_at).getTime() - new Date(message.created_at).getTime());
        if (diff < 3000) return true;
      }
      return false;
    });

    if (!isDuplicate) {
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
 * Replaces an existing message in browser shared memory (e.g. replacing a temp ID with server ID).
 */
export function replaceLocalMessage(convId: string, oldMsgId: string, newMsg: Message): void {
  try {
    if (!convId) return;
    const current = getLocalMessages(convId);

    // Ensure thumbnail cache is updated for newMsg ID
    if (newMsg.image_url && newMsg.image_url.startsWith('data:')) {
      saveThumbnailCache(newMsg.id, newMsg.image_url);
      saveThumbnailCache(oldMsgId, newMsg.image_url);
    }

    const updated = current.map(m => (m.id === oldMsgId ? newMsg : m));
    saveLocalMessages(convId, updated);
  } catch (err) {
    console.warn(`[MonuScript Storage] Error replacing message in ${convId}:`, err);
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
 * 1. Matches exact IDs.
 * 2. Matches temporary client IDs (user_* or asst_*) with corresponding remote messages
 *    by role, image presence, and close timestamp.
 * 3. CRITICAL: Preserves robust data URL thumbnails so photographs NEVER disappear on refresh!
 * 4. Deduplicates any redundant adjacent messages with identical roles and timestamps.
 */
export function mergeMessages(local: Message[], remote: Message[]): Message[] {
  const mergedList: Message[] = [];
  const localCopy = [...local];

  for (const r of remote) {
    // 1. Check for exact ID match in local
    const exactLocalIdx = localCopy.findIndex(l => l.id === r.id);
    if (exactLocalIdx >= 0) {
      const loc = localCopy[exactLocalIdx];
      localCopy.splice(exactLocalIdx, 1);

      // Prioritize data URL thumbnails (loc or cached) over remote /uploads/ URLs that could 404
      let resolvedImg = loc.image_url;
      if (!resolvedImg || !resolvedImg.startsWith('data:')) {
        if (r.image_url && r.image_url.startsWith('data:')) {
          resolvedImg = r.image_url;
        } else {
          const cached = getThumbnailCache(r.id) || (r.image_url ? getThumbnailCache(r.image_url) : null) || getThumbnailCache(loc.id);
          resolvedImg = cached || loc.image_url || r.image_url;
        }
      }

      if (resolvedImg && resolvedImg.startsWith('data:')) {
        saveThumbnailCache(r.id, resolvedImg);
      }

      mergedList.push({
        ...loc,
        ...r,
        image_url: resolvedImg,
        results: (r.results && r.results.length > 0) ? r.results : (loc.results || [])
      });
      continue;
    }

    // 2. Check for temp client message in local corresponding to this remote message
    const tempLocalIdx = localCopy.findIndex(l => {
      if (l.role !== r.role) return false;
      const isClientTemp = l.id.startsWith('user_') || l.id.startsWith('asst_') || l.id.startsWith('error_');
      if (!isClientTemp) return false;
      const lTime = new Date(l.created_at).getTime();
      const rTime = new Date(r.created_at).getTime();
      const timeDiff = Math.abs(lTime - rTime);
      return timeDiff < 45000;
    });

    if (tempLocalIdx >= 0) {
      const loc = localCopy[tempLocalIdx];
      localCopy.splice(tempLocalIdx, 1);

      let resolvedImg = loc.image_url;
      if (!resolvedImg || !resolvedImg.startsWith('data:')) {
        if (r.image_url && r.image_url.startsWith('data:')) {
          resolvedImg = r.image_url;
        } else {
          const cached = getThumbnailCache(loc.id) || getThumbnailCache(r.id) || (r.image_url ? getThumbnailCache(r.image_url) : null);
          resolvedImg = cached || loc.image_url || r.image_url;
        }
      }

      if (resolvedImg && resolvedImg.startsWith('data:')) {
        saveThumbnailCache(r.id, resolvedImg);
        saveThumbnailCache(loc.id, resolvedImg);
      }

      mergedList.push({
        ...loc,
        ...r,
        id: r.id, // adopt the authoritative server ID
        image_url: resolvedImg,
        results: (r.results && r.results.length > 0) ? r.results : (loc.results || [])
      });
      continue;
    }

    // 3. New remote message without local counterpart:
    // Check if thumbnail is cached
    let remoteImg = r.image_url;
    if (remoteImg && !remoteImg.startsWith('data:')) {
      const cached = getThumbnailCache(r.id) || getThumbnailCache(remoteImg);
      if (cached) {
        remoteImg = cached;
      }
    }

    mergedList.push({
      ...r,
      image_url: remoteImg
    });
  }

  // 4. Any remaining local messages that were not matched (e.g. offline messages waiting to sync)
  for (const remaining of localCopy) {
    mergedList.push(remaining);
  }

  // Sort chronologically
  mergedList.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());

  // 5. Final deduplication pass: filter out duplicates with identical role and content created within 5s
  const deduplicated: Message[] = [];
  for (const msg of mergedList) {
    const isDup = deduplicated.some(existing => {
      if (existing.id === msg.id) return true;
      if (existing.role === msg.role && existing.content === msg.content) {
        const diff = Math.abs(new Date(existing.created_at).getTime() - new Date(msg.created_at).getTime());
        if (diff < 5000) return true;
      }
      return false;
    });
    if (!isDup) {
      deduplicated.push(msg);
    }
  }

  return deduplicated;
}
