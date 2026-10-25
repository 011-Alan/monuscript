import React, { useState, useEffect, useRef } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatHeader } from './components/ChatHeader';
import { ChatMessage } from './components/ChatMessage';
import { EmptyState } from './components/EmptyState';
import { ImageUploader } from './components/ImageUploader';
import { FeatureInspectionModal } from './components/FeatureInspectionModal';
import { ThreeDModal } from './components/ThreeDModal';
import { SupportedMonumentsModal } from './components/SupportedMonumentsModal';
import { 
  Conversation, 
  Message, 
  MonumentMetadata, 
  SystemHealth, 
  FeatureSummary,
  StoredResult 
} from './types';
import * as api from './services/api';
import * as storage from './services/storage';

// Helper to generate a lightweight persistent JPEG thumbnail data URL (< 40KB)
// so that uploaded monument photos render instantly and survive browser refreshes even offline.
async function createThumbnailDataUrl(file: File): Promise<string> {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const maxDim = 640;
        let w = img.width;
        let h = img.height;
        if (w > maxDim || h > maxDim) {
          if (w > h) {
            h = Math.round((h * maxDim) / w);
            w = maxDim;
          } else {
            w = Math.round((w * maxDim) / h);
            h = maxDim;
          }
        }
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(img, 0, 0, w, h);
          resolve(canvas.toDataURL('image/jpeg', 0.82));
        } else {
          resolve(e.target?.result as string);
        }
      };
      img.onerror = () => resolve(e.target?.result as string);
      img.src = e.target?.result as string;
    };
    reader.onerror = () => resolve(URL.createObjectURL(file));
    reader.readAsDataURL(file);
  });
}

export const App: React.FC = () => {
  // 1. Instant Synchronous Initialization from Browser Shared Memory (zero flicker on refresh)
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    return storage.getLocalConversations();
  });

  const [activeConvId, setActiveConvId] = useState<string | null>(() => {
    const localActive = storage.getLocalActiveConvId();
    if (localActive) return localActive;
    const localConvs = storage.getLocalConversations();
    return localConvs.length > 0 ? localConvs[0].id : null;
  });

  const [messages, setMessages] = useState<Message[]>(() => {
    const localActive = storage.getLocalActiveConvId();
    if (localActive) {
      return storage.getLocalMessages(localActive);
    }
    const localConvs = storage.getLocalConversations();
    if (localConvs.length > 0) {
      return storage.getLocalMessages(localConvs[0].id);
    }
    return [];
  });

  const [monuments, setMonuments] = useState<MonumentMetadata[]>([]);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [multiMode, setMultiMode] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Modals state
  const [featureModalOpen, setFeatureModalOpen] = useState(false);
  const [activeFeatureSummary, setActiveFeatureSummary] = useState<FeatureSummary | undefined>();
  const [threeDModal, setThreeDModal] = useState<{ isOpen: boolean; url: string | null; name: string }>({
    isOpen: false,
    url: null,
    name: ''
  });
  const [monumentsModalOpen, setMonumentsModalOpen] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initial load & background synchronization
  useEffect(() => {
    loadInitialData();
  }, []);

  // Multi-tab shared memory listener: keep all tabs in the same browser synchronized in real time
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (!e.key || !e.key.startsWith('monuscript_')) return;
      const updatedConvs = storage.getLocalConversations();
      setConversations(updatedConvs);
      const currentActiveId = storage.getLocalActiveConvId();
      if (currentActiveId) {
        if (currentActiveId !== activeConvId) {
          setActiveConvId(currentActiveId);
        }
        setMessages(storage.getLocalMessages(currentActiveId));
      }
    };
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, [activeConvId]);

  const loadInitialData = async () => {
    try {
      // 1. Ensure local active conversation is selected if present
      const localConvs = storage.getLocalConversations();
      const localActive = storage.getLocalActiveConvId() || (localConvs.length > 0 ? localConvs[0].id : null);
      if (localConvs.length > 0) {
        setConversations(localConvs);
        if (localActive) {
          setActiveConvId(localActive);
          setMessages(storage.getLocalMessages(localActive));
        }
      }

      // 2. Fetch backend metadata and remote conversations in background
      const [healthData, monumentsData, remoteConvs] = await Promise.all([
        api.fetchHealth().catch(() => null),
        api.fetchMonuments().catch(() => []),
        api.fetchConversations().catch(() => null)
      ]);

      if (healthData) setHealth(healthData);
      if (monumentsData) setMonuments(monumentsData);

      // 3. Reconcile browser shared memory with backend SQLite database
      if (remoteConvs !== null && Array.isArray(remoteConvs)) {
        const merged = storage.mergeConversations(localConvs, remoteConvs);
        storage.saveLocalConversations(merged);
        setConversations(merged);

        // Sync any local conversations missing on remote (e.g. if backend container was restarted)
        const missingOnRemote = localConvs.filter(lc => !remoteConvs.some(rc => rc.id === lc.id));
        if (missingOnRemote.length > 0) {
          const localMsgsMap: Record<string, Message[]> = {};
          for (const mc of missingOnRemote) {
            localMsgsMap[mc.id] = storage.getLocalMessages(mc.id);
          }
          api.syncConversations({ conversations: missingOnRemote, messages: localMsgsMap }).catch(() => {});
        }

        const targetId = localActive || (merged.length > 0 ? merged[0].id : null);
        if (targetId) {
          setActiveConvId(targetId);
          storage.setLocalActiveConvId(targetId);

          // Fetch latest remote message history and merge with local
          try {
            const details = await api.fetchConversationDetails(targetId);
            const localMsgs = storage.getLocalMessages(targetId);
            const mergedMsgs = storage.mergeMessages(localMsgs, details.messages);
            storage.saveLocalMessages(targetId, mergedMsgs);
            setMessages(mergedMsgs);
          } catch {
            // Keep local messages intact if remote detail fetch fails
            setMessages(storage.getLocalMessages(targetId));
          }
        } else {
          createNewConversation();
        }
      } else if (localConvs.length === 0) {
        // First visit with empty storage and unreachable/empty backend
        createNewConversation();
      }
    } catch (err) {
      console.error('Failed to initialize application:', err);
    }
  };

  const selectConversation = async (id: string) => {
    setActiveConvId(id);
    storage.setLocalActiveConvId(id);

    // Immediately restore messages from browser shared memory (zero latency)
    const localMsgs = storage.getLocalMessages(id);
    setMessages(localMsgs);

    // Fetch remote details in background to sync any updates
    try {
      const details = await api.fetchConversationDetails(id);
      const merged = storage.mergeMessages(localMsgs, details.messages);
      storage.saveLocalMessages(id, merged);
      setMessages(merged);
    } catch (err) {
      console.warn('Backend unavailable, using browser shared memory:', err);
    }
  };

  const createNewConversation = async () => {
    const newId = `conv_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const newConv: Conversation = {
      id: newId,
      title: 'New Monument Exploration',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };

    // 1. Immediately persist in browser shared memory
    storage.upsertLocalConversation(newConv);
    storage.setLocalActiveConvId(newId);
    storage.saveLocalMessages(newId, []);

    // 2. Immediately update local UI
    setConversations(prev => [newConv, ...prev.filter(c => c.id !== newId)]);
    setActiveConvId(newId);
    setMessages([]);

    // 3. Create on backend in background
    try {
      const backendConv = await api.createConversation('New Monument Exploration');
      if (backendConv && backendConv.id) {
        storage.deleteLocalConversation(newId);
        storage.upsertLocalConversation(backendConv);
        storage.setLocalActiveConvId(backendConv.id);
        setConversations(prev => [backendConv, ...prev.filter(c => c.id !== newId && c.id !== backendConv.id)]);
        setActiveConvId(backendConv.id);
      }
    } catch (err) {
      console.warn('Backend conversation creation delayed, persisted locally in browser:', err);
    }
  };

  const handleRename = async (id: string, newTitle: string) => {
    // 1. Update browser shared memory immediately
    storage.renameLocalConversation(id, newTitle);
    setConversations(prev => prev.map(c => (c.id === id ? { ...c, title: newTitle } : c)));

    // 2. Sync to backend
    try {
      await api.renameConversation(id, newTitle);
    } catch (err) {
      console.warn('Failed to rename on backend, retained in browser storage:', err);
    }
  };

  const handleDelete = async (id: string) => {
    // 1. Delete from browser shared memory
    storage.deleteLocalConversation(id);
    const remaining = conversations.filter(c => c.id !== id);
    setConversations(remaining);

    if (activeConvId === id) {
      if (remaining.length > 0) {
        selectConversation(remaining[0].id);
      } else {
        createNewConversation();
      }
    }

    // 2. Sync deletion to backend
    try {
      await api.deleteConversation(id);
    } catch (err) {
      console.warn('Failed to delete on backend:', err);
    }
  };

  const handleUploadImage = async (file: File) => {
    setIsLoading(true);

    let targetConvId = activeConvId;
    if (!targetConvId) {
      const fallbackId = `conv_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      const fallbackConv: Conversation = {
        id: fallbackId,
        title: 'New Monument Exploration',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      };
      storage.upsertLocalConversation(fallbackConv);
      storage.setLocalActiveConvId(fallbackId);
      setConversations(prev => [fallbackConv, ...prev]);
      setActiveConvId(fallbackId);
      targetConvId = fallbackId;
    }

    // 1. Create immediate lightweight thumbnail data URL so photo persists across page refreshes
    let initialImageUrl: string;
    try {
      initialImageUrl = await createThumbnailDataUrl(file);
    } catch {
      initialImageUrl = URL.createObjectURL(file);
    }

    const tempUserMsgId = `user_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const tempUserMsg: Message = {
      id: tempUserMsgId,
      conversation_id: targetConvId,
      role: 'user',
      content: 'Uploaded monument photograph for architectural analysis.',
      image_url: initialImageUrl,
      created_at: new Date().toISOString()
    };

    // Immediately save and display user message
    storage.appendLocalMessage(targetConvId, tempUserMsg);
    setMessages(prev => [...prev, tempUserMsg]);

    try {
      const response = await api.recognizeImage(
        file,
        targetConvId,
        multiMode ? 'multiple' : 'single'
      );

      if (response.feature_summary) {
        setActiveFeatureSummary(response.feature_summary);
      }

      // 2. Identify authoritative server IDs and persistent upload URL
      const serverUserMsgId = response.message_ids?.user_message_id || tempUserMsgId;
      const serverAsstMsgId = response.message_ids?.assistant_message_id || `asst_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      const persistentImageUrl = response.image?.url ? api.formatImageUrl(response.image.url) : initialImageUrl;

      // 3. Update the temporary user message with the server message ID and persistent image URL
      const finalUserMsg: Message = {
        ...tempUserMsg,
        id: serverUserMsgId,
        image_url: persistentImageUrl || initialImageUrl
      };

      // 4. Construct candidate results or rejection results
      const resultsList: StoredResult[] = (response.results && response.results.length > 0)
        ? response.results.map((r: any) => ({
            id: `res_${Math.random().toString(36).slice(2, 9)}`,
            monument_id: r.monument_id,
            score: r.score,
            score_type: r.score_type || 'recognition_score',
            accepted: response.accepted,
            metadata: r
          }))
        : [
            {
              id: `res_rej_${Date.now()}`,
              monument_id: response.top_class || 'unknown',
              score: response.top_score || 0,
              score_type: 'rejected_candidate',
              accepted: false,
              metadata: {
                reason: response.message,
                runner_up_class: response.runner_up_class,
                runner_up_score: response.runner_up_score,
                margin: response.margin,
                top_class: response.top_class,
                top_score: response.top_score,
                processing_time_ms: response.processing_time_ms
              }
            }
          ];

      const asstMsg: Message = {
        id: serverAsstMsgId,
        conversation_id: targetConvId,
        role: 'assistant',
        content: response.message,
        image_url: null,
        created_at: new Date().toISOString(),
        results: resultsList
      };

      // Replace tempUserMsg with finalUserMsg, and append asstMsg in local storage
      storage.replaceLocalMessage(targetConvId, tempUserMsgId, finalUserMsg);
      storage.appendLocalMessage(targetConvId, asstMsg);

      // Update state without duplicate keys
      setMessages(prev => {
        const withoutTemp = prev.filter(m => m.id !== tempUserMsgId && m.id !== serverUserMsgId && m.id !== serverAsstMsgId);
        return [...withoutTemp, finalUserMsg, asstMsg];
      });

      // 5. Update conversation title if default
      const currentConv = conversations.find(c => c.id === targetConvId);
      if (currentConv && currentConv.title === 'New Monument Exploration' && response.top_class) {
        const readableTitle = response.top_class.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
        const newTitle = `${readableTitle} Exploration`;
        handleRename(targetConvId, newTitle);
      }

      // 6. Background synchronization with backend details (cleanly deduplicated because IDs now match)
      try {
        const details = await api.fetchConversationDetails(targetConvId);
        const localCurrent = storage.getLocalMessages(targetConvId);
        const merged = storage.mergeMessages(localCurrent, details.messages);
        storage.saveLocalMessages(targetConvId, merged);
        setMessages(merged);
      } catch {
        // Local messages remain 100% intact
      }
    } catch (err: any) {
      console.error('Recognition error:', err);
      const errorText = err.message || "MonuScript is having trouble reaching the recognition service. Please check your backend connection.";
      const errorMsg: Message = {
        id: `error_${Date.now()}`,
        conversation_id: targetConvId,
        role: 'assistant',
        content: `⚠️ Recognition issue: ${errorText}`,
        image_url: null,
        created_at: new Date().toISOString()
      };
      storage.appendLocalMessage(targetConvId, errorMsg);
      setMessages(prev => [...prev, errorMsg]);
      throw err;
    } finally {
      setIsLoading(false);
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }
  };

  const activeConv = conversations.find(c => c.id === activeConvId);

  return (
    <div className="flex h-screen bg-[#FAF8F5] overflow-hidden text-[#1F2923]">
      {/* Sidebar */}
      <Sidebar
        conversations={conversations}
        activeId={activeConvId}
        onSelect={selectConversation}
        onNew={createNewConversation}
        onRename={handleRename}
        onDelete={handleDelete}
        onOpenMonuments={() => setMonumentsModalOpen(true)}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />

      {/* Main Workspace Area */}
      <main className="flex-1 flex flex-col min-w-0 bg-[#FAF8F5] relative">
        <ChatHeader
          title={activeConv?.title || 'MonuScript Workspace'}
          onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
          health={health}
          multiMode={multiMode}
          onToggleMultiMode={() => setMultiMode(!multiMode)}
        />

        {/* Scrollable Conversation Stream */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <EmptyState
              onOpenSupportedModal={() => setMonumentsModalOpen(true)}
            />
          ) : (
            <div className="divide-y divide-[#E5DFD7]/60 max-w-4xl mx-auto w-full py-4">
              {messages.map((msg) => (
                <ChatMessage
                  key={msg.id}
                  message={msg}
                  onOpen3D={(url, name) => setThreeDModal({ isOpen: true, url, name })}
                  onInspectFeatures={() => setFeatureModalOpen(true)}
                />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Floating Upload Dock */}
        <ImageUploader
          onUpload={handleUploadImage}
          isLoading={isLoading}
        />
      </main>

      {/* Modals */}
      <FeatureInspectionModal
        isOpen={featureModalOpen}
        onClose={() => setFeatureModalOpen(false)}
        summary={activeFeatureSummary}
      />

      <ThreeDModal
        isOpen={threeDModal.isOpen}
        onClose={() => setThreeDModal({ isOpen: false, url: null, name: '' })}
        modelUrl={threeDModal.url}
        monumentName={threeDModal.name}
      />

      <SupportedMonumentsModal
        isOpen={monumentsModalOpen}
        onClose={() => setMonumentsModalOpen(false)}
        monuments={monuments}
        onOpen3D={(url, name) => {
          setMonumentsModalOpen(false);
          setThreeDModal({ isOpen: true, url, name });
        }}
      />
    </div>
  );
};

export default App;
