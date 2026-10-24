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
  FeatureSummary 
} from './types';
import * as api from './services/api';

export const App: React.FC = () => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
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

  // Initial load
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      const [healthData, monumentsData, convsData] = await Promise.all([
        api.fetchHealth().catch(() => null),
        api.fetchMonuments().catch(() => []),
        api.fetchConversations().catch(() => [])
      ]);

      setHealth(healthData);
      setMonuments(monumentsData);
      setConversations(convsData);

      if (convsData.length > 0) {
        selectConversation(convsData[0].id);
      } else {
        createNewConversation();
      }
    } catch (err) {
      console.error('Failed to initialize application:', err);
    }
  };

  const selectConversation = async (id: string) => {
    setActiveConvId(id);
    try {
      const details = await api.fetchConversationDetails(id);
      setMessages(details.messages);
    } catch (err) {
      console.error('Failed to load conversation details:', err);
    }
  };

  const createNewConversation = async () => {
    try {
      const newConv = await api.createConversation('New Monument Exploration');
      setConversations(prev => [newConv, ...prev]);
      setActiveConvId(newConv.id);
      setMessages([]);
    } catch (err) {
      console.error('Failed to create new conversation:', err);
    }
  };

  const handleRename = async (id: string, newTitle: string) => {
    try {
      const updated = await api.renameConversation(id, newTitle);
      setConversations(prev => prev.map(c => (c.id === id ? updated : c)));
    } catch (err) {
      console.error('Failed to rename conversation:', err);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await api.deleteConversation(id);
      const remaining = conversations.filter(c => c.id !== id);
      setConversations(remaining);
      if (activeConvId === id) {
        if (remaining.length > 0) {
          selectConversation(remaining[0].id);
        } else {
          createNewConversation();
        }
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err);
    }
  };

  const handleUploadImage = async (file: File) => {
    if (!activeConvId) return;
    setIsLoading(true);

    try {
      const response = await api.recognizeImage(
        file,
        activeConvId,
        multiMode ? 'multiple' : 'single'
      );

      if (response.feature_summary) {
        setActiveFeatureSummary(response.feature_summary);
      }

      // Re-fetch message history to guarantee full synchronization with SQLite database
      const details = await api.fetchConversationDetails(activeConvId);
      setMessages(details.messages);

      // Auto-update conversation title if it was default
      const currentConv = conversations.find(c => c.id === activeConvId);
      if (currentConv && currentConv.title === 'New Monument Exploration' && response.top_class) {
        const readableTitle = response.top_class.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
        handleRename(activeConvId, `${readableTitle} Exploration`);
      }
    } catch (err: any) {
      console.error('Recognition error:', err);
      // Append local system error message
      setMessages(prev => [
        ...prev,
        {
          id: 'error-' + Date.now(),
          conversation_id: activeConvId,
          role: 'assistant',
          content: "MonuScript is having trouble reaching the recognition service. Please check your backend connection.",
          image_url: null,
          created_at: new Date().toISOString()
        }
      ]);
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
