import React, { useState } from 'react';
import { 
  Plus, 
  MessageSquare, 
  Trash2, 
  Edit3, 
  Check, 
  X, 
  Search, 
  Landmark, 
  BookOpen, 
  ShieldCheck,
  ChevronRight,
  Menu
} from 'lucide-react';
import { Conversation } from '../types';

interface SidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (id: string, newTitle: string) => void;
  onDelete: (id: string) => void;
  onOpenMonuments: () => void;
  isOpen: boolean;
  onToggle: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  conversations,
  activeId,
  onSelect,
  onNew,
  onRename,
  onDelete,
  onOpenMonuments,
  isOpen,
  onToggle
}) => {
  const [search, setSearch] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');

  const filtered = conversations.filter(c => 
    c.title.toLowerCase().includes(search.toLowerCase())
  );

  const startRename = (conv: Conversation, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(conv.id);
    setEditTitle(conv.title);
  };

  const handleSaveRename = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (editTitle.trim()) {
      onRename(id, editTitle.trim());
    }
    setEditingId(null);
  };

  const handleCancelRename = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(null);
  };

  const handleDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm('Delete this conversation and its analysis history?')) {
      onDelete(id);
    }
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-stone-900/30 backdrop-blur-xs z-30 md:hidden"
          onClick={onToggle}
        />
      )}

      <aside className={`
        fixed md:static inset-y-0 left-0 z-40
        w-72 bg-[#F3EFEA] border-r border-[#E5DFD7] flex flex-col
        transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
      `}>
        {/* Brand Header */}
        <div className="p-4 border-b border-[#E5DFD7] flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="w-9 h-9 rounded-lg bg-[#C85A32] text-white flex items-center justify-center shadow-xs">
              <Landmark className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-1.5">
                <span className="font-serif font-bold text-lg text-[#1F2923] tracking-tight">MonuScript</span>
                <span className="text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 border border-amber-200">
                  SVM
                </span>
              </div>
              <p className="text-[11px] text-[#667268] leading-tight">Monument Vision Workspace</p>
            </div>
          </div>
          <button 
            onClick={onToggle}
            className="md:hidden p-1.5 rounded-md text-stone-500 hover:bg-stone-200"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Action Controls */}
        <div className="p-3 space-y-2">
          <button
            onClick={onNew}
            className="w-full flex items-center justify-center space-x-2 py-2 px-3 rounded-lg bg-[#C85A32] hover:bg-[#b04d28] text-white font-medium text-sm transition-colors shadow-xs active:scale-[0.99]"
          >
            <Plus className="w-4 h-4" />
            <span>New Exploration</span>
          </button>

          {/* Search bar */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-stone-400" />
            <input
              type="text"
              placeholder="Search conversations..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 text-xs rounded-md border border-[#E5DFD7] bg-white text-stone-800 placeholder-stone-400 focus:outline-none focus:ring-1 focus:ring-amber-600 focus:border-amber-600"
            />
          </div>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto px-2 py-1 space-y-1">
          <div className="px-2 py-1 text-[11px] font-semibold text-stone-400 uppercase tracking-wider">
            History ({filtered.length})
          </div>

          {filtered.length === 0 ? (
            <div className="px-3 py-8 text-center space-y-1.5">
              <p className="text-xs font-medium text-stone-500">
                {conversations.length === 0 ? 'No past explorations' : 'No matching explorations found.'}
              </p>
              {conversations.length === 0 && (
                <p className="text-[11px] text-stone-400">
                  Upload an image to start a new exploration.
                </p>
              )}
            </div>
          ) : (
            filtered.map((conv) => {
              const isActive = conv.id === activeId;
              const isEditing = editingId === conv.id;

              return (
                <div
                  key={conv.id}
                  onClick={() => onSelect(conv.id)}
                  className={`
                    group relative flex items-center justify-between px-3 py-2 rounded-lg text-xs cursor-pointer transition-all
                    ${isActive 
                      ? 'bg-white text-[#1F2923] font-medium shadow-xs border border-[#E5DFD7]' 
                      : 'text-stone-600 hover:bg-[#EBE5DE] hover:text-[#1F2923]'}
                  `}
                >
                  <div className="flex items-center space-x-2.5 min-w-0 flex-1 mr-1">
                    <MessageSquare className={`w-3.5 h-3.5 shrink-0 ${isActive ? 'text-[#C85A32]' : 'text-stone-400'}`} />
                    {isEditing ? (
                      <input
                        type="text"
                        autoFocus
                        value={editTitle}
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => setEditTitle(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleSaveRename(conv.id, e as any);
                          if (e.key === 'Escape') setEditingId(null);
                        }}
                        className="w-full bg-stone-50 border border-amber-500 rounded px-1.5 py-0.5 text-xs text-stone-900 focus:outline-none"
                      />
                    ) : (
                      <span className="truncate">{conv.title}</span>
                    )}
                  </div>

                  {/* Actions */}
                  <div className="flex items-center space-x-1 shrink-0">
                    {isEditing ? (
                      <>
                        <button
                          onClick={(e) => handleSaveRename(conv.id, e)}
                          className="p-1 rounded text-emerald-700 hover:bg-emerald-50"
                          title="Save"
                        >
                          <Check className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={handleCancelRename}
                          className="p-1 rounded text-stone-500 hover:bg-stone-200"
                          title="Cancel"
                        >
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </>
                    ) : (
                      <div className="opacity-0 group-hover:opacity-100 flex items-center space-x-1 transition-opacity">
                        <button
                          onClick={(e) => startRename(conv, e)}
                          className="p-1 rounded text-stone-500 hover:text-stone-800 hover:bg-stone-200"
                          title="Rename"
                        >
                          <Edit3 className="w-3 h-3" />
                        </button>
                        <button
                          onClick={(e) => handleDelete(conv.id, e)}
                          className="p-1 rounded text-stone-500 hover:text-rose-600 hover:bg-rose-50"
                          title="Delete"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Footer Utilities */}
        <div className="p-3 border-t border-[#E5DFD7] space-y-1.5 bg-[#EFEAE3]/70">
          <button
            onClick={onOpenMonuments}
            className="w-full flex items-center justify-between px-2.5 py-2 rounded-lg text-xs font-medium text-stone-700 hover:bg-white hover:text-stone-900 border border-transparent hover:border-[#E5DFD7] transition-all"
          >
            <div className="flex items-center space-x-2">
              <BookOpen className="w-4 h-4 text-[#C85A32]" />
              <span>Supported Monuments</span>
            </div>
            <ChevronRight className="w-3.5 h-3.5 text-stone-400" />
          </button>

          <div className="p-2 rounded-md bg-stone-100/80 border border-stone-200/60 text-[11px] text-stone-600 flex items-start space-x-2">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-600 mt-0.5 shrink-0" />
            <div>
              <p className="font-semibold text-stone-800">Traditional DIP Pipeline</p>
              <p className="text-[10px] text-stone-500">Sobel Edges • HSV • Histograms • SVM</p>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};
