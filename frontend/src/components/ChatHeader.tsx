import React from 'react';
import { Menu, Layers, Cpu, ShieldCheck } from 'lucide-react';
import { SystemHealth } from '../types';

interface ChatHeaderProps {
  title: string;
  onToggleSidebar: () => void;
  health: SystemHealth | null;
  multiMode: boolean;
  onToggleMultiMode: () => void;
}

export const ChatHeader: React.FC<ChatHeaderProps> = ({
  title,
  onToggleSidebar,
  health,
  multiMode,
  onToggleMultiMode,
}) => {
  return (
    <header className="h-16 border-b border-[#E5DFD7] bg-[#FAF8F5]/90 backdrop-blur-sm px-4 md:px-6 flex items-center justify-between shrink-0">
      <div className="flex items-center space-x-3">
        <button
          onClick={onToggleSidebar}
          className="md:hidden p-2 rounded-lg text-stone-600 hover:bg-stone-200"
          aria-label="Open Navigation"
        >
          <Menu className="w-5 h-5" />
        </button>
        <div>
          <h1 className="font-serif font-bold text-base md:text-lg text-[#1F2923] truncate max-w-[200px] sm:max-w-xs md:max-w-md">
            {title}
          </h1>
          <p className="text-[11px] text-[#667268] hidden sm:block">
            Image Processing Pipeline & SVM Exploration
          </p>
        </div>
      </div>

      <div className="flex items-center space-x-3">
        {/* Multi-candidate sliding window toggle */}
        <button
          onClick={onToggleMultiMode}
          className={`flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
            multiMode
              ? 'bg-amber-50 text-amber-900 border-amber-300'
              : 'bg-white text-stone-600 border-[#E5DFD7] hover:bg-stone-100'
          }`}
          title="Enable sliding-window candidate inspection for images containing multiple monuments"
        >
          <Layers className="w-3.5 h-3.5 text-[#C85A32]" />
          <span className="hidden sm:inline">Sliding Window Multi-Scan</span>
          <span className={`text-[10px] px-1 rounded ${multiMode ? 'bg-amber-200' : 'bg-stone-200'}`}>
            {multiMode ? 'ON' : 'OFF'}
          </span>
        </button>

        {/* Backend health status pill */}
        <div className="flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg bg-white border border-[#E5DFD7] text-xs">
          <div className={`w-2 h-2 rounded-full ${health?.model_loaded ? 'bg-emerald-500' : 'bg-amber-500 animate-pulse'}`} />
          <span className="font-medium text-stone-700 hidden lg:inline">
            {health?.model_loaded ? `SVM Ready (${health.supported_classes} Classes)` : 'Connecting...'}
          </span>
          <span className="font-medium text-stone-700 lg:hidden">
            {health?.model_loaded ? 'Ready' : '...'}
          </span>
        </div>
      </div>
    </header>
  );
};
