import React from 'react';
import { X, Box, ExternalLink } from 'lucide-react';

interface ThreeDModalProps {
  isOpen: boolean;
  onClose: () => void;
  modelUrl: string | null;
  monumentName: string;
}

export const ThreeDModal: React.FC<ThreeDModalProps> = ({
  isOpen,
  onClose,
  modelUrl,
  monumentName
}) => {
  if (!isOpen || !modelUrl) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-stone-900/50 backdrop-blur-xs">
      <div className="w-full max-w-4xl bg-white rounded-2xl shadow-2xl border border-[#E5DFD7] overflow-hidden flex flex-col h-[85vh]">
        {/* Header */}
        <div className="px-5 py-4 border-b border-[#E5DFD7] flex items-center justify-between bg-sand-50">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 rounded-lg bg-amber-100 text-[#C85A32]">
              <Box className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-serif font-bold text-base text-[#1F2923]">
                Interactive 3D Architectural Model
              </h3>
              <p className="text-xs text-stone-500">
                {monumentName} • Rotatable Spatial Geometry
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <a
              href={modelUrl.replace('/embed', '')}
              target="_blank"
              rel="noopener noreferrer"
              className="p-1.5 rounded-lg text-stone-500 hover:text-stone-800 hover:bg-stone-200 transition-colors"
              title="Open Source in New Tab"
            >
              <ExternalLink className="w-4 h-4" />
            </a>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-stone-400 hover:text-stone-700 hover:bg-stone-200"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* 3D Viewport */}
        <div className="flex-1 bg-stone-950 relative">
          <iframe
            title={`3D Model of ${monumentName}`}
            src={modelUrl}
            className="w-full h-full border-0"
            allow="autoplay; fullscreen; xr-spatial-tracking"
            allowFullScreen
          />
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-[#E5DFD7] bg-sand-50 flex items-center justify-between text-xs text-stone-500">
          <span>Use mouse click + drag to rotate, scroll to zoom in/out.</span>
          <button
            onClick={onClose}
            className="px-3.5 py-1.5 rounded-lg text-xs font-medium bg-stone-200 hover:bg-stone-300 text-stone-800 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
