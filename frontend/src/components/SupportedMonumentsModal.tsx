import React from 'react';
import { X, Landmark, MapPin, Box } from 'lucide-react';
import { MonumentMetadata } from '../types';

interface SupportedMonumentsModalProps {
  isOpen: boolean;
  onClose: () => void;
  monuments: MonumentMetadata[];
  onOpen3D: (url: string, name: string) => void;
}

export const SupportedMonumentsModal: React.FC<SupportedMonumentsModalProps> = ({
  isOpen,
  onClose,
  monuments,
  onOpen3D
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-stone-900/40 backdrop-blur-xs">
      <div className="w-full max-w-3xl bg-white rounded-2xl shadow-xl border border-[#E5DFD7] overflow-hidden flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="px-5 py-4 border-b border-[#E5DFD7] flex items-center justify-between bg-sand-50">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 rounded-lg bg-amber-100 text-[#C85A32]">
              <Landmark className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-serif font-bold text-base text-[#1F2923]">
                Supported Indian Monuments ({monuments.length})
              </h3>
              <p className="text-xs text-stone-500">
                Trained SVM Recognition Classes
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-stone-400 hover:text-stone-700 hover:bg-stone-200"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* List of Monuments */}
        <div className="p-5 overflow-y-auto space-y-3">
          {monuments.map((m) => (
            <div
              key={m.id}
              className="p-3.5 rounded-xl border border-stone-200 hover:border-amber-300 transition-colors bg-sand-50/50 space-y-2"
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <h4 className="font-serif font-bold text-sm text-stone-900">{m.name}</h4>
                  <div className="flex items-center space-x-1.5 text-xs text-stone-500">
                    <MapPin className="w-3 h-3 text-[#C85A32]" />
                    <span>{m.location}</span>
                  </div>
                </div>

                {m.model_url && (
                  <button
                    onClick={() => onOpen3D(m.model_url!, m.name)}
                    className="flex items-center space-x-1.5 px-2.5 py-1 rounded-md text-xs font-medium text-amber-900 bg-amber-100 hover:bg-amber-200 border border-amber-300 w-fit"
                  >
                    <Box className="w-3 h-3 text-[#C85A32]" />
                    <span>3D Model</span>
                  </button>
                )}
              </div>

              <p className="text-xs text-stone-600 leading-relaxed">
                {m.description}
              </p>

              {m.key_features && (
                <div className="flex flex-wrap gap-1 pt-1">
                  {m.key_features.map((feat, i) => (
                    <span
                      key={i}
                      className="px-2 py-0.5 rounded text-[10px] bg-white border border-stone-200 text-stone-700"
                    >
                      {feat}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-[#E5DFD7] bg-sand-50 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-xs font-medium bg-stone-800 hover:bg-stone-900 text-white transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
