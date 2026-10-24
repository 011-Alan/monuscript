import React from 'react';
import { Landmark, Sparkles, Sliders, ShieldCheck } from 'lucide-react';

interface EmptyStateProps {
  onSelectSample?: (monumentName: string) => void;
  onOpenSupportedModal: () => void;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ onOpenSupportedModal }) => {
  const monuments = [
    { name: 'Taj Mahal', city: 'Agra, UP' },
    { name: 'Qutub Minar', city: 'Delhi' },
    { name: 'Red Fort', city: 'Delhi' },
    { name: 'India Gate', city: 'New Delhi' },
    { name: 'Charminar', city: 'Hyderabad' },
    { name: 'Hawa Mahal', city: 'Jaipur' },
  ];

  return (
    <div className="max-w-2xl mx-auto py-8 px-4 text-center space-y-6">
      {/* Icon Badge */}
      <div className="inline-flex p-3 rounded-2xl bg-amber-50 border border-amber-200 text-[#C85A32] shadow-xs">
        <Landmark className="w-8 h-8" />
      </div>

      {/* Main Title */}
      <div className="space-y-2">
        <h2 className="font-serif text-2xl md:text-3xl font-bold text-[#1F2923]">
          Syllabus-Based Indian Monument Exploration
        </h2>
        <p className="text-sm md:text-base text-stone-600 max-w-lg mx-auto">
          Upload any photograph of a supported Indian heritage monument. The system extracts deterministic mathematical features and evaluates them using a trained Support Vector Machine.
        </p>
      </div>

      {/* Core Syllabus Features Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-left pt-2">
        <div className="p-3.5 rounded-xl bg-white border border-[#E5DFD7] shadow-2xs space-y-1.5">
          <div className="flex items-center space-x-2 text-[#C85A32]">
            <Sliders className="w-4 h-4" />
            <span className="text-xs font-semibold text-stone-800">Spatial Derivatives</span>
          </div>
          <p className="text-[11px] text-stone-500 leading-relaxed">
            Horizontal & vertical Sobel convolutions (<code className="text-amber-800 font-mono">Sx, Sy</code>) generate gradient magnitude & orientation histograms.
          </p>
        </div>

        <div className="p-3.5 rounded-xl bg-white border border-[#E5DFD7] shadow-2xs space-y-1.5">
          <div className="flex items-center space-x-2 text-[#C85A32]">
            <Sparkles className="w-4 h-4" />
            <span className="text-xs font-semibold text-stone-800">Statistical Moments</span>
          </div>
          <p className="text-[11px] text-stone-500 leading-relaxed">
            Gray-level normalized histograms (<code className="text-amber-800 font-mono">h(k)</code>), intensity variance, uniformity, entropy, and 3x3 spatial layout grids.
          </p>
        </div>

        <div className="p-3.5 rounded-xl bg-white border border-[#E5DFD7] shadow-2xs space-y-1.5">
          <div className="flex items-center space-x-2 text-[#C85A32]">
            <ShieldCheck className="w-4 h-4" />
            <span className="text-xs font-semibold text-stone-800">Calibrated Rejection</span>
          </div>
          <p className="text-[11px] text-stone-500 leading-relaxed">
            Confidence margin thresholds and out-of-distribution distance metrics reject non-monument or ambiguous uploads.
          </p>
        </div>
      </div>

      {/* Supported Monuments Chips */}
      <div className="pt-2">
        <p className="text-xs font-medium text-stone-500 mb-2.5">
          Supported Monuments in Database:
        </p>
        <div className="flex flex-wrap items-center justify-center gap-1.5">
          {monuments.map((m) => (
            <button
              key={m.name}
              onClick={onOpenSupportedModal}
              className="px-2.5 py-1 rounded-full text-xs bg-[#F3EFEA] hover:bg-[#EBE5DE] text-stone-700 border border-[#E5DFD7] transition-colors"
            >
              <span className="font-medium">{m.name}</span>
              <span className="text-stone-400 text-[10px] ml-1">({m.city})</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
