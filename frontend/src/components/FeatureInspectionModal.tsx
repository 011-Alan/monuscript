import React from 'react';
import { X, Activity, Sliders, Layers, Eye, ShieldCheck } from 'lucide-react';
import { FeatureSummary } from '../types';

interface FeatureInspectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  summary?: FeatureSummary;
}

export const FeatureInspectionModal: React.FC<FeatureInspectionModalProps> = ({
  isOpen,
  onClose,
  summary
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-stone-900/40 backdrop-blur-xs">
      <div className="w-full max-w-2xl bg-white rounded-2xl shadow-xl border border-[#E5DFD7] overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-5 py-4 border-b border-[#E5DFD7] flex items-center justify-between bg-sand-50">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 rounded-lg bg-amber-100 text-[#C85A32]">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-serif font-bold text-base text-[#1F2923]">
                DIP Syllabus Feature Pipeline Inspector
              </h3>
              <p className="text-xs text-stone-500">
                139-Dimensional Deterministic Mathematical Representation
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

        {/* Content */}
        <div className="p-6 overflow-y-auto space-y-6 text-xs text-stone-700">
          <div className="p-3.5 rounded-xl bg-amber-50/60 border border-amber-200 space-y-1">
            <h4 className="font-semibold text-amber-950 flex items-center space-x-1.5">
              <ShieldCheck className="w-4 h-4 text-[#C85A32]" />
              <span>Strict Syllabus Constraint Compliance</span>
            </h4>
            <p className="text-stone-600 leading-relaxed text-[11px]">
              No deep learning architectures, CNNs, Transformers, or external vision APIs are used. 
              The classifier runs on pure mathematical features extracted strictly via classical image processing formulas.
            </p>
          </div>

          {/* Feature Groups Breakdown */}
          <div className="space-y-4">
            <h4 className="font-serif font-bold text-sm text-stone-900 border-b pb-1.5">
              Extracted Feature Group Composition
            </h4>

            {/* Group 1: Histogram */}
            <div className="p-3.5 rounded-xl border border-stone-200 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-stone-800 text-xs">
                  1. Grayscale Histogram & Statistical Moments
                </span>
                <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-stone-100 text-stone-700">
                  40 Dimensions
                </span>
              </div>
              <p className="text-[11px] text-stone-500 leading-relaxed">
                Normalized gray-level histogram <code className="font-mono bg-stone-100 px-1">h(k) = n_k / (M*N)</code> across 32 intensity bins. 
                Coupled with 8 central moments: Mean (μ), Variance (σ²), Standard Deviation (σ), Smoothness (R = 1 - 1/(1+σ²)), Uniformity/Energy (Σ h(k)²), Entropy (-Σ h log₂(h)), Skewness (μ₃), and Median.
              </p>
            </div>

            {/* Group 2: Edge */}
            <div className="p-3.5 rounded-xl border border-stone-200 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-stone-800 text-xs">
                  2. Sobel Spatial Gradient Derivatives
                </span>
                <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-stone-100 text-stone-700">
                  36 Dimensions
                </span>
              </div>
              <p className="text-[11px] text-stone-500 leading-relaxed">
                Convolutions with 3x3 Sobel kernels <code className="font-mono bg-stone-100 px-1">Gx = I * Sx</code> and <code className="font-mono bg-stone-100 px-1">Gy = I * Sy</code>.
                Yields Gradient Magnitude <code className="font-mono bg-stone-100 px-1">G = √(Gx² + Gy²)</code> and Orientation <code className="font-mono bg-stone-100 px-1">θ = atan2(Gy, Gx)</code>.
                Extracts 16 magnitude bins, 16 magnitude-weighted orientation bins ([0°, 180°)), Otsu edge density, and horizontal-to-vertical anisotropy ratio.
              </p>
            </div>

            {/* Group 3: Texture & Spatial Grid */}
            <div className="p-3.5 rounded-xl border border-stone-200 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-stone-800 text-xs">
                  3. Texture Variance & 3x3 Spatial Grid Layout
                </span>
                <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-stone-100 text-stone-700">
                  31 Dimensions
                </span>
              </div>
              <p className="text-[11px] text-stone-500 leading-relaxed">
                Local standard deviation filter <code className="font-mono bg-stone-100 px-1">σ_local(x,y)</code> measuring neighborhood spatial variation (4 global moments).
                Augmented with a 3x3 spatial grid partition (9 cells): each cell extracts local mean, local standard deviation, and local edge density to encode architectural layout (e.g. dome on top, minarets at sides, plinth at bottom).
              </p>
            </div>

            {/* Group 4: Color */}
            <div className="p-3.5 rounded-xl border border-stone-200 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-stone-800 text-xs">
                  4. HSV Color Space Distributions
                </span>
                <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-stone-100 text-stone-700">
                  32 Dimensions
                </span>
              </div>
              <p className="text-[11px] text-stone-500 leading-relaxed">
                Normalized histograms of Hue (16 bins), Saturation (8 bins), and Value (8 bins).
                Distinguishes materials like white Makrana marble (Taj Mahal), red/buff sandstone (Red Fort, Qutub Minar), and pink sandstone (Hawa Mahal).
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-[#E5DFD7] bg-sand-50 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-xs font-medium bg-stone-800 hover:bg-stone-900 text-white transition-colors"
          >
            Close Inspector
          </button>
        </div>
      </div>
    </div>
  );
};
