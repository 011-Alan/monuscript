import React from 'react';
import { ShieldAlert, Info, Sparkles, CheckCircle2 } from 'lucide-react';
import { RecognitionCandidate, FeatureSummary } from '../types';
import { MonumentCard } from './MonumentCard';

interface RecognitionResultProps {
  accepted: boolean;
  isUncertain?: boolean;
  message: string;
  results: RecognitionCandidate[];
  topScore?: number;
  runnerUpClass?: string | null;
  runnerUpScore?: number | null;
  processingTimeMs?: number;
  featureSummary?: FeatureSummary;
  onOpen3D?: (url: string, name: string) => void;
  onInspectFeatures?: () => void;
}

export const RecognitionResult: React.FC<RecognitionResultProps> = ({
  accepted,
  isUncertain,
  message,
  results,
  runnerUpClass,
  runnerUpScore,
  processingTimeMs,
  onOpen3D,
  onInspectFeatures
}) => {
  if (!accepted) {
    return (
      <div className="space-y-3">
        {/* Friendly Rejection Box */}
        <div className="p-4 rounded-xl bg-amber-50/80 border border-amber-200 text-stone-800 space-y-2.5">
          <div className="flex items-start space-x-2.5">
            <ShieldAlert className="w-5 h-5 text-[#C85A32] shrink-0 mt-0.5" />
            <div className="space-y-1">
              <h4 className="font-semibold text-xs text-amber-950 uppercase tracking-wider">
                {isUncertain ? 'Uncertain / Ambiguous Candidate' : 'Rejection Filter Triggered'}
              </h4>
              <p className="text-xs text-stone-700 whitespace-pre-line leading-relaxed">
                {message}
              </p>
            </div>
          </div>

          {runnerUpClass && runnerUpScore && (
            <div className="mt-2 pt-2 border-t border-amber-200/60 text-[11px] text-stone-600">
              Closest candidate tested: <span className="font-medium text-stone-800">{runnerUpClass.replace('_', ' ').toUpperCase()}</span> ({Math.round(runnerUpScore * 100)}% score). Margin insufficient for confident identification.
            </div>
          )}
        </div>

        {onInspectFeatures && (
          <button
            onClick={onInspectFeatures}
            className="text-xs text-[#C85A32] hover:underline font-medium"
          >
            Inspect extracted feature vectors for this image →
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Result Cards */}
      <div className="space-y-3">
        {results.map((cand, idx) => (
          <MonumentCard
            key={`${cand.monument_id}-${idx}`}
            candidate={cand}
            onOpen3D={onOpen3D}
            onInspectFeatures={onInspectFeatures}
          />
        ))}
      </div>

      {/* Latency and Pipeline Badge */}
      <div className="flex flex-wrap items-center justify-between text-[11px] text-stone-500 pt-1">
        <div className="flex items-center space-x-1.5 text-emerald-800">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
          <span>Accepted by SVM Rejection Mechanism</span>
        </div>
        {processingTimeMs !== undefined && (
          <span>Inference latency: {processingTimeMs} ms</span>
        )}
      </div>
    </div>
  );
};
