import React, { useState } from 'react';
import { 
  MapPin, 
  Calendar, 
  User, 
  Box, 
  ChevronDown, 
  ChevronUp, 
  Sparkles, 
  Activity,
  Layers
} from 'lucide-react';
import { RecognitionCandidate } from '../types';
import { ConfidenceIndicator } from './ConfidenceIndicator';

interface MonumentCardProps {
  candidate: RecognitionCandidate;
  onOpen3D?: (url: string, name: string) => void;
  onInspectFeatures?: () => void;
}

export const MonumentCard: React.FC<MonumentCardProps> = ({
  candidate,
  onOpen3D,
  onInspectFeatures
}) => {
  const [showHistory, setShowHistory] = useState(false);

  return (
    <div className="rounded-xl border border-[#E5DFD7] bg-white p-4 space-y-4 shadow-2xs hover:border-amber-300 transition-colors">
      {/* Header and Score */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3 border-b border-stone-100 pb-3">
        <div>
          <div className="flex items-center space-x-2">
            <h3 className="font-serif font-bold text-lg text-[#1F2923]">
              {candidate.name}
            </h3>
            {candidate.detected_region && (
              <span className="px-2 py-0.5 rounded text-[10px] bg-amber-100 text-amber-900 font-medium">
                {candidate.detected_region}
              </span>
            )}
          </div>
          <div className="flex items-center space-x-1.5 text-xs text-[#667268] mt-0.5">
            <MapPin className="w-3.5 h-3.5 text-[#C85A32] shrink-0" />
            <span>{candidate.location}</span>
          </div>
        </div>

        <div className="sm:w-44 shrink-0">
          <ConfidenceIndicator
            score={candidate.score}
            scoreType={candidate.score_type}
          />
        </div>
      </div>

      {/* Description */}
      <p className="text-xs text-stone-700 leading-relaxed">
        {candidate.description}
      </p>

      {/* Architectural Badges */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px] text-stone-600 bg-sand-50/70 p-2.5 rounded-lg border border-sand-200">
        {candidate.architectural_style && (
          <div className="flex items-center space-x-1.5">
            <span className="font-semibold text-stone-800">Style:</span>
            <span className="truncate">{candidate.architectural_style}</span>
          </div>
        )}
        {candidate.built_in && (
          <div className="flex items-center space-x-1.5">
            <Calendar className="w-3 h-3 text-stone-400" />
            <span className="font-semibold text-stone-800">Built:</span>
            <span>{candidate.built_in}</span>
          </div>
        )}
        {candidate.builder && (
          <div className="flex items-center space-x-1.5 sm:col-span-2">
            <User className="w-3 h-3 text-stone-400" />
            <span className="font-semibold text-stone-800">Commissioned by:</span>
            <span className="truncate">{candidate.builder}</span>
          </div>
        )}
      </div>

      {/* Key Architectural Features */}
      {candidate.key_features && candidate.key_features.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-[11px] font-semibold text-stone-500 uppercase tracking-wider">
            Key Architectural Elements
          </p>
          <div className="flex flex-wrap gap-1.5">
            {candidate.key_features.map((feat, i) => (
              <span
                key={i}
                className="px-2 py-0.5 rounded-md text-[11px] bg-stone-100 text-stone-700 border border-stone-200"
              >
                {feat}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Expandable Historical Details Accordion */}
      {candidate.historical_details && candidate.historical_details.length > 0 && (
        <div className="border border-stone-200 rounded-lg overflow-hidden">
          <button
            onClick={() => setShowHistory(!showHistory)}
            className="w-full px-3 py-2 text-xs font-medium text-stone-700 bg-stone-50 hover:bg-stone-100 flex items-center justify-between transition-colors"
          >
            <span>Historical Significance ({candidate.historical_details.length} facts)</span>
            {showHistory ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
          {showHistory && (
            <div className="p-3 bg-white space-y-2 text-xs text-stone-600 border-t border-stone-200">
              <ul className="list-disc pl-4 space-y-1.5">
                {candidate.historical_details.map((fact, idx) => (
                  <li key={idx} className="leading-relaxed">
                    {fact}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Action Footer */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-stone-100">
        <div className="flex items-center space-x-2">
          {onInspectFeatures && (
            <button
              onClick={onInspectFeatures}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-stone-700 bg-stone-100 hover:bg-stone-200 border border-stone-200 transition-colors"
            >
              <Activity className="w-3.5 h-3.5 text-[#C85A32]" />
              <span>Inspect DIP Features</span>
            </button>
          )}
        </div>

        {candidate.model_url && (
          <button
            onClick={() => onOpen3D && onOpen3D(candidate.model_url!, candidate.name)}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-amber-900 bg-amber-100 hover:bg-amber-200 border border-amber-300 transition-colors shadow-2xs"
          >
            <Box className="w-3.5 h-3.5 text-[#C85A32]" />
            <span>View 3D Monument Model</span>
          </button>
        )}
      </div>
    </div>
  );
};
