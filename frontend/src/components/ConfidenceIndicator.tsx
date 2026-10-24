import React from 'react';

interface ConfidenceIndicatorProps {
  score: number;
  scoreType?: string;
  showBar?: boolean;
}

export const ConfidenceIndicator: React.FC<ConfidenceIndicatorProps> = ({
  score,
  scoreType = 'recognition_score',
  showBar = true
}) => {
  const percentage = Math.round(score * 100);

  // Determine color theme based on score level
  let badgeColor = 'bg-emerald-50 text-emerald-800 border-emerald-200';
  let barColor = 'bg-emerald-500';

  if (percentage < 45) {
    badgeColor = 'bg-stone-100 text-stone-700 border-stone-200';
    barColor = 'bg-stone-500';
  } else if (percentage < 65) {
    badgeColor = 'bg-amber-50 text-amber-800 border-amber-200';
    barColor = 'bg-amber-500';
  }

  const formatScoreType = (type: string) => {
    switch (type) {
      case 'calibrated_score':
        return 'Calibrated Score';
      case 'region_score':
        return 'Region Scan';
      default:
        return 'Recognition Score';
    }
  };

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-stone-500 text-[11px] font-medium">
          {formatScoreType(scoreType)}
        </span>
        <span className={`px-2 py-0.5 rounded-full font-semibold text-[11px] border ${badgeColor}`}>
          {percentage}%
        </span>
      </div>

      {showBar && (
        <div className="w-full h-1.5 rounded-full bg-stone-100 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${barColor}`}
            style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }}
          />
        </div>
      )}
    </div>
  );
};
