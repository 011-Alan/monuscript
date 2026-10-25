import React, { useState, useEffect } from 'react';
import { X, Box, ExternalLink, Loader2, AlertCircle } from 'lucide-react';

interface ThreeDModalProps {
  isOpen: boolean;
  onClose: () => void;
  modelUrl: string | null;
  monumentName: string;
}

// Verified live Sketchfab 3D models with active embed authorizations
const VERIFIED_MODELS: Record<string, { embed: string; source: string; artist: string; style: string }> = {
  taj_mahal: {
    embed: 'https://sketchfab.com/models/33149233cefd492b9abdd50fe5a8c921/embed?autostart=1',
    source: 'https://sketchfab.com/3d-models/taj-mahal-33149233cefd492b9abdd50fe5a8c921',
    artist: 'uday14viru',
    style: 'Mughal Architecture'
  },
  qutub_minar: {
    embed: 'https://sketchfab.com/models/e165e25ec8f1497aa83ab72419f40ddc/embed?autostart=1',
    source: 'https://sketchfab.com/3d-models/qutub-minar-e165e25ec8f1497aa83ab72419f40ddc',
    artist: '3D Dreamscapes',
    style: 'Indo-Islamic (Ghurid / Mamluk)'
  },
  red_fort: {
    embed: 'https://sketchfab.com/models/74ff6d703a174f9fb1ba266003f5c4fc/embed?autostart=1',
    source: 'https://sketchfab.com/3d-models/red-fort-delhi-74ff6d703a174f9fb1ba266003f5c4fc',
    artist: 'STHAPATYA',
    style: 'Mughal Fortified Architecture'
  },
  india_gate: {
    embed: 'https://sketchfab.com/models/59fe55328271479d82acb65310178d99/embed?autostart=1',
    source: 'https://sketchfab.com/3d-models/india-gate-59fe55328271479d82acb65310178d99',
    artist: 'Macwelshman',
    style: 'Neo-Classical Triumphal Arch'
  },
  charminar: {
    embed: 'https://sketchfab.com/models/82603f1fe3ed4c31b37786e2bc2b63b0/embed?autostart=1',
    source: 'https://sketchfab.com/3d-models/charminar-hyderabad-82603f1fe3ed4c31b37786e2bc2b63b0',
    artist: 'Dolores',
    style: 'Indo-Islamic (Qutb Shahi)'
  },
  hawa_mahal: {
    embed: 'https://sketchfab.com/models/c6bbf5822010490d995869ae625b1be0/embed?autostart=1',
    source: 'https://sketchfab.com/3d-models/hawa-mahal-3d-model-c6bbf5822010490d995869ae625b1be0',
    artist: 'meigandan',
    style: 'Rajput Architecture'
  }
};

function getMonumentKey(name: string): string {
  const norm = name.toLowerCase().replace(/[^a-z0-9]/g, '_');
  if (norm.includes('taj')) return 'taj_mahal';
  if (norm.includes('qutub') || norm.includes('qutb')) return 'qutub_minar';
  if (norm.includes('red') || norm.includes('qila') || norm.includes('lal')) return 'red_fort';
  if (norm.includes('india') || norm.includes('gate')) return 'india_gate';
  if (norm.includes('charminar')) return 'charminar';
  if (norm.includes('hawa') || norm.includes('wind')) return 'hawa_mahal';
  return 'taj_mahal';
}

function resolveModelUrl(rawUrl: string | null, name: string): { embedUrl: string; sourceUrl: string; artist: string; style: string } {
  const key = getMonumentKey(name);
  const info = VERIFIED_MODELS[key];

  // If rawUrl is missing, or is one of the old broken 404 dummy hashes or old wall model, use verified live model
  const isOutdatedOrBroken = !rawUrl || 
    rawUrl.includes('31f47285') || 
    rawUrl.includes('bf041ca4') || 
    rawUrl.includes('a5113eb7') || 
    rawUrl.includes('ca6c729e') || 
    rawUrl.includes('2d2fe62f') || 
    rawUrl.includes('8333bc7b') ||
    rawUrl.includes('47c0ea2a759544729c137d8b86ba5b9a'); // Old Hawa Mahal wall slice

  if (isOutdatedOrBroken && info) {
    return { embedUrl: info.embed, sourceUrl: info.source, artist: info.artist, style: info.style };
  }

  let finalEmbed = rawUrl || info?.embed || '';
  if (!finalEmbed.includes('autostart=')) {
    finalEmbed += (finalEmbed.includes('?') ? '&' : '?') + 'autostart=1';
  }
  const finalSource = finalEmbed.replace('/embed', '').split('?')[0];
  return { 
    embedUrl: finalEmbed, 
    sourceUrl: finalSource, 
    artist: info?.artist || 'Sketchfab Community', 
    style: info?.style || 'Heritage Architecture' 
  };
}

export const ThreeDModal: React.FC<ThreeDModalProps> = ({
  isOpen,
  onClose,
  modelUrl,
  monumentName,
}) => {
  const [iframeLoading, setIframeLoading] = useState(true);
  const [iframeError, setIframeError] = useState(false);

  const resolved = resolveModelUrl(modelUrl, monumentName);

  useEffect(() => {
    if (isOpen) {
      setIframeLoading(true);
      setIframeError(false);
    }
  }, [isOpen, modelUrl, monumentName]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-stone-900/80 backdrop-blur-sm animate-fade-in">
      <div 
        className="bg-stone-900 rounded-2xl w-full max-w-5xl h-[85vh] flex flex-col shadow-2xl overflow-hidden border border-stone-800"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header Bar */}
        <div className="px-6 py-4 border-b border-stone-800 flex items-center justify-between bg-stone-950/60">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-[#C85A32]/20 border border-[#C85A32]/30 flex items-center justify-center text-[#C85A32]">
              <Box className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h3 className="font-serif font-bold text-lg text-stone-100">
                  {monumentName}
                </h3>
                <span className="text-[11px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-stone-800 text-stone-300 border border-stone-700">
                  Sketchfab 3D
                </span>
              </div>
              <p className="text-xs text-stone-400">
                {resolved.style} • Model by <span className="text-amber-400 font-medium">{resolved.artist}</span>
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <a
              href={resolved.sourceUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium text-stone-300 hover:text-white bg-stone-800/80 hover:bg-stone-700/80 border border-stone-700/60 rounded-lg transition-colors"
              title="Open full interactive model on Sketchfab"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Open on Sketchfab</span>
            </a>
            <button
              onClick={onClose}
              className="p-2 text-stone-400 hover:text-white hover:bg-stone-800 rounded-lg transition-colors"
              aria-label="Close 3D Viewport"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Viewport Area */}
        <div className="flex-1 w-full h-full relative bg-stone-950 overflow-hidden">
          {iframeLoading && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-stone-950 text-white space-y-3">
              <Loader2 className="w-8 h-8 animate-spin text-[#C85A32]" />
              <p className="text-xs text-stone-400 font-medium">
                Initializing Sketchfab 3D WebGL runtime...
              </p>
            </div>
          )}

          {iframeError && (
            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-stone-900 p-6 text-center space-y-3">
              <AlertCircle className="w-10 h-10 text-amber-500" />
              <h4 className="font-semibold text-sm text-stone-200">
                Network issue connecting to Sketchfab
              </h4>
              <p className="text-xs text-stone-400 max-w-sm">
                External 3D embeds may be blocked by your network or browser adblocker. You can explore the model directly on Sketchfab.
              </p>
              <a
                href={resolved.sourceUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-semibold bg-[#C85A32] text-white hover:bg-[#b04d28] transition-colors"
              >
                <span>View on Sketchfab</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </div>
          )}

          <iframe
            title={`3D Model of ${monumentName}`}
            src={resolved.embedUrl}
            className="w-full h-full border-0"
            allow="autoplay; fullscreen; xr-spatial-tracking; execution-while-out-of-viewport; execution-while-not-rendered; web-share"
            allowFullScreen
            onLoad={() => setIframeLoading(false)}
            onError={() => {
              setIframeLoading(false);
              setIframeError(true);
            }}
          />
        </div>

        {/* Footer Bar */}
        <div className="px-6 py-3 border-t border-stone-800 bg-stone-950/90 flex flex-wrap items-center justify-between gap-3 text-xs text-stone-400">
          <div className="flex items-center space-x-2">
            <span className="font-medium text-stone-200">Navigation:</span>
            <span>Left-click & drag to orbit • Scroll to zoom • Right-click & drag to pan</span>
          </div>

          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-stone-800 hover:bg-stone-700 text-stone-200 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
