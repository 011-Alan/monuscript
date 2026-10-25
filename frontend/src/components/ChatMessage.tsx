import React, { useState, useEffect } from 'react';
import { Landmark, User, AlertCircle, Info, Sparkles, RefreshCw } from 'lucide-react';
import { Message, RecognitionCandidate } from '../types';
import { RecognitionResult } from './RecognitionResult';
import { formatImageUrl } from '../services/api';
import { getThumbnailCache } from '../services/storage';

interface ChatMessageProps {
  message: Message;
  onOpen3D?: (url: string, name: string) => void;
  onInspectFeatures?: () => void;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  message,
  onOpen3D,
  onInspectFeatures
}) => {
  const isUser = message.role === 'user';
  const [imageError, setImageError] = useState(false);
  const [retryCount, setRetryCount] = useState(0);

  // Compute best available initial image source (prioritizing base64 data URLs)
  const resolveInitialSrc = (): string | undefined => {
    if (!message.image_url) return undefined;
    if (message.image_url.startsWith('data:')) return message.image_url;
    const cached = getThumbnailCache(message.id) || getThumbnailCache(message.image_url);
    if (cached) return cached;
    return formatImageUrl(message.image_url);
  };

  const [currentSrc, setCurrentSrc] = useState<string | undefined>(resolveInitialSrc);

  // Auto-sync currentSrc if message.image_url or message.id updates
  useEffect(() => {
    setImageError(false);
    setRetryCount(0);
    setCurrentSrc(resolveInitialSrc());
  }, [message.id, message.image_url]);

  const handleImageError = () => {
    // 1. If currently failing on a remote URL, try falling back to the thumbnail cache
    if (currentSrc && !currentSrc.startsWith('data:')) {
      const cached = getThumbnailCache(message.id) || getThumbnailCache(message.image_url || '');
      if (cached && currentSrc !== cached) {
        setCurrentSrc(cached);
        return;
      }

      // 2. Retry up to 2 times with a slight delay (useful if server was waking up from cold-start)
      if (retryCount < 2) {
        setRetryCount(prev => prev + 1);
        setTimeout(() => {
          const formatted = formatImageUrl(message.image_url);
          if (formatted) {
            const sep = formatted.includes('?') ? '&' : '?';
            setCurrentSrc(`${formatted}${sep}_retry=${Date.now()}`);
          }
        }, 1200);
        return;
      }
    }
    setImageError(true);
  };

  // Format candidate data if present in stored message results
  const candidates: RecognitionCandidate[] = (message.results || [])
    .filter(r => r.accepted)
    .map(r => ({
      monument_id: r.monument_id,
      name: r.metadata?.name || r.monument_id.replace('_', ' ').toUpperCase(),
      score: r.score,
      score_type: r.score_type,
      location: r.metadata?.location || 'India',
      description: r.metadata?.description || 'Historical Monument',
      historical_details: r.metadata?.historical_details || [],
      key_features: r.metadata?.key_features || [],
      architectural_style: r.metadata?.architectural_style,
      built_in: r.metadata?.built_in,
      builder: r.metadata?.builder,
      model_url: r.metadata?.model_url || null,
      detected_region: (r.metadata as any)?.detected_region
    }));

  const rejectedResult = (message.results || []).find(r => !r.accepted);
  const hasRejectedResults = Boolean(rejectedResult);

  // Determine if content describes rejection or ambiguous prediction
  const lowerContent = (message.content || '').toLowerCase();
  const isRejection =
    hasRejectedResults ||
    lowerContent.includes('rejection filter') ||
    lowerContent.includes('rejection mechanism') ||
    lowerContent.includes('confidence margin') ||
    lowerContent.includes('margin between') ||
    lowerContent.includes('ambiguous candidate') ||
    lowerContent.includes('margin insufficient') ||
    lowerContent.includes('resemblance to');

  const isError =
    message.content?.startsWith('⚠️') ||
    lowerContent.includes('recognition issue') ||
    lowerContent.includes('trouble reaching') ||
    lowerContent.includes('couldn\'t open this image') ||
    lowerContent.includes('error:');

  const runnerUpClass = rejectedResult?.metadata?.runner_up_class;
  const runnerUpScore = rejectedResult?.metadata?.runner_up_score;
  const processingTimeMs = rejectedResult?.metadata?.processing_time_ms;

  return (
    <div className={`py-4 px-4 md:px-6 flex space-x-3.5 ${isUser ? 'bg-transparent' : 'bg-[#FAF8F5]'}`}>
      {/* Avatar */}
      <div className={`
        w-8 h-8 rounded-full flex items-center justify-center shrink-0 text-white shadow-2xs
        ${isUser ? 'bg-stone-700' : 'bg-[#C85A32]'}
      `}>
        {isUser ? <User className="w-4 h-4" /> : <Landmark className="w-4 h-4" />}
      </div>

      {/* Content Body */}
      <div className="flex-1 space-y-3 min-w-0">
        <div className="flex items-center space-x-2">
          <span className="font-semibold text-xs text-stone-900">
            {isUser ? 'Explorer' : 'MonuScript Vision'}
          </span>
          <span className="text-[10px] text-stone-400">
            {new Date(message.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>
        </div>

        {/* User uploaded image thumbnail */}
        {(currentSrc || message.image_url) && (
          <div className="rounded-xl overflow-hidden border border-[#E5DFD7] max-w-sm bg-stone-100 shadow-2xs">
            {!imageError && currentSrc ? (
              <img
                src={currentSrc}
                alt="Uploaded monument"
                className="w-full h-auto max-h-72 object-contain bg-stone-900/5 transition-opacity duration-200"
                onError={handleImageError}
              />
            ) : (
              <div className="p-3.5 flex items-center justify-between space-x-2.5 text-stone-500 text-xs bg-stone-50">
                <div className="flex items-center space-x-2">
                  <Sparkles className="w-4 h-4 text-[#C85A32] shrink-0" />
                  <span>Uploaded monument photograph</span>
                </div>
                {message.image_url && (
                  <button
                    onClick={() => {
                      setImageError(false);
                      setRetryCount(0);
                      const fallback = resolveInitialSrc();
                      setCurrentSrc(fallback ? `${fallback}${fallback.includes('?') ? '&' : '?'}retry=${Date.now()}` : undefined);
                    }}
                    className="inline-flex items-center space-x-1 text-[11px] font-medium text-[#C85A32] hover:underline cursor-pointer"
                    title="Retry loading photo"
                  >
                    <RefreshCw className="w-3 h-3" />
                    <span>Reload</span>
                  </button>
                )}
              </div>
            )}
          </div>
        )}

        {/* User Prompt Bubble */}
        {isUser && message.content && (
          <div className="inline-block p-3 rounded-xl bg-[#F0ECE1] border border-[#E2DBD0] text-stone-800 text-xs md:text-sm leading-relaxed max-w-2xl">
            {message.content}
          </div>
        )}

        {/* Assistant Output: Accepted Recognition Result */}
        {!isUser && candidates.length > 0 && (
          <RecognitionResult
            accepted={true}
            message={message.content}
            results={candidates}
            processingTimeMs={processingTimeMs}
            onOpen3D={onOpen3D}
            onInspectFeatures={onInspectFeatures}
          />
        )}

        {/* Assistant Output: Rejection Filter Outer Panel */}
        {!isUser && candidates.length === 0 && isRejection && (
          <RecognitionResult
            accepted={false}
            message={message.content}
            results={[]}
            runnerUpClass={runnerUpClass}
            runnerUpScore={runnerUpScore}
            processingTimeMs={processingTimeMs}
            onInspectFeatures={onInspectFeatures}
          />
        )}

        {/* Assistant Output: Error Notice Outer Panel */}
        {!isUser && candidates.length === 0 && !isRejection && isError && (
          <div className="p-4 rounded-xl bg-rose-50/90 border border-rose-200 text-rose-950 space-y-2 max-w-2xl shadow-2xs">
            <div className="flex items-start space-x-2.5">
              <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <h4 className="font-semibold text-xs text-rose-950 uppercase tracking-wider">
                  Recognition Notice
                </h4>
                <p className="text-xs md:text-sm text-rose-800 whitespace-pre-line leading-relaxed">
                  {message.content}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Assistant Output: General Informational Outer Panel */}
        {!isUser && candidates.length === 0 && !isRejection && !isError && message.content && (
          <div className="p-4 rounded-xl bg-white border border-[#E5DFD7] text-stone-800 space-y-2 max-w-2xl shadow-2xs">
            <div className="flex items-start space-x-2.5">
              <Info className="w-4 h-4 text-[#C85A32] shrink-0 mt-0.5" />
              <p className="text-xs md:text-sm text-stone-800 whitespace-pre-line leading-relaxed">
                {message.content}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
