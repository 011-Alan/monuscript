import React, { useState } from 'react';
import { Landmark, User, AlertCircle, Info, Sparkles } from 'lucide-react';
import { Message, RecognitionCandidate } from '../types';
import { RecognitionResult } from './RecognitionResult';
import { formatImageUrl } from '../services/api';

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
        {message.image_url && (
          <div className="rounded-xl overflow-hidden border border-[#E5DFD7] max-w-sm bg-stone-100 shadow-2xs">
            {!imageError ? (
              <img
                src={formatImageUrl(message.image_url)}
                alt="Uploaded monument"
                className="w-full h-auto max-h-72 object-contain bg-stone-900/5 transition-opacity duration-200"
                onError={() => setImageError(true)}
              />
            ) : (
              <div className="p-3.5 flex items-center space-x-2.5 text-stone-500 text-xs bg-stone-50">
                <Sparkles className="w-4 h-4 text-[#C85A32] shrink-0" />
                <span>Uploaded monument photograph</span>
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
