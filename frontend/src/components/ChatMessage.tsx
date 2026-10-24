import React from 'react';
import { Landmark, User, ExternalLink } from 'lucide-react';
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

  const hasRejectedResults = (message.results || []).some(r => !r.accepted);

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
            <img
              src={formatImageUrl(message.image_url)}
              alt="Uploaded monument"
              className="w-full h-auto max-h-72 object-contain bg-stone-900/5"
            />
          </div>
        )}

        {/* Text content */}
        {message.content && (!candidates.length && !hasRejectedResults || isUser) && (
          <p className="text-xs md:text-sm text-stone-800 whitespace-pre-line leading-relaxed">
            {message.content}
          </p>
        )}

        {/* Assistant Recognition Output */}
        {!isUser && message.results && message.results.length > 0 && (
          <RecognitionResult
            accepted={candidates.length > 0}
            message={message.content}
            results={candidates}
            onOpen3D={onOpen3D}
            onInspectFeatures={onInspectFeatures}
          />
        )}
      </div>
    </div>
  );
};
