export interface MonumentMetadata {
  id: string;
  name: string;
  location: string;
  description: string;
  historical_details: string[];
  key_features?: string[];
  architectural_style?: string;
  built_in?: string;
  builder?: string;
  model_url: string | null;
  supported: boolean;
}

export interface RecognitionCandidate {
  monument_id: string;
  name: string;
  score: number;
  score_type: string;
  location: string;
  description: string;
  historical_details: string[];
  key_features?: string[];
  architectural_style?: string;
  built_in?: string;
  builder?: string;
  model_url: string | null;
  detected_region?: string;
  bounding_box?: [number, number, number, number];
}

export interface FeatureSummary {
  total_features: number;
  groups: {
    [key: string]: {
      count: number;
      description: string;
      weight_applied: number;
    };
  };
}

export interface RecognitionResponse {
  success: boolean;
  accepted: boolean;
  is_uncertain?: boolean;
  message: string;
  results: RecognitionCandidate[];
  top_class?: string;
  top_score?: number;
  runner_up_class?: string | null;
  runner_up_score?: number | null;
  margin?: number;
  processing_time_ms: number;
  message_ids?: {
    user_message_id?: string;
    assistant_message_id?: string;
  };
  image: {
    filename?: string;
    url?: string;
    width: number;
    height: number;
  };
  feature_summary?: FeatureSummary;
}

export interface StoredResult {
  id: string;
  monument_id: string;
  score: number;
  score_type: string;
  accepted: boolean;
  metadata?: (Partial<RecognitionCandidate> & Record<string, any>);
}

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  image_url: string | null;
  created_at: string;
  results?: StoredResult[];
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface SystemHealth {
  status: string;
  model_loaded: boolean;
  supported_classes: number;
  classes: string[];
  service: string;
}
