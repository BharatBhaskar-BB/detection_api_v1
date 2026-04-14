/* ── API types matching backend schemas ── */

export interface User {
  id: string;
  email?: string;
  phone?: string;
  name: string;
  picture?: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface Room {
  id: string;
  name: string;
  emoji: string;
  is_custom: boolean;
  has_video: boolean;
  status: "pending" | "processing" | "completed" | "failed";
  order: number;
}

export interface Video {
  id: string;
  room_name?: string;
  filename: string;
  size_bytes: number;
  duration_s?: number;
  uploaded_at: string;
}

export interface Scan {
  id: string;
  display_id: string;
  scan_mode: "room_by_room" | "all_at_once";
  llm_provider: "openai" | "gemini";
  status: "recording" | "submitted" | "processing" | "completed" | "failed";
  progress: number;
  progress_message: string;
  pipeline_step: string;
  total_items: number;
  total_rooms: number;
  processing_time_s?: number;
  cost_usd?: number;
  created_at: string;
  completed_at?: string;
  move_summary?: MoveSummary | null;
  rooms: Room[];
  videos: Video[];
}

export interface ScanListItem {
  id: string;
  display_id: string;
  scan_mode: string;
  status: string;
  progress: number;
  progress_message: string;
  total_items: number;
  total_rooms: number;
  created_at: string;
  completed_at?: string;
  move_summary?: MoveSummary | null;
}

export interface MoveSummaryRoom {
  name: string;
  summary: string;
  instructions: string[];
}

export interface MoveSummary {
  has_audio: boolean;
  audio_summary: string;
  overall: string;
  rooms: MoveSummaryRoom[];
  special_instructions: string[];
}

export type Disposition =
  | "going"
  | "staying"
  | "scrap"
  | "haul"
  | "sell"
  | null;

export interface InventoryItem {
  id: string;
  name: string;
  count: number;
  room_name: string;
  source: "llm_draft" | "manual";
  is_new: boolean;
  confidence: number;
  disposition: Disposition;
  image_url?: string | null;
}

export interface EvidenceAnchor {
  x: number;
  y: number;
}

export interface EvidenceFrameItem {
  name: string;
  count: number;
  anchors: EvidenceAnchor[];
}

export interface EvidenceFrame {
  frame_index: number;
  image_url: string;
  items: EvidenceFrameItem[];
}

export interface Inventory {
  scan_id: string;
  display_id: string;
  status: string;
  items: InventoryItem[];
  evidence_frames: EvidenceFrame[];
  total_items: number;
  total_count: number;
}

/* ── WebSocket message types ── */

export interface WSProgress {
  type: "progress";
  scan_id: string;
  room_name: string;
  step: string;
  step_number: number;
  total_steps: number;
  progress: number;
  message: string;
  items_found: string[];
  elapsed_s: number;
  eta_s?: number;
}

export interface WSItemAdded {
  type: "item_added";
  scan_id: string;
  room_name: string;
  item_name: string;
}

export interface WSComplete {
  type: "complete";
  scan_id: string;
  room_name: string;
  total_items: number;
  total_rooms: number;
  processing_time_s: number;
  cost_usd: number;
}

export interface WSError {
  type: "error";
  scan_id: string;
  room_name: string;
  message: string;
}

export type WSMessage = WSProgress | WSItemAdded | WSComplete | WSError;

/* ── Room presets ── */

export const ROOM_PRESETS = [
  { name: "Living Room", emoji: "🛋️" },
  { name: "Bedroom", emoji: "🛏️" },
  { name: "Kitchen", emoji: "🍳" },
  { name: "Bathroom", emoji: "🚿" },
  { name: "Dining Room", emoji: "🍽️" },
  { name: "Office / Study", emoji: "🏢" },
  { name: "Laundry", emoji: "🧺" },
  { name: "Garage", emoji: "🏡" },
] as const;

/* ── Survey Report ── */

export interface ReportItem {
  name: string;
  count: number;
  room: string;
  going: boolean | null;
  disposition: Disposition;
  notes: string;
  size: string;
  dimensions: { length_in: number; width_in: number; height_in: number } | null;
  best_frame_ts: number | null;
  volume_cuft: number;
  weight_lbs: number;
  total_volume_cuft: number;
  total_weight_lbs: number;
  evidence_image: string | null;
  special_handling: string[];
}

export interface SurveyReport {
  generated_at: string;
  video_duration_s: number;
  survey_mode: string;
  num_frames_analyzed: number;
  items: ReportItem[];
  rooms: Record<string, number[]>;
  summary: {
    total_items: number;
    total_item_types: number;
    total_volume_cuft: number;
    total_weight_lbs: number;
    truck_recommendation: string;
  };
  packing_materials: Record<string, number>;
  special_handling: string[];
  usage: { tokens_in: number; tokens_out: number; model: string };
}
