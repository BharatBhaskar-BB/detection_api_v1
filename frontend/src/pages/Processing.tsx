/* ── Processing Page — real-time pipeline progress ── */

import { useState, useCallback, useRef, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useWebSocket } from "../hooks/useWebSocket";
import { useNotificationStore } from "../stores/notificationStore";
import { api } from "../services/api";
import type {
  WSMessage,
  WSProgress,
  WSItemAdded,
  WSComplete,
  Scan,
  Room,
} from "../types";

const PIPELINE_STEPS = [
  { key: "video_processing", label: "Processing Video", icon: "🎞️" },
  { key: "frame_selection", label: "Selecting Key Frames", icon: "🔍" },
  { key: "ai_analysis", label: "AI Inventory Analysis", icon: "🧠" },
  { key: "report", label: "Generating Report", icon: "📋" },
];

interface RoomProgress {
  step: number;
  progress: number;
  message: string;
  items: string[];
  status: "pending" | "uploading" | "processing" | "completed" | "failed";
  totalItems: number;
  error?: string;
}

export default function ProcessingPage() {
  const { id: scanId } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const addNotification = useNotificationStore((s) => s.add);
  const notifiedSteps = useRef(new Set<string>());

  const [scan, setScan] = useState<Scan | null>(null);
  const [rooms, setRooms] = useState<Room[]>([]);
  const isRoomByRoom = scan?.scan_mode === "room_by_room" && rooms.length > 0;

  // All-at-once state
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [message, setMessage] = useState("Preparing…");
  const [feed, setFeed] = useState<string[]>([]);
  const [complete, setComplete] = useState<WSComplete | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Room-by-room state
  const [roomProgress, setRoomProgress] = useState<
    Record<string, RoomProgress>
  >({});

  // Load scan data
  useEffect(() => {
    if (!scanId) return;
    api.scans.get(scanId).then((s) => {
      setScan(s);
      // If scan already completed or failed, redirect immediately
      if (s.status === "completed") {
        navigate(`/scans/${scanId}/inventory`, { replace: true });
        return;
      }
      if (s.status === "failed") {
        setError(s.progress_message || "Pipeline failed");
        return;
      }
      if (s.scan_mode === "room_by_room" && s.rooms.length > 0) {
        const sorted = [...s.rooms].sort((a, b) => a.order - b.order);
        setRooms(sorted);
        // Init room progress from existing room status
        const init: Record<string, RoomProgress> = {};
        for (const r of sorted) {
          init[r.name] = {
            step: 0,
            progress: r.status === "completed" ? 100 : 0,
            message:
              r.status === "completed"
                ? "Complete"
                : r.has_video
                  ? "Processing…"
                  : "Waiting for video",
            items: [],
            status:
              r.status === "completed"
                ? "completed"
                : r.status === "processing"
                  ? "processing"
                  : r.has_video
                    ? "processing"
                    : "pending",
            totalItems: 0,
          };
        }
        setRoomProgress(init);
      }
    });
  }, [scanId, navigate]);

  const handleMessage = useCallback(
    (msg: WSMessage) => {
      switch (msg.type) {
        case "progress": {
          const p = msg as WSProgress;
          if (p.room_name) {
            // Per-room progress
            setRoomProgress((prev) => ({
              ...prev,
              [p.room_name]: {
                ...prev[p.room_name],
                step: p.step_number - 1,
                progress: p.progress <= 1 ? p.progress * 100 : p.progress,
                message: p.message,
                status: "processing",
                items: p.items_found?.length
                  ? [
                      ...(prev[p.room_name]?.items || []),
                      ...p.items_found.filter(
                        (i) => !(prev[p.room_name]?.items || []).includes(i),
                      ),
                    ]
                  : prev[p.room_name]?.items || [],
              },
            }));
            // Also add to the global feed
            if (p.items_found?.length) {
              setFeed((prev) => {
                const next = [...prev];
                for (const item of p.items_found) {
                  if (!next.includes(item)) next.push(item);
                }
                return next;
              });
            }
            const nKey = `${p.room_name}_${p.step_number}`;
            if (!notifiedSteps.current.has(nKey)) {
              notifiedSteps.current.add(nKey);
              addNotification({
                type: "scan_progress",
                title: `${p.room_name}: ${PIPELINE_STEPS[p.step_number - 1]?.label}`,
                message: p.message,
                scanId: p.scan_id,
              });
            }
          } else {
            // All-at-once progress
            setProgress(p.progress <= 1 ? p.progress * 100 : p.progress);
            setCurrentStep(p.step_number - 1);
            setMessage(p.message);
            if (p.items_found?.length) {
              setFeed((prev) => {
                const next = [...prev];
                for (const item of p.items_found) {
                  if (!next.includes(item)) next.push(item);
                }
                return next;
              });
            }
            const stepIdx = p.step_number - 1;
            const nKey = `all_${stepIdx}`;
            if (
              stepIdx >= 0 &&
              stepIdx < PIPELINE_STEPS.length &&
              !notifiedSteps.current.has(nKey)
            ) {
              notifiedSteps.current.add(nKey);
              addNotification({
                type: "scan_progress",
                title: PIPELINE_STEPS[stepIdx].label,
                message: p.message,
                scanId: p.scan_id,
              });
            }
          }
          break;
        }
        case "item_added": {
          const ia = msg as WSItemAdded;
          setFeed((prev) =>
            prev.includes(ia.item_name) ? prev : [...prev, ia.item_name],
          );
          if (ia.room_name) {
            setRoomProgress((prev) => ({
              ...prev,
              [ia.room_name]: {
                ...prev[ia.room_name],
                items: prev[ia.room_name]?.items?.includes(ia.item_name)
                  ? prev[ia.room_name].items
                  : [...(prev[ia.room_name]?.items || []), ia.item_name],
              },
            }));
          }
          break;
        }
        case "complete": {
          const c = msg as WSComplete;
          if (c.room_name) {
            // Single room completed
            setRoomProgress((prev) => ({
              ...prev,
              [c.room_name]: {
                ...prev[c.room_name],
                progress: 100,
                status: "completed",
                message: `${c.total_items} items found`,
                totalItems: c.total_items,
              },
            }));
            addNotification({
              type: "scan_progress",
              title: `${c.room_name} Complete`,
              message: `Found ${c.total_items} items`,
              scanId: c.scan_id,
            });
          } else {
            // Overall scan completed
            setComplete(c);
            setProgress(100);
            addNotification({
              type: "scan_complete",
              title: "Scan Complete!",
              message: `Found ${c.total_items} items in ${c.processing_time_s.toFixed(1)}s`,
              scanId: c.scan_id,
            });
          }
          break;
        }
        case "error":
          if (msg.room_name) {
            setRoomProgress((prev) => ({
              ...prev,
              [msg.room_name]: {
                ...prev[msg.room_name],
                status: "failed",
                error: msg.message,
              },
            }));
          } else {
            setError(msg.message);
          }
          addNotification({
            type: "scan_failed",
            title: msg.room_name ? `${msg.room_name} Failed` : "Scan Failed",
            message: msg.message,
            scanId: msg.scan_id,
          });
          break;
      }
    },
    [addNotification],
  );

  useWebSocket({
    scanId: scanId || "",
    onMessage: handleMessage,
    enabled: !complete && !error,
  });

  if (!scanId) return null;

  // Calculate overall room-by-room progress
  const roomEntries = Object.values(roomProgress);
  const roomsDone = roomEntries.filter((r) => r.status === "completed").length;
  const totalRoomsTracked = rooms.length || 1;
  const overallProgress = isRoomByRoom
    ? Math.round(
        roomEntries.reduce((sum, r) => sum + r.progress, 0) / totalRoomsTracked,
      )
    : Math.round(progress);

  return (
    <div className="app-shell flex flex-col min-h-screen bg-white">
      {/* Header */}
      <header className="flex items-center gap-3 px-5 py-4 border-b border-gray-100 sticky top-0 bg-white z-30 safe-top">
        <button
          onClick={() => navigate("/")}
          className="text-gray-400 hover:text-gray-600"
        >
          ←
        </button>
        <h1 className="font-bold text-base">Processing</h1>
        {isRoomByRoom && (
          <span className="ml-auto text-xs font-semibold text-gray-400">
            {roomsDone}/{totalRoomsTracked} rooms
          </span>
        )}
      </header>

      <div className="flex-1 px-5 py-5 overflow-y-auto page-enter">
        {/* Progress ring */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-28 h-28 rounded-full border-[6px] border-gray-100 relative mb-4">
            <svg className="absolute inset-0 -rotate-90" viewBox="0 0 100 100">
              <circle
                cx="50"
                cy="50"
                r="44"
                fill="none"
                stroke="url(#progress-gradient)"
                strokeWidth="8"
                strokeDasharray={`${overallProgress * 2.76} 276`}
                strokeLinecap="round"
                className="transition-all duration-700"
              />
              <defs>
                <linearGradient
                  id="progress-gradient"
                  x1="0%"
                  y1="0%"
                  x2="100%"
                  y2="0%"
                >
                  <stop offset="0%" stopColor="#dc2626" />
                  <stop offset="100%" stopColor="#f87171" />
                </linearGradient>
              </defs>
            </svg>
            <span className="text-2xl font-extrabold">{overallProgress}%</span>
          </div>
          <p className="text-sm font-semibold text-gray-700">
            {isRoomByRoom
              ? `${roomsDone} of ${totalRoomsTracked} rooms complete`
              : message}
          </p>
          {complete && (
            <p className="text-xs text-gray-400 mt-1.5">
              {complete.processing_time_s > 0
                ? `${complete.processing_time_s.toFixed(1)}s · `
                : ""}
              ${complete.cost_usd.toFixed(3)}
            </p>
          )}
        </div>

        {/* ═══ Room-by-room progress cards ═══ */}
        {isRoomByRoom ? (
          <div className="space-y-3 mb-8">
            {rooms.map((room) => {
              const rp = roomProgress[room.name];
              if (!rp) return null;
              const isDone = rp.status === "completed";
              const isFailed = rp.status === "failed";
              const isProcessing = rp.status === "processing";
              const isPending = rp.status === "pending";
              return (
                <div
                  key={room.id}
                  className={`rounded-2xl border p-4 transition-all ${
                    isDone
                      ? "border-green-200 bg-green-50"
                      : isFailed
                        ? "border-red-200 bg-red-50"
                        : isProcessing
                          ? "border-gray-200 bg-white shadow-sm"
                          : "border-gray-100 bg-gray-50"
                  }`}
                >
                  <div className="flex items-center gap-3 mb-2">
                    <span className="text-xl">{room.emoji}</span>
                    <span className="text-sm font-bold flex-1">
                      {room.name}
                    </span>
                    {isDone && (
                      <span className="text-xs font-bold text-green-600 bg-green-100 px-2 py-0.5 rounded-full">
                        ✓ {rp.totalItems} items
                      </span>
                    )}
                    {isFailed && (
                      <span className="text-xs font-bold text-red-600 bg-red-100 px-2 py-0.5 rounded-full">
                        ✕ Failed
                      </span>
                    )}
                    {isPending && (
                      <span className="text-xs font-medium text-gray-400">
                        Waiting
                      </span>
                    )}
                    {isProcessing && (
                      <span className="text-xs font-medium text-gray-500 flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
                        {PIPELINE_STEPS[rp.step]?.label || "Processing"}
                      </span>
                    )}
                  </div>
                  {/* Room progress bar */}
                  {(isProcessing || isDone) && (
                    <div className="w-full h-1.5 bg-gray-100 rounded-full overflow-hidden mb-2">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          isDone
                            ? "bg-green-400"
                            : "bg-gradient-to-r from-red-500 to-red-400"
                        }`}
                        style={{ width: `${rp.progress}%` }}
                      />
                    </div>
                  )}
                  {/* Room message */}
                  {isProcessing && (
                    <p className="text-[11px] text-gray-500">{rp.message}</p>
                  )}
                  {/* Room items */}
                  {rp.items.length > 0 && (isDone || isProcessing) && (
                    <div className="flex flex-wrap gap-1 mt-2">
                      {rp.items.slice(0, 8).map((item) => (
                        <span
                          key={item}
                          className="px-2 py-0.5 bg-white border border-gray-100 rounded-full text-[10px] font-medium text-gray-600"
                        >
                          {item}
                        </span>
                      ))}
                      {rp.items.length > 8 && (
                        <span className="px-2 py-0.5 text-[10px] font-medium text-gray-400">
                          +{rp.items.length - 8} more
                        </span>
                      )}
                    </div>
                  )}
                  {isFailed && rp.error && (
                    <p className="text-[11px] text-red-600 mt-1">{rp.error}</p>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <>
            {/* ═══ All-at-once pipeline steps ═══ */}
            <div className="mb-8 bg-white rounded-2xl border border-gray-100 p-4 shadow-sm">
              {PIPELINE_STEPS.map((step, i) => {
                const isDone = i < currentStep || !!complete;
                const isCurrent = i === currentStep && !complete;
                return (
                  <div
                    key={step.key}
                    className={`flex items-center gap-3 py-3 ${
                      i < PIPELINE_STEPS.length - 1
                        ? "border-b border-gray-50"
                        : ""
                    } transition-all`}
                  >
                    <span
                      className={`w-8 h-8 flex items-center justify-center rounded-xl text-xs font-bold transition-all ${
                        isDone
                          ? "bg-green-100 text-green-600"
                          : isCurrent
                            ? "bg-red-100 text-brand-red pulse-glow"
                            : "bg-gray-50 text-gray-300"
                      }`}
                    >
                      {isDone ? "✓" : step.icon}
                    </span>
                    <span
                      className={`text-sm font-medium transition-colors ${
                        isDone
                          ? "text-green-600"
                          : isCurrent
                            ? "text-gray-900 font-semibold"
                            : "text-gray-300"
                      }`}
                    >
                      {step.label}
                    </span>
                    {isCurrent && (
                      <span className="ml-auto w-1.5 h-1.5 rounded-full bg-brand-red animate-pulse" />
                    )}
                  </div>
                );
              })}
            </div>
          </>
        )}

        {/* Live feed */}
        {feed.length > 0 && (
          <div className="mb-6">
            <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">
              Items Found ({feed.length})
            </h3>
            <div className="flex flex-wrap gap-2 stagger">
              {feed.map((item) => (
                <span
                  key={item}
                  className="px-3 py-1.5 bg-white border border-gray-100 rounded-full text-xs font-semibold text-gray-700 shadow-sm fade-in-up"
                >
                  {item}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-2xl mb-6 scale-enter">
            <p className="text-sm text-red-700 font-medium">Error: {error}</p>
            <button
              onClick={() => navigate("/")}
              className="mt-3 text-sm text-red-600 underline font-semibold"
            >
              Back to Dashboard
            </button>
          </div>
        )}

        {/* Complete CTA */}
        {complete && (
          <div className="space-y-3 scale-enter">
            <button
              onClick={() => navigate(`/scans/${scanId}/inventory`)}
              className="w-full py-4 bg-gradient-to-r from-gray-900 to-gray-800 text-white rounded-2xl text-sm font-bold hover:from-black hover:to-gray-900 transition-all active:scale-[0.98] shadow-lg shadow-gray-900/20"
            >
              Review Inventory · {complete.total_items} Items
            </button>
            <button
              onClick={() => navigate("/")}
              className="w-full py-3 text-sm font-semibold text-gray-400 hover:text-gray-600 transition-colors"
            >
              Back to Dashboard
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
