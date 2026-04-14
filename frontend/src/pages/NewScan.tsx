/* ── New Scan Page — mode toggle + room grid ── */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../services/api";
import { ROOM_PRESETS } from "../types";

interface SelectedRoom {
  name: string;
  emoji: string;
  is_custom: boolean;
}

export default function NewScanPage() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<"room_by_room" | "all_at_once">(
    "all_at_once",
  );
  const [selected, setSelected] = useState<SelectedRoom[]>([]);
  const [showCustom, setShowCustom] = useState(false);
  const [customName, setCustomName] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggle = (name: string, emoji: string) => {
    setSelected((prev) =>
      prev.some((r) => r.name === name)
        ? prev.filter((r) => r.name !== name)
        : [...prev, { name, emoji, is_custom: false }],
    );
  };

  const addCustomRoom = () => {
    const trimmed = customName.trim();
    if (!trimmed) return;
    setSelected((prev) => [
      ...prev,
      { name: trimmed, emoji: "🏠", is_custom: true },
    ]);
    setCustomName("");
    setShowCustom(false);
  };

  const canStart = mode === "all_at_once" ? true : selected.length > 0;

  const handleStart = async () => {
    if (!canStart) return;
    setCreating(true);
    setError(null);
    try {
      const rooms = mode === "all_at_once" ? [] : selected;
      const scan = await api.scans.create(mode, rooms);
      navigate(`/scans/${scan.id}/camera`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create scan");
      setCreating(false);
    }
  };

  return (
    <div className="px-5 py-5 page-enter">
      {/* Header */}
      <h2 className="text-xl font-bold mb-1">New Scan</h2>
      <p className="text-sm text-gray-400 mb-4">
        Choose how to scan your space
      </p>

      {/* Mode toggle */}
      <div className="flex bg-gray-100 rounded-xl p-1 mb-6">
        {(["all_at_once", "room_by_room"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`flex-1 py-2.5 rounded-lg text-xs font-bold transition-all ${
              mode === m ? "bg-white shadow-sm text-gray-900" : "text-gray-400"
            }`}
          >
            {m === "all_at_once" ? "All at Once" : "Room by Room"}
          </button>
        ))}
      </div>

      <p className="text-xs text-gray-400 mb-4">
        {mode === "all_at_once"
          ? "Record or upload a single walkthrough video of your entire space"
          : "Select rooms first, then record or upload a video"}
      </p>

      {/* Room grid — only for room_by_room mode */}
      {mode === "room_by_room" && (
        <>
          <div className="grid grid-cols-2 gap-3 mb-5">
            {ROOM_PRESETS.map((r) => {
              const isSelected = selected.some((s) => s.name === r.name);
              return (
                <button
                  key={r.name}
                  onClick={() => toggle(r.name, r.emoji)}
                  className={`flex items-center gap-3 p-3.5 rounded-2xl border-[1.5px] transition-all ${
                    isSelected
                      ? "border-gray-900 bg-gray-900 text-white"
                      : "border-gray-100 bg-white text-gray-700 hover:shadow-sm"
                  }`}
                >
                  <span className="text-xl">{r.emoji}</span>
                  <span className="text-xs font-semibold">{r.name}</span>
                </button>
              );
            })}

            {/* Custom rooms already added */}
            {selected
              .filter((r) => r.is_custom)
              .map((r) => (
                <button
                  key={r.name}
                  onClick={() =>
                    setSelected((prev) => prev.filter((s) => s.name !== r.name))
                  }
                  className="flex items-center gap-3 p-3.5 rounded-2xl border-[1.5px] border-gray-900 bg-gray-900 text-white transition-all"
                >
                  <span className="text-xl">{r.emoji}</span>
                  <span className="text-xs font-semibold">{r.name}</span>
                </button>
              ))}

            {/* Add custom room button */}
            <button
              onClick={() => setShowCustom(true)}
              className="flex items-center gap-3 p-3.5 rounded-2xl border-[1.5px] border-dashed border-gray-200 text-gray-400 hover:border-gray-400 transition-colors"
            >
              <span className="text-xl">+</span>
              <span className="text-xs font-semibold">Custom Room</span>
            </button>
          </div>

          {/* Custom room modal */}
          {showCustom && (
            <div className="fixed inset-0 bg-black/40 flex items-end justify-center z-50 backdrop-enter">
              <div className="bg-white w-full max-w-[430px] rounded-t-3xl p-6 slide-up">
                <h3 className="text-base font-bold mb-4">Add Custom Room</h3>
                <input
                  type="text"
                  value={customName}
                  onChange={(e) => setCustomName(e.target.value)}
                  placeholder="e.g. Pantry, Attic, Nursery"
                  className="w-full px-3.5 py-3 border-[1.5px] border-gray-200 rounded-xl text-sm focus:outline-none focus:border-gray-900 transition-colors mb-4"
                  autoFocus
                  onKeyDown={(e) => e.key === "Enter" && addCustomRoom()}
                />
                <div className="flex gap-3">
                  <button
                    onClick={() => {
                      setShowCustom(false);
                      setCustomName("");
                    }}
                    className="flex-1 py-3 border border-gray-200 rounded-xl text-sm font-semibold"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={addCustomRoom}
                    disabled={!customName.trim()}
                    className="flex-1 py-3 bg-gray-900 text-white rounded-xl text-sm font-semibold disabled:opacity-50"
                  >
                    Add Room
                  </button>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* Error message */}
      {error && (
        <div className="mb-3 p-3 bg-red-50 border border-red-200 rounded-xl text-red-700 text-xs">
          {error}
        </div>
      )}

      {/* Start button */}
      <button
        onClick={handleStart}
        disabled={!canStart || creating}
        className="w-full py-4 bg-gradient-to-r from-gray-900 to-gray-800 text-white rounded-2xl font-bold text-sm hover:from-black hover:to-gray-900 transition-all active:scale-[0.98] disabled:opacity-40 disabled:active:scale-100 mt-2 shadow-lg shadow-gray-900/20"
      >
        {creating
          ? "Creating…"
          : mode === "all_at_once"
            ? "Continue"
            : `Continue · ${selected.length} Room${selected.length !== 1 ? "s" : ""}`}
      </button>
    </div>
  );
}
