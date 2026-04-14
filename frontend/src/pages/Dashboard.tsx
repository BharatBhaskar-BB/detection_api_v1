/* ── Dashboard Page ── */

import { useEffect, useState, useCallback } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuthStore } from "../stores/authStore";
import { api } from "../services/api";
import type { ScanListItem } from "../types";

const STATUS_BADGES: Record<string, { label: string; class: string }> = {
  recording: { label: "Recording", class: "bg-yellow-100 text-yellow-700" },
  submitted: { label: "Submitted", class: "bg-blue-100 text-blue-700" },
  processing: { label: "Processing", class: "bg-purple-100 text-purple-700" },
  completed: { label: "Completed", class: "bg-green-100 text-green-700" },
  failed: { label: "Failed", class: "bg-red-100 text-red-700" },
};

export default function DashboardPage() {
  const user = useAuthStore((s) => s.user);
  const navigate = useNavigate();
  const [scans, setScans] = useState<ScanListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [selectMode, setSelectMode] = useState(false);

  useEffect(() => {
    api.scans
      .list()
      .then(setScans)
      .finally(() => setLoading(false));
  }, []);

  const handleDelete = useCallback(
    async (scanId: string, displayId: string) => {
      if (!confirm(`Delete "${displayId}"? This cannot be undone.`)) return;
      try {
        await api.scans.delete(scanId);
        setScans((prev) => prev.filter((s) => s.id !== scanId));
      } catch {
        alert("Failed to delete scan");
      }
    },
    [],
  );

  const toggleSelect = useCallback((scanId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(scanId)) next.delete(scanId);
      else next.add(scanId);
      return next;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    setSelected((prev) =>
      prev.size === scans.length ? new Set() : new Set(scans.map((s) => s.id)),
    );
  }, [scans]);

  const handleBulkDelete = useCallback(async () => {
    if (selected.size === 0) return;
    if (
      !confirm(
        `Delete ${selected.size} scan${selected.size > 1 ? "s" : ""}? This cannot be undone.`,
      )
    )
      return;
    try {
      await api.scans.bulkDelete(Array.from(selected));
      setScans((prev) => prev.filter((s) => !selected.has(s.id)));
      setSelected(new Set());
      setSelectMode(false);
    } catch {
      alert("Failed to delete scans");
    }
  }, [selected]);

  const exitSelectMode = useCallback(() => {
    setSelectMode(false);
    setSelected(new Set());
  }, []);

  const inProgress = scans.filter((s) =>
    ["recording", "submitted", "processing"].includes(s.status),
  );
  const completed = scans.filter((s) => s.status === "completed");

  const firstName = user?.name?.split(" ")[0] || "there";

  return (
    <div className="px-5 py-5 page-enter">
      {/* Greeting */}
      <h2 className="text-xl font-bold mb-0.5">Hey, {firstName} 👋</h2>
      <p className="text-sm text-gray-400 mb-6">
        {completed.length
          ? `${completed.length} scan${completed.length > 1 ? "s" : ""} completed`
          : "Ready to scan your space?"}
      </p>

      {/* Quick Actions */}
      <button
        onClick={() => navigate("/scans/new")}
        className="w-full flex items-center gap-4 p-4 bg-gradient-to-r from-gray-900 to-gray-800 text-white rounded-2xl mb-6 hover:from-black hover:to-gray-900 transition-all active:scale-[0.98] shadow-lg shadow-gray-900/20"
      >
        <span className="w-11 h-11 flex items-center justify-center bg-brand-red rounded-xl text-xl shadow-sm">
          📷
        </span>
        <div className="text-left">
          <p className="font-bold text-sm">New Scan</p>
          <p className="text-xs text-gray-400">Record rooms and scan items</p>
        </div>
        <span className="ml-auto text-gray-500 text-lg">→</span>
      </button>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3 mb-7 stagger">
        <div className="bg-white rounded-xl border border-gray-100 p-3 text-center fade-in-up shadow-sm">
          <p className="text-lg font-extrabold">{scans.length}</p>
          <p className="text-[10px] text-gray-400 font-medium">Total Scans</p>
        </div>
        <div className="bg-white rounded-xl border border-gray-100 p-3 text-center fade-in-up shadow-sm">
          <p className="text-lg font-extrabold text-purple-600">
            {inProgress.length}
          </p>
          <p className="text-[10px] text-gray-400 font-medium">In Progress</p>
        </div>
        <div className="bg-white rounded-xl border border-gray-100 p-3 text-center fade-in-up shadow-sm">
          <p className="text-lg font-extrabold text-brand-red">
            {completed.reduce((sum, s) => sum + s.total_items, 0)}
          </p>
          <p className="text-[10px] text-gray-400 font-medium">Items Found</p>
        </div>
      </div>

      {/* In-progress scans */}
      {inProgress.length > 0 && (
        <section className="mb-7">
          <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">
            In Progress
          </h3>
          <div className="space-y-3">
            {inProgress.map((scan) => (
              <ScanCard
                key={scan.id}
                scan={scan}
                onDelete={handleDelete}
                selectMode={selectMode}
                isSelected={selected.has(scan.id)}
                onToggleSelect={toggleSelect}
              />
            ))}
          </div>
        </section>
      )}

      {/* All scans */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider">
            {completed.length ? "Recent Scans" : "No Scans Yet"}
          </h3>
          {scans.length > 0 && !selectMode && (
            <button
              onClick={() => setSelectMode(true)}
              className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
            >
              Select
            </button>
          )}
          {selectMode && (
            <button
              onClick={exitSelectMode}
              className="text-xs text-brand-red hover:text-red-700 font-medium transition-colors"
            >
              Cancel
            </button>
          )}
        </div>

        {/* Select-all bar */}
        {selectMode && scans.length > 0 && (
          <div className="flex items-center justify-between bg-gray-50 rounded-xl px-4 py-2.5 mb-3">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={selected.size === scans.length}
                onChange={toggleSelectAll}
                className="w-4 h-4 rounded accent-brand-red"
              />
              <span className="text-xs font-medium text-gray-600">
                {selected.size === scans.length ? "Deselect all" : "Select all"}
              </span>
            </label>
            <button
              onClick={handleBulkDelete}
              disabled={selected.size === 0}
              className="text-xs font-bold text-red-500 hover:text-red-700 disabled:text-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              Delete {selected.size > 0 ? `(${selected.size})` : ""}
            </button>
          </div>
        )}
        {loading ? (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="flex items-center gap-3.5 p-4 bg-white border border-gray-100 rounded-2xl"
              >
                <div className="flex-1">
                  <div className="skeleton h-4 w-32 mb-2" />
                  <div className="skeleton h-3 w-20" />
                </div>
                <div className="skeleton h-5 w-16 rounded-full" />
              </div>
            ))}
          </div>
        ) : scans.length === 0 ? (
          <div className="text-center py-10">
            <p className="text-3xl mb-2">📦</p>
            <p className="text-sm text-gray-400">
              Start your first scan to see results here
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {scans.map((scan) => (
              <ScanCard
                key={scan.id}
                scan={scan}
                onDelete={handleDelete}
                selectMode={selectMode}
                isSelected={selected.has(scan.id)}
                onToggleSelect={toggleSelect}
              />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function ScanCard({
  scan,
  onDelete,
  selectMode,
  isSelected,
  onToggleSelect,
}: {
  scan: ScanListItem;
  onDelete: (id: string, displayId: string) => void;
  selectMode: boolean;
  isSelected: boolean;
  onToggleSelect: (id: string) => void;
}) {
  const badge = STATUS_BADGES[scan.status] || STATUS_BADGES.completed;
  const dest =
    scan.status === "processing"
      ? `/scans/${scan.id}/processing`
      : scan.status === "completed"
        ? `/scans/${scan.id}`
        : scan.status === "recording"
          ? `/scans/${scan.id}/camera`
          : `/scans/${scan.id}`;

  return (
    <div className="flex items-center gap-2 fade-in-up">
      {selectMode && (
        <input
          type="checkbox"
          checked={isSelected}
          onChange={() => onToggleSelect(scan.id)}
          className="shrink-0 w-4 h-4 rounded accent-brand-red cursor-pointer"
        />
      )}
      <Link
        to={dest}
        className="flex-1 flex items-center gap-3.5 p-4 bg-white border border-gray-100 rounded-2xl hover:shadow-lg hover:border-gray-200 transition-all active:scale-[0.98] min-w-0"
      >
        <div className="flex-1 min-w-0">
          <p className="font-bold text-sm truncate">{scan.display_id}</p>
          <p className="text-xs text-gray-400 mt-0.5">
            {scan.total_rooms} room{scan.total_rooms !== 1 ? "s" : ""} ·{" "}
            {scan.total_items} item{scan.total_items !== 1 ? "s" : ""}
            {scan.status === "completed" && scan.move_summary && (
              <span
                className={`ml-1.5 inline-flex items-center gap-0.5 text-[10px] font-semibold ${
                  scan.move_summary.has_audio
                    ? "text-emerald-600"
                    : "text-gray-400"
                }`}
              >
                {scan.move_summary.has_audio ? "🎙️ Audio" : "👁️ Visual only"}
              </span>
            )}
          </p>
          {scan.status === "processing" && (
            <div className="mt-2 h-1.5 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-brand-red to-red-400 rounded-full transition-all duration-500"
                style={{ width: `${Math.max(scan.progress, 5)}%` }}
              />
            </div>
          )}
        </div>
        <span
          className={`text-[10px] font-bold px-2.5 py-1 rounded-full whitespace-nowrap ${badge.class}`}
        >
          {badge.label}
        </span>
      </Link>
      {!selectMode && (
        <button
          onClick={(e) => {
            e.preventDefault();
            onDelete(scan.id, scan.display_id);
          }}
          className="shrink-0 w-9 h-9 flex items-center justify-center rounded-xl text-gray-300 hover:text-red-500 hover:bg-red-50 transition-colors"
          title="Delete scan"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          >
            <path d="M2 4h12M5.33 4V2.67a1.33 1.33 0 011.34-1.34h2.66a1.33 1.33 0 011.34 1.34V4M6.67 7.33v4M9.33 7.33v4" />
            <path d="M3.33 4h9.34l-.67 9.33a1.33 1.33 0 01-1.33 1.34H5.33A1.33 1.33 0 014 13.33L3.33 4z" />
          </svg>
        </button>
      )}
    </div>
  );
}
