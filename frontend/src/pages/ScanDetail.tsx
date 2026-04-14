/* ── Scan Detail Page — completed scan summary ── */

import { useEffect, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { api } from "../services/api";
import type { Scan, InventoryItem, Disposition, MoveSummary } from "../types";

const DISPOSITION_OPTIONS: {
  value: Disposition;
  label: string;
  color: string;
  bg: string;
}[] = [
  { value: null, label: "—", color: "text-gray-400", bg: "bg-gray-50" },
  {
    value: "going",
    label: "Going",
    color: "text-green-600",
    bg: "bg-green-50",
  },
  {
    value: "staying",
    label: "Staying",
    color: "text-gray-500",
    bg: "bg-gray-100",
  },
  { value: "scrap", label: "Scrap", color: "text-red-500", bg: "bg-red-50" },
  {
    value: "haul",
    label: "Haul",
    color: "text-orange-500",
    bg: "bg-orange-50",
  },
  { value: "sell", label: "Sell", color: "text-blue-500", bg: "bg-blue-50" },
];

export default function ScanDetailPage() {
  const { id: scanId } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [scan, setScan] = useState<Scan | null>(null);
  const [items, setItems] = useState<InventoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [previewItem, setPreviewItem] = useState<InventoryItem | null>(null);

  useEffect(() => {
    if (!scanId) return;
    Promise.all([
      api.scans.get(scanId),
      api.inventory.get(scanId).catch(() => ({ items: [] as InventoryItem[] })),
    ])
      .then(([s, inv]) => {
        setScan(s);
        setItems(inv.items);
      })
      .finally(() => setLoading(false));
  }, [scanId]);

  if (loading || !scan) {
    return (
      <div className="px-5 py-5">
        <div className="skeleton h-6 w-40 mb-2" />
        <div className="skeleton h-4 w-28 mb-6" />
        <div className="grid grid-cols-3 gap-3 mb-6">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="bg-white rounded-xl border border-gray-100 p-3"
            >
              <div className="skeleton h-5 w-8 mx-auto mb-1" />
              <div className="skeleton h-3 w-12 mx-auto" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  // If still processing, redirect
  if (scan.status === "processing") {
    navigate(`/scans/${scanId}/processing`, { replace: true });
    return null;
  }

  // Group items by room
  const byRoom = items.reduce<Record<string, InventoryItem[]>>((acc, item) => {
    (acc[item.room_name] ||= []).push(item);
    return acc;
  }, {});

  const totalCount = items.reduce((s, i) => s + i.count, 0);

  return (
    <div className="px-5 py-5 page-enter">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-xl font-bold">{scan.display_id}</h2>
        <p className="text-xs text-gray-400 mt-0.5">
          {scan.scan_mode === "room_by_room" ? "Room by Room" : "All at Once"} ·{" "}
          {new Date(scan.created_at).toLocaleDateString()}
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-3 mb-6 stagger">
        <div className="bg-white rounded-xl border border-gray-100 p-3 text-center shadow-sm fade-in-up">
          <p className="text-lg font-extrabold">{items.length}</p>
          <p className="text-[10px] text-gray-400 font-medium">Unique Items</p>
        </div>
        <div className="bg-white rounded-xl border border-gray-100 p-3 text-center shadow-sm fade-in-up">
          <p className="text-lg font-extrabold text-brand-red">{totalCount}</p>
          <p className="text-[10px] text-gray-400 font-medium">Total Count</p>
        </div>
        <div className="bg-white rounded-xl border border-gray-100 p-3 text-center shadow-sm fade-in-up">
          <p className="text-lg font-extrabold">{Object.keys(byRoom).length}</p>
          <p className="text-[10px] text-gray-400 font-medium">Rooms</p>
        </div>
      </div>

      {/* Processing info */}
      {scan.processing_time_s != null && (
        <div className="flex gap-3 mb-6">
          <div className="flex-1 bg-gray-50 rounded-xl p-3 text-center">
            <p className="text-sm font-bold">
              {scan.processing_time_s.toFixed(1)}s
            </p>
            <p className="text-[10px] text-gray-400 font-medium">
              Processing Time
            </p>
          </div>
          {scan.cost_usd != null && (
            <div className="flex-1 bg-gray-50 rounded-xl p-3 text-center">
              <p className="text-sm font-bold">${scan.cost_usd.toFixed(3)}</p>
              <p className="text-[10px] text-gray-400 font-medium">Cost</p>
            </div>
          )}
        </div>
      )}

      {/* Move Summary */}
      {scan.move_summary && <MoveSummaryCard summary={scan.move_summary} />}

      {/* Room breakdown */}
      {Object.entries(byRoom).map(([room, roomItems]) => (
        <section key={room} className="mb-5">
          <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2.5">
            {room} ({roomItems.length})
          </h3>
          <div className="space-y-2">
            {roomItems.map((item) => (
              <div
                key={item.id}
                className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold border ${
                  item.is_new
                    ? "bg-red-50 border-red-200"
                    : "bg-white border-gray-100"
                }`}
              >
                {item.image_url && (
                  <button
                    onClick={() => setPreviewItem(item)}
                    className="w-8 h-8 rounded-lg overflow-hidden flex-shrink-0"
                  >
                    <img
                      src={item.image_url}
                      alt=""
                      className="w-full h-full object-cover"
                    />
                  </button>
                )}
                <div className="flex-1 min-w-0">
                  <span className="capitalize text-gray-800">{item.name}</span>
                  {item.count > 1 && (
                    <span className="text-[10px] text-gray-400 ml-1">
                      ×{item.count}
                    </span>
                  )}
                </div>
                <select
                  value={item.disposition || ""}
                  onChange={async (e) => {
                    const val = (e.target.value || null) as Disposition;
                    // Optimistic update
                    setItems((prev) =>
                      prev.map((i) =>
                        i.id === item.id ? { ...i, disposition: val } : i,
                      ),
                    );
                    try {
                      await api.inventory.updateItem(scanId!, item.id, {
                        disposition: val,
                      });
                    } catch {
                      // Revert on error
                      setItems((prev) =>
                        prev.map((i) =>
                          i.id === item.id
                            ? { ...i, disposition: item.disposition }
                            : i,
                        ),
                      );
                    }
                  }}
                  className={`text-[10px] font-bold rounded-lg px-1.5 py-1 border-0 outline-none cursor-pointer ${
                    DISPOSITION_OPTIONS.find(
                      (o) => o.value === item.disposition,
                    )?.bg || "bg-gray-50"
                  } ${
                    DISPOSITION_OPTIONS.find(
                      (o) => o.value === item.disposition,
                    )?.color || "text-gray-400"
                  }`}
                >
                  {DISPOSITION_OPTIONS.map((opt) => (
                    <option key={opt.label} value={opt.value || ""}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>
        </section>
      ))}

      {/* Actions */}
      <div className="space-y-3 mt-6">
        <Link
          to={`/scans/${scanId}/inventory`}
          className="block w-full py-3.5 bg-gradient-to-r from-gray-900 to-gray-800 text-white rounded-2xl text-sm font-bold text-center hover:from-black hover:to-gray-900 transition-all active:scale-[0.98] shadow-lg shadow-gray-900/20"
        >
          View Survey Report
        </Link>
        <Link
          to={`/scans/${scanId}/inventory`}
          className="block w-full py-3 border border-gray-200 rounded-2xl text-sm font-semibold text-center text-gray-700 hover:bg-gray-50 transition-colors"
        >
          Edit Inventory
        </Link>
        <button
          onClick={() => navigate("/")}
          className="w-full py-3 text-sm font-semibold text-gray-400"
        >
          Back to Dashboard
        </button>
        <button
          onClick={async () => {
            if (!confirm(`Delete "${scan.display_id}"? This cannot be undone.`))
              return;
            try {
              await api.scans.delete(scan.id);
              navigate("/", { replace: true });
            } catch {
              alert("Failed to delete scan");
            }
          }}
          className="w-full py-3 text-sm font-semibold text-red-400 hover:text-red-600 transition-colors"
        >
          Delete Scan
        </button>
      </div>

      {/* Image preview modal */}
      {previewItem?.image_url && (
        <div
          className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-6 backdrop-enter"
          onClick={() => setPreviewItem(null)}
        >
          <div
            className="bg-white rounded-2xl overflow-hidden max-w-sm w-full scale-enter shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={previewItem.image_url}
              alt={previewItem.name}
              className="w-full object-contain max-h-72"
            />
            <div className="p-4 text-center">
              <p className="font-bold text-sm capitalize">{previewItem.name}</p>
              <p className="text-xs text-gray-400 mt-0.5">
                {previewItem.room_name} · ×{previewItem.count}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Move Summary Card ── */

function MoveSummaryCard({ summary }: { summary: MoveSummary }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="mb-6 bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
      {/* Header — always visible */}
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-3 p-4 text-left"
      >
        <span className="text-xl">
          {summary.has_audio ? "🎙️" : "👁️"}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-bold text-gray-800">Move Summary</p>
          <p className="text-[10px] text-gray-400 mt-0.5 truncate">
            {summary.audio_summary}
          </p>
        </div>
        <span
          className={`text-gray-400 text-xs transition-transform ${
            expanded ? "rotate-180" : ""
          }`}
        >
          ▼
        </span>
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="px-4 pb-4 space-y-4 border-t border-gray-50">
          {/* Overall summary */}
          {summary.overall && (
            <p className="text-xs text-gray-600 leading-relaxed pt-3">
              {summary.overall}
            </p>
          )}

          {/* Per-room summaries */}
          {summary.rooms.length > 0 && (
            <div>
              <p className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">
                Room Details
              </p>
              <div className="space-y-2">
                {summary.rooms.map((room) => (
                  <div
                    key={room.name}
                    className="bg-gray-50 rounded-xl p-3"
                  >
                    <p className="text-[11px] font-bold text-gray-700 capitalize">
                      {room.name}
                    </p>
                    <p className="text-[10px] text-gray-500 mt-0.5 leading-relaxed">
                      {room.summary}
                    </p>
                    {room.instructions.length > 0 && (
                      <div className="mt-1.5 space-y-0.5">
                        {room.instructions.map((inst, i) => (
                          <p
                            key={i}
                            className="text-[10px] text-amber-600 flex items-start gap-1"
                          >
                            <span className="mt-px">⚠️</span>
                            <span>{inst}</span>
                          </p>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Special instructions */}
          {summary.special_instructions.length > 0 && (
            <div>
              <p className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">
                Special Instructions
              </p>
              <div className="bg-amber-50 rounded-xl p-3 space-y-1.5">
                {summary.special_instructions.map((inst, i) => (
                  <p
                    key={i}
                    className="text-[10px] text-amber-800 flex items-start gap-1.5"
                  >
                    <span className="mt-px">📋</span>
                    <span>{inst}</span>
                  </p>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
