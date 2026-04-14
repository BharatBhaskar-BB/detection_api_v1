/* ── Survey Report Page — report viewer with Internal/Customer toggle + inline edit ── */

import { useEffect, useState, useMemo, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api } from "../services/api";
import type {
  SurveyReport,
  ReportItem,
  Disposition,
  Scan,
  MoveSummary,
  InventoryItem,
  Inventory,
} from "../types";

const DISPOSITION_CONFIG: Record<
  string,
  { label: string; color: string; bg: string }
> = {
  going: { label: "GOING", color: "text-green-600", bg: "bg-green-50" },
  staying: { label: "STAYING", color: "text-gray-500", bg: "bg-gray-100" },
  scrap: { label: "SCRAP", color: "text-red-500", bg: "bg-red-50" },
  haul: { label: "HAUL", color: "text-orange-500", bg: "bg-orange-50" },
  sell: { label: "SELL", color: "text-blue-500", bg: "bg-blue-50" },
};

function DispositionBadge({ value }: { value: Disposition }) {
  if (!value) return null;
  const cfg = DISPOSITION_CONFIG[value];
  if (!cfg) return null;
  return (
    <span
      className={`text-[10px] font-bold ${cfg.color} ${cfg.bg} px-1.5 py-0.5 rounded`}
    >
      {cfg.label}
    </span>
  );
}

const DISPOSITION_OPTIONS: { value: string; label: string }[] = [
  { value: "", label: "—" },
  { value: "going", label: "Going" },
  { value: "staying", label: "Staying" },
  { value: "scrap", label: "Scrap" },
  { value: "haul", label: "Haul" },
  { value: "sell", label: "Sell" },
];

type ViewMode = "internal" | "customer";

export default function InventoryPage() {
  const { id: scanId } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [report, setReport] = useState<SurveyReport | null>(null);
  const [scan, setScan] = useState<Scan | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("internal");
  const [expandedRoom, setExpandedRoom] = useState<string | null>(null);
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);
  const [allExpanded, setAllExpanded] = useState(true);
  const [filterRoom, setFilterRoom] = useState<string | null>(null); // null = "All"

  // ── Edit mode state ──
  const [editing, setEditing] = useState(false);
  const [inventoryItems, setInventoryItems] = useState<InventoryItem[]>([]);
  const [editLoading, setEditLoading] = useState(false);
  const [addingToRoom, setAddingToRoom] = useState<string | null>(null);
  const [newItemName, setNewItemName] = useState("");
  const [newItemCount, setNewItemCount] = useState(1);

  // Build lookup from (name, room) → InventoryItem for edit operations
  const inventoryLookup = useMemo(() => {
    const map = new Map<string, InventoryItem>();
    for (const item of inventoryItems) {
      map.set(
        `${item.name.toLowerCase()}::${item.room_name.toLowerCase()}`,
        item,
      );
    }
    return map;
  }, [inventoryItems]);

  const findInventoryItem = useCallback(
    (reportItem: ReportItem, room: string): InventoryItem | undefined => {
      return inventoryLookup.get(
        `${reportItem.name.toLowerCase()}::${room.toLowerCase()}`,
      );
    },
    [inventoryLookup],
  );

  const enterEditMode = async () => {
    if (!scanId) return;
    setEditLoading(true);
    try {
      const inv: Inventory = await api.inventory.get(scanId);
      setInventoryItems(inv.items);
      setEditing(true);
    } catch (e: any) {
      console.error("Failed to load inventory for editing:", e);
    } finally {
      setEditLoading(false);
    }
  };

  const exitEditMode = () => {
    setEditing(false);
    setAddingToRoom(null);
    setNewItemName("");
    setNewItemCount(1);
  };

  const updateItemField = async (
    invItem: InventoryItem,
    updates: Partial<{ count: number; disposition: Disposition; name: string }>,
  ) => {
    if (!scanId) return;
    // Optimistic update
    setInventoryItems((prev) =>
      prev.map((i) => (i.id === invItem.id ? { ...i, ...updates } : i)),
    );
    try {
      await api.inventory.updateItem(scanId, invItem.id, updates);
    } catch {
      // Revert
      setInventoryItems((prev) =>
        prev.map((i) => (i.id === invItem.id ? invItem : i)),
      );
    }
  };

  const deleteItem = async (invItem: InventoryItem) => {
    if (!scanId) return;
    setInventoryItems((prev) => prev.filter((i) => i.id !== invItem.id));
    try {
      await api.inventory.deleteItem(scanId, invItem.id);
    } catch {
      setInventoryItems((prev) => [...prev, invItem]);
    }
  };

  const addItem = async (room: string) => {
    if (!scanId || !newItemName.trim()) return;
    try {
      const created = await api.inventory.addItem(scanId, {
        name: newItemName.trim().toLowerCase(),
        count: Math.max(1, newItemCount),
        room_name: room,
      });
      setInventoryItems((prev) => [...prev, created]);
      setNewItemName("");
      setNewItemCount(1);
      setAddingToRoom(null);
    } catch (e: any) {
      console.error("Failed to add item:", e);
    }
  };

  useEffect(() => {
    if (!scanId) return;
    Promise.all([
      api.scans.report(scanId),
      api.scans.get(scanId).catch(() => null),
    ])
      .then(([r, s]) => {
        setReport(r);
        if (s) setScan(s);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [scanId]);

  // Group items by room
  const roomData = useMemo(() => {
    if (!report) return {};
    const result: Record<string, ReportItem[]> = {};
    for (const [room, indices] of Object.entries(report.rooms)) {
      result[room] = indices.map((i) => report.items[i]);
    }
    return result;
  }, [report]);

  const roomNames = useMemo(() => Object.keys(roomData), [roomData]);
  const filteredRoomData = useMemo(() => {
    if (!filterRoom) return roomData;
    return filterRoom in roomData ? { [filterRoom]: roomData[filterRoom] } : roomData;
  }, [roomData, filterRoom]);

  if (!scanId) return null;

  if (loading) {
    return (
      <div className="app-shell flex flex-col min-h-screen bg-gray-50">
        <header className="flex items-center gap-3 px-5 py-4 border-b border-gray-100 bg-white">
          <div className="skeleton h-5 w-5 rounded" />
          <div className="flex-1">
            <div className="skeleton h-5 w-32" />
          </div>
        </header>
        <div className="flex-1 px-5 py-5">
          <div className="grid grid-cols-2 gap-3 mb-6">
            {[1, 2, 3, 4].map((i) => (
              <div
                key={i}
                className="bg-white rounded-xl border border-gray-100 p-4"
              >
                <div className="skeleton h-7 w-16 mx-auto mb-2" />
                <div className="skeleton h-3 w-12 mx-auto" />
              </div>
            ))}
          </div>
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="bg-white rounded-xl border border-gray-100 p-4 mb-3"
            >
              <div className="skeleton h-4 w-28 mb-2" />
              <div className="skeleton h-3 w-36" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="px-5 py-10 text-center">
        <p className="text-sm text-red-500 mb-4">
          {error || "Report not available"}
        </p>
        <button
          onClick={() => navigate(`/scans/${scanId}`)}
          className="text-sm text-gray-500 underline"
        >
          Back to Scan
        </button>
      </div>
    );
  }

  const { summary } = report;
  const isInternal = viewMode === "internal";

  return (
    <div className="app-shell flex flex-col min-h-screen bg-gray-50 print:bg-white">
      {/* Header */}
      <header className="flex items-center gap-3 px-5 py-4 border-b border-gray-100/80 sticky top-0 glass z-30 safe-top print:static">
        <button
          onClick={() => navigate(`/scans/${scanId}`)}
          className="w-8 h-8 flex items-center justify-center rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors print:hidden active:scale-95"
        >
          ←
        </button>
        <div className="flex-1">
          <h1 className="font-bold text-base">Survey Report</h1>
          <p className="text-[11px] text-gray-400">
            {new Date(report.generated_at).toLocaleDateString()} ·{" "}
            {scan?.move_summary
              ? scan.move_summary.has_audio
                ? "🎤 Voice Assisted"
                : "👁️ Visual Only"
              : report.survey_mode === "voice_assisted"
                ? "🎤 Voice Assisted"
                : "👁️ Visual Only"}
          </p>
        </div>
        <button
          onClick={() => window.print()}
          className="w-8 h-8 flex items-center justify-center rounded-lg text-gray-500 hover:bg-gray-100 transition-colors print:hidden active:scale-95"
        >
          🖨️
        </button>
        {editing ? (
          <button
            onClick={exitEditMode}
            className="px-3 py-1.5 text-xs font-bold text-white bg-green-600 rounded-lg shadow-sm hover:bg-green-700 transition-colors print:hidden active:scale-95"
          >
            ✓ Done
          </button>
        ) : (
          <button
            onClick={enterEditMode}
            disabled={editLoading}
            className="px-3 py-1.5 text-xs font-bold text-white bg-gray-800 rounded-lg shadow-sm hover:bg-gray-700 transition-colors print:hidden active:scale-95 disabled:opacity-50"
          >
            {editLoading ? "…" : "✏️ Edit"}
          </button>
        )}
      </header>

      <div className="flex-1 px-5 py-5 overflow-y-auto page-enter">
        {/* View toggle */}
        <div className="flex bg-white rounded-xl p-1 mb-5 border border-gray-100 shadow-sm print:hidden">
          {(["internal", "customer"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setViewMode(m)}
              className={`flex-1 py-2.5 rounded-lg text-xs font-bold transition-all active:scale-[0.97] ${
                viewMode === m
                  ? "bg-gradient-to-r from-gray-900 to-gray-800 text-white shadow-sm"
                  : "text-gray-400 hover:text-gray-600"
              }`}
            >
              {m === "internal" ? "🔧 Internal View" : "👤 Customer View"}
            </button>
          ))}
        </div>

        {/* Summary cards */}
        <div className="grid grid-cols-2 gap-3 mb-6 stagger">
          <div className="bg-white rounded-xl border border-gray-100 p-3.5 text-center shadow-sm fade-in-up">
            <p className="text-2xl font-extrabold text-brand-red">
              {summary.total_items}
            </p>
            <p className="text-[10px] text-gray-400 font-medium mt-0.5">
              Total Items
            </p>
          </div>
          <div className="bg-white rounded-xl border border-gray-100 p-3.5 text-center shadow-sm fade-in-up">
            <p className="text-2xl font-extrabold">
              {summary.total_volume_cuft.toLocaleString()}
            </p>
            <p className="text-[10px] text-gray-400 font-medium mt-0.5">
              Total Cu Ft
            </p>
          </div>
          <div className="bg-white rounded-xl border border-gray-100 p-3.5 text-center shadow-sm fade-in-up">
            <p className="text-2xl font-extrabold">
              {summary.total_weight_lbs.toLocaleString()}
            </p>
            <p className="text-[10px] text-gray-400 font-medium mt-0.5">
              Total Lbs
            </p>
          </div>
          <div className="bg-white rounded-xl border border-gray-100 p-3.5 text-center shadow-sm fade-in-up">
            <p className="text-lg font-extrabold leading-tight">
              {summary.truck_recommendation}
            </p>
            <p className="text-[10px] text-gray-400 font-medium mt-0.5">
              🚛 Truck Size
            </p>
          </div>
        </div>

        {/* Move Summary */}
        {scan?.move_summary && <MoveSummaryCard summary={scan.move_summary} />}

        {/* Room filter tabs */}
        {roomNames.length > 1 && (
          <div className="flex gap-2 mb-5 overflow-x-auto pb-1 scrollbar-hide print:hidden">
            <button
              onClick={() => setFilterRoom(null)}
              className={`px-3 py-1.5 rounded-full text-xs font-bold whitespace-nowrap transition-all ${
                !filterRoom
                  ? "bg-gray-900 text-white shadow-sm"
                  : "bg-white border border-gray-200 text-gray-500 hover:border-gray-300"
              }`}
            >
              All ({report.items.length})
            </button>
            {roomNames.map((room) => (
              <button
                key={room}
                onClick={() => setFilterRoom(filterRoom === room ? null : room)}
                className={`px-3 py-1.5 rounded-full text-xs font-bold whitespace-nowrap transition-all ${
                  filterRoom === room
                    ? "bg-gray-900 text-white shadow-sm"
                    : "bg-white border border-gray-200 text-gray-500 hover:border-gray-300"
                }`}
              >
                {room} ({roomData[room]?.length || 0})
              </button>
            ))}
          </div>
        )}

        {/* Room sections */}
        {Object.entries(filteredRoomData).map(([room, items]) => {
          const roomVol = items.reduce((s, i) => s + i.total_volume_cuft, 0);
          const roomWt = items.reduce((s, i) => s + i.total_weight_lbs, 0);
          const roomCount = items.reduce((s, i) => s + i.count, 0);
          const isExpanded = allExpanded || expandedRoom === room;

          return (
            <section key={room} className="mb-4">
              {/* Room header — clickable accordion */}
              <button
                onClick={() => {
                  if (allExpanded) {
                    setAllExpanded(false);
                    setExpandedRoom(isExpanded ? null : room);
                  } else {
                    setExpandedRoom(expandedRoom === room ? null : room);
                  }
                }}
                className="w-full flex items-center justify-between bg-white rounded-xl border border-gray-100 px-4 py-3.5 mb-1 hover:shadow-md hover:border-gray-200 transition-all active:scale-[0.99]"
              >
                <div className="text-left">
                  <h3 className="text-sm font-bold capitalize">{room}</h3>
                  <p className="text-[10px] text-gray-400 mt-0.5">
                    {roomCount} item{roomCount !== 1 ? "s" : ""} ·{" "}
                    {roomVol.toFixed(0)} cu ft · {roomWt.toFixed(0)} lbs
                  </p>
                </div>
                <span
                  className={`text-gray-300 text-xs transition-transform duration-200 ${
                    isExpanded ? "rotate-180" : ""
                  }`}
                >
                  ▼
                </span>
              </button>

              {/* Room items table */}
              {isExpanded && (
                <div className="bg-white rounded-xl border border-gray-100 overflow-hidden">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="bg-gray-50 text-gray-500 text-[10px] uppercase tracking-wider">
                        <th className="text-left px-3 py-2">Item</th>
                        <th className="text-center px-2 py-2">Qty</th>
                        {!editing && (
                          <th className="text-center px-2 py-2">Size</th>
                        )}
                        {isInternal && !editing && (
                          <>
                            <th className="text-right px-2 py-2">Cu Ft</th>
                            <th className="text-right px-2 py-2">Lbs</th>
                          </>
                        )}
                        <th className="text-center px-2 py-2">Status</th>
                        {editing && (
                          <th className="text-center px-2 py-2 w-8"></th>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {items.map((item, idx) => {
                        const invItem = editing
                          ? findInventoryItem(item, room)
                          : undefined;
                        const displayCount =
                          editing && invItem ? invItem.count : item.count;
                        const displayDisp =
                          editing && invItem
                            ? invItem.disposition
                            : item.disposition;

                        return (
                          <tr
                            key={`${room}-${idx}`}
                            className="border-t border-gray-50 hover:bg-gray-50/50"
                          >
                            <td className="px-3 py-2.5">
                              <div className="flex items-center gap-2">
                                {item.evidence_image && (
                                  <button
                                    onClick={() =>
                                      setLightboxSrc(
                                        `data:image/jpeg;base64,${item.evidence_image}`,
                                      )
                                    }
                                    className="w-8 h-8 rounded-lg overflow-hidden flex-shrink-0 bg-gray-100"
                                  >
                                    <img
                                      src={`data:image/jpeg;base64,${item.evidence_image}`}
                                      alt={item.name}
                                      className="w-full h-full object-cover"
                                    />
                                  </button>
                                )}
                                <div>
                                  <p className="font-semibold text-gray-800 capitalize">
                                    {item.name}
                                  </p>
                                  {!editing &&
                                    item.special_handling.length > 0 && (
                                      <p className="text-[10px] text-amber-600 mt-0.5">
                                        ⚠️ {item.special_handling[0]}
                                      </p>
                                    )}
                                  {!editing && item.notes && isInternal && (
                                    <p className="text-[10px] text-gray-400 mt-0.5 italic">
                                      {item.notes}
                                    </p>
                                  )}
                                </div>
                              </div>
                            </td>

                            {/* Quantity — editable in edit mode */}
                            <td className="text-center px-2">
                              {editing && invItem ? (
                                <div className="flex items-center justify-center gap-1">
                                  <button
                                    onClick={() =>
                                      updateItemField(invItem, {
                                        count: Math.max(1, displayCount - 1),
                                      })
                                    }
                                    className="w-5 h-5 rounded bg-gray-100 text-gray-600 text-xs font-bold hover:bg-gray-200 active:scale-90"
                                  >
                                    −
                                  </button>
                                  <span className="w-5 text-center font-bold text-gray-800">
                                    {displayCount}
                                  </span>
                                  <button
                                    onClick={() =>
                                      updateItemField(invItem, {
                                        count: displayCount + 1,
                                      })
                                    }
                                    className="w-5 h-5 rounded bg-gray-100 text-gray-600 text-xs font-bold hover:bg-gray-200 active:scale-90"
                                  >
                                    +
                                  </button>
                                </div>
                              ) : (
                                <span className="font-bold text-gray-800">
                                  {displayCount}
                                </span>
                              )}
                            </td>

                            {/* Size — hidden in edit mode */}
                            {!editing && (
                              <td className="text-center text-gray-500 px-2 capitalize">
                                {item.size}
                              </td>
                            )}

                            {/* Volume/Weight — hidden in edit mode */}
                            {isInternal && !editing && (
                              <>
                                <td className="text-right text-gray-600 px-2">
                                  {item.total_volume_cuft}
                                </td>
                                <td className="text-right text-gray-600 px-2">
                                  {item.total_weight_lbs}
                                </td>
                              </>
                            )}

                            {/* Disposition — dropdown in edit mode, badge otherwise */}
                            <td className="text-center px-2">
                              {editing && invItem ? (
                                <select
                                  value={displayDisp || ""}
                                  onChange={(e) => {
                                    const val = (e.target.value ||
                                      null) as Disposition;
                                    updateItemField(invItem, {
                                      disposition: val,
                                    });
                                  }}
                                  className={`text-[10px] font-bold rounded px-1.5 py-1 border border-gray-200 outline-none ${
                                    displayDisp
                                      ? DISPOSITION_CONFIG[displayDisp]?.bg ||
                                        "bg-white"
                                      : "bg-white"
                                  }`}
                                >
                                  {DISPOSITION_OPTIONS.map((opt) => (
                                    <option key={opt.value} value={opt.value}>
                                      {opt.label}
                                    </option>
                                  ))}
                                </select>
                              ) : (
                                <DispositionBadge value={displayDisp} />
                              )}
                            </td>

                            {/* Delete button — edit mode only */}
                            {editing && (
                              <td className="text-center px-2">
                                {invItem ? (
                                  <button
                                    onClick={() => deleteItem(invItem)}
                                    className="w-5 h-5 rounded text-red-400 hover:text-red-600 hover:bg-red-50 text-xs active:scale-90"
                                    title="Remove item"
                                  >
                                    ✕
                                  </button>
                                ) : null}
                              </td>
                            )}
                          </tr>
                        );
                      })}

                      {/* Manually added items not in report */}
                      {editing &&
                        inventoryItems
                          .filter(
                            (inv) =>
                              inv.room_name.toLowerCase() ===
                                room.toLowerCase() &&
                              !items.some(
                                (ri) =>
                                  ri.name.toLowerCase() ===
                                  inv.name.toLowerCase(),
                              ),
                          )
                          .map((inv) => (
                            <tr
                              key={`new-${inv.id}`}
                              className="border-t border-gray-50 bg-green-50/30"
                            >
                              <td className="px-3 py-2.5">
                                <div className="flex items-center gap-2">
                                  <span className="text-[10px] text-green-600 bg-green-100 px-1 rounded font-bold">
                                    NEW
                                  </span>
                                  <p className="font-semibold text-gray-800 capitalize">
                                    {inv.name}
                                  </p>
                                </div>
                              </td>
                              <td className="text-center px-2">
                                <div className="flex items-center justify-center gap-1">
                                  <button
                                    onClick={() =>
                                      updateItemField(inv, {
                                        count: Math.max(1, inv.count - 1),
                                      })
                                    }
                                    className="w-5 h-5 rounded bg-gray-100 text-gray-600 text-xs font-bold hover:bg-gray-200 active:scale-90"
                                  >
                                    −
                                  </button>
                                  <span className="w-5 text-center font-bold text-gray-800">
                                    {inv.count}
                                  </span>
                                  <button
                                    onClick={() =>
                                      updateItemField(inv, {
                                        count: inv.count + 1,
                                      })
                                    }
                                    className="w-5 h-5 rounded bg-gray-100 text-gray-600 text-xs font-bold hover:bg-gray-200 active:scale-90"
                                  >
                                    +
                                  </button>
                                </div>
                              </td>
                              <td className="text-center px-2">
                                <select
                                  value={inv.disposition || ""}
                                  onChange={(e) => {
                                    const val = (e.target.value ||
                                      null) as Disposition;
                                    updateItemField(inv, { disposition: val });
                                  }}
                                  className={`text-[10px] font-bold rounded px-1.5 py-1 border border-gray-200 outline-none ${
                                    inv.disposition
                                      ? DISPOSITION_CONFIG[inv.disposition]
                                          ?.bg || "bg-white"
                                      : "bg-white"
                                  }`}
                                >
                                  {DISPOSITION_OPTIONS.map((opt) => (
                                    <option key={opt.value} value={opt.value}>
                                      {opt.label}
                                    </option>
                                  ))}
                                </select>
                              </td>
                              <td className="text-center px-2">
                                <button
                                  onClick={() => deleteItem(inv)}
                                  className="w-5 h-5 rounded text-red-400 hover:text-red-600 hover:bg-red-50 text-xs active:scale-90"
                                >
                                  ✕
                                </button>
                              </td>
                            </tr>
                          ))}
                    </tbody>
                    {!editing && isInternal && (
                      <tfoot>
                        <tr className="border-t-2 border-gray-200 bg-gray-50 font-bold text-[11px]">
                          <td className="px-3 py-2">
                            Subtotal ({roomCount} items)
                          </td>
                          <td className="text-center px-2">{roomCount}</td>
                          <td className="px-2" />
                          <td className="text-right px-2">
                            {roomVol.toFixed(0)}
                          </td>
                          <td className="text-right px-2">
                            {roomWt.toFixed(0)}
                          </td>
                          <td className="px-2" />
                        </tr>
                      </tfoot>
                    )}
                  </table>

                  {/* Add item row — edit mode only */}
                  {editing && (
                    <div className="border-t border-gray-100 px-3 py-2">
                      {addingToRoom === room ? (
                        <div className="flex items-center gap-2">
                          <input
                            type="text"
                            value={newItemName}
                            onChange={(e) => setNewItemName(e.target.value)}
                            placeholder="Item name"
                            className="flex-1 text-xs border border-gray-200 rounded px-2 py-1.5 outline-none focus:border-gray-400"
                            autoFocus
                            onKeyDown={(e) => {
                              if (e.key === "Enter") addItem(room);
                              if (e.key === "Escape") setAddingToRoom(null);
                            }}
                          />
                          <input
                            type="number"
                            value={newItemCount}
                            onChange={(e) =>
                              setNewItemCount(
                                Math.max(1, parseInt(e.target.value) || 1),
                              )
                            }
                            className="w-12 text-xs text-center border border-gray-200 rounded px-1 py-1.5 outline-none focus:border-gray-400"
                            min={1}
                          />
                          <button
                            onClick={() => addItem(room)}
                            className="text-xs font-bold text-green-600 hover:text-green-700 px-2 py-1"
                          >
                            Add
                          </button>
                          <button
                            onClick={() => setAddingToRoom(null)}
                            className="text-xs text-gray-400 hover:text-gray-600 px-1"
                          >
                            ✕
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={() => {
                            setAddingToRoom(room);
                            setNewItemName("");
                            setNewItemCount(1);
                          }}
                          className="text-[10px] font-bold text-gray-400 hover:text-gray-600 transition-colors"
                        >
                          + Add Item
                        </button>
                      )}
                    </div>
                  )}
                </div>
              )}
            </section>
          );
        })}

        {/* Packing Materials — internal only */}
        {isInternal &&
          report.packing_materials &&
          Object.keys(report.packing_materials).length > 0 && (
            <section className="mb-6">
              <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">
                Packing Materials Estimate
              </h3>
              <div className="bg-white rounded-xl border border-gray-100 p-4">
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(report.packing_materials).map(
                    ([material, qty]) => (
                      <div
                        key={material}
                        className="flex justify-between text-xs py-1"
                      >
                        <span className="text-gray-600 capitalize">
                          {material.replace(/_/g, " ")}
                        </span>
                        <span className="font-bold">{qty as number}</span>
                      </div>
                    ),
                  )}
                </div>
              </div>
            </section>
          )}

        {/* Special Handling — internal only */}
        {isInternal &&
          report.special_handling &&
          report.special_handling.length > 0 && (
            <section className="mb-6">
              <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">
                Special Handling Notes
              </h3>
              <div className="bg-white rounded-xl border border-gray-100 p-4 space-y-2">
                {report.special_handling.map((note, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs">
                    <span className="text-amber-500 mt-0.5">⚠️</span>
                    <span className="text-gray-700">{note}</span>
                  </div>
                ))}
              </div>
            </section>
          )}

        {/* LLM Usage — internal only */}
        {isInternal && report.usage && (
          <section className="mb-6">
            <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">
              Analysis Details
            </h3>
            <div className="bg-white rounded-xl border border-gray-100 p-4">
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex justify-between py-1">
                  <span className="text-gray-500">Model</span>
                  <span className="font-medium">{report.usage.model}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-gray-500">Frames Analyzed</span>
                  <span className="font-medium">
                    {report.num_frames_analyzed}
                  </span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-gray-500">Tokens In</span>
                  <span className="font-medium">
                    {report.usage.tokens_in?.toLocaleString() || "—"}
                  </span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-gray-500">Tokens Out</span>
                  <span className="font-medium">
                    {report.usage.tokens_out?.toLocaleString() || "—"}
                  </span>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Footer spacer */}
        <div className="h-8 print:hidden" />
      </div>

      {/* Lightbox */}
      {lightboxSrc && (
        <div
          className="fixed inset-0 bg-black/85 flex items-center justify-center z-50 p-6 print:hidden backdrop-enter"
          onClick={() => setLightboxSrc(null)}
        >
          <div className="relative scale-enter">
            <img
              src={lightboxSrc}
              alt="Evidence"
              className="max-w-full max-h-[80vh] rounded-2xl object-contain shadow-2xl"
            />
            <button
              onClick={() => setLightboxSrc(null)}
              className="absolute -top-3 -right-3 w-8 h-8 bg-white rounded-full text-gray-600 font-bold text-sm flex items-center justify-center shadow-lg hover:bg-gray-100 transition-colors"
            >
              ✕
            </button>
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
    <div className="mb-5 bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-3 p-4 text-left"
      >
        <span className="text-xl">{summary.has_audio ? "🎙️" : "👁️"}</span>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-bold text-gray-800">Move Summary</p>
          <p className="text-[10px] text-gray-400 mt-0.5 truncate">
            {summary.audio_summary}
          </p>
        </div>
        <span
          className={`text-gray-400 text-xs transition-transform ${expanded ? "rotate-180" : ""}`}
        >
          ▼
        </span>
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-4 border-t border-gray-50">
          {summary.overall && (
            <p className="text-xs text-gray-600 leading-relaxed pt-3">
              {summary.overall}
            </p>
          )}

          {summary.rooms.length > 0 && (
            <div>
              <p className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">
                Room Details
              </p>
              <div className="space-y-2">
                {summary.rooms.map((room) => (
                  <div key={room.name} className="bg-gray-50 rounded-xl p-3">
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
