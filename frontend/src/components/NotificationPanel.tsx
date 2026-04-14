/* ── Notification Panel — dropdown from bell icon ── */

import { useNavigate } from "react-router-dom";
import {
  useNotificationStore,
  type AppNotification,
} from "../stores/notificationStore";

const ICONS: Record<string, string> = {
  scan_complete: "✅",
  scan_failed: "❌",
  scan_progress: "🔄",
  welcome: "👋",
  pwa_update: "🆕",
};

function timeAgo(ts: number): string {
  const diff = Date.now() - ts;
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return `${Math.floor(diff / 86_400_000)}d ago`;
}

function NotificationItem({
  n,
  onNavigate,
}: {
  n: AppNotification;
  onNavigate: (path: string) => void;
}) {
  const { markRead, remove } = useNotificationStore();

  const handleClick = () => {
    markRead(n.id);
    if (n.scanId) {
      if (n.type === "scan_complete") {
        onNavigate(`/scans/${n.scanId}/inventory`);
      } else {
        onNavigate(`/scans/${n.scanId}`);
      }
    }
  };

  return (
    <div
      onClick={handleClick}
      className={`flex items-start gap-3 px-4 py-3 border-b border-gray-50 cursor-pointer hover:bg-gray-50 transition-colors ${
        !n.read ? "bg-red-50/40" : ""
      }`}
    >
      <span className="text-lg mt-0.5">{ICONS[n.type] || "🔔"}</span>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-gray-900 truncate">
          {n.title}
        </p>
        <p className="text-xs text-gray-500 mt-0.5 line-clamp-2">{n.message}</p>
        <p className="text-[10px] text-gray-300 mt-1">{timeAgo(n.timestamp)}</p>
      </div>
      {!n.read && (
        <span className="w-2 h-2 rounded-full bg-brand-red mt-2 flex-shrink-0" />
      )}
      <button
        onClick={(e) => {
          e.stopPropagation();
          remove(n.id);
        }}
        className="text-gray-300 hover:text-gray-500 text-xs mt-1 flex-shrink-0"
        title="Dismiss"
      >
        ✕
      </button>
    </div>
  );
}

export default function NotificationPanel() {
  const navigate = useNavigate();
  const { notifications, markAllRead, clear, closePanel } =
    useNotificationStore();

  const handleNavigate = (path: string) => {
    closePanel();
    navigate(path);
  };

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 z-[60]" onClick={closePanel} />

      {/* Panel */}
      <div className="absolute top-14 right-3 w-[340px] max-h-[70vh] bg-white rounded-2xl shadow-xl border border-gray-100 z-[70] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
          <h3 className="text-sm font-bold">Notifications</h3>
          <div className="flex items-center gap-3">
            {notifications.length > 0 && (
              <>
                <button
                  onClick={markAllRead}
                  className="text-[11px] text-brand-red font-semibold hover:underline"
                >
                  Mark all read
                </button>
                <button
                  onClick={clear}
                  className="text-[11px] text-gray-400 font-semibold hover:underline"
                >
                  Clear
                </button>
              </>
            )}
          </div>
        </div>

        {/* List */}
        <div className="flex-1 overflow-y-auto">
          {notifications.length === 0 ? (
            <div className="py-12 text-center text-sm text-gray-300">
              No notifications yet
            </div>
          ) : (
            notifications.map((n) => (
              <NotificationItem key={n.id} n={n} onNavigate={handleNavigate} />
            ))
          )}
        </div>
      </div>
    </>
  );
}
