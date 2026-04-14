/* ── App shell layout with bottom tab bar ── */

import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useAuthStore } from "../stores/authStore";
import { useNotificationStore } from "../stores/notificationStore";
import NotificationPanel from "./NotificationPanel";

const tabs = [
  { to: "/", icon: "🏠", label: "Home" },
  { to: "/scans", icon: "📦", label: "Scans" },
  { to: "/settings", icon: "⚙️", label: "Settings" },
];

export default function Layout() {
  const user = useAuthStore((s) => s.user);
  const { unreadCount, panelOpen, togglePanel } = useNotificationStore();
  const navigate = useNavigate();

  return (
    <div className="app-shell flex flex-col min-h-screen">
      {/* Top bar */}
      <header className="flex items-center justify-between px-5 py-3 glass border-b border-gray-100/80 sticky top-0 z-50 safe-top">
        <button
          onClick={() => navigate("/")}
          className="flex items-center gap-2.5 active:scale-95 transition-transform"
        >
          <img src="/icons/icon-32.png" alt="" className="w-7 h-7 rounded-lg" />
          <h1 className="text-lg font-extrabold tracking-tight">
            Bundle<span className="text-brand-red">Box</span>
          </h1>
        </button>
        <div className="flex items-center gap-2">
          <button
            onClick={togglePanel}
            className="w-9 h-9 flex items-center justify-center rounded-xl bg-gray-50/80 border border-gray-100 text-sm relative hover:bg-gray-100 transition-colors active:scale-95"
          >
            🔔
            {unreadCount > 0 && (
              <span className="absolute -top-1 -right-1 w-[18px] h-[18px] bg-brand-red text-white text-[10px] font-bold rounded-full flex items-center justify-center pulse-glow">
                {unreadCount > 9 ? "9+" : unreadCount}
              </span>
            )}
          </button>
          <div className="w-9 h-9 flex items-center justify-center rounded-xl bg-gradient-to-br from-gray-800 to-gray-950 text-white text-xs font-bold shadow-sm">
            {user?.name?.slice(0, 2).toUpperCase() || "BB"}
          </div>
        </div>
      </header>

      {/* Notification panel */}
      {panelOpen && <NotificationPanel />}

      {/* Page content */}
      <main className="flex-1 overflow-y-auto pb-20">
        <Outlet />
      </main>

      {/* Bottom tab bar */}
      <nav className="flex items-center justify-around glass border-t border-gray-100/80 py-2 pb-[max(0.5rem,env(safe-area-inset-bottom))] fixed bottom-0 left-1/2 -translate-x-1/2 w-full max-w-[430px] z-50">
        {tabs.map((tab) => (
          <NavLink
            key={tab.to}
            to={tab.to}
            end={tab.to === "/"}
            className={({ isActive }) =>
              `flex flex-col items-center gap-0.5 px-4 py-1.5 rounded-xl text-[10px] font-semibold transition-all active:scale-95 ${
                isActive
                  ? "text-brand-red bg-red-50/60"
                  : "text-gray-400 hover:text-gray-600"
              }`
            }
          >
            <span className="text-xl leading-none">{tab.icon}</span>
            {tab.label}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
