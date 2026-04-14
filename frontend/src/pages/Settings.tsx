/* ── Settings Page ── */

import { useNavigate } from "react-router-dom";
import { useAuthStore } from "../stores/authStore";

export default function SettingsPage() {
  const navigate = useNavigate();
  const { user, logout } = useAuthStore();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <div className="px-5 py-6 page-enter">
      <h2 className="text-lg font-bold mb-6">Settings</h2>

      {/* Profile section */}
      <div className="bg-white rounded-xl border border-gray-100 p-4 mb-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-full bg-gradient-to-br from-gray-800 to-gray-950 text-white flex items-center justify-center text-sm font-bold shadow-sm">
            {user?.name?.slice(0, 2).toUpperCase() || "BB"}
          </div>
          <div>
            <p className="font-semibold text-sm">{user?.name || "User"}</p>
            <p className="text-xs text-gray-400">
              {user?.email || user?.phone || ""}
            </p>
          </div>
        </div>
      </div>

      {/* App info */}
      <div className="bg-white rounded-xl border border-gray-100 p-4 mb-6">
        <div className="flex justify-between text-sm py-2">
          <span className="text-gray-500">Version</span>
          <span className="font-medium">1.0.0</span>
        </div>
        <div className="flex justify-between text-sm py-2 border-t border-gray-50">
          <span className="text-gray-500">Pipeline</span>
          <span className="font-medium">Gemini 2.0 Flash</span>
        </div>
      </div>

      {/* Logout */}
      <button
        onClick={handleLogout}
        className="w-full py-3.5 rounded-xl bg-red-50 text-red-600 font-semibold text-sm hover:bg-red-100 transition-colors"
      >
        Log Out
      </button>
    </div>
  );
}
