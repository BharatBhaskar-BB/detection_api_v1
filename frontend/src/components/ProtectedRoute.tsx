/* ── Protected route — redirects to /login if unauthenticated ── */

import { Navigate } from "react-router-dom";
import { useAuthStore } from "../stores/authStore";

export default function ProtectedRoute({
  children,
}: {
  children: React.ReactNode;
}) {
  const { isAuthenticated, isLoading, needsName } = useAuthStore();

  if (isLoading) {
    return (
      <div className="app-shell flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="text-3xl mb-3 animate-pulse">📦</div>
          <p className="text-sm text-gray-400 font-medium">Loading…</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  // Force name capture for phone users who haven't set their name yet
  if (needsName) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}
