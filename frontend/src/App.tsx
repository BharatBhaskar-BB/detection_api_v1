/* ── App routes ── */

import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { useEffect } from "react";
import { useAuthStore } from "./stores/authStore";
import { usePushSubscription } from "./hooks/usePushSubscription";
import Layout from "./components/Layout";
import ProtectedRoute from "./components/ProtectedRoute";
import LoginPage from "./pages/Login";
import DashboardPage from "./pages/Dashboard";
import NewScanPage from "./pages/NewScan";
import CameraPage from "./pages/Camera";
import ProcessingPage from "./pages/Processing";
import InventoryPage from "./pages/Inventory";
import ScanDetailPage from "./pages/ScanDetail";
import SettingsPage from "./pages/Settings";

export default function App() {
  const loadUser = useAuthStore((s) => s.loadUser);
  usePushSubscription();

  useEffect(() => {
    loadUser();
  }, [loadUser]);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />

        {/* All authenticated routes share the Layout shell */}
        <Route
          element={
            <ProtectedRoute>
              <Layout />
            </ProtectedRoute>
          }
        >
          <Route index element={<DashboardPage />} />
          <Route path="scans" element={<DashboardPage />} />
          <Route path="scans/new" element={<NewScanPage />} />
          <Route path="scans/:id" element={<ScanDetailPage />} />
          <Route path="scans/:id/inventory" element={<InventoryPage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>

        {/* Full-screen pages (no bottom nav) */}
        <Route
          path="scans/:id/camera"
          element={
            <ProtectedRoute>
              <CameraPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="scans/:id/processing"
          element={
            <ProtectedRoute>
              <ProcessingPage />
            </ProtectedRoute>
          }
        />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
