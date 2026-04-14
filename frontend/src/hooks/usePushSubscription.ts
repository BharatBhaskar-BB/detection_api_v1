/* ── Hook to subscribe to Web Push notifications ── */

import { useEffect, useRef } from "react";
import { api } from "../services/api";
import { useAuthStore } from "../stores/authStore";

function urlBase64ToUint8Array(base64String: string): Uint8Array {
  const padding = "=".repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, "+").replace(/_/g, "/");
  const raw = atob(base64);
  const arr = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);
  return arr;
}

export function usePushSubscription() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const attempted = useRef(false);

  useEffect(() => {
    if (!isAuthenticated || attempted.current) return;
    if (!("serviceWorker" in navigator) || !("PushManager" in window)) return;

    attempted.current = true;

    (async () => {
      try {
        const reg = await navigator.serviceWorker.ready;

        // Check if already subscribed
        const existing = await reg.pushManager.getSubscription();
        if (existing) return; // Already subscribed

        // Get VAPID public key from backend
        const { public_key } = await api.push.getVapidKey();
        const appServerKey = urlBase64ToUint8Array(public_key);

        // Ask permission and subscribe
        const subscription = await reg.pushManager.subscribe({
          userVisibleOnly: true,
          applicationServerKey: appServerKey as BufferSource,
        });

        // Send subscription to backend
        await api.push.subscribe(subscription.toJSON());
      } catch {
        // User denied or push not supported — silent fail
      }
    })();
  }, [isAuthenticated]);
}
