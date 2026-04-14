/* ── API client for BundleBox backend ── */

import type {
  Inventory,
  InventoryItem,
  Scan,
  ScanListItem,
  SurveyReport,
  TokenResponse,
  User,
} from "../types";

const BASE = "/api/v1";

function getToken(): string | null {
  return localStorage.getItem("bb_token");
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers,
    signal: options.signal ?? AbortSignal.timeout(30000),
  });

  if (res.status === 401 || res.status === 403) {
    localStorage.removeItem("bb_token");
    window.location.href = "/login";
    throw new Error("Unauthorized");
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }

  if (res.status === 204) return undefined as T;
  return res.json();
}

/* ── Auth ── */

export const api = {
  auth: {
    google: (credential: string) =>
      request<TokenResponse>("/auth/google", {
        method: "POST",
        body: JSON.stringify({ credential }),
      }),
    signup: (email: string, password: string, name: string) =>
      request<TokenResponse>("/auth/signup", {
        method: "POST",
        body: JSON.stringify({ email, password, name }),
      }),
    login: (email: string, password: string) =>
      request<TokenResponse>("/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      }),
    me: () => request<User>("/auth/me"),

    // Phone auth (Twilio WhatsApp OTP)
    sendOTP: (phone: string, channel: "sms" | "whatsapp" = "sms") =>
      request<{ status: string; message: string }>("/auth/phone/send-otp", {
        method: "POST",
        body: JSON.stringify({ phone, channel }),
      }),
    verifyOTP: (phone: string, code: string) =>
      request<TokenResponse>("/auth/phone/verify-otp", {
        method: "POST",
        body: JSON.stringify({ phone, code }),
      }),
    setName: (name: string, email?: string, password?: string) =>
      request<User>("/auth/phone/set-name", {
        method: "POST",
        body: JSON.stringify({
          name,
          email: email || undefined,
          password: password || undefined,
        }),
      }),
  },

  /* ── Scans ── */

  scans: {
    list: () => request<ScanListItem[]>("/scans"),
    get: (id: string) => request<Scan>(`/scans/${encodeURIComponent(id)}`),
    create: (
      scan_mode: string,
      rooms: { name: string; emoji: string; is_custom: boolean }[],
    ) =>
      request<Scan>("/scans", {
        method: "POST",
        body: JSON.stringify({ scan_mode, rooms }),
      }),
    report: (id: string) =>
      request<SurveyReport>(`/scans/${encodeURIComponent(id)}/report`),
    delete: (id: string) =>
      request<void>(`/scans/${encodeURIComponent(id)}`, { method: "DELETE" }),
    bulkDelete: (scanIds: string[]) =>
      request<void>("/scans/bulk-delete", {
        method: "POST",
        body: JSON.stringify({ scan_ids: scanIds }),
      }),
    submit: (id: string) =>
      request<Scan>(`/scans/${encodeURIComponent(id)}/submit`, {
        method: "POST",
      }),
    uploadVideo: (scanId: string, file: File, roomName?: string) => {
      const form = new FormData();
      form.append("file", file);
      if (roomName) form.append("room_name", roomName);
      return request<{ id: string; filename: string; size_bytes: number }>(
        `/scans/${encodeURIComponent(scanId)}/videos`,
        { method: "POST", body: form, signal: AbortSignal.timeout(300000) },
      );
    },
  },

  /* ── Inventory ── */

  inventory: {
    get: (scanId: string) =>
      request<Inventory>(`/scans/${encodeURIComponent(scanId)}/inventory`),
    addItem: (
      scanId: string,
      item: { name: string; count: number; room_name: string },
    ) =>
      request<InventoryItem>(`/scans/${encodeURIComponent(scanId)}/inventory`, {
        method: "POST",
        body: JSON.stringify(item),
      }),
    updateItem: (
      scanId: string,
      itemId: string,
      updates: Partial<InventoryItem>,
    ) =>
      request<InventoryItem>(
        `/scans/${encodeURIComponent(scanId)}/inventory/${encodeURIComponent(itemId)}`,
        { method: "PATCH", body: JSON.stringify(updates) },
      ),
    deleteItem: (scanId: string, itemId: string) =>
      request<void>(
        `/scans/${encodeURIComponent(scanId)}/inventory/${encodeURIComponent(itemId)}`,
        { method: "DELETE" },
      ),
    save: (scanId: string) =>
      request<Scan>(`/scans/${encodeURIComponent(scanId)}/inventory/save`, {
        method: "POST",
      }),
  },

  /* ── Push Notifications ── */

  push: {
    getVapidKey: () =>
      request<{ public_key: string }>("/push/vapid-public-key"),
    subscribe: (subscription: PushSubscriptionJSON) =>
      request<{ status: string }>("/push/subscribe", {
        method: "POST",
        body: JSON.stringify({
          endpoint: subscription.endpoint,
          keys: subscription.keys,
        }),
      }),
    unsubscribe: (subscription: PushSubscriptionJSON) =>
      request<{ status: string }>("/push/unsubscribe", {
        method: "POST",
        body: JSON.stringify({
          endpoint: subscription.endpoint,
          keys: subscription.keys,
        }),
      }),
  },
};
