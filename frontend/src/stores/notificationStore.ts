/* ── Notification store using Zustand ── */

import { create } from "zustand";

export type NotificationType =
  | "scan_complete"
  | "scan_failed"
  | "scan_progress"
  | "welcome"
  | "pwa_update";

export interface AppNotification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  read: boolean;
  timestamp: number; // epoch ms
  scanId?: string;
}

interface NotificationState {
  notifications: AppNotification[];
  unreadCount: number;
  panelOpen: boolean;

  add: (n: Omit<AppNotification, "id" | "read" | "timestamp">) => void;
  markRead: (id: string) => void;
  markAllRead: () => void;
  remove: (id: string) => void;
  clear: () => void;
  togglePanel: () => void;
  closePanel: () => void;
}

let nextId = 1;

export const useNotificationStore = create<NotificationState>((set, _get) => ({
  notifications: [],
  unreadCount: 0,
  panelOpen: false,

  add: (n) => {
    const notification: AppNotification = {
      ...n,
      id: `notif-${nextId++}`,
      read: false,
      timestamp: Date.now(),
    };
    set((s) => ({
      notifications: [notification, ...s.notifications].slice(0, 50), // keep max 50
      unreadCount: s.unreadCount + 1,
    }));
  },

  markRead: (id) =>
    set((s) => {
      const notifications = s.notifications.map((n) =>
        n.id === id ? { ...n, read: true } : n,
      );
      return {
        notifications,
        unreadCount: notifications.filter((n) => !n.read).length,
      };
    }),

  markAllRead: () =>
    set((s) => ({
      notifications: s.notifications.map((n) => ({ ...n, read: true })),
      unreadCount: 0,
    })),

  remove: (id) =>
    set((s) => {
      const notifications = s.notifications.filter((n) => n.id !== id);
      return {
        notifications,
        unreadCount: notifications.filter((n) => !n.read).length,
      };
    }),

  clear: () => set({ notifications: [], unreadCount: 0 }),

  togglePanel: () => set((s) => ({ panelOpen: !s.panelOpen })),
  closePanel: () => set({ panelOpen: false }),
}));
