/* ── Auth store using Zustand ── */

import { create } from "zustand";
import type { User } from "../types";
import { api } from "../services/api";
import { useNotificationStore } from "./notificationStore";

interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  needsName: boolean; // true when phone user needs to complete profile

  setToken: (token: string) => void;
  loadUser: () => Promise<void>;
  loginWithGoogle: (credential: string) => Promise<void>;
  loginWithEmail: (email: string, password: string) => Promise<void>;
  loginWithPhone: (phone: string, code: string) => Promise<void>;
  completeProfile: (
    name: string,
    email?: string,
    password?: string,
  ) => Promise<void>;
  signup: (email: string, password: string, name: string) => Promise<void>;
  loginDemo: () => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  token: localStorage.getItem("bb_token"),
  isLoading: true,
  isAuthenticated: false,
  needsName: false,

  setToken: (token: string) => {
    localStorage.setItem("bb_token", token);
    set({ token, isAuthenticated: true });
  },

  loadUser: async () => {
    const token = get().token;
    if (!token) {
      set({ isLoading: false, isAuthenticated: false });
      return;
    }
    // Demo mode — skip API call
    if (token === "demo") {
      set({
        user: {
          id: "demo",
          email: "demo@bundlebox.app",
          name: "Bharat Bhaskar",
        },
        isLoading: false,
        isAuthenticated: true,
      });
      return;
    }
    try {
      const user = await api.auth.me();
      const needsName = !user.name || !user.email;
      set({ user, isLoading: false, isAuthenticated: true, needsName });
    } catch {
      localStorage.removeItem("bb_token");
      set({
        user: null,
        token: null,
        isLoading: false,
        isAuthenticated: false,
      });
    }
  },

  loginWithGoogle: async (credential: string) => {
    const res = await api.auth.google(credential);
    get().setToken(res.access_token);
    await get().loadUser();
    useNotificationStore.getState().add({
      type: "welcome",
      title: "Welcome!",
      message: "Signed in with Google. Ready to scan!",
    });
  },

  loginWithPhone: async (phone: string, code: string) => {
    const res = await api.auth.verifyOTP(phone, code);
    get().setToken(res.access_token);
    await get().loadUser();
    // needsName is now set atomically inside loadUser
    const state = get();
    if (!state.needsName) {
      useNotificationStore.getState().add({
        type: "welcome",
        title: "Welcome back!",
        message: "Signed in with phone. Ready to scan!",
      });
    }
  },

  completeProfile: async (name: string, email?: string, password?: string) => {
    const user = await api.auth.setName(name, email, password);
    set({ user, needsName: false });
    useNotificationStore.getState().add({
      type: "welcome",
      title: `Welcome, ${name}!`,
      message: "Your account is set up. Start your first scan!",
    });
  },

  loginWithEmail: async (email: string, password: string) => {
    const res = await api.auth.login(email, password);
    get().setToken(res.access_token);
    await get().loadUser();
    useNotificationStore.getState().add({
      type: "welcome",
      title: "Welcome back!",
      message: "Ready to scan? Tap + to start a new inventory.",
    });
  },

  signup: async (email: string, password: string, name: string) => {
    const res = await api.auth.signup(email, password, name);
    get().setToken(res.access_token);
    await get().loadUser();
    useNotificationStore.getState().add({
      type: "welcome",
      title: "Welcome to BundleBox!",
      message: "Start your first scan — tap + to begin.",
    });
  },

  loginDemo: () => {
    const demoUser: User = {
      id: "demo",
      email: "demo@bundlebox.app",
      name: "Bharat Bhaskar",
    };
    localStorage.setItem("bb_token", "demo");
    set({
      user: demoUser,
      token: "demo",
      isLoading: false,
      isAuthenticated: true,
    });
    useNotificationStore.getState().add({
      type: "welcome",
      title: "Welcome to BundleBox!",
      message: "You're in demo mode. Start your first scan!",
    });
  },

  logout: () => {
    localStorage.removeItem("bb_token");
    set({ user: null, token: null, isAuthenticated: false });
  },
}));
