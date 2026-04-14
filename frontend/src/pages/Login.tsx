/* ── Login Page — Google, Phone (WhatsApp OTP), Email ── */

import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useAuthStore } from "../stores/authStore";
import { api } from "../services/api";

declare global {
  interface Window {
    google?: {
      accounts: {
        id: {
          initialize: (config: Record<string, unknown>) => void;
          renderButton: (
            el: HTMLElement,
            config: Record<string, unknown>,
          ) => void;
        };
      };
    };
  }
}

type AuthTab = "phone" | "email";

export default function LoginPage() {
  const navigate = useNavigate();
  const {
    loginWithEmail,
    loginWithGoogle,
    loginWithPhone,
    completeProfile,
    signup,
    needsName,
    isAuthenticated,
  } = useAuthStore();

  const [tab, setTab] = useState<AuthTab>("phone");
  const [isSignup, setIsSignup] = useState(false);

  // Email fields
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [signupName, setSignupName] = useState("");

  // Phone fields
  const [phone, setPhone] = useState("");
  const [otpChannel, setOtpChannel] = useState<"sms" | "whatsapp">("sms");
  const [otpSent, setOtpSent] = useState(false);
  const [otp, setOtp] = useState("");
  const [cooldown, setCooldown] = useState(0);

  // Name capture (after phone verify)
  const [captureName, setCaptureName] = useState("");
  const [captureEmail, setCaptureEmail] = useState("");
  const [capturePassword, setCapturePassword] = useState("");

  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const googleBtnRef = useRef<HTMLDivElement>(null);

  // Redirect if already authenticated (and not needing name)
  useEffect(() => {
    if (isAuthenticated && !needsName) navigate("/");
  }, [isAuthenticated, needsName, navigate]);

  // Google Identity Services
  const handleGoogleResponse = useCallback(
    async (response: { credential: string }) => {
      setError("");
      setLoading(true);
      try {
        await loginWithGoogle(response.credential);
        navigate("/");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Google sign-in failed");
      } finally {
        setLoading(false);
      }
    },
    [loginWithGoogle, navigate],
  );

  useEffect(() => {
    const initGoogle = () => {
      if (!window.google || !googleBtnRef.current) return;
      window.google.accounts.id.initialize({
        client_id: import.meta.env.VITE_GOOGLE_CLIENT_ID || "",
        callback: handleGoogleResponse,
      });
      window.google.accounts.id.renderButton(googleBtnRef.current, {
        theme: "outline",
        size: "large",
        width: 340,
        text: "continue_with",
        shape: "pill",
      });
    };
    // Script may load after component mount
    if (window.google) {
      initGoogle();
    } else {
      const timer = setInterval(() => {
        if (window.google) {
          clearInterval(timer);
          initGoogle();
        }
      }, 200);
      return () => clearInterval(timer);
    }
  }, [handleGoogleResponse]);

  // OTP cooldown timer
  useEffect(() => {
    if (cooldown <= 0) return;
    const t = setTimeout(() => setCooldown(cooldown - 1), 1000);
    return () => clearTimeout(t);
  }, [cooldown]);

  // ── Handlers ──

  const handleSendOTP = async () => {
    setError("");
    setLoading(true);
    try {
      await api.auth.sendOTP(phone, otpChannel);
      setOtpSent(true);
      setCooldown(60);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send OTP");
    } finally {
      setLoading(false);
    }
  };

  const handleVerifyOTP = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await loginWithPhone(phone, otp);
      // If needsName is set, stay on this page — name capture will show
      if (!useAuthStore.getState().needsName) navigate("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Invalid OTP");
    } finally {
      setLoading(false);
    }
  };

  const handleSetName = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await completeProfile(
        captureName,
        captureEmail || undefined,
        capturePassword || undefined,
      );
      navigate("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save profile");
    } finally {
      setLoading(false);
    }
  };

  const handleEmailSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (isSignup) {
        await signup(email, password, signupName);
      } else {
        await loginWithEmail(email, password);
      }
      navigate("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  // ── Profile Capture Screen ──
  if (needsName) {
    return (
      <div className="app-shell flex flex-col min-h-screen bg-white">
        <div className="flex-1 flex flex-col items-center justify-center px-8">
          <div className="w-[72px] h-[72px] rounded-[18px] flex items-center justify-center mb-6 shadow-lg overflow-hidden">
            <img
              src="/icons/icon-128.png"
              alt="BundleBox"
              className="w-full h-full object-contain"
            />
          </div>
          <h1 className="text-2xl font-extrabold mb-2">
            Complete Your Profile
          </h1>
          <p className="text-sm text-gray-400 mb-8 text-center">
            Set up your account so you can log in with email next time
          </p>
          <form onSubmit={handleSetName} className="w-full max-w-[340px]">
            <label className="block text-xs font-semibold text-gray-500 mb-1.5">
              Name <span className="text-red-400">*</span>
            </label>
            <input
              type="text"
              value={captureName}
              onChange={(e) => setCaptureName(e.target.value)}
              className="w-full px-3.5 py-3 border-[1.5px] border-gray-200 rounded-xl text-sm focus:outline-none focus:border-gray-900 transition-colors mb-4"
              placeholder="Your name"
              required
              autoFocus
            />
            <label className="block text-xs font-semibold text-gray-500 mb-1.5">
              Email <span className="text-gray-300">(recommended)</span>
            </label>
            <input
              type="email"
              value={captureEmail}
              onChange={(e) => setCaptureEmail(e.target.value)}
              className="w-full px-3.5 py-3 border-[1.5px] border-gray-200 rounded-xl text-sm focus:outline-none focus:border-gray-900 transition-colors mb-4"
              placeholder="you@email.com"
            />
            {captureEmail && (
              <>
                <label className="block text-xs font-semibold text-gray-500 mb-1.5">
                  Password{" "}
                  <span className="text-gray-300">(min 8 characters)</span>
                </label>
                <input
                  type="password"
                  value={capturePassword}
                  onChange={(e) => setCapturePassword(e.target.value)}
                  className="w-full px-3.5 py-3 border-[1.5px] border-gray-200 rounded-xl text-sm focus:outline-none focus:border-gray-900 transition-colors mb-4"
                  placeholder="••••••••"
                  minLength={8}
                  required
                />
              </>
            )}
            {error && (
              <p className="text-xs text-brand-red font-medium mb-3">{error}</p>
            )}
            <button
              type="submit"
              disabled={
                loading ||
                !captureName.trim() ||
                (!!captureEmail && capturePassword.length < 8)
              }
              className="w-full py-3.5 bg-gray-900 text-white rounded-xl text-sm font-semibold hover:bg-black transition-colors disabled:opacity-50"
            >
              {loading ? "Saving…" : "Continue"}
            </button>
          </form>
        </div>
      </div>
    );
  }

  // ── Main Login Screen ──
  return (
    <div className="app-shell flex flex-col min-h-screen bg-white">
      <span className="absolute top-4 right-4 text-[10px] font-bold text-gray-300 tracking-widest">
        PWA v1.0
      </span>

      <div className="flex-1 flex flex-col items-center justify-center px-8 pt-10 pb-5">
        {/* Logo */}
        <div className="w-[72px] h-[72px] rounded-[18px] flex items-center justify-center mb-6 shadow-lg overflow-hidden">
          <img
            src="/icons/icon-128.png"
            alt="BundleBox"
            className="w-full h-full object-contain"
          />
        </div>
        <h1 className="text-[28px] font-extrabold tracking-tight mb-1">
          Bundle<span className="text-brand-red">Box</span>
        </h1>
        <p className="text-sm text-gray-400 mb-7">
          Smart room inventory in minutes
        </p>

        <div className="w-full max-w-[340px]">
          {/* Google Sign-In (rendered by Google Identity Services) */}
          <div ref={googleBtnRef} className="flex justify-center mb-4" />

          <div className="flex items-center gap-3 mb-5">
            <div className="flex-1 h-px bg-gray-200" />
            <span className="text-xs text-gray-400 font-medium">or</span>
            <div className="flex-1 h-px bg-gray-200" />
          </div>

          {/* Tab switcher: Phone / Email */}
          <div className="flex rounded-xl bg-gray-100 p-1 mb-5">
            <button
              type="button"
              onClick={() => {
                setTab("phone");
                setError("");
              }}
              className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-colors ${
                tab === "phone"
                  ? "bg-white shadow text-gray-900"
                  : "text-gray-400"
              }`}
            >
              📱 Phone
            </button>
            <button
              type="button"
              onClick={() => {
                setTab("email");
                setError("");
              }}
              className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-colors ${
                tab === "email"
                  ? "bg-white shadow text-gray-900"
                  : "text-gray-400"
              }`}
            >
              ✉️ Email
            </button>
          </div>

          {/* ── Phone Tab ── */}
          {tab === "phone" && (
            <>
              {!otpSent ? (
                <div>
                  <label className="block text-xs font-semibold text-gray-500 mb-1.5">
                    Phone number (with country code)
                  </label>
                  <input
                    type="tel"
                    value={phone}
                    onChange={(e) => setPhone(e.target.value)}
                    className="w-full px-3.5 py-3 border-[1.5px] border-gray-200 rounded-xl text-sm focus:outline-none focus:border-gray-900 transition-colors mb-3"
                    placeholder="+91 98765 43210"
                  />

                  {/* Channel selector */}
                  <label className="block text-xs font-semibold text-gray-500 mb-1.5">
                    Send code via
                  </label>
                  <div className="flex gap-2 mb-4">
                    <button
                      type="button"
                      onClick={() => setOtpChannel("sms")}
                      className={`flex-1 py-2.5 rounded-xl text-xs font-semibold border-[1.5px] transition-all ${
                        otpChannel === "sms"
                          ? "border-gray-900 bg-gray-900 text-white"
                          : "border-gray-200 text-gray-500 hover:border-gray-300"
                      }`}
                    >
                      📱 SMS
                    </button>
                    <button
                      type="button"
                      onClick={() => setOtpChannel("whatsapp")}
                      className={`flex-1 py-2.5 rounded-xl text-xs font-semibold border-[1.5px] transition-all ${
                        otpChannel === "whatsapp"
                          ? "border-green-600 bg-green-600 text-white"
                          : "border-gray-200 text-gray-500 hover:border-gray-300"
                      }`}
                    >
                      💬 WhatsApp
                    </button>
                  </div>

                  <button
                    type="button"
                    onClick={handleSendOTP}
                    disabled={loading || phone.length < 10}
                    className="w-full py-3.5 bg-gray-900 text-white rounded-xl text-sm font-semibold hover:bg-black transition-colors disabled:opacity-50 mb-3"
                  >
                    {loading ? "Sending…" : "Send OTP"}
                  </button>
                </div>
              ) : (
                <form onSubmit={handleVerifyOTP}>
                  <p className="text-xs text-gray-500 mb-3">
                    OTP sent to <strong>{phone}</strong> via{" "}
                    {otpChannel === "whatsapp" ? "WhatsApp" : "SMS"}
                  </p>
                  <label className="block text-xs font-semibold text-gray-500 mb-1.5">
                    Enter 6-digit code
                  </label>
                  <input
                    type="text"
                    inputMode="numeric"
                    maxLength={6}
                    value={otp}
                    onChange={(e) => setOtp(e.target.value.replace(/\D/g, ""))}
                    className="w-full px-3.5 py-3 border-[1.5px] border-gray-200 rounded-xl text-sm text-center tracking-[0.5em] font-mono focus:outline-none focus:border-gray-900 transition-colors mb-4"
                    placeholder="000000"
                    autoFocus
                  />
                  <button
                    type="submit"
                    disabled={loading || otp.length < 4}
                    className="w-full py-3.5 bg-gray-900 text-white rounded-xl text-sm font-semibold hover:bg-black transition-colors disabled:opacity-50 mb-3"
                  >
                    {loading ? "Verifying…" : "Verify & Sign In"}
                  </button>
                  <div className="flex justify-between items-center">
                    <button
                      type="button"
                      onClick={() => {
                        setOtpSent(false);
                        setOtp("");
                        setError("");
                      }}
                      className="text-xs text-gray-400 underline"
                    >
                      Change number
                    </button>
                    <button
                      type="button"
                      onClick={handleSendOTP}
                      disabled={cooldown > 0 || loading}
                      className="text-xs text-brand-red font-semibold disabled:text-gray-300"
                    >
                      {cooldown > 0 ? `Resend in ${cooldown}s` : "Resend OTP"}
                    </button>
                  </div>
                </form>
              )}
            </>
          )}

          {/* ── Email Tab ── */}
          {tab === "email" && (
            <form onSubmit={handleEmailSubmit}>
              {isSignup && (
                <div className="mb-4">
                  <label className="block text-xs font-semibold text-gray-500 mb-1.5">
                    Name
                  </label>
                  <input
                    type="text"
                    value={signupName}
                    onChange={(e) => setSignupName(e.target.value)}
                    className="w-full px-3.5 py-3 border-[1.5px] border-gray-200 rounded-xl text-sm focus:outline-none focus:border-gray-900 transition-colors"
                    placeholder="Your name"
                    required
                  />
                </div>
              )}
              <div className="mb-4">
                <label className="block text-xs font-semibold text-gray-500 mb-1.5">
                  Email address
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full px-3.5 py-3 border-[1.5px] border-gray-200 rounded-xl text-sm focus:outline-none focus:border-gray-900 transition-colors"
                  placeholder="you@email.com"
                  required
                />
              </div>
              <div className="mb-4">
                <label className="block text-xs font-semibold text-gray-500 mb-1.5">
                  Password
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-3.5 py-3 border-[1.5px] border-gray-200 rounded-xl text-sm focus:outline-none focus:border-gray-900 transition-colors"
                  placeholder="••••••••"
                  required
                  minLength={8}
                />
              </div>
              <button
                type="submit"
                disabled={loading}
                className="w-full py-3.5 bg-gray-900 text-white rounded-xl text-sm font-semibold hover:bg-black transition-colors disabled:opacity-50"
              >
                {loading ? "Loading…" : isSignup ? "Sign Up" : "Sign In"}
              </button>
            </form>
          )}

          {/* Error */}
          {error && (
            <p className="text-xs text-brand-red font-medium mt-3">{error}</p>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="px-8 pb-8 text-center">
        {tab === "email" && (
          <p className="text-[13px] text-gray-400">
            {isSignup ? "Already have an account?" : "Don't have an account?"}{" "}
            <button
              onClick={() => {
                setIsSignup(!isSignup);
                setError("");
              }}
              className="text-gray-900 font-semibold underline underline-offset-2"
            >
              {isSignup ? "Sign In" : "Sign Up"}
            </button>
          </p>
        )}
        <p className="text-[11px] text-gray-300 mt-3">
          By continuing, you agree to the Terms of Service and Privacy Policy.
        </p>
      </div>
    </div>
  );
}
