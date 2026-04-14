/* ── Camera Page — record or upload video ── */

import { useRef, useState, useCallback, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api } from "../services/api";
import type { Scan, Room } from "../types";

type RecState = "idle" | "recording" | "paused" | "done";

export default function CameraPage() {
  const { id: scanId } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const videoRef = useRef<HTMLVideoElement>(null);
  const reviewVideoRef = useRef<HTMLVideoElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [state, setState] = useState<RecState>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [facingMode, setFacingMode] = useState<"environment" | "user">(
    "environment",
  );
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [blob, setBlob] = useState<Blob | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Room-by-room state
  const [scan, setScan] = useState<Scan | null>(null);
  const [rooms, setRooms] = useState<Room[]>([]);
  const [currentRoomIdx, setCurrentRoomIdx] = useState(0);
  const isRoomByRoom = scan?.scan_mode === "room_by_room" && rooms.length > 0;
  const currentRoom = isRoomByRoom ? rooms[currentRoomIdx] : null;

  const timerRef = useRef<ReturnType<typeof setInterval> | undefined>(
    undefined,
  );

  /* ── load scan data (for room-by-room) ── */
  useEffect(() => {
    if (!scanId) return;
    api.scans.get(scanId).then((s) => {
      setScan(s);
      if (s.scan_mode === "room_by_room" && s.rooms.length > 0) {
        const sorted = [...s.rooms].sort((a, b) => a.order - b.order);
        setRooms(sorted);
      }
    });
  }, [scanId]);

  /* ── start camera ── */
  const startCamera = useCallback(async () => {
    setCameraError(null);

    // Check for secure context (getUserMedia requires HTTPS or localhost)
    if (!window.isSecureContext) {
      setCameraError(
        "Camera requires a secure connection (HTTPS). You can still upload a video from your device.",
      );
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError(
        "Camera not supported on this browser. You can still upload a video from your device.",
      );
      return;
    }

    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: true,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = s;
      }
      setStream(s);
    } catch (err) {
      let msg =
        "Could not access camera. You can still upload a video from your device.";
      if (err instanceof DOMException) {
        switch (err.name) {
          case "NotAllowedError":
            msg =
              "Camera access was denied. Please go to your browser settings, allow camera & microphone for this site, then refresh the page. Or upload a video instead.";
            break;
          case "NotFoundError":
            msg =
              "No camera found on this device. You can still upload a video from your device.";
            break;
          case "NotReadableError":
            msg =
              "Camera is in use by another app. Close other apps using the camera and try again, or upload a video instead.";
            break;
        }
      }
      setCameraError(msg);
    }
  }, [facingMode]);

  useEffect(() => {
    startCamera();
    return () => {
      stream?.getTracks().forEach((t) => t.stop());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [facingMode]);

  /* ── recording controls ── */
  const startRecording = () => {
    if (!stream) {
      setCameraError(
        "Camera is not active. Please grant camera permission and try again, or upload a video instead.",
      );
      return;
    }
    chunksRef.current = [];

    // Pick a supported mimeType (Safari doesn't support webm)
    const codecs = [
      "video/webm;codecs=vp9,opus",
      "video/webm;codecs=vp8,opus",
      "video/webm",
      "video/mp4",
    ];
    const mimeType = codecs.find((c) => MediaRecorder.isTypeSupported(c)) ?? "";
    const mr = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    mr.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    mr.onstop = () => {
      const recType = mr.mimeType || "video/webm";
      const b = new Blob(chunksRef.current, { type: recType });
      setBlob(b);
      setState("done");
    };
    mr.start(1000); // 1s chunks
    recorderRef.current = mr;
    setState("recording");
    setElapsed(0);
    timerRef.current = setInterval(() => setElapsed((t) => t + 1), 1000);
  };

  const pauseRecording = () => {
    recorderRef.current?.pause();
    setState("paused");
    clearInterval(timerRef.current);
  };

  const resumeRecording = () => {
    recorderRef.current?.resume();
    setState("recording");
    timerRef.current = setInterval(() => setElapsed((t) => t + 1), 1000);
  };

  const stopRecording = () => {
    recorderRef.current?.stop();
    clearInterval(timerRef.current);
    stream?.getTracks().forEach((t) => t.stop());
  };

  const retake = () => {
    setBlob(null);
    setUploadedFile(null);
    setState("idle");
    setElapsed(0);
    startCamera();
  };

  /* ── handle file upload from device ── */
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    // Stop camera if active
    stream?.getTracks().forEach((t) => t.stop());
    setUploadedFile(file);
    setBlob(null);
    setState("done");
  };

  /* ── upload & proceed ── */
  const accept = async () => {
    if (!scanId) return;
    const fileToUpload =
      uploadedFile ||
      (blob
        ? new File(
            [blob],
            blob.type.includes("mp4") ? "walkthrough.mp4" : "walkthrough.webm",
            { type: blob.type },
          )
        : null);
    if (!fileToUpload) return;
    setUploading(true);
    setError(null);
    // Pause video playback while uploading
    reviewVideoRef.current?.pause();
    try {
      const roomName = currentRoom?.name;
      await api.scans.uploadVideo(scanId, fileToUpload, roomName);

      if (isRoomByRoom) {
        // Mark this room as uploaded locally
        setRooms((prev) =>
          prev.map((r, i) =>
            i === currentRoomIdx ? { ...r, has_video: true, status: "processing" as const } : r,
          ),
        );

        if (currentRoomIdx < rooms.length - 1) {
          // More rooms to go — reset for next room
          setCurrentRoomIdx((i) => i + 1);
          setBlob(null);
          setUploadedFile(null);
          setState("idle");
          setUploading(false);
          startCamera();
          return;
        }
        // All rooms uploaded — go to processing
        navigate(`/scans/${scanId}/processing`);
      } else {
        // All-at-once: submit and go to processing
        await api.scans.submit(scanId);
        navigate(`/scans/${scanId}/processing`);
      }
    } catch (err) {
      console.error("Upload failed:", err);
      setError(err instanceof Error ? err.message : "Upload failed");
      setUploading(false);
    }
  };

  const flipCamera = () => {
    stream?.getTracks().forEach((t) => t.stop());
    setFacingMode((f) => (f === "environment" ? "user" : "environment"));
  };

  const fmt = (s: number) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  return (
    <div className="fixed inset-0 bg-black flex flex-col">
      {/* Hidden file input for video upload */}
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        onChange={handleFileSelect}
        className="hidden"
      />

      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between p-4 safe-top">
        <button
          onClick={() => navigate(-1)}
          className="w-9 h-9 flex items-center justify-center bg-black/40 rounded-full text-white text-sm backdrop-blur"
        >
          ←
        </button>
        <span className="px-3 py-1.5 bg-black/40 backdrop-blur rounded-full text-white text-xs font-semibold">
          {isRoomByRoom
            ? `${currentRoom?.emoji} ${currentRoom?.name} · ${currentRoomIdx + 1}/${rooms.length}`
            : "Walkthrough Video"}
        </span>
        <div className="w-9" />
      </div>

      {/* Room progress bar (room-by-room mode) */}
      {isRoomByRoom && (
        <div className="absolute top-16 left-0 right-0 z-20 px-4 safe-top">
          <div className="flex gap-1.5">
            {rooms.map((r, i) => (
              <div
                key={r.id}
                className={`flex-1 h-1.5 rounded-full transition-all ${
                  i < currentRoomIdx
                    ? "bg-green-400"
                    : i === currentRoomIdx
                      ? "bg-white"
                      : "bg-white/30"
                }`}
              />
            ))}
          </div>
          <div className="flex justify-between mt-1.5">
            {rooms.map((r, i) => (
              <span
                key={r.id}
                className={`text-[9px] font-medium ${
                  i <= currentRoomIdx ? "text-white" : "text-white/40"
                }`}
              >
                {r.emoji}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Video preview */}
      {state !== "done" ? (
        <>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="flex-1 object-cover"
          />
          {cameraError && (
            <div className="absolute inset-0 flex items-center justify-center z-10 px-6">
              <div className="bg-gray-900/90 backdrop-blur-xl rounded-2xl p-6 max-w-sm text-center">
                <div className="text-4xl mb-3">📷</div>
                <p className="text-white text-sm leading-relaxed">
                  {cameraError}
                </p>
              </div>
            </div>
          )}
        </>
      ) : uploadedFile ? (
        <video
          ref={reviewVideoRef}
          src={URL.createObjectURL(uploadedFile)}
          controls
          autoPlay
          playsInline
          className="flex-1 object-contain bg-black"
        />
      ) : blob ? (
        <video
          ref={reviewVideoRef}
          src={URL.createObjectURL(blob)}
          controls
          autoPlay
          playsInline
          className="flex-1 object-contain bg-black"
        />
      ) : null}

      {/* Full-screen upload overlay */}
      {uploading && (
        <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-black/80 backdrop-blur-sm">
          <div className="w-16 h-16 border-4 border-white/30 border-t-white rounded-full animate-spin" />
          <p className="mt-4 text-white text-lg font-semibold">
            Uploading video…
          </p>
          <p className="mt-1 text-white/60 text-sm">Please wait</p>
        </div>
      )}

      {/* Timer */}
      {(state === "recording" || state === "paused") && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-20">
          <span
            className={`px-4 py-1.5 rounded-full text-white text-sm font-mono font-bold backdrop-blur ${
              state === "recording" ? "bg-red-600/80" : "bg-yellow-500/80"
            }`}
          >
            {state === "paused" && "⏸ "}
            {fmt(elapsed)}
          </span>
        </div>
      )}

      {/* Controls */}
      <div className="absolute bottom-0 left-0 right-0 z-20 px-6 pb-10 pt-4 safe-bottom">
        {state === "idle" && (
          <div className="flex flex-col items-center gap-5">
            <div className="flex items-center justify-center gap-8">
              <button
                onClick={flipCamera}
                className="w-12 h-12 flex items-center justify-center bg-white/20 rounded-full text-white backdrop-blur"
              >
                🔄
              </button>
              <button
                onClick={startRecording}
                className="w-[72px] h-[72px] rounded-full border-4 border-white flex items-center justify-center"
              >
                <div className="w-14 h-14 bg-red-500 rounded-full" />
              </button>
              <div className="w-12" />
            </div>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="w-full py-3 bg-white/20 backdrop-blur rounded-2xl text-white text-sm font-semibold flex items-center justify-center gap-2"
            >
              📁 Upload Video Instead
            </button>
          </div>
        )}

        {(state === "recording" || state === "paused") && (
          <div className="flex items-center justify-center gap-8">
            <button
              onClick={state === "recording" ? pauseRecording : resumeRecording}
              className="w-12 h-12 flex items-center justify-center bg-white/20 rounded-full text-white text-xl backdrop-blur"
            >
              {state === "recording" ? "⏸" : "▶"}
            </button>
            <button
              onClick={stopRecording}
              className="w-[72px] h-[72px] rounded-full border-4 border-white flex items-center justify-center"
            >
              <div className="w-8 h-8 bg-red-500 rounded-lg" />
            </button>
            <div className="w-12" />
          </div>
        )}

        {state === "done" && (
          <div className="flex gap-3">
            <button
              onClick={retake}
              className="flex-1 py-4 bg-white/20 text-white rounded-2xl text-sm font-bold backdrop-blur active:scale-[0.97] transition-transform"
            >
              Retake
            </button>
            <button
              onClick={accept}
              disabled={uploading}
              className="flex-1 py-4 bg-white text-gray-900 rounded-2xl text-sm font-bold disabled:opacity-50 active:scale-[0.97] transition-transform shadow-lg"
            >
              {uploading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-4 h-4 border-2 border-gray-400 border-t-gray-900 rounded-full animate-spin" />
                  Uploading…
                </span>
              ) : isRoomByRoom && currentRoomIdx < rooms.length - 1 ? (
                `Upload & Next Room →`
              ) : (
                "Accept & Process"
              )}
            </button>
            {error && (
              <p className="text-red-400 text-xs text-center mt-2">{error}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
