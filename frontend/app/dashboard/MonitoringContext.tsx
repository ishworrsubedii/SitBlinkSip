"use client";

import * as React from "react";
import { toast } from "sonner";
import { API_URL } from "@/lib/constants";
import { cameraService, type CameraSession } from "./services/cameraService";
import { canFireAlert, playAlertTone } from "@/lib/alertSounds";

const STORAGE_KEYS = {
  CAMERA_SESSION: "camera_session",
  MONITOR_SETTINGS: "monitor_settings",
  CAMERA_ID: "selected_camera",
  IS_STREAMING: "is_streaming",
  EYE_BLINK_DATA: "eye_blink_data",
  POSTURE_DATA: "posture_data",
};

const saveToStorage = (key: string, value: any) => {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.error(`Failed to save ${key} to storage:`, error);
  }
};

const getFromStorage = (key: string) => {
  try {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : null;
  } catch (error) {
    console.error(`Failed to get ${key} from storage:`, error);
    return null;
  }
};

interface MonitoringContextValue {
  isCameraActive: boolean;
  isStreaming: boolean;
  isLoading: boolean;
  stream: MediaStream | null;
  selectedCamera: string;
  setSelectedCamera: (id: string) => void;
  cameras: MediaDeviceInfo[];
  monitorPosture: boolean;
  setMonitorPosture: (v: boolean) => void;
  monitorEyeBlink: boolean;
  setMonitorEyeBlink: (v: boolean) => void;
  eyeBlinkFrame: string | null;
  postureFrame: string | null;
  backendStatus: string;
  toggleMonitoring: () => Promise<void>;
}

const MonitoringContext = React.createContext<MonitoringContextValue | null>(null);

export function useMonitoring() {
  const ctx = React.useContext(MonitoringContext);
  if (!ctx) {
    throw new Error("useMonitoring must be used within a MonitoringProvider");
  }
  return ctx;
}

// Owns the camera stream, frame-capture loop, and detection WebSocket at the
// dashboard-layout level (mirrors PersonContext's water-break socket) so that
// navigating between dashboard pages no longer tears monitoring down — only
// unmounting the whole /dashboard subtree does.
export function MonitoringProvider({ children }: { children: React.ReactNode }) {
  const videoRef = React.useRef<HTMLVideoElement>(null);
  const streamRef = React.useRef<MediaStream | null>(null);
  const [stream, setStream] = React.useState<MediaStream | null>(null);
  const [isCameraActive, setIsCameraActive] = React.useState(false);
  const [selectedCamera, setSelectedCamera] = React.useState<string>("");
  const [cameras, setCameras] = React.useState<MediaDeviceInfo[]>([]);

  const [isStreaming, setIsStreaming] = React.useState(false);
  const [monitorPosture, setMonitorPosture] = React.useState(false);
  const [monitorEyeBlink, setMonitorEyeBlink] = React.useState(false);
  const [eyeBlinkFrame, setEyeBlinkFrame] = React.useState<string | null>(null);
  const [postureFrame, setPostureFrame] = React.useState<string | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);

  const [ws, setWs] = React.useState<WebSocket | null>(null);
  const [frameIntervalId, setFrameIntervalId] = React.useState<NodeJS.Timeout | null>(null);

  const badPostureStreakRef = React.useRef(0);
  const lastBlinkAtRef = React.useRef(Date.now());

  const [backendStatus, setBackendStatus] = React.useState("Not Checked");

  const handleFrameAlerts = (data: any) => {
    if (data.posture_data) {
      if (data.posture_data.status === "Bad Posture") {
        badPostureStreakRef.current += 1;
        if (badPostureStreakRef.current >= 5 && canFireAlert("posture", 45000)) {
          toast.error("Bad posture detected — sit up straight!");
          playAlertTone("posture");
        }
      } else {
        badPostureStreakRef.current = 0;
      }
    }

    if (data.eye_blink_data?.blink) {
      lastBlinkAtRef.current = Date.now();
    }
  };

  // Warn if no blink has been detected for over 60 seconds while eye-blink monitoring is active
  React.useEffect(() => {
    if (!isStreaming || !monitorEyeBlink) return;

    lastBlinkAtRef.current = Date.now();
    const interval = setInterval(() => {
      if (Date.now() - lastBlinkAtRef.current > 60000 && canFireAlert("blink", 60000)) {
        toast.error("You haven't blinked in a while — rest your eyes!");
        playAlertTone("blink");
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [isStreaming, monitorEyeBlink]);

  // Get available cameras
  React.useEffect(() => {
    async function getCameras() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter((device) => device.kind === "videoinput");
        setCameras(videoDevices);
        if (videoDevices.length) {
          setSelectedCamera(videoDevices[0].deviceId);
        }
      } catch (error) {
        toast.error("Failed to get camera devices");
      }
    }
    getCameras();
  }, []);

  const checkBackendStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/status`);
      setBackendStatus(response.ok ? "Ready" : "Not Ready");
    } catch (error) {
      setBackendStatus("Not Ready");
    }
  };

  React.useEffect(() => {
    checkBackendStatus();
    const interval = setInterval(checkBackendStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const cleanup = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setStream(null);

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (frameIntervalId) {
      clearInterval(frameIntervalId);
      setFrameIntervalId(null);
    }

    if (ws) {
      ws.close();
      setWs(null);
    }

    setIsCameraActive(false);
    setIsStreaming(false);
    setEyeBlinkFrame(null);
    setPostureFrame(null);

    localStorage.removeItem(STORAGE_KEYS.IS_STREAMING);
    localStorage.removeItem(STORAGE_KEYS.CAMERA_SESSION);
    localStorage.removeItem("camera_active");
  };

  // Cleanup only when the provider itself unmounts (i.e. leaving /dashboard
  // entirely), not when an individual dashboard page unmounts.
  React.useEffect(() => {
    return () => cleanup();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const initializeCamera = async (deviceId: string) => {
    const mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: { exact: deviceId },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });

    streamRef.current = mediaStream;
    setStream(mediaStream);
    if (videoRef.current) {
      videoRef.current.srcObject = mediaStream;
      await videoRef.current.play();
    }
    setIsCameraActive(true);
    saveToStorage("camera_active", true);
    saveToStorage(STORAGE_KEYS.CAMERA_ID, deviceId);
  };

  const startFrameCapture = (websocket: WebSocket) => {
    if (!videoRef.current) return;

    const frameInterval = setInterval(() => {
      if (!videoRef.current || websocket.readyState !== WebSocket.OPEN) return;

      const canvas = document.createElement("canvas");
      canvas.width = 1280;
      canvas.height = 720;
      const ctx = canvas.getContext("2d");

      if (!ctx) return;

      try {
        const videoAspect = videoRef.current.videoWidth / videoRef.current.videoHeight;
        const canvasAspect = canvas.width / canvas.height;
        let drawWidth = canvas.width;
        let drawHeight = canvas.height;
        let offsetX = 0;
        let offsetY = 0;

        if (videoAspect > canvasAspect) {
          drawHeight = canvas.width / videoAspect;
          offsetY = (canvas.height - drawHeight) / 2;
        } else {
          drawWidth = canvas.height * videoAspect;
          offsetX = (canvas.width - drawWidth) / 2;
        }

        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Mirror the captured pixels (not just the CSS preview) so the processed
        // output the backend sends back matches what's shown in the live preview
        // instead of appearing flipped relative to it. Any overlay text the backend
        // draws is applied after this, so it stays right-reading.
        ctx.save();
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(videoRef.current, offsetX, offsetY, drawWidth, drawHeight);
        ctx.restore();

        canvas.toBlob(
          (blob) => {
            if (blob && websocket.readyState === WebSocket.OPEN) {
              websocket.send(blob);
            }
          },
          "image/jpeg",
          0.85
        );
      } catch (error) {
        console.error("Frame capture error:", error);
      }
    }, 100);

    setFrameIntervalId(frameInterval);
  };

  const attachSocketHandlers = (socket: WebSocket) => {
    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.eye_blink_image) {
          setEyeBlinkFrame(data.eye_blink_image);
          saveToStorage(STORAGE_KEYS.EYE_BLINK_DATA, data.eye_blink_image);
        }
        if (data.posture_image) {
          setPostureFrame(data.posture_image);
          saveToStorage(STORAGE_KEYS.POSTURE_DATA, data.posture_image);
        }
        handleFrameAlerts(data);
      } catch (error) {
        console.error("Error processing message:", error);
      }
    };

    socket.onclose = () => {
      setIsStreaming(false);
    };

    socket.onerror = () => {
      toast.error("Connection error occurred");
      setIsStreaming(false);
    };
  };

  // Single action: turns the camera on AND starts whichever detectors are toggled
  // on (posture only, eye-blink only, or both) in one step — no separate preview.
  const toggleMonitoring = async () => {
    if (!selectedCamera) {
      toast.error("Please select a camera");
      return;
    }
    if (!monitorPosture && !monitorEyeBlink) {
      toast.error("Please enable Posture and/or Eye Blink detection first");
      return;
    }

    if (isStreaming) {
      setIsLoading(true);
      try {
        await cameraService.stopBackgroundMonitoring();
      } catch (error) {
        console.error("Failed to stop monitoring:", error);
      } finally {
        cleanup();
        setIsLoading(false);
      }
      return;
    }

    try {
      setIsLoading(true);

      if (!streamRef.current) {
        await initializeCamera(selectedCamera);
      }

      const newSession = await cameraService.startBackgroundMonitoring({
        posture: monitorPosture,
        eye_blink: monitorEyeBlink,
      });

      saveToStorage(STORAGE_KEYS.CAMERA_SESSION, newSession);
      saveToStorage(STORAGE_KEYS.IS_STREAMING, true);
      saveToStorage(STORAGE_KEYS.MONITOR_SETTINGS, {
        posture: monitorPosture,
        eye_blink: monitorEyeBlink,
      });

      const newWs = cameraService.createWebSocket(newSession.user_id);
      newWs.onopen = () => {
        toast.success("Monitoring started");
        startFrameCapture(newWs);
      };
      attachSocketHandlers(newWs);

      setWs(newWs);
      setIsStreaming(true);
    } catch (error) {
      toast.error("Failed to start monitoring — check camera permissions");
      console.error("Monitoring error:", error);
      cleanup();
    } finally {
      setIsLoading(false);
    }
  };

  const resumeMonitoring = async (savedSession: CameraSession) => {
    try {
      setIsLoading(true);
      const newWs = cameraService.createWebSocket(savedSession.user_id);

      newWs.onopen = () => {
        toast.success("Monitoring service reconnected");
        startFrameCapture(newWs);
        setIsStreaming(true);
        saveToStorage(STORAGE_KEYS.IS_STREAMING, true);
      };
      attachSocketHandlers(newWs);

      setWs(newWs);
    } catch (error) {
      console.error("Failed to resume monitoring:", error);
      cleanup();
    } finally {
      setIsLoading(false);
    }
  };

  const restoreSession = async () => {
    const settings = getFromStorage(STORAGE_KEYS.MONITOR_SETTINGS);
    const savedCameraId = getFromStorage(STORAGE_KEYS.CAMERA_ID);
    const wasActive = getFromStorage("camera_active");

    if (!savedCameraId || !wasActive) return;

    try {
      if (settings) {
        setMonitorPosture(settings.posture);
        setMonitorEyeBlink(settings.eye_blink);
      }

      await initializeCamera(savedCameraId);

      const wasStreaming = getFromStorage(STORAGE_KEYS.IS_STREAMING);
      const savedSession = getFromStorage(STORAGE_KEYS.CAMERA_SESSION);
      if (wasStreaming && savedSession) {
        await resumeMonitoring(savedSession);
      }

      const eyeBlinkData = getFromStorage(STORAGE_KEYS.EYE_BLINK_DATA);
      const postureData = getFromStorage(STORAGE_KEYS.POSTURE_DATA);
      if (eyeBlinkData && settings?.eye_blink) setEyeBlinkFrame(eyeBlinkData);
      if (postureData && settings?.posture) setPostureFrame(postureData);
    } catch (error) {
      console.error("Failed to restore camera session:", error);
      cleanup();
    }
  };

  // Runs once when the dashboard is first entered — resumes any monitoring
  // session left running (e.g. after a page refresh), independent of which
  // dashboard page happens to be open.
  React.useEffect(() => {
    restoreSession();

    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible" && !streamRef.current) {
        restoreSession();
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => document.removeEventListener("visibilitychange", handleVisibilityChange);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <MonitoringContext.Provider
      value={{
        isCameraActive,
        isStreaming,
        isLoading,
        stream,
        selectedCamera,
        setSelectedCamera,
        cameras,
        monitorPosture,
        setMonitorPosture,
        monitorEyeBlink,
        setMonitorEyeBlink,
        eyeBlinkFrame,
        postureFrame,
        backendStatus,
        toggleMonitoring,
      }}
    >
      {children}
      {/* Persistent capture element — stays mounted for the whole dashboard
          session (not just the Services page) so the frame-capture loop and
          detection WebSocket keep running while navigating between dashboard
          pages. Kept off-screen rather than unmounted; visible previews on
          individual pages attach the same `stream` to their own <video>. */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ position: "fixed", top: -9999, left: -9999, width: 1, height: 1, opacity: 0, pointerEvents: "none" }}
      />
    </MonitoringContext.Provider>
  );
}
