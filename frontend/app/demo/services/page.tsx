'use client';

import { useState, useEffect, useRef } from 'react';
import useWebSocket from 'react-use-websocket';
import { toast } from 'sonner';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Camera, Eye, ShieldCheck, Waves, Video, Play, Square, Settings, Loader2, Brain, AlertCircle } from 'lucide-react';
import { API_URL, WS_URL } from '@/lib/constants';

const WellnessMonitor = () => {
  // Camera states
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);

  // Monitoring states
  const [isStreaming, setIsStreaming] = useState(false);
  const [monitorPosture, setMonitorPosture] = useState(true);
  const [monitorEyeBlink, setMonitorEyeBlink] = useState(true);
  const [imageData, setImageData] = useState<string | null>(null);
  const [waterBreakInterval, setWaterBreakInterval] = useState(30);
  const [isWaterBreakActive, setIsWaterBreakActive] = useState(false);

  // WebSocket connection
  const { sendMessage, lastMessage } = useWebSocket(
    `${WS_URL}/sitblink/ws?posture=${monitorPosture}&eye_blink=${monitorEyeBlink}`,
    {
      shouldReconnect: () => isStreaming,
    }
  );

  // Get available cameras
  useEffect(() => {
    async function getCameras() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        setCameras(videoDevices);
        if (videoDevices.length) setSelectedCamera(videoDevices[0].deviceId);
      } catch (error) {
        toast.error('Failed to get camera devices');
      }
    }
    getCameras();
  }, []);

  // Handle camera preview
  const toggleCamera = async () => {
    try {
      if (isCameraActive) {
        if (videoRef.current?.srcObject) {
          const stream = videoRef.current.srcObject as MediaStream;
          stream.getTracks().forEach(track => track.stop());
          videoRef.current.srcObject = null;
        }
        setIsCameraActive(false);
      } else {
        const constraints = {
          video: {
            deviceId: selectedCamera ? { exact: selectedCamera } : undefined,
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setIsCameraActive(true);
        }
      }
    } catch (error) {
      console.error('Camera access error:', error);
      toast.error('Failed to access camera. Please check permissions.');
    }
  };

  // Handle stream control
  const toggleStream = async () => {
    if (!selectedCamera) {
      toast.error('Please select a camera first');
      return;
    }

    const endpoint = isStreaming ? 'stop_sitblink_stream' : 'start_sitblink_stream';
    try {
      const response = await fetch(`${API_URL}/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          posture: monitorPosture, 
          eye_blink: monitorEyeBlink 
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setIsStreaming(!isStreaming);
      toast.success(isStreaming ? 'Frame capture stopped' : 'Frame capture started');
    } catch (error) {
      console.error('Stream toggle error:', error);
      toast.error('Failed to toggle frame capture. Please try again.');
    }
  };

  // Handle water break notifications
  const toggleWaterBreakNotifications = async () => {
    try {
      if (!isWaterBreakActive) {
        await fetch(`${API_URL}/sip/set_water_break_interval`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ interval_minutes: waterBreakInterval }),
        });
        setIsWaterBreakActive(true);
        toast.success('Water break notifications enabled');
      } else {
        await fetch(`${API_URL}/sip/stop_water_break_notifications`, {
          method: 'POST',
        });
        setIsWaterBreakActive(false);
        toast.success('Water break notifications disabled');
      }
    } catch (error) {
      toast.error('Failed to update water break notifications');
    }
  };

  // Add this after the existing WebSocket connection
  useEffect(() => {
    if (lastMessage?.data) {
      try {
        const data = JSON.parse(lastMessage.data);
        if (data.image) {
          setImageData(data.image);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    }
  }, [lastMessage]);

  const [isLoading, setIsLoading] = useState(false);

  return (
    <div className="container mx-auto space-y-6 p-6">
      {/* Camera Setup Card */}
      <Card className="backdrop-blur-sm bg-white/95 border border-blue-100">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div className="flex items-center gap-2">
            <div className="rounded-full bg-blue-50 p-2">
              <Camera className="h-4 w-4 text-blue-600" />
            </div>
            <CardTitle className="text-lg font-medium">Camera Setup</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4">
            <div className="flex items-center justify-between">
              <Label>Select Camera</Label>
              <select 
                className="rounded-md border p-2 bg-white"
                value={selectedCamera}
                onChange={(e) => setSelectedCamera(e.target.value)}
              >
                <option value="">Choose a camera...</option>
                {cameras.map((camera) => (
                  <option key={camera.deviceId} value={camera.deviceId}>
                    {camera.label || `Camera ${camera.deviceId.slice(0, 5)}`}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="flex gap-4">
              <Button
                variant="outline"
                className="flex-1"
                onClick={toggleCamera}
                disabled={!selectedCamera}
              >
                <Eye className="mr-2 h-4 w-4" />
                {isCameraActive ? 'Stop Preview' : 'Preview Camera'}
              </Button>

              <Button
                variant={isStreaming ? "destructive" : "default"}
                className="flex-1"
                onClick={toggleStream}
                disabled={!selectedCamera}
              >
                <Video className="mr-2 h-4 w-4" />
                {isStreaming ? 'Stop Capture' : 'Start Capture'}
              </Button>
            </div>

            {/* Preview Window */}
            {isCameraActive && (
              <div className="aspect-video rounded-lg border overflow-hidden bg-black">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-contain"
                />
              </div>
            )}

            {/* Frame Capture Window */}
            {isStreaming && imageData && (
              <div className="aspect-video rounded-lg border overflow-hidden bg-black">
                <img
                  src={`data:image/jpeg;base64,${imageData}`}
                  alt="Frame Capture"
                  className="w-full h-full object-contain"
                />
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Monitoring Pipeline Card */}
      <Card className="backdrop-blur-sm bg-white/95 border border-violet-100">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div className="flex items-center gap-2">
            <div className="rounded-full bg-violet-50 p-2">
              <Brain className="h-4 w-4 text-violet-600" />
            </div>
            <CardTitle className="text-lg font-medium">Monitoring Pipeline</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="rounded-full bg-emerald-50 p-1.5">
                  <ShieldCheck className="h-4 w-4 text-emerald-600" />
                </div>
                <Label>Posture Detection</Label>
              </div>
              <Switch
                checked={monitorPosture}
                onCheckedChange={setMonitorPosture}
              />
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="rounded-full bg-blue-50 p-1.5">
                  <Eye className="h-4 w-4 text-blue-600" />
                </div>
                <Label>Eye Blink Detection</Label>
              </div>
              <Switch
                checked={monitorEyeBlink}
                onCheckedChange={setMonitorEyeBlink}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Water Break Settings Card */}
      <Card className="backdrop-blur-sm bg-white/95 border border-cyan-100">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div className="flex items-center gap-2">
            <div className="rounded-full bg-cyan-50 p-2">
              <Waves className="h-4 w-4 text-cyan-600" />
            </div>
            <CardTitle className="text-lg font-medium">Water Break Settings</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Interval (minutes)</Label>
            <Input
              type="number"
              value={waterBreakInterval}
              onChange={(e) => setWaterBreakInterval(Number(e.target.value))}
              min={1}
              max={120}
            />
          </div>
          <Button
            variant={isWaterBreakActive ? "destructive" : "default"}
            className="w-full"
            onClick={toggleWaterBreakNotifications}
          >
            <Waves className="mr-2 h-4 w-4" />
            {isWaterBreakActive ? 'Stop Water Break Notifications' : 'Start Water Break Notifications'}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};

export default WellnessMonitor;
