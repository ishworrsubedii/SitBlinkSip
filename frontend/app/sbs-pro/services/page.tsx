'use client';

import { useState, useEffect, useRef } from 'react';
import { toast } from 'sonner';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Camera, Eye, ShieldCheck, Waves, Video, Play, Square, Settings, Loader2, Brain, AlertCircle, Activity } from 'lucide-react';
import { API_URL, WS_URL } from '@/lib/constants';
import { apiClient, ApiError } from '@/lib/api-client';

const WellnessMonitor = () => {
  // Camera states
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);

  // Monitoring states
  const [isStreaming, setIsStreaming] = useState(false);
  const [monitorPosture, setMonitorPosture] = useState(false);
  const [monitorEyeBlink, setMonitorEyeBlink] = useState(false);
  const [imageData, setImageData] = useState<string | null>(null);
  const [waterBreakInterval, setWaterBreakInterval] = useState(30);
  const [isWaterBreakActive, setIsWaterBreakActive] = useState(false);

  // Add these state variables near the other state declarations
  const [startPosture, setStartPosture] = useState(true);
  const [startEyeBlink, setStartEyeBlink] = useState(true);

  // Add these state variables for frame capture settings
  const [framePosture, setFramePosture] = useState(false);
  const [frameEyeBlink, setFrameEyeBlink] = useState(false);

  // Add these state variables for pipeline settings
  const [pipelinePosture, setPipelinePosture] = useState(false);
  const [pipelineEyeBlink, setPipelineEyeBlink] = useState(false);

  // Replace isStreaming with isPipelineActive
  // Replace isStreaming with isPipelineActive/
  const [isPipelineActive, setIsPipelineActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // WebSocket connection
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [lastMessage, setLastMessage] = useState<MessageEvent | null>(null);

  // Get available cameras
  useEffect(() => {
    async function getCameras() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        setCameras(videoDevices);
        if (videoDevices.length) {
          setSelectedCamera(videoDevices[0].deviceId);
          console.log('Available cameras:', videoDevices);
        }
      } catch (error) {
        toast.error('Failed to get camera devices');
      }
    }
    getCameras();
  }, []);

  const toggleCamera = async () => {
    if (!selectedCamera) {
      toast.error('Please select a camera before starting preview');
      return;
    }

    if (isCameraActive) {
        // Stop the camera
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => {
                track.stop();
            });
            streamRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
        setIsCameraActive(false);
    } else {
        try {
            setIsInitializing(true);
            setIsCameraActive(true);
            
            console.log('Attempting to access camera...');
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    deviceId: selectedCamera ? { exact: selectedCamera } : undefined
                }
            }).catch(error => {
                if (error.name === 'NotAllowedError') {
                    throw new Error('Camera access denied. Please allow camera access in your browser settings.');
                }
                throw error;
            });
            
            streamRef.current = stream;
            
            if (videoRef.current) {
                console.log('Setting video source...');
                videoRef.current.srcObject = stream;
                // Wait for video to be ready
                await new Promise((resolve) => {
                    videoRef.current!.onloadedmetadata = () => {
                        videoRef.current!.play()
                            .then(resolve)
                            .catch(err => console.error('Error playing video:', err));
                    };
                });
                console.log('Video ready for capture');
            }
        } catch (error) {
            console.error('Error accessing the camera:', error);
            toast.error(error instanceof Error ? error.message : 'Failed to start camera preview');
            setIsCameraActive(false);
        } finally {
            setIsInitializing(false);
        }
    }
};

  // Cleanup effect
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => {
          track.stop();
        });
      }
    };
  }, []);

  // Handle stream control
  const toggleStream = async () => {
    // Add validation checks at the start
    if (!selectedCamera) {
      toast.error('Please select a camera first');
      return;
    }

    if (!isCameraActive) {
      toast.error('Please start the camera preview first');
      return;
    }

    if (!monitorPosture && !monitorEyeBlink) {
      toast.error('Please enable at least one monitoring option (Posture or Eye Blink)');
      return;
    }

    if (!isStreaming) {
      const newWs = new WebSocket(`${WS_URL}/ws?posture=${monitorPosture}&eye_blink=${monitorEyeBlink}`);
      
      await new Promise((resolve, reject) => {
        newWs.onopen = () => {
          toast.success('Connected to server');
          resolve(true);
        };

        newWs.onmessage = (event) => setLastMessage(event);
        newWs.onerror = (error) => reject(error);
        newWs.onclose = () => stopStreaming();
        setTimeout(() => reject(new Error('Connection timeout')), 5000);
      });

      setWs(newWs);
      
      const frameInterval = setInterval(() => {
        if (!videoRef.current || !newWs || newWs.readyState !== WebSocket.OPEN) return;

        const canvas = document.createElement('canvas');
        // Match the target dimensions from backend
        canvas.width = 1280;  // Increased for better quality
        canvas.height = 720;
        const ctx = canvas.getContext('2d');
        
        if (!ctx) return;

        try {
          // Draw maintaining aspect ratio
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

          ctx.fillStyle = '#000000';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(videoRef.current, offsetX, offsetY, drawWidth, drawHeight);
          
          canvas.toBlob(
            (blob) => {
              if (blob && newWs.readyState === WebSocket.OPEN) {
                newWs.send(blob);
              }
            },
            'image/jpeg',
            0.85  // Increased quality
          );
        } catch (error) {
          console.error('Frame capture error:', error);
        }
      }, 100);

      setFrameIntervalId(frameInterval);
      setIsStreaming(true);
    }
  };

  // Handle water break notifications
  const toggleWaterBreakNotifications = async () => {
    if (waterBreakInterval < 1) {
        toast.error('Please set a valid water break interval (minimum 1 minute)');
        return;
    }

    try {
      if (!isWaterBreakActive) {
        await fetch(`${API_URL}/set_water_break_interval`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ interval_minutes: waterBreakInterval }),
        });
        setIsWaterBreakActive(true);
        toast.success('Water break notifications enabled');
      } else {
        await fetch(`${API_URL}/stop_water_break_notifications`, {
          method: 'POST',
        });
        setIsWaterBreakActive(false);
        toast.success('Water break notifications disabled');
      }
    } catch (error) {
      toast.error('Failed to update water break notifications. Please try again.');
      console.error('Water break error:', error);
    }
  };

  // Add this after the existing WebSocket connection
  useEffect(() => {
    if (lastMessage?.data) {
        console.log('Processing WebSocket message:', {
            timestamp: new Date().toISOString(),
            rawDataType: typeof lastMessage.data,
            rawDataLength: lastMessage.data.length
        });

        try {
            const data = JSON.parse(lastMessage.data);
            console.log('Processed WebSocket message:', {
                timestamp: new Date().toISOString(),
                messageType: {
                    hasPostureImage: !!data.posture_image,
                    hasEyeBlinkImage: !!data.eye_blink_image,
                    hasGenericImage: !!data.image
                },
                data
            });

            if (monitorPosture && data.posture_image) {
                setImageData(data.posture_image);
                console.log('Updated posture image data');
            } else if (monitorEyeBlink && data.eye_blink_image) {
                setImageData(data.eye_blink_image);
                console.log('Updated eye blink image data');
            } else if (data.image) {
                setImageData(data.image.split(',')[1]);
                console.log('Updated generic image data');
            }
        } catch (error) {
            console.error('Failed to process WebSocket message:', {
                error,
                rawData: lastMessage.data
            });
        }
    }
  }, [lastMessage, monitorPosture, monitorEyeBlink]);

  const [backendStatus, setBackendStatus] = useState('Not Checked');
  const checkBackendStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/status`);
      if (response.ok) {
        setBackendStatus('Ready');
      } else {
        throw new Error('Backend not ready');
      }
    } catch (error) {
      setBackendStatus('Not Ready');
      console.error('Error checking backend status:', error);
    }
  };

  useEffect(() => {
    checkBackendStatus();
    const interval = setInterval(checkBackendStatus, 30000); // Check every 3 seconds
    return () => clearInterval(interval);
  }, []);

  // Add a new state for loading
  const [isInitializing, setIsInitializing] = useState(false);



  // Pipeline control function
  const togglePipeline = async () => {
    if (!selectedCamera) {
      toast.error('Please select a camera first');
      return;
    }

    if (!monitorPosture && !monitorEyeBlink) {
      toast.error('Please enable at least one monitoring option (Posture or Eye Blink)');
      return;
    }

    const endpoint = isPipelineActive ? 'stop_sitblink_stream' : 'start_sitblink_stream';
    setIsLoading(true);
    
    try {
      await apiClient.fetch(endpoint, {
        method: 'POST',
        body: {
          posture: monitorPosture,
          eye_blink: monitorEyeBlink
        },
      });

      setIsPipelineActive(!isPipelineActive);
      toast.success(isPipelineActive ? 'Pipeline stopped' : 'Pipeline started');
      
    } catch (error) {
      console.error('Pipeline toggle error:', error);
      if (error instanceof ApiError) {
        toast.error(error.message);
      } else {
        toast.error('An unexpected error occurred. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Add this function to handle stopping the stream
  const stopStreaming = () => {
    console.log('Stopping stream...');
    setIsStreaming(false);
    if (frameIntervalId) {
      clearInterval(frameIntervalId);
      setFrameIntervalId(null);
    }
    if (ws) {
      ws.close();
      setWs(null);
    }
    console.log('Stream stopped');
  };

  // Add these state variables
  const [frameIntervalId, setFrameIntervalId] = useState<NodeJS.Timeout | null>(null);
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const [eyeBlinkFrame, setEyeBlinkFrame] = useState<string | null>(null);
  const [postureFrame, setPostureFrame] = useState<string | null>(null);

  // Add cleanup effect
  useEffect(() => {
    return () => {
      if (frameIntervalId) {
        clearInterval(frameIntervalId);
      }
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [frameIntervalId, eventSource]);

  // Add WebSocket setup function
  const setupWebSocket = () => {
    const newWs = new WebSocket(`${WS_URL}/ws?posture=${monitorPosture}&eye_blink=${monitorEyeBlink}`);

    newWs.onopen = () => {
      console.log('WebSocket connected');
      toast.success('Connected to server');
    };

    newWs.onclose = () => {
      console.log('WebSocket disconnected');
      stopStreaming();
    };

    newWs.onerror = (error) => {
      console.error('WebSocket error:', error);
      stopStreaming();
    };

    newWs.onmessage = (event) => {
      setLastMessage(event);
    };

    setWs(newWs);
    return newWs;
  };

  const requestCameraPermission = async () => {
    try {
      const result = await navigator.permissions.query({ name: 'camera' as PermissionName });
      if (result.state === 'denied') {
        toast.error('Camera access is blocked. Please allow camera access in your browser settings.');
        return false;
      }
      if (result.state === 'prompt') {
        toast.info('Please allow camera access when prompted');
      }
      return true;
    } catch (error) {
      console.error('Error checking camera permission:', error);
      toast.error('Unable to check camera permissions');
      return false;
    }
  };

  useEffect(() => {
    const initCamera = async () => {
      const hasPermission = await requestCameraPermission();
      if (!hasPermission) return;

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 1280 },
            height: { ideal: 720 }
          } 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        toast.error('Failed to access camera');
        console.error(err);
      }
    };

    initCamera();
  }, []);

  const connectWebSocket = () => {
    const newWs = new WebSocket(`${WS_URL}/ws?posture=${monitorPosture}&eye_blink=${monitorEyeBlink}`);
    
    newWs.onopen = () => {
      console.log('WebSocket connected successfully');
      toast.success('Connected to server');
    };

    newWs.onerror = (error) => {
      console.error('WebSocket error:', error);
      toast.error('Connection error. Retrying...');
      setTimeout(connectWebSocket, 3000);
    };

    newWs.onclose = () => {
      console.log('WebSocket closed. Attempting to reconnect...');
      setTimeout(connectWebSocket, 3000);
    };

    setWs(newWs);
  };

  // Add connection status notifications
  useEffect(() => {
    if (ws) {
        ws.onclose = () => {
            toast.error('Connection lost. Attempting to reconnect...');
            stopStreaming();
        };

        ws.onerror = () => {
            toast.error('Connection error occurred');
            stopStreaming();
        };
    }
  }, [ws]);

  // Add backend status notifications
  useEffect(() => {
    const prevStatus = backendStatus;
    if (prevStatus !== 'Not Checked' && backendStatus === 'Not Ready') {
        toast.error('Backend service is not responding. Some features may be unavailable.');
    }
  }, [backendStatus]);

  return (
    <div className="h-[calc(100vh-4rem)] overflow-y-auto">
      <div className="w-full p-6 space-y-6">
        {/* Monitoring Options Card */}
        <Card className="backdrop-blur-sm bg-white/95 border border-purple-100">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <div className="flex items-center gap-2">
              <Settings className="h-4 w-4 text-purple-600" />
              <CardTitle className="text-lg font-medium">Monitoring Options</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <div className="rounded-full bg-gradient-to-br from-emerald-50 to-emerald-100 p-1.5">
                  <ShieldCheck className="h-4 w-4 text-emerald-600" />
                </div>
                <div className="flex flex-col">
                  <Label className="text-sm font-medium text-blue-900">Posture Detection</Label>
                  <span className="text-xs text-gray-500">Monitor and alert for poor posture</span>
                </div>
                <Switch
                  checked={monitorPosture}
                  onCheckedChange={setMonitorPosture}
                  disabled={isStreaming}
                  className="ml-2"
                />
              </div>

              <div className="flex items-center gap-2">
                <div className="rounded-full bg-gradient-to-br from-violet-50 to-violet-100 p-1.5">
                  <Eye className="h-4 w-4 text-violet-600" />
                </div>
                <div className="flex flex-col">
                  <Label className="text-sm font-medium text-blue-900">Eye Blink Detection</Label>
                  <span className="text-xs text-gray-500">Monitor eye strain and blink rate</span>
                </div>
                <Switch
                  checked={monitorEyeBlink}
                  onCheckedChange={setMonitorEyeBlink}
                  disabled={isStreaming}
                  className="ml-2"
                />
              </div>
            </div>
            
            {!monitorPosture && !monitorEyeBlink && (
              <div className="mt-4 flex items-center gap-2 text-amber-600 bg-amber-50 p-2 rounded">
                <AlertCircle className="h-4 w-4" />
                <span className="text-sm">Please enable at least one monitoring option</span>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="border border-blue-100">
                  <CardHeader className="pb-2">
                    <div className="flex items-center gap-2">
                      <Brain className="h-4 w-4 text-blue-600" />
                      <CardTitle className="text-sm font-medium">Monitoring Controls</CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center justify-between gap-4">
                      <div className="flex items-center gap-6">
                        <div className="flex items-center gap-2">
                          <div className="rounded-full bg-gradient-to-br from-emerald-50 to-emerald-100 p-1.5">
                            <ShieldCheck className="h-4 w-4 text-emerald-600" />
                          </div>
                          <Label className="text-sm">Posture Detection</Label>
                          <Switch
                            checked={monitorPosture}
                            onCheckedChange={setMonitorPosture}
                            disabled={isStreaming}
                            className="
                              relative h-6 w-11 rounded-full border-2 border-gray-200 bg-white
                              data-[state=checked]:border-blue-600 data-[state=checked]:bg-gradient-to-r from-blue-600 to-blue-600
                              transition-colors duration-200
                            "
                          />
                        </div>

                        <div className="flex items-center gap-2">
                          <div className="rounded-full bg-gradient-to-br from-violet-50 to-violet-100 p-1.5">
                            <Eye className="h-4 w-4 text-violet-600" />
                          </div>
                          <Label className="text-sm">Eye Blink Detection</Label>
                          <Switch
                            checked={monitorEyeBlink}
                            onCheckedChange={setMonitorEyeBlink}
                            disabled={isStreaming}
                            className="
                              relative h-6 w-11 rounded-full border-2 border-gray-200 bg-white
                              data-[state=checked]:border-blue-600 data-[state=checked]:bg-gradient-to-r from-blue-600 to-blue-600
                              transition-colors duration-200
                            "
                          />
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

        {/* Camera Setup Card */}
        <Card className="backdrop-blur-sm bg-white/95 border border-blue-100 overflow-hidden">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <div className="flex items-center gap-2">
              <Camera className="h-4 w-4 text-blue-600" />
              <CardTitle className="text-lg font-medium">Camera Setup</CardTitle>
            </div>
            <div className="flex items-center gap-2">
              <div
                style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  backgroundColor: backendStatus === 'Ready' ? 'green' : 'red',
                }}
              />
              <div className="text-sm text-blue-900/70">
                Backend Status: {backendStatus}
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4 max-h-[calc(100vh-12rem)] overflow-y-auto">
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
                  disabled={!selectedCamera || isLoading}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      {isStreaming ? 'Stopping...' : 'Starting...'}
                    </>
                  ) : (
                    <>
                      <Video className="mr-2 h-4 w-4" />
                      {isStreaming ? 'Stop Capture' : 'Start Pipeline'}
                    </>
                  )}
                </Button>
              </div>
              

              {isCameraActive && (
                <div className="grid grid-cols-2 gap-4 overflow-x-hidden">
                  {/* Preview Window */}
                  <div className="relative overflow-hidden bg-black rounded-lg">
                    <div className="aspect-video relative">
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="absolute inset-0 w-full h-full object-cover [transform:scaleX(-1)]"
                        style={{
                          transform: 'scaleX(-1)',
                          WebkitTransform: 'scaleX(-1)',
                        }}
                      />
                      {isInitializing && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/80 backdrop-blur-sm">
                          <div className="flex flex-col items-center gap-4">
                            <div className="relative">
                              <div className="h-16 w-16 animate-spin rounded-full border-4 border-blue-100 border-t-blue-500" />
                              <div className="absolute inset-0 flex items-center justify-center">
                                <Camera className="h-6 w-6 text-blue-500" />
                              </div>
                            </div>
                            <div className="flex flex-col items-center gap-2">
                              <span className="text-lg font-medium text-white">Initializing Camera</span>
                              <span className="text-sm text-gray-300">Please wait while we set up your camera...</span>
                            </div>
                          </div>
                        </div>
                      )}
                      <div className="absolute top-4 right-4">
                        <div className="flex items-center gap-2 bg-black/50 backdrop-blur-sm px-3 py-1.5 rounded-full">
                          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                          <span className="text-xs text-white font-medium">Live Preview</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Processed Output */}
                  {isCameraActive && isStreaming && imageData && (
                    <div className="relative overflow-hidden bg-black rounded-lg">
                      <div className="aspect-video relative">
                        <img
                          className="absolute inset-0 w-full h-full object-cover"
                          src={imageData}
                          alt="Processed output"
                        />
                        <div className="absolute top-4 right-4">
                          <div className="flex items-center gap-2 bg-black/50 backdrop-blur-sm px-3 py-1.5 rounded-full">
                            <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                            <span className="text-xs text-white font-medium">
                              {monitorPosture ? 'Posture Detection' : 'Eye Blink Detection'}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
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
    </div>
  );
};

export default WellnessMonitor;
