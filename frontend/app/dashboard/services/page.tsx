'use client';

import { useState, useEffect, useRef } from 'react';
import { toast } from 'sonner';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Camera, Eye, ShieldCheck, Video, Settings, Loader2, AlertCircle, SlidersHorizontal } from 'lucide-react';
import { personService } from './personService';
import { usePerson } from '../PersonContext';
import { useMonitoring } from '../MonitoringContext';

const WellnessMonitor = () => {
  const person = usePerson();
  const {
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
  } = useMonitoring();

  // Local preview-only video element — the actual capture happens off-screen
  // in MonitoringProvider so it keeps running across dashboard navigation;
  // this just mirrors the same MediaStream for a visible preview here.
  const previewRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (previewRef.current) {
      previewRef.current.srcObject = stream;
    }
  }, [stream]);

  // Detection sensitivity thresholds — per-person setting, configurable before
  // starting monitoring. Defaults come from the person profile.
  const [thresholdsOpen, setThresholdsOpen] = useState(false);
  const [earThreshold, setEarThreshold] = useState(0.15);
  const [postureAngleThreshold, setPostureAngleThreshold] = useState(145);
  const [postureDisplacementThreshold, setPostureDisplacementThreshold] = useState(0.65);
  const [savingThresholds, setSavingThresholds] = useState(false);

  useEffect(() => {
    if (person) {
      setEarThreshold(person.ear_threshold);
      setPostureAngleThreshold(person.posture_angle_threshold);
      setPostureDisplacementThreshold(person.posture_displacement_threshold);
    }
  }, [person]);

  const saveThresholds = async () => {
    if (!person) return;
    try {
      setSavingThresholds(true);
      await personService.updateDetectionThresholds(person.id, {
        ear_threshold: earThreshold,
        posture_angle_threshold: postureAngleThreshold,
        posture_displacement_threshold: postureDisplacementThreshold,
      });
      toast.success('Detection thresholds saved');
      setThresholdsOpen(false);
    } catch (error) {
      toast.error('Failed to save thresholds');
      console.error('Threshold save error:', error);
    } finally {
      setSavingThresholds(false);
    }
  };

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
                <span className="text-sm">Enable posture, eye-blink, or both — monitoring runs whichever you pick</span>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Camera Setup Card */}
        <Card className="backdrop-blur-sm bg-white/95 border border-blue-100 overflow-hidden">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <div className="flex items-center gap-2">
              <Camera className="h-4 w-4 text-blue-600" />
              <CardTitle className="text-lg font-medium">Camera</CardTitle>
            </div>
            <div className="flex items-center gap-3">
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

              <Dialog open={thresholdsOpen} onOpenChange={setThresholdsOpen}>
                <DialogTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    disabled={isStreaming}
                    title="Detection sensitivity settings"
                  >
                    <SlidersHorizontal className="h-4 w-4 text-blue-600" />
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Detection Sensitivity</DialogTitle>
                    <DialogDescription>
                      Tune how sensitive posture and eye-blink detection are. Saved to your profile and applied the next time you start monitoring.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 py-2">
                    <div className="space-y-1.5">
                      <Label>Eye Blink Threshold (EAR)</Label>
                      <Input
                        type="number"
                        step="0.01"
                        min={0.05}
                        max={0.5}
                        value={earThreshold}
                        onChange={(e) => setEarThreshold(Number(e.target.value))}
                      />
                      <p className="text-xs text-gray-500">Lower catches blinks more easily but may trigger on partial closes. Default: 0.15</p>
                    </div>
                    <div className="space-y-1.5">
                      <Label>Posture Angle Threshold (degrees)</Label>
                      <Input
                        type="number"
                        step="1"
                        min={90}
                        max={180}
                        value={postureAngleThreshold}
                        onChange={(e) => setPostureAngleThreshold(Number(e.target.value))}
                      />
                      <p className="text-xs text-gray-500">Head-to-shoulder angle below this counts as bad posture. Default: 145</p>
                    </div>
                    <div className="space-y-1.5">
                      <Label>Posture Displacement Threshold</Label>
                      <Input
                        type="number"
                        step="0.01"
                        min={0.1}
                        max={1.5}
                        value={postureDisplacementThreshold}
                        onChange={(e) => setPostureDisplacementThreshold(Number(e.target.value))}
                      />
                      <p className="text-xs text-gray-500">Forward head displacement below this counts as bad posture. Default: 0.65</p>
                    </div>
                  </div>
                  <DialogFooter>
                    <Button
                      variant="outline"
                      onClick={() => {
                        setEarThreshold(0.15);
                        setPostureAngleThreshold(145);
                        setPostureDisplacementThreshold(0.65);
                      }}
                    >
                      Reset to Defaults
                    </Button>
                    <Button onClick={saveThresholds} disabled={savingThresholds || !person}>
                      {savingThresholds ? 'Saving...' : 'Save'}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
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
                  disabled={isStreaming}
                >
                  <option value="">Choose a camera...</option>
                  {cameras.map((camera) => (
                    <option key={camera.deviceId} value={camera.deviceId}>
                      {camera.label || `Camera ${camera.deviceId.slice(0, 5)}`}
                    </option>
                  ))}
                </select>
              </div>

              <Button
                variant={isStreaming ? "destructive" : "default"}
                onClick={toggleMonitoring}
                disabled={isLoading}
                className={isStreaming ? undefined : "bg-blue-600 hover:bg-blue-700 text-white"}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    {isStreaming ? 'Stopping...' : 'Starting...'}
                  </>
                ) : (
                  <>
                    <Video className="mr-2 h-4 w-4" />
                    {isStreaming ? 'Stop Monitoring' : 'Start Monitoring'}
                  </>
                )}
              </Button>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Live Preview — mirrors the shared camera stream owned by
                    MonitoringProvider, purely for display on this page. */}
                <div className="relative overflow-hidden bg-black rounded-lg">
                  <div className="aspect-video relative">
                    <video
                      ref={previewRef}
                      autoPlay
                      playsInline
                      muted
                      className={`absolute inset-0 w-full h-full object-cover [transform:scaleX(-1)] ${isCameraActive ? '' : 'invisible'}`}
                    />
                    {!isCameraActive && (
                      <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-500">
                        {isLoading ? (
                          <div className="flex flex-col items-center gap-3">
                            <div className="h-10 w-10 animate-spin rounded-full border-4 border-blue-100 border-t-blue-500" />
                            <span>Starting camera...</span>
                          </div>
                        ) : (
                          <span>Camera preview appears here once you start monitoring</span>
                        )}
                      </div>
                    )}
                    {isCameraActive && (
                      <div className="absolute top-4 right-4">
                        <div className="flex items-center gap-2 bg-black/50 backdrop-blur-sm px-3 py-1.5 rounded-full">
                          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                          <span className="text-xs text-white font-medium">Live Preview</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Posture Output */}
                {isStreaming && monitorPosture && (
                  <div className="relative overflow-hidden bg-black rounded-lg">
                    <div className="aspect-video relative">
                      {postureFrame ? (
                        <img
                          className="absolute inset-0 w-full h-full object-cover"
                          src={postureFrame}
                          alt="Posture detection output"
                        />
                      ) : (
                        <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-400">
                          Waiting for posture data...
                        </div>
                      )}
                      <div className="absolute top-4 right-4">
                        <div className="flex items-center gap-2 bg-black/50 backdrop-blur-sm px-3 py-1.5 rounded-full">
                          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                          <span className="text-xs text-white font-medium">Posture Detection</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Eye Blink Output */}
                {isStreaming && monitorEyeBlink && (
                  <div className="relative overflow-hidden bg-black rounded-lg">
                    <div className="aspect-video relative">
                      {eyeBlinkFrame ? (
                        <img
                          className="absolute inset-0 w-full h-full object-cover"
                          src={eyeBlinkFrame}
                          alt="Eye blink detection output"
                        />
                      ) : (
                        <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-400">
                          Waiting for eye-blink data...
                        </div>
                      )}
                      <div className="absolute top-4 right-4">
                        <div className="flex items-center gap-2 bg-black/50 backdrop-blur-sm px-3 py-1.5 rounded-full">
                          <div className="w-2 h-2 rounded-full bg-violet-500 animate-pulse" />
                          <span className="text-xs text-white font-medium">Eye Blink Detection</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default WellnessMonitor;
