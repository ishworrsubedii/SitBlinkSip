"use client";

import * as React from "react";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Waves, Volume2, VolumeX, UserRound } from "lucide-react";
import { usePerson } from "../PersonContext";
import { personService } from "../services/personService";
import { isSoundEnabled, setSoundEnabled } from "@/lib/alertSounds";

export default function SettingsPage() {
  const person = usePerson();
  const [interval, setIntervalValue] = React.useState(30);
  const [savingInterval, setSavingInterval] = React.useState(false);
  const [soundEnabled, setSoundEnabledState] = React.useState(true);
  const saveTimeout = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  React.useEffect(() => {
    setSoundEnabledState(isSoundEnabled());
  }, []);

  React.useEffect(() => {
    if (person) {
      setIntervalValue(person.water_break_interval);
    }
  }, [person]);

  React.useEffect(() => {
    return () => {
      if (saveTimeout.current) clearTimeout(saveTimeout.current);
    };
  }, []);

  const toggleSound = (enabled: boolean) => {
    setSoundEnabledState(enabled);
    setSoundEnabled(enabled);
  };

  const handleIntervalChange = (raw: string) => {
    const value = Number(raw);
    if (!Number.isFinite(value)) return;

    setIntervalValue(value);
    if (!person || value < 1) return;

    if (saveTimeout.current) clearTimeout(saveTimeout.current);
    saveTimeout.current = setTimeout(async () => {
      try {
        setSavingInterval(true);
        await personService.updateWaterBreakInterval(person.id, value);
        toast.success("Reminder interval saved");
      } catch (error) {
        toast.error("Failed to save water break interval");
        console.error("Water break interval save error:", error);
      } finally {
        setSavingInterval(false);
      }
    }, 600);
  };

  if (!person) return null;

  return (
    <div className="w-full p-6 space-y-6 max-w-2xl">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900">Settings</h1>
        <p className="text-sm text-gray-500">Profile and reminder preferences</p>
      </div>

      <Card className="backdrop-blur-sm bg-white/95 border border-blue-100">
        <CardHeader className="flex flex-row items-center gap-2 pb-2">
          <UserRound className="h-4 w-4 text-blue-600" />
          <CardTitle className="text-lg font-medium">Profile</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
              <span className="text-sm font-medium text-blue-700">
                {person.name.trim().split(/\s+/).slice(0, 2).map((p) => p[0]?.toUpperCase()).join("")}
              </span>
            </div>
            <span className="text-sm font-medium text-gray-900">{person.name}</span>
          </div>
        </CardContent>
      </Card>

      <Card className="backdrop-blur-sm bg-white/95 border border-cyan-100">
        <CardHeader className="flex flex-row items-center gap-2 pb-2">
          <div className="rounded-full bg-cyan-50 p-2">
            <Waves className="h-4 w-4 text-cyan-600" />
          </div>
          <CardTitle className="text-lg font-medium">Water Break Reminders</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Remind me every (minutes)</Label>
            <Input
              type="number"
              value={interval}
              onChange={(e) => handleIntervalChange(e.target.value)}
              min={1}
              max={120}
            />
            <p className="text-xs text-gray-500">
              {savingInterval
                ? "Saving..."
                : `Runs automatically in the background — a tone plays every ${interval} min while the backend is connected, even across page refreshes.`}
            </p>
          </div>
          <Separator />
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {soundEnabled ? (
                <Volume2 className="h-4 w-4 text-cyan-600" />
              ) : (
                <VolumeX className="h-4 w-4 text-gray-400" />
              )}
              <div className="flex flex-col">
                <Label className="text-sm font-medium">Sound Alerts</Label>
                <span className="text-xs text-gray-500">Play a tone for posture, blink and water-break alerts</span>
              </div>
            </div>
            <Switch checked={soundEnabled} onCheckedChange={toggleSound} />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
