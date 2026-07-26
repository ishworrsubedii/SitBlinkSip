"use client";

import * as React from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, Eye, Timer, BrainCircuit } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { apiClient } from "@/lib/api-client";
import { usePerson } from "../PersonContext";

const timeRangeOptions = [
  { value: '60', label: 'Last Hour' },
  { value: '120', label: 'Last 2 Hours' },
  { value: '240', label: 'Last 4 Hours' },
  { value: '480', label: 'Last 8 Hours' },
  { value: '720', label: 'Last 12 Hours' },
  { value: '1440', label: 'Last 24 Hours' },
];

const POSTURE_THRESHOLD = 80;
const BLINK_THRESHOLD = 12;

interface PostureRow {
  timestamp: string;
  head_tilt: number;
  displacement_ratio: number;
  posture_status: boolean | null;
}

interface EyeRow {
  timestamp: string;
  ear: number;
  blink: boolean | null;
}

function toMinuteBucket(timestamp: string) {
  return new Date(`${timestamp.replace(' ', 'T')}Z`).setSeconds(0, 0);
}

function formatXAxis(timestamp: number) {
  return new Date(timestamp).toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });
}

function formatDuration(ms: number) {
  const minutes = Math.round(ms / 60000);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

export default function DashboardPage() {
  const person = usePerson();
  const [minutes, setMinutes] = React.useState('60');
  const [postureRows, setPostureRows] = React.useState<PostureRow[]>([]);
  const [eyeRows, setEyeRows] = React.useState<EyeRow[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!person) return;
    let cancelled = false;
    setLoading(true);
    setError(null);

    Promise.all([
      apiClient.fetch(`get_posture_data?minutes=${minutes}&person_id=${person.id}`),
      apiClient.fetch(`get_eye_data?minutes=${minutes}&person_id=${person.id}`),
    ])
      .then(([postureRes, eyeRes]) => {
        if (cancelled) return;
        setPostureRows((postureRes.data ?? []).slice().reverse());
        setEyeRows((eyeRes.data ?? []).slice().reverse());
      })
      .catch(() => {
        if (!cancelled) setError("Couldn't load your data. Is the tracking service running?");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [person, minutes]);

  const hasData = postureRows.length > 0 || eyeRows.length > 0;

  const avgPostureScore = postureRows.length
    ? Math.round(
        (postureRows.filter((r) => r.posture_status).length / postureRows.length) * 100
      )
    : null;

  const totalBlinks = eyeRows.filter((r) => r.blink).length;

  const blinkRatePerMinute = React.useMemo(() => {
    const buckets = new Map<number, number>();
    for (const row of eyeRows) {
      if (!row.blink) continue;
      const bucket = toMinuteBucket(row.timestamp);
      buckets.set(bucket, (buckets.get(bucket) ?? 0) + 1);
    }
    return Array.from(buckets.entries())
      .sort(([a], [b]) => a - b)
      .map(([timestamp, blinkRate]) => ({ timestamp, blinkRate }));
  }, [eyeRows]);

  const avgBlinkRate = blinkRatePerMinute.length
    ? Math.round(
        blinkRatePerMinute.reduce((sum, d) => sum + d.blinkRate, 0) / blinkRatePerMinute.length
      )
    : null;

  const postureScoreOverTime = React.useMemo(() => {
    const buckets = new Map<number, { good: number; total: number }>();
    for (const row of postureRows) {
      const bucket = toMinuteBucket(row.timestamp);
      const entry = buckets.get(bucket) ?? { good: 0, total: 0 };
      entry.total += 1;
      if (row.posture_status) entry.good += 1;
      buckets.set(bucket, entry);
    }
    return Array.from(buckets.entries())
      .sort(([a], [b]) => a - b)
      .map(([timestamp, { good, total }]) => ({
        timestamp,
        score: Math.round((good / total) * 100),
      }));
  }, [postureRows]);

  const trackedSpan = React.useMemo(() => {
    const timestamps = [...postureRows, ...eyeRows].map((r) => new Date(`${r.timestamp.replace(' ', 'T')}Z`).getTime());
    if (timestamps.length < 2) return null;
    return Math.max(...timestamps) - Math.min(...timestamps);
  }, [postureRows, eyeRows]);

  if (!person) return null;

  return (
    <div className="w-full p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900">Welcome back, {person.name}</h1>
          <p className="text-sm text-gray-500">Here's how your posture and eye health are trending.</p>
        </div>
        <Select value={minutes} onValueChange={setMinutes}>
          <SelectTrigger className="w-[180px] bg-white">
            <SelectValue placeholder="Select time window" />
          </SelectTrigger>
          <SelectContent className="bg-white">
            {timeRangeOptions.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {loading && (
        <div className="grid gap-4 md:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-[120px] w-full rounded-xl" />
          ))}
        </div>
      )}

      {!loading && error && (
        <Card className="border-red-100 bg-red-50">
          <CardContent className="py-6 text-sm text-red-700">{error}</CardContent>
        </Card>
      )}

      {!loading && !error && !hasData && (
        <Card className="border-blue-100 bg-blue-50/50">
          <CardContent className="py-10 text-center">
            <BrainCircuit className="mx-auto mb-3 h-8 w-8 text-blue-500" />
            <p className="text-gray-700">No monitoring data yet for {person.name}.</p>
            <Link href="/dashboard/services" className="mt-3 inline-block text-sm font-medium text-blue-600 hover:underline">
              Start a monitoring session →
            </Link>
          </CardContent>
        </Card>
      )}

      {!loading && !error && hasData && (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            <Card className="bg-primary/5">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Posture Score</CardTitle>
                <Brain className="h-4 w-4 text-green-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{avgPostureScore ?? '--'}%</div>
                <Progress value={avgPostureScore ?? 0} className="mt-2" />
                <p className="text-xs text-muted-foreground mt-2">
                  {avgPostureScore !== null && avgPostureScore >= POSTURE_THRESHOLD
                    ? 'Good posture detected'
                    : 'Room for improvement'}
                </p>
              </CardContent>
            </Card>

            <Card className="bg-primary/5">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Blink Rate</CardTitle>
                <Eye className="h-4 w-4 text-blue-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{avgBlinkRate ?? '--'} bpm</div>
                <Progress value={avgBlinkRate ? Math.min(avgBlinkRate * 5, 100) : 0} className="mt-2" />
                <p className="text-xs text-muted-foreground mt-2">Normal range: 10-20 bpm</p>
              </CardContent>
            </Card>

            <Card className="bg-primary/5">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Time Tracked</CardTitle>
                <Timer className="h-4 w-4 text-orange-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{trackedSpan !== null ? formatDuration(trackedSpan) : '--'}</div>
                <p className="text-xs text-muted-foreground mt-2">Within the selected window</p>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            <Card className="backdrop-blur-sm bg-white/95 border border-blue-100">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <div className="flex items-center gap-2">
                  <Eye className="h-4 w-4 text-blue-600" />
                  <CardTitle className="text-lg font-medium">Blink Rate</CardTitle>
                </div>
                <div className="text-sm text-blue-900/70">{totalBlinks} blinks</div>
              </CardHeader>
              <CardContent>
                <div className="h-[280px]">
                  {blinkRatePerMinute.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={blinkRatePerMinute} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" className="opacity-20" />
                        <XAxis dataKey="timestamp" tickFormatter={formatXAxis} type="number" domain={['dataMin', 'dataMax']} fontSize={11} stroke="#64748b" />
                        <YAxis fontSize={11} stroke="#64748b" label={{ value: 'Blinks/min', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                        <Tooltip labelFormatter={formatXAxis} contentStyle={{ backgroundColor: 'rgba(255,255,255,0.95)', borderRadius: '6px', border: '1px solid rgba(37,99,235,0.1)' }} />
                        <Line type="monotone" dataKey="blinkRate" stroke="#2563eb" strokeWidth={2} dot={false} />
                        <ReferenceLine y={BLINK_THRESHOLD} stroke="#dc2626" strokeDasharray="3 3" label={{ value: 'Normal', position: 'right', fontSize: 11 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex h-full items-center justify-center text-sm text-gray-400">No blink data in this window</div>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card className="backdrop-blur-sm bg-white/95 border border-blue-100">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <div className="flex items-center gap-2">
                  <Brain className="h-4 w-4 text-blue-600" />
                  <CardTitle className="text-lg font-medium">Posture</CardTitle>
                </div>
                <div className="text-sm text-blue-900/70">{avgPostureScore ?? '--'}% average</div>
              </CardHeader>
              <CardContent>
                <div className="h-[280px]">
                  {postureScoreOverTime.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={postureScoreOverTime} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" className="opacity-20" />
                        <XAxis dataKey="timestamp" tickFormatter={formatXAxis} type="number" domain={['dataMin', 'dataMax']} fontSize={11} stroke="#64748b" />
                        <YAxis domain={[0, 100]} fontSize={11} stroke="#64748b" label={{ value: 'Posture Score %', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                        <Tooltip labelFormatter={formatXAxis} contentStyle={{ backgroundColor: 'rgba(255,255,255,0.95)', borderRadius: '6px', border: '1px solid rgba(37,99,235,0.1)' }} />
                        <Line type="monotone" dataKey="score" stroke="#2563eb" strokeWidth={2} dot={false} />
                        <ReferenceLine y={POSTURE_THRESHOLD} stroke="#dc2626" strokeDasharray="3 3" label={{ value: 'Target', position: 'right', fontSize: 11 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex h-full items-center justify-center text-sm text-gray-400">No posture data in this window</div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
