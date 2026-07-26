"use client";

import * as React from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, Eye, Activity, BrainCircuit } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { apiClient } from "@/lib/api-client";
import { usePerson } from "../PersonContext";

const rangeOptions = [
  { value: '1440', label: 'Today' },
  { value: '10080', label: 'This Week' },
  { value: '43200', label: 'This Month' },
];

interface PostureRow {
  timestamp: string;
  posture_status: boolean | null;
}

interface EyeRow {
  timestamp: string;
  blink: boolean | null;
}

function dayKey(timestamp: string) {
  return timestamp.slice(0, 10); // YYYY-MM-DD
}

function dayLabel(key: string) {
  return new Date(`${key}T00:00:00Z`).toLocaleDateString('en-US', { weekday: 'short' });
}

export default function AnalyticsPage() {
  const person = usePerson();
  const [range, setRange] = React.useState('10080');
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
      apiClient.fetch(`get_posture_data?minutes=${range}&person_id=${person.id}`),
      apiClient.fetch(`get_eye_data?minutes=${range}&person_id=${person.id}`),
    ])
      .then(([postureRes, eyeRes]) => {
        if (cancelled) return;
        setPostureRows(postureRes.data ?? []);
        setEyeRows(eyeRes.data ?? []);
      })
      .catch(() => {
        if (!cancelled) setError("Couldn't load your analytics. Is the tracking service running?");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [person, range]);

  const hasData = postureRows.length > 0 || eyeRows.length > 0;

  const avgPostureScore = postureRows.length
    ? Math.round((postureRows.filter((r) => r.posture_status).length / postureRows.length) * 100)
    : null;

  const totalBlinks = eyeRows.filter((r) => r.blink).length;
  const dayCount = new Set([...postureRows, ...eyeRows].map((r) => dayKey(r.timestamp))).size || 1;
  const avgBlinkRate = eyeRows.length ? Math.round(totalBlinks / (dayCount * 24 * 60) * 60) : null;

  const dailyTrends = React.useMemo(() => {
    const byDay = new Map<string, { postureGood: number; postureTotal: number; blinks: number }>();
    for (const row of postureRows) {
      const key = dayKey(row.timestamp);
      const entry = byDay.get(key) ?? { postureGood: 0, postureTotal: 0, blinks: 0 };
      entry.postureTotal += 1;
      if (row.posture_status) entry.postureGood += 1;
      byDay.set(key, entry);
    }
    for (const row of eyeRows) {
      const key = dayKey(row.timestamp);
      const entry = byDay.get(key) ?? { postureGood: 0, postureTotal: 0, blinks: 0 };
      if (row.blink) entry.blinks += 1;
      byDay.set(key, entry);
    }
    return Array.from(byDay.entries())
      .sort(([a], [b]) => (a < b ? -1 : 1))
      .map(([key, v]) => ({
        day: dayLabel(key),
        posture: v.postureTotal ? Math.round((v.postureGood / v.postureTotal) * 100) : 0,
        blinks: v.blinks,
      }));
  }, [postureRows, eyeRows]);

  if (!person) return null;

  return (
    <div className="w-full p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900">Analytics Overview</h1>
          <p className="text-sm text-gray-500">{person.name}'s posture and eye health trends</p>
        </div>
        <Select value={range} onValueChange={setRange}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Select time range" />
          </SelectTrigger>
          <SelectContent>
            {rangeOptions.map((option) => (
              <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {loading && (
        <div className="grid gap-4 md:grid-cols-3">
          {[0, 1, 2].map((i) => <Skeleton key={i} className="h-[100px] w-full rounded-xl" />)}
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
            <p className="text-gray-700">No data yet for {person.name} in this range.</p>
            <Link href="/dashboard/services" className="mt-3 inline-block text-sm font-medium text-blue-600 hover:underline">
              Start a monitoring session →
            </Link>
          </CardContent>
        </Card>
      )}

      {!loading && !error && hasData && (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium">Avg Posture Score</CardTitle>
                <Brain className="h-4 w-4 text-blue-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{avgPostureScore ?? '--'}%</div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium">Avg Blink Rate</CardTitle>
                <Eye className="h-4 w-4 text-green-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{avgBlinkRate ?? '--'} bpm</div>
                <p className="text-xs text-muted-foreground mt-1">Normal range: 10-20 bpm</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium">Total Blinks</CardTitle>
                <Activity className="h-4 w-4 text-orange-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{totalBlinks}</div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-blue-500" />
                Daily Trends
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={dailyTrends} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" className="opacity-20" />
                    <XAxis dataKey="day" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="posture" stroke="#2563eb" name="Posture Score %" />
                    <Line type="monotone" dataKey="blinks" stroke="#16a34a" name="Blinks" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
