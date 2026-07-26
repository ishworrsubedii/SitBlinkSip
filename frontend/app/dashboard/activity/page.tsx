"use client";

import * as React from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, Eye, BrainCircuit } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { apiClient } from "@/lib/api-client";
import { usePerson } from "../PersonContext";

interface ActivityEvent {
  timestamp: string;
  type: 'posture' | 'eye_blink';
  status: string;
  good: boolean;
}

function formatTimestamp(timestamp: string) {
  return new Date(`${timestamp.replace(' ', 'T')}Z`).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });
}

export default function ActivityPage() {
  const person = usePerson();
  const [events, setEvents] = React.useState<ActivityEvent[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!person) return;
    let cancelled = false;
    setLoading(true);
    setError(null);

    Promise.all([
      apiClient.fetch(`get_posture_data?minutes=1440&person_id=${person.id}`),
      apiClient.fetch(`get_eye_data?minutes=1440&person_id=${person.id}`),
    ])
      .then(([postureRes, eyeRes]) => {
        if (cancelled) return;
        const postureEvents: ActivityEvent[] = (postureRes.data ?? []).map((r: any) => ({
          timestamp: r.timestamp,
          type: 'posture' as const,
          status: r.posture_status ? 'Good posture' : 'Poor posture detected',
          good: !!r.posture_status,
        }));
        const eyeEvents: ActivityEvent[] = (eyeRes.data ?? [])
          .filter((r: any) => r.blink)
          .map((r: any) => ({
            timestamp: r.timestamp,
            type: 'eye_blink' as const,
            status: 'Blink recorded',
            good: true,
          }));
        const merged = [...postureEvents, ...eyeEvents]
          .sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1))
          .slice(0, 100);
        setEvents(merged);
      })
      .catch(() => {
        if (!cancelled) setError("Couldn't load your activity. Is the tracking service running?");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [person]);

  if (!person) return null;

  return (
    <div className="w-full p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900">Activity</h1>
        <p className="text-sm text-gray-500">{person.name}'s recent posture and eye-blink events (last 24 hours)</p>
      </div>

      {loading && (
        <div className="space-y-2">
          {[0, 1, 2, 3, 4].map((i) => <Skeleton key={i} className="h-12 w-full rounded-lg" />)}
        </div>
      )}

      {!loading && error && (
        <Card className="border-red-100 bg-red-50">
          <CardContent className="py-6 text-sm text-red-700">{error}</CardContent>
        </Card>
      )}

      {!loading && !error && events.length === 0 && (
        <Card className="border-blue-100 bg-blue-50/50">
          <CardContent className="py-10 text-center">
            <BrainCircuit className="mx-auto mb-3 h-8 w-8 text-blue-500" />
            <p className="text-gray-700">No activity yet for {person.name}.</p>
            <Link href="/dashboard/services" className="mt-3 inline-block text-sm font-medium text-blue-600 hover:underline">
              Start a monitoring session →
            </Link>
          </CardContent>
        </Card>
      )}

      {!loading && !error && events.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base font-medium">Recent Events</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="divide-y divide-gray-100">
              {events.map((event, i) => (
                <div key={i} className="flex items-center gap-3 px-6 py-3">
                  {event.type === 'posture' ? (
                    <Brain className={`h-4 w-4 shrink-0 ${event.good ? 'text-emerald-500' : 'text-amber-500'}`} />
                  ) : (
                    <Eye className="h-4 w-4 shrink-0 text-blue-500" />
                  )}
                  <span className="flex-1 text-sm text-gray-700">{event.status}</span>
                  <span className="text-xs text-gray-400">{formatTimestamp(event.timestamp)}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
