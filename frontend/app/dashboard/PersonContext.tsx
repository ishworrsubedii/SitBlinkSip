"use client";

import * as React from "react";
import { toast } from "sonner";
import { personService, type Person } from "./services/personService";
import { playAlertTone } from "@/lib/alertSounds";
import { WS_URL } from "@/lib/constants";

interface PersonContextValue {
  person: Person | null;
}

const PersonContext = React.createContext<PersonContextValue>({ person: null });

export function usePerson() {
  return React.useContext(PersonContext).person;
}

export function PersonProvider({ children }: { children: React.ReactNode }) {
  const [person, setPerson] = React.useState<Person | null>(null);
  const [name, setName] = React.useState("");
  const [submitting, setSubmitting] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [ready, setReady] = React.useState(false);

  React.useEffect(() => {
    setPerson(personService.getStoredPerson());
    setReady(true);
  }, []);

  // Water-break reminders: one persistent connection for the whole dashboard session.
  // The backend tracks the actual schedule, so a refresh (which just reconnects) never
  // resets the countdown — it picks up exactly where the server says it should be.
  React.useEffect(() => {
    if (!person) return;

    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let cancelled = false;

    const connect = () => {
      socket = new WebSocket(`${WS_URL}/ws/water-break/${person.id}`);

      socket.onopen = () => {
        console.log(`[water-break] connected for person ${person.id}`);
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "water_break") {
            toast.info("Time for a water break! 💧");
            playAlertTone("water");
          }
        } catch {
          // ignore malformed messages
        }
      };

      socket.onclose = (event) => {
        console.log(`[water-break] disconnected (code ${event.code}), retrying in 5s`);
        if (!cancelled) {
          reconnectTimer = setTimeout(connect, 5000);
        }
      };

      socket.onerror = () => {
        console.error('[water-break] websocket error');
        socket?.close();
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [person?.id]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      const created = await personService.createPerson(name.trim());
      setPerson(created);
    } catch (err) {
      setError("Couldn't save your name. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  if (!ready) {
    return null;
  }

  if (!person) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
        <div className="w-full max-w-sm rounded-xl bg-white p-6 shadow-xl">
          <h2 className="text-lg font-semibold text-gray-900">Who's using this dashboard?</h2>
          <p className="mt-1 text-sm text-gray-600">
            Tell us your name so we can keep your posture and eye-blink data separate from anyone else's.
          </p>
          <form onSubmit={handleSubmit} className="mt-4 space-y-3">
            <input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Your name"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            {error && <p className="text-sm text-red-600">{error}</p>}
            <button
              type="submit"
              disabled={submitting || !name.trim()}
              className="w-full rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:opacity-50"
            >
              {submitting ? "Saving..." : "Continue"}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <PersonContext.Provider value={{ person }}>
      {children}
    </PersonContext.Provider>
  );
}
