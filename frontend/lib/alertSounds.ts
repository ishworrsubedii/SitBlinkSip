'use client';

export type AlertKind = 'posture' | 'blink' | 'water';

const SOUND_ENABLED_KEY = 'sitblinksip_sound_enabled';

// Distinct short tone patterns per alert type, so users can tell them apart by ear.
const TONE_PATTERNS: Record<AlertKind, { frequencies: number[]; beepMs: number }> = {
  posture: { frequencies: [440, 440], beepMs: 140 },
  blink: { frequencies: [660], beepMs: 180 },
  water: { frequencies: [523, 659, 784], beepMs: 130 },
};

let audioContext: AudioContext | null = null;
const lastFiredAt: Partial<Record<AlertKind, number>> = {};

function getAudioContext(): AudioContext | null {
  if (typeof window === 'undefined') return null;
  const Ctor = window.AudioContext || (window as any).webkitAudioContext;
  if (!Ctor) return null;
  if (!audioContext) {
    audioContext = new Ctor();
  }
  if (audioContext.state === 'suspended') {
    audioContext.resume().catch(() => {});
  }
  return audioContext;
}

export function isSoundEnabled(): boolean {
  if (typeof window === 'undefined') return true;
  const stored = localStorage.getItem(SOUND_ENABLED_KEY);
  return stored === null ? true : stored === 'true';
}

export function setSoundEnabled(enabled: boolean) {
  if (typeof window === 'undefined') return;
  localStorage.setItem(SOUND_ENABLED_KEY, String(enabled));
}

export function canFireAlert(kind: AlertKind, cooldownMs: number): boolean {
  const now = Date.now();
  const last = lastFiredAt[kind] ?? 0;
  if (now - last < cooldownMs) return false;
  lastFiredAt[kind] = now;
  return true;
}

export function playAlertTone(kind: AlertKind) {
  if (!isSoundEnabled()) return;
  const ctx = getAudioContext();
  if (!ctx) return;

  const { frequencies, beepMs } = TONE_PATTERNS[kind];
  const gap = 0.05;
  let startTime = ctx.currentTime;

  frequencies.forEach((freq) => {
    const oscillator = ctx.createOscillator();
    const gainNode = ctx.createGain();
    oscillator.type = 'sine';
    oscillator.frequency.value = freq;

    gainNode.gain.setValueAtTime(0, startTime);
    gainNode.gain.linearRampToValueAtTime(0.2, startTime + 0.01);
    gainNode.gain.linearRampToValueAtTime(0, startTime + beepMs / 1000);

    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);

    oscillator.start(startTime);
    oscillator.stop(startTime + beepMs / 1000);

    startTime += beepMs / 1000 + gap;
  });
}
