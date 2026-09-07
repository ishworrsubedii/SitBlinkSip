"use client";

import { useEffect, useState } from "react";
import {
  AppWindowMac,
  AppWindow,
  Terminal,
  Download,
  ShieldCheck,
  Github,
} from "lucide-react";
import { formatDownloadCount, type DownloadCounts } from "@/lib/github";

const RELEASES = "https://github.com/ishworrsubedii/desktop-sitblinksip/releases/latest/download";
const RELEASES_PAGE = "https://github.com/ishworrsubedii/desktop-sitblinksip/releases";

type OS = "macos" | "windows" | "linux";

const builds: Record<
  OS,
  {
    name: string;
    icon: typeof AppWindowMac;
    file: string;
    format: string;
    detail: string;
    note: string;
  }
> = {
  macos: {
    name: "macOS",
    icon: AppWindowMac,
    file: `${RELEASES}/SitBlinkSip-Desktop-macos-arm64.dmg`,
    format: ".dmg",
    detail: "Apple Silicon · macOS 11+",
    note: "Unsigned build — right-click the app → Open on first launch.",
  },
  windows: {
    name: "Windows",
    icon: AppWindow,
    file: `${RELEASES}/SitBlinkSip-Desktop-windows-setup.exe`,
    format: "setup.exe",
    detail: "Windows 10 / 11 · 64-bit",
    note: "Unsigned build — SmartScreen: click More info → Run anyway.",
  },
  linux: {
    name: "Linux",
    icon: Terminal,
    file: `${RELEASES}/sitblinksip-desktop-linux-amd64.deb`,
    format: ".deb",
    detail: "Any desktop with a system tray",
    note: "Install with sudo apt install ./sitblinksip-desktop*.deb",
  },
};

function detectOS(): OS {
  if (typeof navigator === "undefined") return "macos";
  const platform = `${navigator.platform ?? ""} ${navigator.userAgent ?? ""}`.toLowerCase();
  if (platform.includes("win")) return "windows";
  if (platform.includes("linux") && !platform.includes("android")) return "linux";
  return "macos";
}

export default function DesktopDownload({ counts }: { counts?: DownloadCounts | null }) {
  const [detected, setDetected] = useState<OS | null>(null);

  useEffect(() => {
    setDetected(detectOS());
  }, []);

  const order: OS[] = detected
    ? [detected, ...(["macos", "windows", "linux"] as OS[]).filter((os) => os !== detected)]
    : ["macos", "windows", "linux"];

  return (
    <section id="desktop" className="relative scroll-mt-24">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="py-16 md:py-24">
          <div className="mx-auto max-w-2xl text-center" data-aos="fade-up">
            <div className="mb-3 inline-flex rounded-full bg-violet-50 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-violet-700">
              Desktop App
            </div>
            <h2 className="text-3xl font-bold text-slate-900 md:text-4xl">
              No browser tab. No server. Just the tray icon.
            </h2>
            <p className="mt-4 text-lg text-slate-600">
              SitBlinkSip Desktop watches your posture, blink rate, and water
              breaks locally through your webcam, packaged as one native app
              that runs quietly in the background — nothing to host, nothing
              to keep open.
            </p>
            {!!counts?.total && (
              <p className="mt-3 text-sm font-medium text-slate-500">
                {formatDownloadCount(counts.total)} downloads across macOS, Windows &amp; Linux
              </p>
            )}
          </div>

          <div className="mt-12 grid gap-5 sm:grid-cols-3">
            {order.map((os, index) => {
              const build = builds[os];
              const isPrimary = detected !== null && os === detected;
              return (
                <div
                  key={os}
                  data-aos="fade-up"
                  data-aos-delay={index * 75}
                  className={`flex flex-col rounded-2xl border p-6 shadow-sm transition-shadow hover:shadow-md ${
                    isPrimary
                      ? "border-blue-200 bg-blue-50/40 ring-1 ring-blue-100"
                      : "border-gray-200 bg-white"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${
                        isPrimary ? "bg-blue-100" : "bg-slate-100"
                      }`}
                    >
                      <build.icon
                        className={`h-5 w-5 ${isPrimary ? "text-blue-600" : "text-slate-600"}`}
                      />
                    </div>
                    <div>
                      <div className="font-semibold text-slate-900">{build.name}</div>
                      <div className="text-xs text-slate-500">{build.detail}</div>
                    </div>
                    {isPrimary && (
                      <span className="ml-auto rounded-full bg-blue-600 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-white">
                        Your OS
                      </span>
                    )}
                  </div>

                  <a
                    href={build.file}
                    className={`mt-5 inline-flex h-11 w-full items-center justify-center gap-2 rounded-lg px-4 text-sm font-semibold shadow-sm transition-all ${
                      isPrimary
                        ? "bg-blue-600 text-white hover:bg-blue-700 hover:shadow-md"
                        : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                    }`}
                  >
                    <Download className="h-4 w-4" />
                    Download {build.format}
                  </a>

                  <p className="mt-3 text-xs leading-relaxed text-slate-500">
                    {build.note}
                  </p>
                  {!!counts?.[os] && (
                    <p className="mt-2 text-[11px] font-medium uppercase tracking-wide text-slate-400">
                      {formatDownloadCount(counts[os])} downloads
                    </p>
                  )}
                </div>
              );
            })}
          </div>

          <div
            className="mt-10 flex flex-col items-center justify-center gap-4 text-center sm:flex-row sm:text-left"
            data-aos="fade-up"
          >
            <span className="inline-flex items-center gap-2 text-sm text-slate-500">
              <ShieldCheck className="h-4 w-4 shrink-0 text-emerald-500" />
              Local processing only — no camera frame ever leaves your machine.
            </span>
            <a
              href={RELEASES_PAGE}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-sm font-medium text-slate-500 underline-offset-4 hover:text-blue-600 hover:underline"
            >
              <Github className="h-4 w-4" />
              All releases &amp; source
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
