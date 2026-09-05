"use client";

import PageIllustration from "@/components/page-illustration";
import { Armchair, Eye, Droplets, Github, Download, Circle } from "lucide-react";

const GITHUB_URL = "https://github.com/ishworrsubedii/desktop-sitblinksip";

export default function HeroHome() {
  return (
    <section className="relative" aria-label="Main Hero Section">
      <PageIllustration />

      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="pb-16 pt-40 md:pt-52">
          <div className="grid items-center gap-12 lg:grid-cols-[1.05fr_0.95fr] lg:gap-8">
            {/* Copy */}
            <div className="text-center lg:text-left">
              <div
                className="mb-5 inline-flex items-center gap-2 rounded-full border border-blue-100 bg-blue-50/80 px-3 py-1 text-xs font-medium text-blue-700"
                data-aos="fade-up"
              >
                <span className="relative flex h-1.5 w-1.5">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-blue-500 opacity-75 motion-reduce:animate-none" />
                  <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-blue-600" />
                </span>
                Open-source wellness companion for screen time
              </div>

              <h1
                className="text-4xl font-bold tracking-tight text-slate-900 md:text-5xl lg:text-[3.25rem] lg:leading-[1.05]"
                data-aos="fade-up"
              >
                <span className="text-blue-600">Sit</span> Better.{" "}
                <span className="text-violet-600">Blink</span> More.{" "}
                <span className="text-cyan-600">Sip</span> Regularly.
              </h1>

              <p
                className="mx-auto mt-5 max-w-xl text-lg text-slate-600 lg:mx-0"
                data-aos="fade-up"
                data-aos-delay="100"
              >
                SitBlinkSip watches your posture, blinking behavior, and hydration
                breaks through your webcam while you work, then nudges you when
                it's time to reset — so healthier habits build themselves.
              </p>

              <div
                className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row lg:justify-start"
                data-aos="fade-up"
                data-aos-delay="150"
              >
                <a
                  className="inline-flex h-12 w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-6 font-medium text-white shadow-sm transition-all duration-150 ease-in-out hover:bg-blue-700 hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 sm:w-auto"
                  href="/#desktop"
                >
                  <Download className="h-4 w-4" />
                  Download Desktop App
                </a>
                <a
                  className="inline-flex h-12 w-full items-center justify-center gap-2 rounded-lg bg-slate-100 px-6 font-medium text-slate-700 transition duration-150 ease-in-out hover:bg-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-400 focus-visible:ring-offset-2 sm:w-auto"
                  href={GITHUB_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Github className="h-4 w-4" />
                  View on GitHub
                </a>
              </div>

              <div
                className="mt-8 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-sm text-slate-500 lg:justify-start"
                data-aos="fade-up"
                data-aos-delay="200"
              >
                <span className="inline-flex items-center gap-1.5">
                  <Circle className="h-2 w-2 fill-emerald-500 text-emerald-500" />
                  Free &amp; open source
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <Circle className="h-2 w-2 fill-emerald-500 text-emerald-500" />
                  Runs on your own machine
                </span>
              </div>
            </div>

            {/* Product visual */}
            <div
              className="relative mx-auto w-full max-w-md lg:max-w-none"
              data-aos="fade-up"
              data-aos-delay="150"
            >
              <div className="pointer-events-none absolute -inset-6 -z-10 rounded-[2rem] bg-gradient-to-tr from-blue-100 via-white to-cyan-50 blur-2xl" />

              <div className="overflow-hidden rounded-2xl border border-gray-200/80 bg-white shadow-2xl shadow-slate-900/10 transition-transform duration-300 [transform:perspective(1400px)_rotateY(-6deg)_rotateX(2deg)] hover:[transform:perspective(1400px)_rotateY(-2deg)_rotateX(1deg)] motion-reduce:[transform:none]">
                {/* App window chrome */}
                <div className="flex items-center gap-2 border-b border-gray-100 bg-gray-50 px-4 py-3">
                  <div className="flex gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-full bg-red-400" />
                    <span className="h-2.5 w-2.5 rounded-full bg-amber-400" />
                    <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
                  </div>
                  <div className="flex-1 text-center text-xs font-medium text-gray-400">
                    SitBlinkSip Desktop — tray widget
                  </div>
                </div>

                {/* Mock HUD widget */}
                <div className="space-y-3 bg-slate-50 p-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Running in background
                    </span>
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-2 py-1 text-[10px] font-medium text-emerald-700">
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500 motion-reduce:animate-none" />
                      MONITORING
                    </span>
                  </div>

                  <div className="rounded-xl border border-gray-100 bg-white p-3 shadow-sm">
                    <div className="flex items-center gap-2.5">
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-blue-50">
                        <Armchair className="h-4 w-4 text-blue-600" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="text-xs text-slate-500">Posture</div>
                        <div className="text-sm font-semibold text-slate-800">Good posture</div>
                      </div>
                      <div className="h-2 w-16 overflow-hidden rounded-full bg-gray-100">
                        <div className="h-full w-[86%] rounded-full bg-blue-500" />
                      </div>
                    </div>
                  </div>

                  <div className="rounded-xl border border-gray-100 bg-white p-3 shadow-sm">
                    <div className="flex items-center gap-2.5">
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-50">
                        <Eye className="h-4 w-4 text-violet-600" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="text-xs text-slate-500">Blink rate</div>
                        <div className="text-sm font-semibold text-slate-800">15 / min</div>
                      </div>
                      <div className="h-2 w-16 overflow-hidden rounded-full bg-gray-100">
                        <div className="h-full w-[70%] rounded-full bg-violet-500" />
                      </div>
                    </div>
                  </div>

                  <div className="rounded-xl border border-gray-100 bg-white p-3 shadow-sm">
                    <div className="flex items-center gap-2.5">
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-cyan-50">
                        <Droplets className="h-4 w-4 text-cyan-600" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="text-xs text-slate-500">Next water break</div>
                        <div className="text-sm font-semibold text-slate-800">in 12 min</div>
                      </div>
                      <div className="h-2 w-16 overflow-hidden rounded-full bg-gray-100">
                        <div className="h-full w-[45%] rounded-full bg-cyan-500" />
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Floating reminder toast */}
              <div className="absolute -bottom-5 -left-4 hidden animate-float items-center gap-2 rounded-xl border border-gray-100 bg-white px-3 py-2 shadow-lg sm:flex motion-reduce:animate-none">
                <Eye className="h-4 w-4 text-violet-600" />
                <span className="text-xs font-medium text-slate-700">Time to blink 👀</span>
              </div>
            </div>
          </div>

          {/* Scroll indicator */}
          <div className="mt-20 text-center" data-aos="fade-up" data-aos-delay="100">
            <a href="#features" className="inline-flex flex-col items-center group">
              <span className="mb-3 text-sm text-slate-500 group-hover:text-slate-700">
                See how it works
              </span>
              <svg
                className="h-6 w-6 animate-bounce text-slate-400 motion-reduce:animate-none"
                fill="none"
                strokeWidth="2"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path d="M19 14l-7 7m0 0l-7-7m7 7V3" />
              </svg>
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
