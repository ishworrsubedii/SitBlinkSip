import { Github, LayoutDashboard } from "lucide-react";

const GITHUB_URL = "https://github.com/ishworrsubedii/SitBlinkSip";

export default function Cta() {
  return (
    <section className="relative overflow-hidden">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div
          className="relative overflow-hidden rounded-2xl bg-gradient-to-b from-slate-900 to-slate-800 text-center shadow-2xl"
          data-aos="zoom-y-out"
        >
          {/* Glow effects */}
          <div className="absolute inset-0">
            <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2">
              <div className="h-40 sm:h-56 w-[280px] sm:w-[480px] rounded-full bg-blue-500/20 blur-3xl" />
            </div>
          </div>

          <div className="relative px-4 py-12 sm:py-16 md:py-20">
            <div className="mx-auto max-w-3xl">
              <h2 className="mb-4 text-2xl sm:text-3xl md:text-4xl font-bold text-white">
                Ready to work healthier?
              </h2>
              <p className="mb-8 text-sm sm:text-base md:text-lg text-gray-300 px-2 sm:px-0">
                Open your dashboard and start building better work habits — sitting, blinking, and sipping included.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8 px-2 sm:px-6">
                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-4 sm:p-5">
                  <span className="block font-bold text-base sm:text-lg text-gray-200 mb-1">Real-time</span>
                  <span className="text-xs sm:text-sm text-gray-400">Health monitoring</span>
                </div>
                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-4 sm:p-5">
                  <span className="block font-bold text-base sm:text-lg text-gray-200 mb-1">Open-source</span>
                  <span className="text-xs sm:text-sm text-gray-400">Computer vision core</span>
                </div>
                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-4 sm:p-5">
                  <span className="block font-bold text-base sm:text-lg text-gray-200 mb-1">Personalized</span>
                  <span className="text-xs sm:text-sm text-gray-400">Health insights</span>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row items-center justify-center gap-3 px-4 sm:px-0">
                <a
                  className="w-full sm:w-auto group inline-flex items-center justify-center rounded-lg bg-blue-600 px-6 sm:px-8 py-3 text-white transition-all hover:bg-blue-700"
                  href="/dashboard"
                >
                  Go to Dashboard
                  <LayoutDashboard className="ml-2 h-4 w-4 transition-transform group-hover:scale-110" />
                </a>
                <a
                  className="w-full sm:w-auto inline-flex items-center justify-center rounded-lg border border-gray-700 bg-gray-800/50 px-6 sm:px-8 py-3 text-gray-200 backdrop-blur-sm transition-colors hover:border-blue-500 hover:text-blue-400"
                  href={GITHUB_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  View Source Code
                  <Github className="ml-2 h-4 w-4" />
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
