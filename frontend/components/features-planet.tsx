import Image from "next/image";
import WorkspaceImg from "@/public/images/workspace-monitor.png";
import EyeTrackingOverlay from "@/public/images/eye-tracking-overlay.svg";
import PostureIndicator from "@/public/images/posture-indicator.svg";

export default function FeaturesPlanet() {
  return (
    <section className="relative before:absolute before:inset-0 before:-z-20 before:bg-gray-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="py-12 md:py-20">
          {/* Section header */}
          <div className="mx-auto max-w-3xl pb-16 text-center md:pb-20">
            <h2 className="text-3xl font-bold text-gray-200 md:text-4xl">
              Real-Time Health Guardian
            </h2>
          </div>
          {/* Workspace Monitoring visualization */}
          <div className="pb-16 md:pb-20" data-aos="zoom-y-out">
            <div className="text-center">
              <div className="relative inline-flex rounded-full before:absolute before:inset-0 before:-z-10 before:scale-[.85] before:animate-[pulse_3s_cubic-bezier(.4,0,.6,1)_infinite] before:bg-gradient-to-b before:from-blue-900 before:to-sky-700/50 before:blur-3xl">
                {/* Main Workspace Image */}
                <div className="relative">
                  <Image
                    className="rounded-full bg-gray-900 shadow-xl"
                    src={WorkspaceImg}
                    width={800}
                    height={800}
                    alt="Workspace Health Monitoring"
                  />

                  {/* Animated Overlays */}
                  <div className="pointer-events-none absolute inset-0" aria-hidden="true">
                    {/* Status Indicators - Top of Display */}
                    <div className="absolute left-1/2 top-[5%] -translate-x-1/2 flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-4">
                      {/* AI Monitoring Status */}
                      <div className="rounded-lg bg-black/30 px-2 sm:px-3 py-1.5 backdrop-blur-sm">
                        <div className="flex items-center space-x-2 text-white">
                          <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
                          <span className="text-[9px] sm:text-[10px] font-medium">LIVE ANALYSIS</span>
                        </div>
                      </div>

                      {/* Processing Status */}
                      <div className="rounded-lg bg-black/30 px-2 sm:px-3 py-1.5 backdrop-blur-sm">
                        <div className="flex items-center space-x-2 text-white">
                          <svg className="h-2.5 sm:h-3 w-2.5 sm:w-3 animate-spin" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                          </svg>
                          <span className="text-[9px] sm:text-[10px] font-medium">PROCESSING</span>
                        </div>
                      </div>
                    </div>

                    {/* Health Metrics Panel - Right Side */}
                    <div className="absolute right-2 sm:right-4 top-1/2 -translate-y-1/2 space-y-2 sm:space-y-3">
                      {/* Blink Rate Card */}
                      <div className="rounded-lg bg-black/30 p-2 sm:p-3 backdrop-blur-sm">
                        <div className="flex items-center space-x-2 sm:space-x-3 text-white">
                          <svg className="h-3 w-3 sm:h-4 sm:w-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                              d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                              d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                          </svg>
                          <div>
                            <div className="text-[8px] sm:text-[10px] font-medium uppercase tracking-wider opacity-75">Blink Rate</div>
                            <div className="font-mono text-xs sm:text-sm">10/min</div>
                          </div>
                        </div>
                      </div>

                      {/* Posture Card */}
                      <div className="rounded-lg bg-black/30 p-2 sm:p-3 backdrop-blur-sm">
                        <div className="flex items-center space-x-2 sm:space-x-3 text-white">
                          <svg className="h-3 w-3 sm:h-4 sm:w-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                              d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                          </svg>
                          <div>
                            <div className="text-[8px] sm:text-[10px] font-medium uppercase tracking-wider opacity-75">Posture</div>
                            <div className="font-mono text-xs sm:text-sm text-emerald-400">Good Posture</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          {/* Grid */}
          <div className="grid gap-4 sm:gap-6 overflow-hidden grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
            <article className="p-4 sm:p-6">
              <h3 className="mb-2 flex items-center space-x-2 font-medium text-gray-200 text-sm sm:text-base">
                <svg className="fill-blue-500" xmlns="http://www.w3.org/2000/svg" width={16} height={16}>
                  <path d="M8 0C3.6 0 0 3.6 0 8s3.6 8 8 8 8-3.6 8-8-3.6-8-8-8zm0 14c-3.3 0-6-2.7-6-6s2.7-6 6-6 6 2.7 6 6-2.7 6-6 6zm3.5-6c0 1.9-1.6 3.5-3.5 3.5S4.5 9.9 4.5 8 6.1 4.5 8 4.5s3.5 1.6 3.5 3.5z" />
                </svg>
                <span>Real-time Health Tracking</span>
              </h3>
              <p className="text-[13px] sm:text-[15px] text-gray-400">
                Instant webcam analysis of your eye blinks and posture, with continuous feedback to help maintain healthy work habits.
              </p>
            </article>
            <article className="p-4 sm:p-6">
              <h3 className="mb-2 flex items-center space-x-2 font-medium text-gray-200 text-sm sm:text-base">
                <svg className="fill-blue-500" xmlns="http://www.w3.org/2000/svg" width={16} height={16}>
                  <path d="M8 0C3.6 0 0 3.6 0 8s3.6 8 8 8 8-3.6 8-8-3.6-8-8-8zm0 2c3.3 0 6 2.7 6 6s-2.7 6-6 6-6-2.7-6-6 2.7-6 6-6zm0 2C6.9 4 6 4.9 6 6s.9 2 2 2 2-.9 2-2-.9-2-2-2z" />
                </svg>
                <span>Enhanced Privacy</span>
              </h3>
              <p className="text-[13px] sm:text-[15px] text-gray-400">
                Zero data storage - all processing happens instantly on your device with no video recording or storage.
              </p>
            </article>
            <article className="p-4 sm:p-6">
              <h3 className="mb-2 flex items-center space-x-2 font-medium text-gray-200 text-sm sm:text-base">
                <svg className="fill-blue-500" xmlns="http://www.w3.org/2000/svg" width={16} height={16}>
                  <path d="M15.5 12L14 8.5V3c0-.6-.4-1-1-1H3c-.6 0-1 .4-1 1v5.5L.5 12c-.3.8.3 1.6 1.1 1.9.2.1.4.1.6.1H3v1c0 .6.4 1 1 1h8c.6 0 1-.4 1-1v-1h.8c.9 0 1.7-.7 1.7-1.6 0-.1 0-.3-.1-.4zM4 14v-1h8v1H4zm9.4-2H2.6L4 8.5V3h8v5.5L13.4 12z" />
                </svg>
                <span>Smart Alerts</span>
              </h3>
              <p className="text-[13px] sm:text-[15px] text-gray-400">
                Get gentle reminders for water breaks, eye exercises, and posture adjustments when you need them most.
              </p>
            </article>
            <article className="p-4 sm:p-6">
              <h3 className="mb-2 flex items-center space-x-2 font-medium text-gray-200 text-sm sm:text-base">
                <svg className="fill-blue-500" xmlns="http://www.w3.org/2000/svg" width={16} height={16}>
                  <path d="M14 0H2C.9 0 0 .9 0 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V2c0-1.1-.9-2-2-2zM2 14V2h12v12H2zm9-9H5c-.6 0-1 .4-1 1v6c0 .6.4 1 1 1h6c.6 0 1-.4 1-1V6c0-.6-.4-1-1-1z" />
                </svg>
                <span>Visual Insights</span>
              </h3>
              <p className="text-[13px] sm:text-[15px] text-gray-400">
                View your health metrics through intuitive charts and graphs, making it easy to track improvements over time.
              </p>
            </article>
            <article className="p-4 sm:p-6">
              <h3 className="mb-2 flex items-center space-x-2 font-medium text-gray-200 text-sm sm:text-base">
                <svg className="fill-blue-500" xmlns="http://www.w3.org/2000/svg" width={16} height={16}>
                  <path d="M8 0C3.6 0 0 3.6 0 8s3.6 8 8 8 8-3.6 8-8-3.6-8-8-8zm0 14c-3.3 0-6-2.7-6-6s2.7-6 6-6 6 2.7 6 6-2.7 6-6 6zm-.5-4.5c0 .3.2.5.5.5s.5-.2.5-.5V8c0-.3-.2-.5-.5-.5s-.5.2-.5.5v1.5zM8 5c-.6 0-1 .4-1 1s.4 1 1 1 1-.4 1-1-.4-1-1-1z" />
                </svg>
                <span>AI Assistant</span>
              </h3>
              <p className="text-[13px] sm:text-[15px] text-gray-400">
                Coming soon: Get personalized health tips and insights from our AI assistant based on your usage patterns.
              </p>
            </article>
            <article className="p-4 sm:p-6">
              <h3 className="mb-2 flex items-center space-x-2 font-medium text-gray-200 text-sm sm:text-base">
                <svg className="fill-blue-500" xmlns="http://www.w3.org/2000/svg" width={16} height={16}>
                  <path d="M8 3.5a.5.5 0 0 0-1 0V9a.5.5 0 0 0 .252.434l3.5 2a.5.5 0 0 0 .496-.868L8 8.71V3.5z" />
                  <path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm7-8A7 7 0 1 1 1 8a7 7 0 0 1 14 0z" />
                </svg>
                <span>Workspace Analysis</span>
              </h3>
              <p className="text-[13px] sm:text-[15px] text-gray-400">
                Coming soon: Monitor lighting conditions and screen distance to optimize your workspace for eye health.
              </p>
            </article>
          </div>
        </div>
      </div>
    </section>
  );
}
