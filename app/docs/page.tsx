import PageIllustration from "@/components/page-illustration";
import Header from "@/components/ui/header";
import Footer from "@/components/ui/footer";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Documentation",
  description:
    "Install SitBlinkSip Desktop and learn how the posture, blink, and hydration reminders work — a free, open-source wellness app for developers.",
  openGraph: {
    title: "SitBlinkSip Documentation",
    description:
      "Install SitBlinkSip Desktop and learn how the posture, blink, and hydration reminders work.",
  },
};

export default function Documentation() {
  return (
    <>
      <Header />

      <main className="relative">
        <PageIllustration />

        {/* Hero section */}
        <section className="relative">
          <div className="mx-auto max-w-6xl px-4 sm:px-6">
            <div className="pt-32 pb-12 md:pt-40 md:pb-20">
              <div className="text-center">
                <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4">
                  Getting Started with
                  <span className="text-blue-600"> SitBlinkSip</span>
                </h1>
                <p className="text-xl text-slate-500 mb-8">
                  Install the desktop app and let it watch your posture, blink rate, and water breaks while you work.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Documentation content */}
        <section className="relative border-t border-slate-100">
          <div className="mx-auto max-w-6xl px-4 sm:px-6 py-12 md:py-20">
            <div className="grid md:grid-cols-12 gap-8">
              {/* Sidebar */}
              <div className="md:col-span-3">
                <nav className="sticky top-24">
                  <ul className="space-y-3">
                    <li>
                      <a href="#installation" className="text-sm font-medium text-slate-600 hover:text-blue-500">
                        Installation
                      </a>
                    </li>
                    <li>
                      <a href="#requirements" className="text-sm font-medium text-slate-600 hover:text-blue-500">
                        System Requirements
                      </a>
                    </li>
                    <li>
                      <a href="#usage" className="text-sm font-medium text-slate-600 hover:text-blue-500">
                        Basic Usage
                      </a>
                    </li>
                  </ul>
                </nav>
              </div>

              {/* Main content */}
              <div className="md:col-span-9">
                {/* Installation section */}
                <div id="installation" className="mb-12">
                  <h2 className="text-2xl font-bold text-slate-900 mb-4">Installation</h2>
                  <p className="text-slate-500 mb-6">
                    SitBlinkSip runs as a native desktop app — no Docker, no server, nothing to host.
                    Grab the latest build for your OS from the{" "}
                    <a
                      href="https://github.com/ishworrsubedii/desktop-sitblinksip/releases"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:underline"
                    >
                      GitHub releases page
                    </a>
                    .
                  </p>
                </div>
                <div className="mx-auto max-w-3xl" data-aos="zoom-y-out" data-aos-delay={600}>
                  <div className="relative aspect-video rounded-2xl bg-[#1E1E1E] px-5 py-3 shadow-2xl">
                    {/* Mac-style Terminal Header */}
                    <div className="relative mb-6 flex items-center">
                      <div className="absolute flex gap-1.5">
                        <div className="h-3 w-3 rounded-full bg-[#FF605C]"></div>
                        <div className="h-3 w-3 rounded-full bg-[#FFBD44]"></div>
                        <div className="h-3 w-3 rounded-full bg-[#00CA4E]"></div>
                      </div>
                      <div className="mx-auto text-[13px] font-medium text-gray-400">
                        sitblinksip-desktop — install (Linux)
                      </div>
                    </div>

                    {/* Terminal Content */}
                    <div className="font-mono text-[15px] [&_span]:opacity-0">
                      <span className="animate-[code-1_15s_infinite] text-white">
                        $ curl -LO https://github.com/ishworrsubedii/desktop-sitblinksip/releases/latest/download/sitblinksip-desktop-linux-amd64.deb
                      </span>
                      <br />
                      <span className="animate-[code-2_15s_infinite] text-gray-400">
                        Downloaded sitblinksip-desktop-linux-amd64.deb
                      </span>
                      <br />
                      <span className="animate-[code-3_15s_infinite] text-white">
                        $ sudo apt install ./sitblinksip-desktop-linux-amd64.deb
                      </span>
                      <br />
                      <span className="animate-[code-4_15s_infinite] text-gray-400">
                        Setting up sitblinksip-desktop ...
                      </span>
                      <br />
                      <span className="animate-[code-5_15s_infinite] text-emerald-400">
                        ✨ Installed successfully
                      </span>
                      <br />
                      <span className="animate-[code-6_15s_infinite] text-emerald-400">
                        🚀 SitBlinkSip is now running in your system tray
                      </span>
                      <br />
                      <span className="animate-[code-7_15s_infinite] text-blue-400">
                        👁  Press F6 anytime to reveal the live preview
                      </span>
                    </div>
                  </div>
                </div>

                {/* Requirements section */}
                <div id="requirements" className="mb-12 mt-12">
                  <h2 className="text-2xl font-bold text-slate-900 mb-4">System Requirements</h2>
                  <ul className="list-disc list-inside space-y-2 text-slate-500">
                    <li>A webcam for posture and blink detection</li>
                    <li>Linux with a system tray — available now (.deb)</li>
                    <li>Windows — on the roadmap</li>
                    <li>No account, no internet connection, and no server required</li>
                  </ul>
                </div>

                {/* Usage section */}
                <div id="usage" className="mb-12">
                  <h2 className="text-2xl font-bold text-slate-900 mb-4">Basic Usage</h2>
                  <div className="prose prose-slate max-w-none">
                    <p className="text-slate-500 mb-4">
                      Once installed, SitBlinkSip counts blinks and tracks posture quietly in the background
                      without showing your camera feed. Press <strong>F6</strong> to reveal a live preview on
                      demand, and the screen briefly blanks if your blink rate drops too low. All detection
                      runs locally using on-device computer-vision models — no video ever leaves your machine.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      <Footer border={true} />
    </>
  );
}
