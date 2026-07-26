import { Github, Star, GitFork, ScanEye } from "lucide-react";

const GITHUB_URL = "https://github.com/ishworrsubedii/SitBlinkSip";

const highlights = [
  "Explore the full computer-vision and API implementation.",
  "Learn how posture, blink, and hydration detection actually work.",
  "Open issues, suggest features, or submit a pull request.",
];

const techStack = [
  "Next.js",
  "FastAPI",
  "Python",
  "OpenCV",
  "MediaPipe",
  "dlib",
  "Docker",
];

export default function OpenSource() {
  return (
    <section id="open-source" className="relative overflow-hidden bg-slate-900 scroll-mt-24">
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.07] [mask-image:radial-gradient(white,transparent_75%)]"
        style={{
          backgroundImage:
            "linear-gradient(to right, white 1px, transparent 1px), linear-gradient(to bottom, white 1px, transparent 1px)",
          backgroundSize: "24px 24px",
        }}
        aria-hidden="true"
      />
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="relative grid gap-12 py-16 md:py-24 lg:grid-cols-2 lg:items-center lg:gap-10">
          {/* Copy */}
          <div data-aos="fade-up">
            <div className="mb-3 inline-flex items-center gap-1.5 rounded-full bg-white/10 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-slate-300">
              <Github className="h-3.5 w-3.5" />
              Open Source
            </div>
            <h2 className="text-3xl font-bold text-white md:text-4xl">
              Built in public. Built for developers.
            </h2>
            <p className="mt-4 text-lg leading-relaxed text-slate-300">
              SitBlinkSip is fully open source. Anyone can read the code,
              understand how the monitoring works, learn from it, or help
              make it better.
            </p>
            <ul className="mt-6 space-y-3">
              {highlights.map((item) => (
                <li key={item} className="flex items-start gap-2.5 text-sm text-slate-300">
                  <ScanEye className="mt-0.5 h-4 w-4 shrink-0 text-blue-400" />
                  {item}
                </li>
              ))}
            </ul>

            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-8 inline-flex items-center gap-2 rounded-lg bg-white px-6 py-3 font-semibold text-slate-900 shadow-sm transition-all hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900"
            >
              <Github className="h-4 w-4" />
              Star on GitHub
            </a>

            <div className="mt-10 flex flex-wrap gap-2">
              {techStack.map((tech) => (
                <span
                  key={tech}
                  className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-slate-400"
                >
                  {tech}
                </span>
              ))}
            </div>
          </div>

          {/* Terminal mockup */}
          <div data-aos="fade-up" data-aos-delay="100">
            <div className="overflow-hidden rounded-2xl border border-white/10 bg-slate-950/80 shadow-2xl">
              <div className="flex items-center gap-2 border-b border-white/10 bg-white/5 px-4 py-3">
                <div className="flex gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-red-400/80" />
                  <span className="h-2.5 w-2.5 rounded-full bg-amber-400/80" />
                  <span className="h-2.5 w-2.5 rounded-full bg-emerald-400/80" />
                </div>
                <span className="ml-2 text-xs text-slate-400">terminal</span>
              </div>
              <div className="space-y-3 p-5 font-mono text-sm">
                <p className="text-slate-500"># clone the repo</p>
                <p className="text-slate-200">
                  <span className="text-emerald-400">$</span> git clone{" "}
                  {GITHUB_URL}.git
                </p>
                <p className="text-slate-200">
                  <span className="text-emerald-400">$</span> cd SitBlinkSip
                </p>
                <p className="mt-4 text-slate-500"># run backend + frontend</p>
                <p className="text-slate-200">
                  <span className="text-emerald-400">$</span> docker compose up
                  --build
                </p>
                <p className="mt-4 flex items-center gap-2 text-xs text-slate-500">
                  <Star className="h-3.5 w-3.5" /> Apache 2.0 Licensed
                  <GitFork className="ml-3 h-3.5 w-3.5" /> Contributions welcome
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
