import { Github, Star, GitFork, Linkedin, Twitter, Globe, Mail } from "lucide-react";

const GITHUB_URL = "https://github.com/ishworrsubedii/desktop-sitblinksip";

const socials = [
  { label: "GitHub", href: "https://github.com/ishworrsubedii", icon: Github },
  { label: "LinkedIn", href: "https://www.linkedin.com/in/ishworrsubedii/", icon: Linkedin },
  { label: "X / Twitter", href: "https://x.com/ishworr_", icon: Twitter },
  { label: "Portfolio", href: "https://ishwor-subedi.com.np/", icon: Globe },
  { label: "Email", href: "mailto:ishworr.subedi@gmail.com", icon: Mail },
];

const techStack = ["Python", "PySide6", "MediaPipe", "OpenCV", "NumPy", "pynput"];

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
              SitBlinkSip Desktop is fully open source, Apache-2.0 licensed.
              Read the code, learn how the on-device detection works, or send
              a pull request.
            </p>

            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-6 inline-flex items-center gap-2 rounded-lg bg-white px-6 py-3 font-semibold text-slate-900 shadow-sm transition-all hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900"
            >
              <Github className="h-4 w-4" />
              Star on GitHub
            </a>

            <div className="mt-8 flex flex-wrap gap-2">
              {techStack.map((tech) => (
                <span
                  key={tech}
                  className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-slate-400"
                >
                  {tech}
                </span>
              ))}
            </div>

            <div className="mt-10 flex items-center gap-3 border-t border-white/10 pt-6">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-blue-600 to-blue-400 text-xs font-bold text-white">
                IS
              </div>
              <div className="text-sm text-slate-400">
                <span className="font-semibold text-white">Ishwor Subedi</span>
                {" "}&middot; Machine Learning Engineer, built solo
              </div>
              <div className="ml-auto flex gap-1.5">
                {socials.map((social) => (
                  <a
                    key={social.label}
                    href={social.href}
                    target={social.href.startsWith("mailto:") ? undefined : "_blank"}
                    rel={social.href.startsWith("mailto:") ? undefined : "noopener noreferrer"}
                    aria-label={social.label}
                    title={social.label}
                    className="flex h-8 w-8 items-center justify-center rounded-full border border-white/10 text-slate-400 transition-colors hover:border-blue-400/40 hover:text-blue-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                  >
                    <social.icon className="h-3.5 w-3.5" />
                  </a>
                ))}
              </div>
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
                  <span className="text-emerald-400">$</span> cd desktop-sitblinksip
                </p>
                <p className="mt-4 text-slate-500"># run from source</p>
                <p className="text-slate-200">
                  <span className="text-emerald-400">$</span> pip install -r requirements.txt
                </p>
                <p className="text-slate-200">
                  <span className="text-emerald-400">$</span> python -m sitblinksip_desktop
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
