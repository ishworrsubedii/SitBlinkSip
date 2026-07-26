import { Armchair, Eye, Droplets, ArrowRight } from "lucide-react";

const pillars = [
  {
    letter: "Sit",
    title: "Posture Monitoring",
    tagline: "Stay aware of your posture while you work.",
    description:
      "SitBlinkSip watches for slouching and leaning through your webcam and lets you know when it's time to sit back up.",
    icon: Armchair,
    tint: "text-blue-600",
    chip: "bg-blue-50",
    ring: "group-hover:ring-blue-100",
    bar: "bg-blue-500",
  },
  {
    letter: "Blink",
    title: "Eye Blink Detection",
    tagline: "Give your eyes the attention they deserve.",
    description:
      "Long focus sessions change how often and how completely we blink. SitBlinkSip keeps track and reminds you when it matters.",
    icon: Eye,
    tint: "text-violet-600",
    chip: "bg-violet-50",
    ring: "group-hover:ring-violet-100",
    bar: "bg-violet-500",
    href: "#research",
    linkLabel: "Why it matters",
  },
  {
    letter: "Sip",
    title: "Water Break Reminders",
    tagline: "Stay hydrated while you stay focused.",
    description:
      "Deep work makes it easy to forget the basics. Get timely nudges to reach for water and keep hydration on track.",
    icon: Droplets,
    tint: "text-cyan-600",
    chip: "bg-cyan-50",
    ring: "group-hover:ring-cyan-100",
    bar: "bg-cyan-500",
  },
];

export default function Pillars() {
  return (
    <section id="features" className="relative scroll-mt-24">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="py-12 md:py-20">
          <div className="mx-auto max-w-3xl pb-12 text-center md:pb-16" data-aos="fade-up">
            <div className="mb-3 inline-flex rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-slate-500">
              Sit · Blink · Sip
            </div>
            <h2 className="text-3xl font-bold text-slate-900 md:text-4xl">
              Three habits. One companion.
            </h2>
            <p className="mt-4 text-lg text-slate-600">
              Long hours at a screen make it easy to forget your body. SitBlinkSip
              keeps an eye on three simple things so you don't have to.
            </p>
          </div>

          <div className="grid gap-6 sm:grid-cols-3">
            {pillars.map((pillar, index) => (
              <div
                key={pillar.title}
                data-aos="fade-up"
                data-aos-delay={index * 100}
                className={`group relative flex flex-col rounded-2xl border border-gray-100 bg-white p-6 shadow-sm ring-1 ring-transparent transition-all duration-300 hover:-translate-y-1 hover:shadow-lg ${pillar.ring}`}
              >
                <div
                  className={`mb-5 flex h-12 w-12 items-center justify-center rounded-xl ${pillar.chip}`}
                >
                  <pillar.icon className={`h-6 w-6 ${pillar.tint}`} />
                </div>
                <h3 className="text-xl font-bold text-slate-900">
                  {pillar.letter}
                  <span className="ml-2 text-sm font-medium text-slate-400">
                    {pillar.title}
                  </span>
                </h3>
                <p className={`mt-2 text-sm font-medium ${pillar.tint}`}>
                  {pillar.tagline}
                </p>
                <p className="mt-3 flex-1 text-sm leading-relaxed text-slate-600">
                  {pillar.description}
                </p>
                {pillar.href && (
                  <a
                    href={pillar.href}
                    className={`mt-4 inline-flex items-center gap-1 text-sm font-semibold ${pillar.tint} transition-colors hover:underline`}
                  >
                    {pillar.linkLabel}
                    <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
                  </a>
                )}
                <div
                  className={`absolute bottom-0 left-6 right-6 h-0.5 scale-x-0 rounded-full ${pillar.bar} transition-transform duration-300 group-hover:scale-x-100`}
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
