import { Play, Focus, BellRing, TrendingUp } from "lucide-react";

const steps = [
  {
    number: "01",
    title: "Start working",
    description: "Open SitBlinkSip before you start your work session.",
    icon: Play,
  },
  {
    number: "02",
    title: "Stay focused",
    description:
      "Keep working normally while SitBlinkSip quietly monitors posture, blinking, and time since your last break.",
    icon: Focus,
  },
  {
    number: "03",
    title: "Get reminded",
    description:
      "Receive a gentle notification when poor posture, prolonged lack of blinking, or a hydration break needs your attention.",
    icon: BellRing,
  },
  {
    number: "04",
    title: "Build better habits",
    description: "Develop healthier work routines over time, session after session.",
    icon: TrendingUp,
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="relative scroll-mt-24">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="py-16 md:py-24">
          <div className="mx-auto max-w-2xl text-center" data-aos="fade-up">
            <div className="mb-3 inline-flex rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-slate-500">
              How It Works
            </div>
            <h2 className="text-3xl font-bold text-slate-900 md:text-4xl">
              Technology that works quietly in the background.
            </h2>
            <p className="mt-4 text-lg text-slate-600">
              SitBlinkSip helps you do your work without forgetting yourself —
              here's what actually happens once you hit start.
            </p>
          </div>

          <div className="relative mt-14 grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
            {/* connecting line */}
            <div
              className="pointer-events-none absolute left-0 right-0 top-6 hidden h-px bg-gradient-to-r from-transparent via-gray-200 to-transparent lg:block"
              aria-hidden="true"
            />
            {steps.map((step, index) => (
              <div
                key={step.number}
                className="relative flex flex-col items-center text-center"
                data-aos="fade-up"
                data-aos-delay={index * 100}
              >
                <div className="relative z-10 flex h-12 w-12 items-center justify-center rounded-full border-2 border-blue-100 bg-white shadow-sm">
                  <step.icon className="h-5 w-5 text-blue-600" />
                </div>
                <span className="mt-4 text-xs font-bold tracking-wider text-blue-300">
                  {step.number}
                </span>
                <h3 className="mt-1 text-lg font-semibold text-slate-900">
                  {step.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-600">
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
