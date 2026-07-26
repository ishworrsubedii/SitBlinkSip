import { LineChart, History, Watch, Sparkles } from "lucide-react";

const upcoming = [
  {
    title: "Detailed analytics",
    description: "Deeper health and productivity insights drawn from your sessions.",
    icon: LineChart,
  },
  {
    title: "Historical trends",
    description: "Track posture and blink patterns over days, weeks, and months.",
    icon: History,
  },
  {
    title: "Wearable integration",
    description: "Support for smartwatches and other wearables for fuller monitoring.",
    icon: Watch,
  },
  {
    title: "Personalized insights",
    description: "Recommendations tailored to your own habits over time.",
    icon: Sparkles,
  },
];

export default function FutureVision() {
  return (
    <section className="relative">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="py-12 md:py-16">
          <div className="mx-auto max-w-2xl text-center" data-aos="fade-up">
            <div className="mb-3 inline-flex rounded-full bg-amber-50 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-amber-700">
              What's Next
            </div>
            <h2 className="text-2xl font-bold text-slate-900 md:text-3xl">
              We're just getting started.
            </h2>
            <p className="mt-3 text-base text-slate-600">
              These are directions we're exploring — not yet part of the app.
            </p>
          </div>

          <div className="mt-10 grid gap-4 grid-cols-2 lg:grid-cols-4">
            {upcoming.map((item, index) => (
              <div
                key={item.title}
                data-aos="fade-up"
                data-aos-delay={index * 75}
                className="flex flex-col items-center rounded-xl border border-dashed border-gray-200 bg-white/60 p-5 text-center transition-colors hover:border-gray-300"
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gray-100">
                  <item.icon className="h-5 w-5 text-gray-500" />
                </div>
                <h3 className="mt-3 text-sm font-semibold text-slate-800">
                  {item.title}
                </h3>
                <p className="mt-1 text-xs leading-relaxed text-slate-500">
                  {item.description}
                </p>
                <span className="mt-3 rounded-full bg-gray-100 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-gray-500">
                  Planned
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
