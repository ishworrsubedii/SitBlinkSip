import type { ElementType } from "react";
import {
  Monitor,
  Focus,
  Eye,
  EyeOff,
  Droplets,
  Frown,
  ChevronRight,
  ChevronDown,
  Info,
  ArrowUpRight,
  Camera,
  ScanEye,
  BellRing,
  Sparkles,
  Quote,
} from "lucide-react";

const researchFlow = [
  { label: "Screen time", icon: Monitor },
  { label: "Intense visual focus", icon: Focus },
  { label: "Changes in blinking behavior", icon: Eye },
  { label: "Less frequent / incomplete blinks", icon: EyeOff },
  { label: "Tear-film instability", icon: Droplets },
  { label: "Dryness & digital eye strain", icon: Frown },
];

const productFlow = [
  { label: "Camera", icon: Camera },
  { label: "Blink detection", icon: ScanEye },
  { label: "Pattern monitoring", icon: Eye },
  { label: "Timely reminder", icon: BellRing },
  { label: "More awareness", icon: Sparkles },
  { label: "Healthier habits", icon: Droplets },
];

const insights = [
  {
    number: "01",
    title: "Blinking changes during screen use",
    summary:
      "Research has identified altered blinking patterns as one factor associated with digital eye strain and dry-eye symptoms during prolonged digital-device use.",
    citation: "Digital Screen Use and Dry Eye: A Review",
    url: "https://pubmed.ncbi.nlm.nih.gov/33181547/",
    cta: "Read the research",
  },
  {
    number: "02",
    title: "Reduced blinking can affect the tear film",
    summary:
      "Reviews describe decreased blink frequency as an important factor in ocular surface dryness, since longer gaps between blinks can let the tear film break up before the next blink.",
    citation: "Digital Eye Strain: Updated Perspectives",
    url: "https://pubmed.ncbi.nlm.nih.gov/39308959/",
    cta: "View study",
  },
  {
    number: "03",
    title: "Blink reminders have been studied",
    summary:
      "A randomized controlled trial evaluating blink-reminder software reported improvements in blink rate and dry-eye-related symptom scores among visual display terminal users.",
    citation:
      "Efficacy of blink software in improving the blink rate and dry eye symptoms in visual display terminal users",
    url: "https://pubmed.ncbi.nlm.nih.gov/34571605/",
    cta: "Read the study",
  },
];

const furtherReading = [
  {
    label: "Blink Animation Software to Improve Blinking and Dry Eye Symptoms",
    url: "https://pubmed.ncbi.nlm.nih.gov/26164310/",
  },
  {
    label: "Blink rate, incomplete blinks and computer vision syndrome",
    url: "https://pubmed.ncbi.nlm.nih.gov/23538437/",
  },
];

function FlowDiagram({
  steps,
  tone,
}: {
  steps: { label: string; icon: ElementType }[];
  tone: "amber" | "blue";
}) {
  const chip =
    tone === "amber"
      ? "border-amber-200 bg-amber-50 text-amber-800"
      : "border-blue-200 bg-blue-50 text-blue-700";
  const iconTone = tone === "amber" ? "text-amber-500" : "text-blue-500";

  return (
    <div className="flex flex-col items-center gap-2 lg:flex-row lg:flex-wrap lg:justify-center lg:gap-2">
      {steps.map((step, index) => (
        <div key={step.label} className="flex flex-col items-center lg:flex-row">
          <div
            className={`flex items-center gap-2 rounded-full border px-3.5 py-2 text-xs font-semibold sm:text-sm ${chip}`}
          >
            <step.icon className="h-4 w-4 shrink-0" />
            <span>{step.label}</span>
          </div>
          {index < steps.length - 1 && (
            <>
              <ChevronDown className={`my-1 h-4 w-4 shrink-0 lg:hidden ${iconTone}`} />
              <ChevronRight className={`mx-1 hidden h-4 w-4 shrink-0 lg:block ${iconTone}`} />
            </>
          )}
        </div>
      ))}
    </div>
  );
}

export default function WhyBlinkMatters() {
  return (
    <section id="research" className="relative bg-slate-50 scroll-mt-24">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="py-16 md:py-24">
          {/* Header */}
          <div className="mx-auto max-w-3xl text-center" data-aos="fade-up">
            <div className="mb-3 inline-flex rounded-full bg-violet-100 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-violet-700">
              The Science
            </div>
            <h2 className="text-3xl font-bold text-slate-900 md:text-4xl">
              Why blinking matters when you work at a screen.
            </h2>
            <p className="mt-4 text-lg leading-relaxed text-slate-600">
              Blinking is one of the eye's natural ways of maintaining the tear
              film across the ocular surface. During prolonged digital-device
              use, blinking behavior can change — and reduced or incomplete
              blinking has been associated with dry-eye symptoms and digital
              eye strain.
            </p>
          </div>

          {/* Research flow */}
          <div className="mt-12 rounded-2xl border border-gray-100 bg-white p-6 shadow-sm md:p-10" data-aos="fade-up">
            <FlowDiagram steps={researchFlow} tone="amber" />
          </div>

          {/* Insight cards */}
          <div className="mt-12 grid gap-6 sm:grid-cols-3">
            {insights.map((insight, index) => (
              <a
                key={insight.number}
                href={insight.url}
                target="_blank"
                rel="noopener noreferrer"
                data-aos="fade-up"
                data-aos-delay={index * 100}
                className="group flex flex-col rounded-2xl border border-gray-100 bg-white p-6 shadow-sm transition-all duration-300 hover:-translate-y-1 hover:shadow-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500"
              >
                <span className="text-sm font-bold text-violet-300">
                  {insight.number}
                </span>
                <h3 className="mt-2 text-base font-semibold text-slate-900">
                  {insight.title}
                </h3>
                <p className="mt-2 flex-1 text-sm leading-relaxed text-slate-600">
                  {insight.summary}
                </p>
                <p className="mt-4 text-xs italic text-slate-400">{insight.citation}</p>
                <span className="mt-3 inline-flex items-center gap-1 text-sm font-semibold text-violet-600">
                  {insight.cta}
                  <ArrowUpRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
                </span>
              </a>
            ))}
          </div>

          {/* Further reading */}
          <div
            className="mt-6 flex flex-col items-center gap-x-6 gap-y-2 text-sm text-slate-500 sm:flex-row sm:justify-center"
            data-aos="fade-up"
          >
            <span className="font-medium text-slate-600">Further reading:</span>
            {furtherReading.map((ref) => (
              <a
                key={ref.url}
                href={ref.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 underline decoration-slate-300 underline-offset-4 hover:text-violet-600 hover:decoration-violet-400"
              >
                {ref.label}
                <ArrowUpRight className="h-3 w-3" />
              </a>
            ))}
          </div>

          {/* Disclaimer */}
          <div
            className="mx-auto mt-10 flex max-w-3xl items-start gap-3 rounded-xl border border-slate-200 bg-white/60 p-4 text-sm text-slate-500"
            data-aos="fade-up"
          >
            <Info className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" />
            <p>
              Digital eye strain has multiple contributing factors. Blinking
              behavior is one part of the picture, alongside things like
              prolonged near-focus work, visual demands, environmental
              conditions, and individual eye health. SitBlinkSip focuses on
              blink awareness as one simple, measurable habit that can support
              healthier screen-time routines — it doesn't diagnose, treat, or
              cure eye conditions.
            </p>
          </div>

          {/* Transition to product */}
          <div className="mx-auto mt-20 max-w-3xl text-center" data-aos="fade-up">
            <h3 className="text-2xl font-bold text-slate-900 md:text-3xl">
              That's where SitBlinkSip comes in.
            </h3>
            <p className="mt-4 text-lg leading-relaxed text-slate-600">
              Instead of waiting until your eyes feel tired, SitBlinkSip helps
              you become more aware of your blinking habits while you work.
            </p>
          </div>

          <div className="mt-10 rounded-2xl border border-blue-100 bg-blue-50/40 p-6 shadow-sm md:p-10" data-aos="fade-up">
            <FlowDiagram steps={productFlow} tone="blue" />
          </div>

          <blockquote
            className="mx-auto mt-12 flex max-w-2xl flex-col items-center gap-3 text-center"
            data-aos="fade-up"
          >
            <Quote className="h-6 w-6 text-blue-300" />
            <p className="text-lg font-medium text-slate-700">
              We don't want to interrupt your work. We want to help you build
              healthier habits while you work.
            </p>
          </blockquote>
        </div>
      </div>
    </section>
  );
}
