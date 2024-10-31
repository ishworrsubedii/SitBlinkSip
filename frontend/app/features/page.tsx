import PageIllustration from "@/components/page-illustration";
import { Brain, Eye, Droplets, Spline, Activity, Bell } from "lucide-react";
import Header from "@/components/ui/header";
import Footer from "@/components/ui/footer";
import Link from "next/link";

export const metadata = {
  title: 'Features - SitBlinkSip',
  description: 'Discover all the powerful features of SitBlinkSip that help maintain your digital health.',
};


export default function Features() {
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
                  Powerful Features for Your
                  <span className="text-blue-600"> Digital Health</span>
                </h1>
                <p className="text-xl text-slate-500 mb-8">
                  Comprehensive tools to monitor and improve your workspace wellness
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Features grid */}
        <section className="relative border-t border-slate-100">
          <div className="mx-auto max-w-6xl px-4 sm:px-6 py-12 md:py-20">
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
              {features.map((feature) => (
                <div 
                  key={feature.title}
                  className="relative flex flex-col p-6 bg-white rounded-xl shadow-sm transition-shadow hover:shadow-md"
                >
                  <div className={`w-12 h-12 rounded-full mb-4 flex items-center justify-center ${feature.iconBg}`}>
                    <feature.icon className={`w-6 h-6 ${feature.iconColor}`} />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                  <p className="text-slate-500 flex-grow">{feature.description}</p>
                  {feature.comingSoon && (
                    <span className="absolute top-4 right-4 text-xs font-semibold px-2 py-1 rounded-full bg-blue-100 text-blue-600">
                      Coming Soon
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>

      <Footer border={true} />
    </>
  );
}

const features = [
  {
    title: "Posture Guardian",
    description: "Real-time AI-powered posture detection to prevent back pain and promote healthy sitting habits throughout your workday.",
    icon: Spline,
    iconBg: "bg-blue-50",
    iconColor: "text-blue-500"
  },
  {
    title: "Eye Care Monitor",
    description: "Track your blink rate and receive timely reminders using the 20-20-20 rule to reduce digital eye strain.",
    icon: Eye,
    iconBg: "bg-violet-50",
    iconColor: "text-violet-500"
  },
  {
    title: "Hydration Coach",
    description: "Smart water intake tracking with personalized reminders based on your activity level and workspace conditions.",
    icon: Droplets,
    iconBg: "bg-cyan-50",
    iconColor: "text-cyan-500"
  },
  {
    title: "Health Analytics",
    description: "Comprehensive dashboard with insights into your daily, weekly, and monthly wellness metrics and trends.",
    icon: Activity,
    iconBg: "bg-emerald-50",
    iconColor: "text-emerald-500",
  },
  {
    title: "Smart Notifications",
    description: "Context-aware alerts that adapt to your work patterns and help maintain optimal health habits.",
    icon: Bell,
    iconBg: "bg-amber-50",
    iconColor: "text-amber-500"
  },
  {
    title: "AI Health Assistant",
    description: "Personalized recommendations and insights powered by machine learning to optimize your workspace wellness.",
    icon: Brain,
    iconBg: "bg-rose-50",
    iconColor: "text-rose-500",
    comingSoon: true
  }
];