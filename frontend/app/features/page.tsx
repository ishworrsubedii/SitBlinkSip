import PageIllustration from "@/components/page-illustration";
import { Brain, Eye, Droplets, Spline, Activity, Bell, Sparkles } from "lucide-react";
import Header from "@/components/ui/header";
import Footer from "@/components/ui/footer";
import Link from "next/link";

export const metadata = {
  title: 'SitBlinkSip Features - Your Digital Wellness Companion',
  description: 'Simple tools for better workplace health: posture tracking, eye care, hydration reminders, and more. Try SitBlinkSip today!',
  keywords: 'workplace wellness, posture tracking, eye care, hydration reminders, digital health, office health',
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
            <div className="pt-40 pb-8 md:pt-52 md:pb-12">
              <div className="text-center">
                <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold text-gray-900 mb-4 font-playfair-display">
                  Transform Your{' '}
                  <span className="text-blue-600">
                    Workspace Health
                  </span>
                </h1>
                <h2 className="text-xl md:text-2xl text-gray-600 mb-8 max-w-3xl mx-auto">
                  Smart tools designed to transform your daily work routine into a healthier experience
                </h2>
              </div>
            </div>
          </div>
        </section>

        {/* Features grid */}
        <section className="relative">
          <div className="mx-auto max-w-6xl px-4 sm:px-6 py-8 md:py-16">
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
              {features.map((feature) => (
                <div 
                  key={feature.title}
                  className="relative flex flex-col p-6 bg-white rounded-2xl shadow-sm transition-all duration-300 hover:shadow-md hover:transform hover:-translate-y-1 border border-blue-100"
                >
                  <div className="w-12 h-12 rounded-xl mb-4 flex items-center justify-center bg-blue-50">
                    <feature.icon className="w-6 h-6 text-blue-600" />
                  </div>
                  <h3 className="text-2xl font-bold mb-4 text-blue-600">
                    {feature.title}
                  </h3>
                  <ul className="space-y-3 text-gray-600 flex-grow list-inside mb-4">
                    {feature.benefits.map((benefit, index) => (
                      <li key={index} className="flex items-start group">
                        <span className="inline-block w-2 h-2 rounded-full bg-blue-600 mt-2 mr-3 group-hover:scale-125 transition-transform"></span>
                        <span className="text-gray-700 group-hover:text-gray-900 transition-colors">
                          {benefit}
                        </span>
                      </li>
                    ))}
                  </ul>
                  {feature.comingSoon && (
                    <span className="absolute top-4 right-4 text-xs font-semibold px-3 py-1 rounded-full bg-blue-600 text-white">
                      Coming Soon
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Bottom CTA Section */}
        <section className="relative py-12 md:py-20">
          <div className="mx-auto max-w-6xl px-4 sm:px-6">
            <div className="relative rounded-2xl bg-blue-600 p-8 md:p-12">
              <div className="relative flex flex-col lg:flex-row justify-between items-center">
                <div className="text-center lg:text-left lg:max-w-2xl mb-8 lg:mb-0">
                  <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
                    Ready to Transform Your Work Health?
                  </h2>
                  <p className="text-blue-100">
                    Experience the future of workplace wellness with AI-powered health monitoring and real-time guidance
                  </p>
                </div>
                <div className="flex flex-col sm:flex-row gap-4">
                  <a 
                    href="/preview" 
                    className="group relative inline-flex items-center justify-center px-8 py-3 overflow-hidden font-medium bg-white rounded-md transition duration-300 ease-out shadow-md hover:shadow-lg"
                  >
                    <span className="absolute inset-0 flex items-center justify-center w-full h-full text-blue-600 duration-300 -translate-x-full bg-white group-hover:translate-x-0 ease">
                      <Eye className="w-5 h-5" />
                    </span>
                    <span className="absolute flex items-center justify-center w-full h-full text-blue-600 transition-all duration-300 transform group-hover:translate-x-full ease">
                      Try Demo
                    </span>
                    <span className="relative invisible">Try Demo</span>
                  </a>
                  <a 
                    className="inline-flex h-12 items-center rounded-lg bg-blue-600 px-6 text-white transition duration-150 ease-in-out hover:bg-blue-700"
                    href="/waitlist"
                  >
                    Get Started Free
                  </a>
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

const features = [
  {
    title: "Better Posture",
    description: "AI helps you sit correctly to prevent back pain. Get friendly reminders when you need to adjust your position.",
    icon: Spline,
    iconBg: "bg-blue-50",
    iconColor: "text-blue-500",
    benefits: ["Track your posture in real-time", "Get reminders to rest your eyes", "Stay hydrated throughout the day"],
  },
  {
    title: "Eye Protection",
    description: "Take care of your eyes with simple reminders. Follow the 20-20-20 rule: look away every 20 minutes.",
    icon: Eye,
    iconBg: "bg-violet-50",
    iconColor: "text-violet-500",
    benefits: ["Take care of your eyes", "Follow the 20-20-20 rule", "Get reminders to look away"],
  },
  {
    title: "Hydration Coach",
    description: "Smart water intake tracking with personalized reminders based on your activity level and workspace conditions.",
    icon: Droplets,
    iconBg: "bg-cyan-50",
    iconColor: "text-cyan-500",
    benefits: ["Track your water intake", "Get personalized reminders", "Stay hydrated throughout the day"],
  },
  {
    title: "Health Analytics",
    description: "Comprehensive dashboard with insights into your daily, weekly, and monthly wellness metrics and trends.",
    icon: Activity,
    iconBg: "bg-emerald-50",
    iconColor: "text-emerald-500",
    benefits: ["Comprehensive dashboard", "Insights into wellness metrics", "Trends analysis"],
  },
  {
    title: "Smart Notifications",
    description: "Context-aware alerts that adapt to your work patterns and help maintain optimal health habits.",
    icon: Bell,
    iconBg: "bg-amber-50",
    iconColor: "text-amber-500",
    benefits: ["Adapt to your work patterns", "Help maintain optimal health habits", "Context-aware alerts"],
  },
  {
    title: "AI Health Assistant",
    description: "Personalized recommendations and insights powered by machine learning to optimize your workspace wellness.",
    icon: Brain,
    iconBg: "bg-rose-50",
    iconColor: "text-rose-500",
    benefits: ["Personalized recommendations", "Machine learning insights", "Optimize workspace wellness"],
    comingSoon: true
  }
];