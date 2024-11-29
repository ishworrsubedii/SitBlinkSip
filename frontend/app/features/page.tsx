import PageIllustration from "@/components/page-illustration";
import { Brain, Eye, Droplets, Spline, Activity, Bell, Sparkles } from "lucide-react";
import Header from "@/components/ui/header";
import Footer from "@/components/ui/footer";
import Link from "next/link";

export const metadata = {
  title: 'Health Monitoring Features - Posture Detection & Eye Care',
  description: 'Advanced features for workplace wellness including AI posture detection, eye strain prevention, and hydration tracking. Transform your digital health with smart monitoring.',
  keywords: [
    'posture detection features',
    'eye strain prevention',
    'workplace wellness features',
    'health monitoring tools',
    'ergonomic workspace',
    'computer vision syndrome prevention',
    'AI health tracking',
    'blink rate monitoring',
    'hydration tracking',
    'workplace health analytics'
  ],
  openGraph: {
    title: 'Digital Health Monitoring Features - SitBlinkSip',
    description: 'Comprehensive workplace wellness features including AI posture detection, eye strain prevention, and health analytics.',
    images: [
      {
        url: 'https://sitblinksip.com/images/features-preview.jpg',
        width: 1200,
        height: 630,
        alt: 'SitBlinkSip Features Overview'
      }
    ]
  }
};

const SitBlinkSipFeatures = () => {
  return (
    <>
      <Header />
      
      <main className="relative">
        <PageIllustration />
        <div className="gap-10 flex items-center justify-center py-20 text-white">
        </div>
        {/* Hero Section */}
        <section className="relative py-16 md:py-4 mb-1">
          <div className="mx-auto max-w-6xl px-4 sm:px-6">
            <div className="text-center">
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold text-gray-900 mb-4 font-playfair-display">
                Transform Your <span className="text-blue-600">Workspace Health</span>
              </h1>
              <h2 className="text-xl md:text-2xl text-gray-600 mb-8 max-w-3xl mx-auto">
                Intelligent tools to optimize posture, eye care, hydration, and more for a healthier work experience.
              </h2>
              <div className="flex justify-center gap-4 mt-8">
                <a 
                  href="/preview" 
                  className="inline-flex h-12 items-center rounded-lg bg-blue-600 px-6 text-white transition duration-150 ease-in-out hover:bg-blue-700"
                >
                  Try Demo
                </a>
                <a 
                  className="inline-flex h-12 items-center rounded-lg bg-blue-200 px-6 text-blue-600 transition duration-150 ease-in-out hover:bg-blue-300"
                  href="/waitlist"
                >
                  Get Started
                </a>
              </div>
            </div>
          </div>
        </section>

        {/* Features Grid */}
        <section className="relative py-12 md:py-20">
          <div className="mx-auto max-w-6xl px-4 sm:px-6">
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
                  <p className="text-gray-600 flex-grow mb-4">
                    {feature.description}
                  </p>
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
                    Experience the future of workplace wellness with AI-powered health monitoring and real-time guidance.
                  </p>
                </div>
                <div className="flex flex-col sm:flex-row gap-4">
                  <a 
                    href="/preview" 
                    className="inline-flex h-12 items-center rounded-lg bg-white px-6 text-blue-600 transition duration-150 ease-in-out hover:bg-blue-100"
                  >
                    Try Demo
                  </a>
                  <a 
                    className="inline-flex h-12 items-center rounded-lg bg-blue-200 px-6 text-blue-600 transition duration-150 ease-in-out hover:bg-blue-300"
                    href="/waitlist"
                  >
                    Get Started
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
};

const features = [
  {
    title: "Better Posture",
    description: "AI-powered posture tracking and personalized reminders help you maintain an ergonomic seated position and prevent back pain.",
    icon: Spline,
    benefits: [
      "Real-time posture monitoring",
      "Customizable posture correction alerts",
      "Metrics to track your progress"
    ]
  },
  {
    title: "Eye Protection",
    description: "Intelligent eye care features like the 20-20-20 rule and screen brightness adjustments ensure your eyes stay healthy and comfortable.",
    icon: Eye,
    benefits: [
      "Automated 20-20-20 break reminders",
      "Dynamic screen brightness adjustment",
      "Personalized eye strain prevention"
    ]
  },
  {
    title: "Hydration Coach",
    description: "Smart water intake tracking with personalized reminders based on your activity level and workspace conditions keeps you optimally hydrated.",
    icon: Droplets,
    benefits: [
      "Automatic water intake logging",
      "Personalized hydration recommendations",
      "Alerts for timely water breaks"
    ]
  },
  {
    title: "Health Analytics",
    description: "Comprehensive dashboard with insights into your daily, weekly, and monthly wellness metrics and trends.",
    icon: Activity,
    benefits: [
      "Detailed health and productivity metrics",
      "Personalized recommendations based on data",
      "Trends analysis for long-term improvement" 
    ]
  },
  {
    title: "Smart Notifications",
    description: "Context-aware alerts that adapt to your work patterns and help maintain optimal health habits.",
    icon: Bell,
    benefits: [
      "Intelligent scheduling of reminders",
      "Customizable notification preferences",
      "Seamless integration with your workflow"
    ]
  },
  {
    title: "AI Health Assistant",
    description: "Personalized recommendations and insights powered by machine learning to optimize your workspace wellness.",
    icon: Brain,
    benefits: [
      "AI-driven wellness coaching",
      "Tailored suggestions for improvement",
      "Continuous learning and adaptation"
    ],
    comingSoon: true
  }
];

export default SitBlinkSipFeatures;