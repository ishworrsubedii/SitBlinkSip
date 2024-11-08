import Image from "next/image";
import Stripes from "@/public/images/stripes-dark.svg";
import { Sparkles, Check } from "lucide-react";
import PricingSection from '@/components/PricingSection';

export default function Cta() {
  return (
    <section className="relative overflow-hidden">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div
          className="relative overflow-hidden rounded-2xl bg-gradient-to-b from-slate-900 to-slate-800 text-center shadow-2xl"
          data-aos="zoom-y-out"
        >
          {/* Glow effects */}
          <div className="absolute inset-0">
            <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2">
              <div className="h-40 sm:h-56 w-[280px] sm:w-[480px] rounded-full bg-blue-500/20 blur-3xl" />
            </div>
            <div className="absolute inset-0 bg-grid-white/5 bg-[size:20px_20px] [mask-image:radial-gradient(white,transparent_70%)]" />
          </div>

          <div className="relative px-4 py-12 sm:py-16 md:py-20">
            <div className="mx-auto max-w-3xl">
              <h2 className="mb-4 text-2xl sm:text-3xl md:text-4xl font-bold text-white">
                Transform Your Workspace Health
              </h2>
              <p className="mb-8 text-sm sm:text-base md:text-lg text-gray-300 px-2 sm:px-0">
                Smart monitoring for better health and productivity
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8 px-2 sm:px-6">
                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-4 sm:p-5">
                  <span className="block font-bold text-base sm:text-lg text-gray-200 mb-1">Real-time</span>
                  <span className="text-xs sm:text-sm text-gray-400">Health monitoring</span>
                </div>
                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-4 sm:p-5">
                  <span className="block font-bold text-base sm:text-lg text-gray-200 mb-1">AI-Powered</span>
                  <span className="text-xs sm:text-sm text-gray-400">Smart analytics</span>
                </div>
                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-4 sm:p-5">
                  <span className="block font-bold text-base sm:text-lg text-gray-200 mb-1">Personalized</span>
                  <span className="text-xs sm:text-sm text-gray-400">Health insights</span>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row items-center justify-center gap-3 px-4 sm:px-0">
                <a
                  className="w-full sm:w-auto group inline-flex items-center justify-center rounded-lg bg-blue-600 px-6 sm:px-8 py-3 text-white transition-all hover:bg-blue-700"
                  href="/signup"
                >
                  Get Started
                  <Sparkles className="ml-2 h-4 w-4 transition-transform group-hover:scale-110" />
                </a>
                <a
                  className="w-full sm:w-auto inline-flex items-center justify-center rounded-lg border border-gray-700 bg-gray-800/50 px-6 sm:px-8 py-3 text-gray-200 backdrop-blur-sm transition-colors hover:border-blue-500 hover:text-blue-400"
                  href="/demo"
                >
                  Watch Demo
                  <svg className="ml-2 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Pricing Section */}
      <PricingSection className="pt-16 md:pt-20" standalone={false} hideTitle={false} />

      {/* Newsletter Section */}
      <div className="mx-auto mt-20 max-w-4xl px-4">
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-b from-blue-600/90 to-blue-800/90 p-8 shadow-2xl">
          {/* Background decoration */}
          <div className="absolute inset-0 bg-grid-white/10 bg-[size:20px_20px] [mask-image:radial-gradient(white,transparent_70%)]" />
          <div className="absolute -left-20 -top-20 h-[400px] w-[400px] rounded-full bg-blue-500/20 blur-3xl" />
          <div className="absolute -right-20 -bottom-20 h-[400px] w-[400px] rounded-full bg-purple-500/20 blur-3xl" />
          
          <div className="relative">
            <div className="flex flex-col items-center text-center">
              <div className="inline-flex rounded-full bg-blue-500/10 px-3 py-1 text-sm font-medium text-blue-100 ring-1 ring-inset ring-blue-400/20 mb-4">
                Newsletter
              </div>
              <h2 className="text-3xl font-bold text-white mb-2">
                Stay Updated with Health Tech
              </h2>
              <p className="max-w-2xl text-lg text-blue-100/80 mb-8">
                Join our community and receive the latest updates on digital wellness, productivity tips, 
                and exclusive insights delivered straight to your inbox.
              </p>
              
              <form className="w-full max-w-md" aria-label="Newsletter Form">
                <div className="flex flex-col sm:flex-row gap-3">
                  <div className="relative flex-grow">
                    <input 
                      type="email" 
                      placeholder="Enter your email" 
                      className="w-full rounded-lg border-0 bg-white/10 px-4 py-3 text-white placeholder-blue-200/60 backdrop-blur-sm ring-1 ring-inset ring-white/20 focus:ring-2 focus:ring-white/30 focus:outline-none"
                      aria-label="Email address"
                    />
                  </div>
                  <button 
                    className="group relative inline-flex items-center justify-center rounded-lg bg-white px-6 py-3 text-blue-600 transition-all hover:bg-blue-50 focus:outline-none focus:ring-2 focus:ring-white/30"
                    aria-label="Subscribe to newsletter"
                  >
                    <span className="font-semibold">Subscribe</span>
                    <Sparkles className="ml-2 h-4 w-4 transition-transform group-hover:scale-110" />
                  </button>
                </div>
              </form>

              <div className="mt-8 flex items-center justify-center gap-x-8 text-sm text-blue-200/80">
                <div className="flex items-center gap-2">
                  <Check className="h-4 w-4" />
                  <span>No spam</span>
                </div>
                <div className="flex items-center gap-2">
                  <Check className="h-4 w-4" />
                  <span>Unsubscribe anytime</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
