import Image from "next/image";
import PageIllustration from "@/components/page-illustration";
import Avatar01 from "@/public/images/avatar-01.jpg";
import Avatar02 from "@/public/images/avatar-02.jpg";
import Avatar03 from "@/public/images/avatar-03.jpg";
import Avatar04 from "@/public/images/avatar-04.jpg";
import Avatar05 from "@/public/images/avatar-05.jpg";
import Avatar06 from "@/public/images/avatar-06.jpg";
import { Brain, Eye, Droplets, Spline } from "lucide-react";

export default function HeroHome() {
  return (
    <section className="relative">
      <PageIllustration />
      
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        
        <div className="pb-12 pt-32 md:pb-20 md:pt-40">
       
          <div className="pb-12 text-center md:pb-16">
            <h1
              className="mb-6 text-5xl font-bold text-slate-900 md:text-6xl"
              data-aos="zoom-y-out"
            >
              Your Digital Wellness
              <br className="hidden sm:block" />
              <span className="text-blue-600"> Companion</span>
            </h1>
            <div className="mx-auto max-w-3xl">
              <p
                className="mb-8 text-lg text-slate-700"
                data-aos="zoom-y-out"
                data-aos-delay={150}
              >
                Transform your work-from-home experience with our AI-powered health companion. 
                Maintain perfect posture, protect your eyes, and stay hydrated while boosting 
                your productivity and well-being.
              </p>
              <div
                className="mx-auto max-w-xs sm:flex sm:max-w-none sm:justify-center"
                data-aos="zoom-y-out"
                data-aos-delay={300}
              >
                <a
                  className="btn group mb-4 w-full bg-blue-600 text-white hover:bg-blue-700 sm:mb-0 sm:w-auto"
                  href="#0"
                >
                  <span className="relative inline-flex items-center">
                    Start Your Health Journey
                    <span className="ml-1 transition-transform group-hover:translate-x-0.5">
                      →
                    </span>
                  </span>
                </a>
                <a
                  className="btn w-full bg-slate-100 text-slate-700 hover:bg-slate-200 sm:ml-4 sm:w-auto"
                  href="#0"
                >
                  View Features
                </a>
              </div>
            </div>
          </div>
          

          {/* Main Features */}
          <div className="mb-10 grid grid-cols-3 gap-6" data-aos="fade-up">
            <div className="group flex flex-col items-center rounded-xl bg-white p-6 shadow-sm transition-all hover:shadow-md">
              <div className="animate-float mb-4 rounded-full bg-blue-50 p-3">
                <Spline className="h-6 w-6 text-blue-600" />
              </div>
              <h3 className="mb-2 font-semibold text-slate-800">Posture Guardian</h3>
              <p className="text-center text-sm text-slate-600">
                AI-powered posture detection to prevent back pain and promote healthy sitting habits
              </p>
            </div>

            <div className="group flex flex-col items-center rounded-xl bg-white p-6 shadow-sm transition-all hover:shadow-md">
              <div className="animate-float mb-4 rounded-full bg-violet-50 p-3">
                <Eye className="h-6 w-6 text-violet-600" />
              </div>
              <h3 className="mb-2 font-semibold text-slate-800">Eye Care Timer</h3>
              <p className="text-center text-sm text-slate-600">
                Smart reminders using the 20-20-20 rule to reduce digital eye strain
              </p>
            </div>

            <div className="group flex flex-col items-center rounded-xl bg-white p-6 shadow-sm transition-all hover:shadow-md">
              <div className="animate-float mb-4 rounded-full bg-cyan-50 p-3">
                <Droplets className="h-6 w-6 text-cyan-600" />
              </div>
              <h3 className="mb-2 font-semibold text-slate-800">Hydration Coach</h3>
              <p className="text-center text-sm text-slate-600">
                Personalized water intake tracking and smart reminder system
              </p>
            </div>
          </div>

          {/* Hero Content */}
          

          <div className="mb-8 text-center">
            <p className="text-sm font-medium text-slate-600">
              Powered by Advanced Technologies
            </p>
            <div className="mt-3 flex items-center justify-center gap-6">
              {["Computer Vision", "Machine Learning", "Data Analytics", "Next Js"].map((tech) => (
                <span key={tech} className="text-xs text-slate-500">{tech}</span>
              ))}
            </div>
          </div>

          
        </div>
      </div>
    </section>
  );
}
