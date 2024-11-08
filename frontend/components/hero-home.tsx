"use client";

import Image from "next/image";
import PageIllustration from "@/components/page-illustration";
import { Brain, Eye, Droplets, Spline } from "lucide-react";
import { useEffect, useState } from "react";

export default function HeroHome() {
  const phrases = [
    "For Tech Experts",
    "For Office Pros",
    "For Creators & Coders",
    "For Learners & Teachers"
  ];
  
  const [currentPhraseIndex, setCurrentPhraseIndex] = useState(0);
  const [currentText, setCurrentText] = useState("");
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    const typingSpeed = 100; // Speed for typing
    const deletingSpeed = 50; // Speed for deleting
    const pauseTime = 2000; // Time to pause at full phrase

    const typeWriter = () => {
      const currentPhrase = phrases[currentPhraseIndex];
      
      if (isDeleting) {
        // Deleting text
        setCurrentText(currentPhrase.substring(0, currentText.length - 1));
        if (currentText === "") {
          setIsDeleting(false);
          setCurrentPhraseIndex((prev) => (prev + 1) % phrases.length);
        }
      } else {
        // Typing text
        setCurrentText(currentPhrase.substring(0, currentText.length + 1));
        if (currentText === currentPhrase) {
          // Pause at full phrase
          setTimeout(() => setIsDeleting(true), pauseTime);
          return;
        }
      }
    };

    const timer = setTimeout(
      typeWriter,
      isDeleting ? deletingSpeed : typingSpeed
    );

    return () => clearTimeout(timer);
  }, [currentText, isDeleting, currentPhraseIndex]);

  return (
    <section className="relative min-h-screen" aria-label="Main Hero Section">
      <PageIllustration />
      
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="pb-12 pt-40 md:pt-52">
          {/* Hero Content */}
          <div className="pb-6 text-center md:pb-8">
            <h1 className="text-4xl font-bold text-slate-900 md:text-5xl lg:text-6xl">
              AI-Powered Health Monitoring
            </h1>
            <div className="mx-auto mb-8 h-14 pt-4">
              <span className="relative inline-block text-blue-600/90 text-3xl font-bold md:text-4xl lg:text-5xl">
                <span className="invisible">For Computer Professionals</span>
                <span className="absolute left-1/2 -translate-x-1/2 whitespace-nowrap">
                  {currentText}
                  <span className="animate-blink ml-1 inline-block h-8 w-[2px] bg-blue-600/90"></span>
                </span>
              </span>
            </div>
            <div className="mx-auto max-w-2xl pt-4">
              <p className="mb-8 text-lg text-slate-600/80">
                Enhance your well-being with advanced AI monitoring for posture, eye health, and work-break balance. 
                Our intelligent system provides real-time tracking and personalized recommendations.
              </p>
              <div className="flex flex-col items-center justify-center space-y-4 sm:flex-row sm:space-x-4 sm:space-y-0">
                <a
                  className="inline-flex h-12 items-center rounded-lg bg-blue-600 px-6 text-white transition duration-150 ease-in-out hover:bg-blue-700"
                  href="/signup"
                >
                  Start Free Trial
                </a>
                <a
                  className="inline-flex h-12 items-center rounded-lg bg-slate-100 px-6 text-slate-600 transition duration-150 ease-in-out hover:bg-slate-200"
                  href="/features"
                >
                  Explore Features
                </a>
              </div>
            </div>
          </div>

          {/* Feature Cards */}
          <div className="mt-12 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
            <div className="group flex flex-col items-center rounded-xl bg-white p-6 shadow-sm transition-all hover:shadow-md">
              <div className="animate-float mb-4 rounded-full bg-blue-50/50 p-3">
                <Spline className="h-6 w-6 text-blue-600/80" />
              </div>
              <h3 className="mb-2 font-semibold text-slate-800">Posture Analysis</h3>
              <p className="text-center text-sm text-slate-600">
                Real-time posture monitoring with AI-driven correction guidance
              </p>
            </div>

            <div className="group flex flex-col items-center rounded-xl bg-white p-6 shadow-sm transition-all hover:shadow-md">
              <div className="animate-float mb-4 rounded-full bg-violet-50/50 p-3">
                <Eye className="h-6 w-6 text-violet-600/80" />
              </div>
              <h3 className="mb-2 font-semibold text-slate-800">Eye Care Monitor</h3>
              <p className="text-center text-sm text-slate-600">
                Smart blink detection and eye strain prevention system
              </p>
            </div>

            <div className="group flex flex-col items-center rounded-xl bg-white p-6 shadow-sm transition-all hover:shadow-md">
              <div className="animate-float mb-4 rounded-full bg-cyan-50/50 p-3">
                <Brain className="h-6 w-6 text-cyan-600/80" />
              </div>
              <h3 className="mb-2 font-semibold text-slate-800">Break Timer</h3>
              <p className="text-center text-sm text-slate-600">
                Intelligent break scheduling with hydration reminders
              </p>
            </div>
       
 
          </div>

          {/* Scroll Indicator */}
          <div className="mt-10 text-center" data-aos="fade-up" data-aos-delay="100">
            <div className="flex flex-col items-center">
              <span className="mb-3 text-sm text-slate-600">Scroll to explore more</span>
              <svg className="h-6 w-6 animate-bounce text-slate-400" fill="none" strokeWidth="2" viewBox="0 0 24 24" stroke="currentColor">
                <path d="M19 14l-7 7m0 0l-7-7m7 7V3"></path>
              </svg>
            </div>
          </div>

          {/* Tech Stack Section */}
          <div className="mt-32" data-aos="fade-up">
            <h3 className="mb-8 text-center text-xl font-semibold text-slate-800">
              Powered by Advanced Technology
            </h3>
            
            <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
              {[
                {
                  icon: <Brain className="h-5 w-5 text-blue-600" />,
                  name: "Computer Vision",
                  description: "Advanced posture tracking",
                  borderColor: "border-blue-200",
                  hoverBorder: "hover:border-blue-400",
                },
                {
                  icon: <Eye className="h-5 w-5 text-violet-600" />,
                  name: "Eye Tracking",
                  description: "Blink detection & monitoring",
                  borderColor: "border-violet-200",
                  hoverBorder: "hover:border-violet-400",
                },
                {
                  icon: <Brain className="h-5 w-5 text-emerald-600" />,
                  name: "AI Analytics",
                  description: "Health pattern recognition",
                  borderColor: "border-emerald-200",
                  hoverBorder: "hover:border-emerald-400",
                },
                {
                  icon: <Spline className="h-5 w-5 text-orange-600" />,
                  name: "Real-time Processing",
                  description: "Instant health insights",
                  borderColor: "border-orange-200",
                  hoverBorder: "hover:border-orange-400",
                },
              ].map((tech) => (
                <div
                  key={tech.name}
                  className={`group relative border-2 ${tech.borderColor} ${tech.hoverBorder} rounded-lg p-4 transition-all duration-300 hover:-translate-y-1`}
                >
                  <div className="flex items-center space-x-3">
                    <div className="rounded-md bg-white p-1.5">
                      {tech.icon}
                    </div>
                    <h4 className="font-medium text-slate-800">{tech.name}</h4>
                  </div>
                  <p className="mt-2 text-sm text-slate-600">
                    {tech.description}
                  </p>
                  <div className="absolute bottom-0 left-0 h-1 w-0 bg-gradient-to-r from-slate-100 to-slate-200 transition-all duration-300 group-hover:w-full"></div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
