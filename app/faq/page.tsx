"use client";

import React, { useState } from 'react';
import { ChevronDown, MessageCircle, Zap, Shield, Clock, HelpCircle } from 'lucide-react';
import PageIllustration from '@/components/page-illustration';
import Header from '@/components/ui/header';
import Footer from '@/components/ui/footer';
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import Script from 'next/script';

// Structured data for FAQ
const structuredData = {
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "What is SitBlinkSip?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "SitBlinkSip is a free, open-source desktop app that uses your webcam to watch sitting posture and eye blink frequency locally on your machine, then nudges you with reminders to fix your posture, blink more, and take water breaks — with no camera frame ever leaving your computer."
      }
    }
  ]
};

const FAQPage = () => {
  const faqs = [
    {
      category: "Getting Started",
      icon: <Zap className="w-5 h-5" />,
      questions: [
        {
          question: "What is SitBlinkSip?",
          answer: "SitBlinkSip is a free, open-source desktop companion for developers and other computer-heavy workers. It watches your posture and blink rate through your webcam, runs quietly in the system tray, and reminds you when it's time to sit up, blink more, or grab some water."
        },
        {
          question: "Does SitBlinkSip require any special equipment?",
          answer: "No — just a computer with a webcam. SitBlinkSip Desktop is a native app that installs once and runs fully offline in the background. No account, no browser tab, and no extra hardware required."
        },
        {
          question: "How does the posture and blink detection work?",
          answer: "SitBlinkSip Desktop runs computer-vision models directly on your machine to track posture and blink rate in real time. It doesn't show a live camera feed by default — press F6 to reveal a preview on demand — and it doesn't stream or save any video."
        },
        {
          question: "Will it slow down my machine while I'm coding?",
          answer: "SitBlinkSip is designed to run lightly in the background alongside your usual dev tools — it only needs the webcam and a small amount of CPU for detection, and it never uploads anything, so there's no network overhead either."
        },
        {
          question: "Which platforms are supported?",
          answer: "SitBlinkSip Desktop is available now for Linux (.deb). Windows support is on the roadmap; macOS isn't currently planned. Check the GitHub releases page for the latest builds."
        }
      ]
    },
    {
      category: "Privacy & Security",
      icon: <Shield className="w-5 h-5" />,
      questions: [
        {
          question: "Does SitBlinkSip send my camera feed anywhere?",
          answer: "No. All posture and blink detection runs locally on your device. No video, frame, or image ever leaves your machine — there's no server for it to go to, since SitBlinkSip has no backend."
        },
        {
          question: "Is my usage data stored anywhere?",
          answer: "Your settings (like reminder intervals and alert preferences) are stored locally on your machine. SitBlinkSip doesn't require an account and doesn't send analytics or personal data to any server."
        }
      ]
    },
    {
      category: "Open Source",
      icon: <Clock className="w-5 h-5" />,
      questions: [
        {
          question: "Is SitBlinkSip free?",
          answer: "Yes. SitBlinkSip is completely free and open source under the MIT license — no subscriptions, no paywalls, no premium tier."
        },
        {
          question: "Can I contribute or self-host it?",
          answer: "Yes — the source for the desktop app is on GitHub, and contributions, bug reports, and feature requests are welcome. See the GitHub repo for build instructions and the contributing guide."
        }
      ]
    }
  ];

  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="grow">
        <div className="relative">
          <PageIllustration />
          
          <div className="pt-32 pb-12 md:pt-40 md:pb-20">
            {/* Page header */}
            <div className="text-center pb-12 md:pb-16 max-w-3xl mx-auto px-4 sm:px-6">
              <Badge 
                variant="outline" 
                className="px-4 py-1 border-blue-200 text-blue-700 bg-blue-50 mb-4"
              >
                Help Center
              </Badge>
              <h1 className="text-4xl md:text-5xl font-bold mb-4">
                Frequently Asked <span className="text-blue-600">Questions</span>
              </h1>
              <p className="text-xl text-slate-600">
                Find answers to common questions about our service
              </p>
            </div>

            {/* Search section */}
            
            {/* FAQ sections */}
            <div className="max-w-3xl mx-auto px-4 sm:px-6 space-y-8">
              {faqs.map((section, index) => (
                <FAQSection key={index} {...section} />
              ))}
            </div>

            {/* Contact support card */}
            <div className="max-w-3xl mx-auto px-4 sm:px-6 mt-16">
              <Card className="bg-gradient-to-br from-blue-50 to-blue-100/50 border-blue-100">
                <CardContent className="p-8">
                  <div className="flex flex-col md:flex-row items-center justify-between gap-6">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 rounded-full bg-blue-600/10 flex items-center justify-center">
                        <MessageCircle className="w-6 h-6 text-blue-600" />
                      </div>
                      <div>
                        <h3 className="text-xl font-bold text-slate-900">Still have questions?</h3>
                        <p className="text-slate-600">Open an issue on GitHub and we'll help out</p>
                      </div>
                    </div>
                    <a
                      href="https://github.com/ishworrsubedii/desktop-sitblinksip/issues"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-colors"
                    >
                      Open an Issue
                    </a>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </main>
      <Footer />
      <Script id="faq-schema" type="application/ld+json">
        {JSON.stringify(structuredData)}
      </Script>
    </div>
  );
};

interface FAQSectionProps {
  category: string;
  icon: React.ReactElement;
  questions: { question: string; answer: string }[];
}

const FAQSection = ({ category, icon, questions }: FAQSectionProps) => {
  return (
    <div>
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 rounded-lg bg-blue-600/10 flex items-center justify-center">
          {React.cloneElement(icon, { className: 'text-blue-600' })}
        </div>
        <h2 className="text-xl font-bold text-slate-900">{category}</h2>
      </div>
      <div className="space-y-3">
        {questions.map((item, index) => (
          <FAQItem key={index} {...item} />
        ))}
      </div>
    </div>
  );
};

interface FAQItemProps {
  question: string;
  answer: string;
}

const FAQItem = ({ question, answer }: FAQItemProps) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="border border-slate-200 rounded-xl bg-white/50 backdrop-blur-sm overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-4 text-left"
      >
        <span className="font-medium text-slate-900">{question}</span>
        <ChevronDown 
          className={`w-5 h-5 text-slate-500 transition-transform ${
            isOpen ? 'rotate-180' : ''
          }`} 
        />
      </button>
      <div 
        className={`overflow-hidden transition-all duration-300 ease-in-out ${
          isOpen ? 'max-h-96' : 'max-h-0'
        }`}
      >
        <div className="p-4 pt-0 text-slate-600">
          {answer}
        </div>
      </div>
    </div>
  );
};

export default FAQPage;