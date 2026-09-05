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
        "text": "SitBlinkSip is an AI-powered health monitoring software that uses your device's camera to analyze sitting posture and eye blink frequency. It provides gentle reminders to improve posture and eye health, helping prevent eye strain and maintain proper ergonomics."
      }
    },
    // Add more FAQ items based on your faqs array
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
          answer: "SitBlinkSip is health monitoring software that uses your device's camera to analyze your sitting posture and eye blink frequency. It provides gentle reminders to improve posture and eye health, helping to prevent eye strain and dryness. The software also suggests exercises and sends notifications to support better overall health."
        },
        {
          question:"Does SitBlinkSip require any special equipment?",
          answer: "No, you only need a device with a camera. The web dashboard runs in any modern browser (Chrome, Firefox, Safari) with a stable internet connection — or, if you'd rather not keep a tab open, SitBlinkSip Desktop is a native app for Windows, macOS, and Linux that runs fully offline in the background. No additional hardware is required either way."
        },
        {
          question: "How does the posture monitoring system work?",
          answer: "Our AI-powered system uses your device's camera to analyze your sitting position in real-time. It tracks key points on your body and provides gentle reminders when it detects poor posture. All processing is done on realtime without saving any data to our servers to ensure privacy."
        },
        {
          question: "What equipment do I need to get started?",
          answer: "Just a device with a camera. Use the web dashboard in any modern browser (Chrome, Firefox, Safari), or install SitBlinkSip Desktop — a native app for Windows, macOS, and Linux that needs no browser or internet connection once installed. No additional hardware is required either way."
        }
      ]
    },
    {
      category: "Privacy & Security",
      icon: <Shield className="w-5 h-5" />,
      questions: [
        {
          question: "How is my data protected?",
          answer: "We take privacy seriously. Frames from your device's camera are securely transmitted to our servers for processing. We ensure that no images are stored on our servers. The only data we retain are your preferences and aggregated statistics to enhance your experience. All data is encrypted using industry-standard protocols."
        },
        {
          question: "Can I delete my data?",
          answer: "Yes, you have full control over your data. You can delete your account and all associated data at any time from your account settings. Once deleted, all your data is permanently removed from our servers."
        }
      ]
    },
    {
      category: "Subscription & Billing",
      icon: <Clock className="w-5 h-5" />,
      questions: [
        {
          question: "Can I cancel my subscription at any time?",
          answer: "Yes, you can cancel your subscription anytime with no questions asked. If you cancel, you'll continue to have access to the service until the end of your current billing period. We also offer a 30-day money-back guarantee for new subscribers."
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
                        <p className="text-slate-600">Our support team is here to help</p>
                      </div>
                    </div>
                    <button className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-colors">
                      Contact Support
                    </button>
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