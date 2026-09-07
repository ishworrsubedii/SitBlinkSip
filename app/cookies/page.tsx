import React from 'react';
import Link from 'next/link';
import { ArrowLeft, Shield } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Cookie Policy | SitBlinkSip',
  description: 'Learn about how SitBlinkSip uses cookies to enhance your browsing experience and protect your privacy.',
  openGraph: {
    title: 'Cookie Policy | SitBlinkSip',
    description: 'Learn about how SitBlinkSip uses cookies to enhance your browsing experience and protect your privacy.',
    url: 'https://sitblinksip.tech/cookies',
    siteName: 'SitBlinkSip',
    type: 'website',
  },
  twitter: {
    card: 'summary',
    title: 'Cookie Policy | SitBlinkSip',
    description: 'Learn about how SitBlinkSip uses cookies to enhance your browsing experience.',
  },
  robots: {
    index: true,
    follow: true,
  }
};

export default function CookiesPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        <Link href="/" className="inline-flex items-center text-blue-600 hover:text-blue-700 mb-6">
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Home
        </Link>

        <Card className="p-6 md:p-8">
          <div className="flex items-center gap-3 mb-6">
            <Shield className="w-8 h-8 text-blue-600" />
            <h1 className="text-3xl font-bold">Cookie Policy</h1>
          </div>

          <div className="space-y-6 text-gray-600">
            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-3">1. Introduction</h2>
              <p>
                At SitBlinkSip, we use cookies to enhance your experience on our website. This policy explains what cookies are, how we use them, and your choices regarding their use.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-3">2. What are Cookies?</h2>
              <p>
                Cookies are small text files stored on your device when you visit a website. They help us remember your preferences and improve your user experience.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-3">3. How We Use Cookies</h2>
              <ul className="list-disc pl-5 space-y-2">
                <li>To remember your preferences and settings</li>
                <li>To analyze site traffic and usage patterns</li>
                <li>To provide personalized content and ads</li>
                <li>To improve site performance and security</li>
              </ul>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-3">4. Types of Cookies We Use</h2>
              <ul className="list-disc pl-5 space-y-2">
                <li><strong>Essential Cookies:</strong> Necessary for the website to function properly.</li>
                <li><strong>Performance Cookies:</strong> Help us understand how visitors interact with our site.</li>
                <li><strong>Functional Cookies:</strong> Enable enhanced functionality and personalization.</li>
                <li><strong>Targeting Cookies:</strong> Used to deliver relevant ads and track ad performance.</li>
              </ul>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-3">5. Managing Cookies</h2>
              <p>
                You can manage your cookie preferences through your browser settings. Please note that disabling cookies may affect your experience on our site.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-3">6. Changes to This Policy</h2>
              <p>
                We may update this cookie policy from time to time. We encourage you to review this policy periodically for any changes.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-3">7. Contact Us</h2>
              <p>
                If you have any questions about our cookie policy, please contact us at support@sitblinksip.com.
              </p>
            </section>
          </div>
        </Card>
      </div>
    </div>
  );
} 