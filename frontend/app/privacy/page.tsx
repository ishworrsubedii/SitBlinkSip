import React from 'react';
import Link from 'next/link';
import { ArrowLeft, Shield } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Privacy Policy | SitBlinkSip',
  description: 'Read about how SitBlinkSip protects your privacy and handles your personal information.',
  openGraph: {
    title: 'Privacy Policy | SitBlinkSip',
    description: 'Read about how SitBlinkSip protects your privacy and handles your personal information.',
    url: 'https://sitblinksip.tech/privacy',
    siteName: 'SitBlinkSip',
    type: 'website',
  },
  twitter: {
    card: 'summary',
    title: 'Privacy Policy | SitBlinkSip',
    description: 'Read about how SitBlinkSip protects your privacy.',
  },
  robots: {
    index: true,
    follow: true,
  }
};

export default function PrivacyPage() {
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
            <h1 className="text-3xl font-bold">Privacy Policy</h1>
          </div>

          <div className="space-y-6 text-gray-600">
            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-3">1. Data Collection & Processing</h2>
              <ul className="list-disc pl-5 space-y-2">
                <li>All video processing occurs on Realtime and never saved to our servers</li>
                <li>No images or video streams are saved or transmitted to our servers</li>
                <li>Only anonymous usage statistics and preferences are stored</li>
                <li>Camera access is required but footage never leaves your device</li>
              </ul>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-3">2. Data Storage</h2>
              <p>We store minimal user data including:</p>
              <ul className="list-disc pl-5 space-y-2">
                <li>Email address (for authentication)</li>
                <li>User preferences and settings</li>
                <li>Usage statistics</li>
              </ul>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-3">3. Data Security</h2>
              <p>We implement several security measures:</p>
              <ul className="list-disc pl-5 space-y-2">
              <li>All video processing occurs on Realtime and never saved to our servers</li>
              <li>End-to-end encryption for any data transmission</li>
                <li>Regular security audits and updates</li>
                <li>No storage of sensitive biometric data</li>
              </ul>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-3">4. Your Rights</h2>
              <p>You have the right to:</p>
              <ul className="list-disc pl-5 space-y-2">
                <li>Access your stored data</li>
                <li>Request data deletion</li>
                <li>Opt-out of anonymous analytics</li>
                <li>Export your data</li>
              </ul>
            </section>
          </div>
        </Card>
      </div>
    </div>
  );
} 