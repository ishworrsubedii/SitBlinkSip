 'use client';

import { Card } from '@/components/ui/card';
import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';

export default function TermsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        <Link href="/" className="inline-flex items-center text-blue-600 hover:text-blue-700 mb-6">
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Home
        </Link>
        
        <Card className="p-6 md:p-8">
          <h1 className="text-3xl font-bold mb-6">Terms of Service</h1>
          
          <div className="space-y-6 text-gray-600">
            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-3">1. Service Description</h2>
              <p>SitBlinkSip provides real-time posture monitoring and eye blink detection services through your device's camera. Our service processes all data locally on your device and does not store or transmit any video or image data to our servers.</p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-3">2. Data Processing</h2>
              <p>All video processing, posture analysis, and eye blink detection are performed locally on your device. No images or video streams are saved or transmitted to external servers.</p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-3">3. User Responsibilities</h2>
              <ul className="list-disc pl-5 space-y-2">
                <li>Ensure proper lighting and camera positioning for accurate monitoring</li>
                <li>Use the service in appropriate environments</li>
                <li>Not attempt to reverse engineer or modify the service</li>
                <li>Maintain the confidentiality of your account credentials</li>
              </ul>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-3">4. Service Limitations</h2>
              <p>The service provides general wellness monitoring and should not be considered medical advice. Consult healthcare professionals for medical concerns.</p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-3">5. Changes to Terms</h2>
              <p>We reserve the right to modify these terms at any time. Continued use of the service constitutes acceptance of updated terms.</p>
            </section>
          </div>
        </Card>
      </div>
    </div>
  );
}