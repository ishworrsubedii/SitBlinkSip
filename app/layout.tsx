import "./css/style.css";
import "./globals.css";

import { Inter } from "next/font/google";
import { Toaster } from 'sonner';
import { Metadata, Viewport } from 'next';

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL('https://sitblinksip.tech'),
  title: {
    default: 'SitBlinkSip - Sit Right, Blink Bright, Sip Well',
    template: '%s | SitBlinkSip'
  },
  description: 'Sit Right, Blink Bright, Sip Well - SitBlinkSip helps you maintain good posture, prevent eye strain, and stay hydrated while working. Get real-time feedback and wellness reminders.',
  keywords: [
    'posture reminder app',
    'blink reminder for developers',
    'eye strain prevention for programmers',
    'developer health app',
    'coding posture app',
    'webcam posture detection',
    'open source wellness app',
    'water break reminder',
    'ergonomics for programmers',
    'digital eye strain',
    'workplace wellness',
    'desktop tray app for developers',
    'privacy-first health app',
    'sit right blink bright sip well',
  ],
  authors: [{ name: 'SitBlinkSip Team' }],
  creator: 'SitBlinkSip',
  publisher: 'SitBlinkSip',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  icons: {
    icon: '/favicon.ico',
  },
  openGraph: {
    title: 'SitBlinkSip - Sit Right, Blink Bright, Sip Well',
    description: 'SitBlinkSip helps you maintain good posture, prevent eye strain, and stay hydrated while working. Get real-time feedback and wellness reminders.',
    url: 'https://sitblinksip.tech',
    siteName: 'SitBlinkSip',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'SitBlinkSip - Sit Right, Blink Bright, Sip Well',
    description: 'SitBlinkSip helps you maintain good posture, prevent eye strain, and stay hydrated while working.',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body
        className={`${inter.variable} bg-gray-50 font-inter tracking-tight text-gray-900 antialiased`}
      >
        <div className="flex min-h-screen flex-col overflow-hidden supports-[overflow:clip]:overflow-clip">
          {children}
          <Toaster richColors position="top-right" />
        </div>
      </body>
    </html>
  );
}
