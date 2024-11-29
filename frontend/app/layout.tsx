import "./css/style.css";

import { Inter } from "next/font/google";
import { Toaster } from 'sonner';

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata = {
  title: {
    default: "SitBlinkSip - AI-Powered Health Monitoring for Computer Users",
    template: "%s | SitBlinkSip"
  },
  description: "AI-powered wellness assistant for proper posture detection, eye strain prevention, and hydration tracking. Prevent computer vision syndrome and maintain healthy work habits.",
  keywords: [
    "posture detection",
    "eye strain prevention",
    "blink detection",
    "computer vision syndrome",
    "workplace wellness",
    "digital health monitoring",
    "back pain prevention",
    "eye health",
    "hydration tracking",
    "ergonomic workspace",
    "AI health assistant",
    "workplace ergonomics"
  ],
  authors: [{ name: "SitBlinkSip" }],
  creator: "SitBlinkSip",
  publisher: "SitBlinkSip",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://sitblinksip.vercel.app",
    title: "SitBlinkSip - AI-Powered Health Monitoring for Computer Users",
    description: "Prevent computer vision syndrome and maintain healthy work habits with AI-powered wellness monitoring.",
    siteName: "SitBlinkSip"
  },
  twitter: {
    card: "summary_large_image",
    title: "SitBlinkSip - AI-Powered Health Monitoring",
    description: "Prevent computer vision syndrome and maintain healthy work habits with AI-powered wellness monitoring.",
    creator: "@sitblinksip"
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
  verification: {
    google: "your-google-verification-code",
  }
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
