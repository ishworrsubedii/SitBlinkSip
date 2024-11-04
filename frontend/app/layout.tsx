import "./css/style.css";

import { Inter } from "next/font/google";
import { Toaster } from 'sonner';

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata = {
  title: "SitBlinkSip",
  description: "SitBlinkSip is a tool that helps you improve your posture and eye health by monitoring your posture and eye movements.",
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
