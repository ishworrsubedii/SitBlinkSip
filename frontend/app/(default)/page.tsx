export const metadata = {
  title: "SitBlinkSip — A Health Companion for Developers",
  description:
    "SitBlinkSip is an open-source wellness companion that watches your posture, blinking behavior, and hydration breaks while you work, then reminds you when it's time to reset.",
  openGraph: {
    title: "SitBlinkSip — A Health Companion for Developers",
    description:
      "Sit better, blink more, sip regularly. Open-source computer-vision monitoring for posture, eye blinking, and water breaks.",
    url: "https://sitblinksip.tech",
    siteName: "SitBlinkSip",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "SitBlinkSip — A Health Companion for Developers",
    description:
      "Sit better, blink more, sip regularly. Open-source computer-vision monitoring for posture, eye blinking, and water breaks.",
  },
};

import Hero from "@/components/hero-home";
import Pillars from "@/components/pillars";
import FeaturesPlanet from "@/components/features-planet";
import WhyBlinkMatters from "@/components/why-blink-matters";
import HowItWorks from "@/components/how-it-works";
import DesktopDownload from "@/components/desktop-download";
import LargeTestimonial from "@/components/large-testimonial";
import FutureVision from "@/components/future-vision";
import OpenSource from "@/components/open-source";
import DeveloperCredit from "@/components/developer-credit";
import Cta from "@/components/cta";

export default function Home() {
  return (
    <>
      <Hero />
      <Pillars />
      <FeaturesPlanet />
      <WhyBlinkMatters />
      <HowItWorks />
      <DesktopDownload />
      <LargeTestimonial />
      <FutureVision />
      <OpenSource />
      <DeveloperCredit />
      <Cta />
    </>
  );
}
