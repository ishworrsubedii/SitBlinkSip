import PageIllustration from "@/components/page-illustration";
import Header from "@/components/ui/header";
import Footer from "@/components/ui/footer";

export const metadata = {
  title: 'Documentation - SitBlinkSip',
  description: 'Setup and usage documentation for SitBlinkSip application.',
};

export default function Documentation() {
  return (
    <>
      <Header />
      
      <main className="relative">
        <PageIllustration />

        {/* Hero section */}
        <section className="relative">
          <div className="mx-auto max-w-6xl px-4 sm:px-6">
            <div className="pt-32 pb-12 md:pt-40 md:pb-20">
              <div className="text-center">
                <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4">
                  Getting Started with
                  <span className="text-blue-600"> SitBlinkSip</span>
                </h1>
                <p className="text-xl text-slate-500 mb-8">
                  Learn how to set up and use SitBlinkSip for your digital wellness journey
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Documentation content */}
        <section className="relative border-t border-slate-100">
          <div className="mx-auto max-w-6xl px-4 sm:px-6 py-12 md:py-20">
            <div className="grid md:grid-cols-12 gap-8">
              {/* Sidebar */}
              <div className="md:col-span-3">
                <nav className="sticky top-24">
                  <ul className="space-y-3">
                    <li>
                      <a href="#installation" className="text-sm font-medium text-slate-600 hover:text-blue-500">
                        Installation
                      </a>
                    </li>
                    <li>
                      <a href="#requirements" className="text-sm font-medium text-slate-600 hover:text-blue-500">
                        System Requirements
                      </a>
                    </li>
                    <li>
                      <a href="#usage" className="text-sm font-medium text-slate-600 hover:text-blue-500">
                        Basic Usage
                      </a>
                    </li>
                  </ul>
                </nav>
              </div>

              {/* Main content */}
              <div className="md:col-span-9">
                {/* Installation section */}
                <div id="installation" className="mb-12">
                  <h2 className="text-2xl font-bold text-slate-900 mb-4">Installation</h2>
                  <p className="text-slate-500 mb-6">
                    SitBlinkSip can be easily installed using Docker. Follow these steps to get started:
                  </p>
                  
                  
                </div>
                <div className="mx-auto max-w-3xl" data-aos="zoom-y-out" data-aos-delay={600}>
  <div className="relative aspect-video rounded-2xl bg-[#1E1E1E] px-5 py-3 shadow-2xl">
    {/* Mac-style Terminal Header */}
    <div className="relative mb-6 flex items-center">
      <div className="absolute flex gap-1.5">
        <div className="h-3 w-3 rounded-full bg-[#FF605C]"></div>
        <div className="h-3 w-3 rounded-full bg-[#FFBD44]"></div>
        <div className="h-3 w-3 rounded-full bg-[#00CA4E]"></div>
      </div>
      <div className="mx-auto text-[13px] font-medium text-gray-400">
        sitblinksip — setup
      </div>
    </div>

    {/* Terminal Content */}
    <div className="font-mono text-[15px] [&_span]:opacity-0">
      <span className="animate-[code-1_15s_infinite] text-white">
        $ docker pull ishworrsubedii/sitblinksip
      </span>
      <br />
      <span className="animate-[code-2_15s_infinite] text-gray-400">
        Using default tag: latest
      </span>
      <br />
      <span className="animate-[code-3_15s_infinite] text-gray-400">
        latest: Pulling from ishworrsubedii/sitblinksip
      </span>
      <br />
      <span className="animate-[code-4_15s_infinite] text-gray-400">
        8b752a1a5ff2: Pulling fs layer
      </span>
      <br />
      <span className="animate-[code-5_15s_infinite] text-gray-400">
        67a4178b7d47: Pull complete
      </span>
      <br />
      <span className="animate-[code-6_15s_infinite] text-gray-400">
        Status: Downloaded newer image
      </span>
      <br />
      <span className="animate-[code-7_15s_infinite] text-white">
        $ docker run -p 8000:8000 ishworrsubedii/sitblinksip
      </span>
      <br />
      <span className="animate-[code-8_15s_infinite] text-emerald-400">
        ✨ Server started successfully
      </span>
      <br />
      <span className="animate-[code-9_15s_infinite] text-emerald-400">
        🚀 Application running at http://localhost:8000
      </span>
      <br />
      <span className="animate-[code-10_15s_infinite] text-blue-400">
        📊 View analytics: http://localhost:8000/dashboard
      </span>
    </div>
  </div>
</div>

                {/* Requirements section */}
                <div id="requirements" className="mb-12">
                  <h2 className="text-2xl font-bold text-slate-900 mb-4">System Requirements</h2>
                  <ul className="list-disc list-inside space-y-2 text-slate-500">
                    <li>Webcam access for posture detection</li>
                    <li>Modern web browser (Chrome, Firefox, Safari)</li>
                    <li>Docker (for local installation)</li>
                    <li>Minimum 4GB RAM recommended</li>
                    <li>Stable internet connection</li>
                  </ul>
                </div>

                {/* Usage section */}
                <div id="usage" className="mb-12">
                  <h2 className="text-2xl font-bold text-slate-900 mb-4">Basic Usage</h2>
                  <div className="prose prose-slate max-w-none">
                    <p className="text-slate-500 mb-4">
                      After installation, SitBlinkSip will request camera permissions to monitor your posture. 
                      The application uses local machine learning models to process all data on your device, 
                      ensuring your privacy.
                    </p>
                    
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      <Footer border={true} />
    </>
  );
}
