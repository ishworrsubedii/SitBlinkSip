import Image from "next/image";
import Stripes from "@/public/images/stripes-dark.svg";

export default function Cta() {
  return (
    <section>
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div
          className="relative overflow-hidden rounded-2xl text-center shadow-xl before:pointer-events-none before:absolute before:inset-0 before:-z-10 before:rounded-2xl before:bg-gray-900"
          data-aos="zoom-y-out"
        >
          {/* Glow */}
          <div
            className="absolute bottom-0 left-1/2 -z-10 -translate-x-1/2 translate-y-1/2"
            aria-hidden="true"
          >
            <div className="h-56 w-[480px] rounded-full border-[20px] border-blue-500 blur-3xl" />
          </div>
          {/* Stripes illustration */}
          <div
            className="pointer-events-none absolute left-1/2 top-0 -z-10 -translate-x-1/2 transform"
            aria-hidden="true"
          >
            <Image
              className="max-w-none"
              src={Stripes}
              width={768}
              height={432}
              alt="Stripes"
            />
          </div>
          <div className="px-4 py-12 md:px-12 md:py-20">
            <h2 className="mb-4 text-3xl font-bold text-gray-200 md:text-4xl">
              Code Smarter, Stay Healthier
            </h2>
            <p className="mb-8 text-lg text-gray-400">
              Monitor your posture and eye strain while you code
            </p>
            <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-3">
              <div className="text-gray-300">
                <span className="block font-bold">AI-Powered</span>
                <span className="text-sm">Smart code suggestions</span>
              </div>
              <div className="text-gray-300">
                <span className="block font-bold">Health-Focused</span>
                <span className="text-sm">Ergonomic monitoring</span>
              </div>
              <div className="text-gray-300">
                <span className="block font-bold">Team-Ready</span>
                <span className="text-sm">Seamless collaboration</span>
              </div>
            </div>
            <div className="mx-auto max-w-xs space-y-4 sm:flex sm:max-w-none sm:justify-center sm:space-x-4 sm:space-y-0">
              <a
                className="btn group flex w-full items-center justify-center bg-gradient-to-t from-blue-600 to-blue-500 bg-[length:100%_100%] bg-[bottom] text-white shadow transition-all hover:bg-[length:100%_150%] sm:w-auto"
                href="/signup"
              >
                Start Install
                <span className="ml-2 text-blue-300 transition-transform group-hover:translate-x-0.5">
                  →
                </span>
              </a>
              <a
                className="btn flex w-full items-center justify-center border border-gray-700 text-gray-300 hover:bg-gray-800 sm:w-auto"
                href="/demo"
              >
                Watch Demo
                <svg className="ml-2 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </a>
            </div>
            <div className="mt-8 text-sm text-gray-400">
              <span>Trusted by developers from</span>
              <div className="mt-2 flex justify-center space-x-6">
                <span className="font-semibold">TCP</span>
                <span className="font-semibold">BrentInnovate</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
