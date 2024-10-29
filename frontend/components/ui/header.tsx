import Link from "next/link";
import Logo from "./logo";

import { Github } from "lucide-react";
export default function Header() {
  return (
    <header className="fixed top-2 z-30 w-full md:top-6">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="relative flex h-14 items-center justify-between gap-3 rounded-2xl bg-white/90 px-3 shadow-lg shadow-black/[0.03] backdrop-blur-sm before:pointer-events-none before:absolute before:inset-0 before:rounded-[inherit] before:border before:border-transparent before:[background:linear-gradient(theme(colors.gray.100),theme(colors.gray.200))_border-box] before:[mask-composite:exclude_!important] before:[mask:linear-gradient(white_0_0)_padding-box,_linear-gradient(white_0_0)]">
          {/* Logo section - reduced flex-1 */}
          <div className="flex items-center gap-4">
            <Logo />
            <h1 className="font-display text-xl font-black tracking-tight ml-2">
              <span className="bg-gradient-to-r from-blue-600 via-blue-500 to-blue-400 bg-clip-text text-transparent">
                Sit<span className="font-extrabold">Blink</span>
                <span className="font-bold text-blue-400">Sip</span>
              </span>
            </h1>
          </div>

          {/* Navigation Links - centered */}
          <nav className="flex-1 flex justify-center">
            <ul className="flex items-center gap-12">
              <li>
                <Link
                  href="/"
                  className="text-sm font-medium text-gray-600 hover:text-blue-500 transition-colors"
                >
                  Home
                </Link>
              </li>
              <li>
                <Link
                  href="/features"
                  className="text-sm font-medium text-gray-600 hover:text-blue-500 transition-colors"
                >
                  Features
                </Link>
              </li>
              <li>
                <Link
                  href="/docs"
                  className="text-sm font-medium text-gray-600 hover:text-blue-500 transition-colors"
                >
                  Documentation
                </Link>
              </li>
            </ul>
          </nav>

          {/* Right side buttons */}
          <div className="flex items-center gap-3">
            <Link
              href="/demo"
              className="hidden md:inline-flex btn-sm bg-white font-medium text-gray-800 shadow transition-colors hover:bg-gray-50"
            >
              Live Demo
            </Link>
            <Link
              href="https://github.com/ishworrsubedii/sitblinksip"
              className="btn-sm bg-gradient-to-r from-gray-800 to-gray-900 text-white shadow-sm hover:from-gray-900 hover:to-black transition-all"
            >
              <Github className="w-4 h-4 mr-2" />
              Star on GitHub
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}
