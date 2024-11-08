"use client";

import Link from "next/link";
import Logo from "./logo";
import { Github, Menu, X } from "lucide-react";
import { useState } from "react";

export default function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="fixed top-2 z-30 w-full md:top-6">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="relative flex h-14 items-center justify-between gap-3 rounded-2xl bg-white/90 px-3 shadow-lg shadow-black/[0.03] backdrop-blur-sm before:pointer-events-none before:absolute before:inset-0 before:rounded-[inherit] before:border before:border-transparent before:[background:linear-gradient(theme(colors.gray.100),theme(colors.gray.200))_border-box] before:[mask-composite:exclude_!important] before:[mask:linear-gradient(white_0_0)_padding-box,_linear-gradient(white_0_0)]">
          {/* Logo section */}
          <div className="flex items-center gap-4">
            <Logo />
            <h1 className="font-display text-xl font-black tracking-tight ml-2">
              <span className="bg-gradient-to-r from-blue-600 via-blue-500 to-blue-400 bg-clip-text text-transparent">
                Sit<span className="font-extrabold">Blink</span>
                <span className="font-bold text-blue-400">Sip</span>
              </span>
            </h1>
          </div>

          {/* Mobile menu button */}
          <button
            className="md:hidden"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? (
              <X className="h-6 w-6 text-gray-600" />
            ) : (
              <Menu className="h-6 w-6 text-gray-600" />
            )}
          </button>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex flex-1 justify-center">
            <ul className="flex items-center gap-8">
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
                  href="/pricing"
                  className="text-sm font-medium text-gray-600 hover:text-blue-500 transition-colors"
                >
                  Pricing
                </Link>
              </li>
              <li>
                <Link
                  href="/blog"
                  className="text-sm font-medium text-gray-600 hover:text-blue-500 transition-colors"
                >
                  Blog
                </Link>
              </li>
              <li>
                <Link
                  href="/faq"
                  className="text-sm font-medium text-gray-600 hover:text-blue-500 transition-colors"
                >
                  FAQ
                </Link>
              </li>
            </ul>
          </nav>

          {/* Desktop buttons */}
          <div className="hidden md:flex items-center gap-3">
            <Link href="/preview" className="btn-sm bg-white font-medium text-gray-800 shadow transition-colors hover:bg-gray-50">
              Preview
            </Link>
            <Link href="/get-started" className="btn-sm bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-sm hover:from-blue-700 hover:to-blue-800 transition-all">
              Get Started
            </Link>
          </div>

          {/* Mobile Navigation Menu */}
          {mobileMenuOpen && (
            <div className="absolute top-full left-0 right-0 mt-2 bg-white rounded-lg shadow-lg p-4 md:hidden">
              <nav className="flex flex-col space-y-4">
                <Link
                  href="/"
                  className="text-sm font-medium text-gray-600 hover:text-blue-500 transition-colors"
                >
                  Home
                </Link>
                <Link
                  href="/features"
                  className="text-sm font-medium text-gray-600 hover:text-blue-500 transition-colors"
                >
                  Features
                </Link>
                <Link
                  href="#2"
                  className="text-sm font-medium text-gray-600 hover:text-blue-500 transition-colors"
                >
                  Pricing
                </Link>
                <Link
                  href="/blog"
                  className="text-sm font-medium text-gray-600 hover:text-blue-500 transition-colors"
                >
                  Blog
                </Link>
                <Link
                  href="/faq"
                  className="text-sm font-medium text-gray-600 hover:text-blue-500 transition-colors"
                >
                  FAQ
                </Link>
                <div className="pt-4 flex flex-col gap-2">
                  <Link href="/demo" className="btn-sm bg-white font-medium text-gray-800 shadow transition-colors hover:bg-gray-50 text-center">
                    Live Demo
                  </Link>
                  <Link href="/get-started" className="btn-sm bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-sm hover:from-blue-700 hover:to-blue-800 transition-all text-center">
                    Get Started
                  </Link>
                </div>
              </nav>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
