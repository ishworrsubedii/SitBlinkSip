"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import Logo from "./logo";
import { Menu, Github, Download } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetClose,
} from "@/components/ui/sheet";

const navLinks = [
  { label: "Home", href: "/" },
  { label: "Features", href: "/#features" },
  { label: "How It Works", href: "/#how-it-works" },
  { label: "Desktop App", href: "/#desktop" },
];

const GITHUB_URL = "https://github.com/ishworrsubedii/SitBlinkSip";

export default function Header() {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header className="fixed top-2 z-40 w-full md:top-6">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div
          className={`relative flex h-16 items-center justify-between gap-3 rounded-2xl border px-4 backdrop-blur-md transition-all duration-300 ${
            scrolled
              ? "border-gray-200/80 bg-white/95 shadow-[0_8px_30px_-12px_rgba(15,23,42,0.18)]"
              : "border-white/60 bg-white/80 shadow-[0_1px_2px_rgba(15,23,42,0.04)]"
          }`}
        >
          <div className="flex shrink-0 items-center gap-2.5">
            <Logo />
            <Link
              href="/"
              className="font-display text-lg tracking-tight sm:text-xl"
            >
              <span className="bg-gradient-to-r from-blue-600 to-blue-500 bg-clip-text font-extrabold text-transparent">
                SitBlinkSip
              </span>
            </Link>
          </div>

          {/* Desktop nav */}
          <nav className="hidden flex-1 items-center justify-center lg:flex">
            <ul className="flex items-center gap-1">
              {navLinks.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="rounded-full px-3.5 py-2 text-sm font-medium text-gray-600 transition-colors hover:bg-gray-50 hover:text-blue-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>

          {/* Desktop actions */}
          <div className="hidden shrink-0 items-center gap-2 lg:flex">
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 rounded-full border border-gray-200 px-3.5 py-2 text-sm font-medium text-gray-700 transition-colors hover:border-gray-300 hover:bg-gray-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            >
              <Github className="h-4 w-4" />
              GitHub
            </a>
            <Link
              href="/#desktop"
              className="inline-flex items-center gap-1.5 rounded-full bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition-all duration-150 hover:-translate-y-0.5 hover:bg-blue-700 hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 active:translate-y-0"
            >
              <Download className="h-4 w-4" />
              Download
            </Link>
          </div>

          {/* Mobile trigger */}
          <div className="flex items-center gap-2 lg:hidden">
            <Link
              href="/#desktop"
              aria-label="Download SitBlinkSip Desktop"
              className="inline-flex items-center gap-1.5 rounded-full bg-blue-600 px-3 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-blue-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
            >
              <Download className="h-4 w-4" />
              <span className="hidden sm:inline">Download</span>
            </Link>
            <Sheet open={open} onOpenChange={setOpen}>
              <button
                type="button"
                aria-label="Open menu"
                onClick={() => setOpen(true)}
                className="inline-flex h-9 w-9 items-center justify-center rounded-full text-gray-600 transition-colors hover:bg-gray-50 hover:text-blue-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
              >
                <Menu className="h-5 w-5" />
              </button>
              <SheetContent
                side="right"
                className="w-full max-w-xs bg-white motion-reduce:transition-none sm:max-w-sm"
              >
                <SheetHeader>
                  <SheetTitle>
                    <span className="bg-gradient-to-r from-blue-600 to-blue-500 bg-clip-text font-extrabold text-transparent">
                      SitBlinkSip
                    </span>
                  </SheetTitle>
                </SheetHeader>
                <nav className="mt-6 flex flex-col gap-1">
                  {navLinks.map((link) => (
                    <SheetClose asChild key={link.href}>
                      <Link
                        href={link.href}
                        className="rounded-lg px-3 py-3 text-base font-medium text-gray-700 transition-colors hover:bg-gray-50 hover:text-blue-600"
                      >
                        {link.label}
                      </Link>
                    </SheetClose>
                  ))}
                  <SheetClose asChild>
                    <a
                      href={GITHUB_URL}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 rounded-lg px-3 py-3 text-base font-medium text-gray-700 transition-colors hover:bg-gray-50 hover:text-blue-600"
                    >
                      <Github className="h-5 w-5" />
                      GitHub
                    </a>
                  </SheetClose>
                </nav>
                <div className="mt-6 border-t border-gray-100 pt-6">
                  <SheetClose asChild>
                    <Link
                      href="/#desktop"
                      className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-3 text-base font-semibold text-white shadow-sm transition-colors hover:bg-blue-700"
                    >
                      <Download className="h-5 w-5" />
                      Download for your OS
                    </Link>
                  </SheetClose>
                </div>
              </SheetContent>
            </Sheet>
          </div>
        </div>
      </div>
    </header>
  );
}
