import Link from "next/link";
import Logo from "./logo";
import {
  MessageCircle,
  Twitter,
  Github,
  Linkedin,
  Heart,
  HelpCircle,
  LayoutDashboard,
  PresentationIcon,
} from "lucide-react";

export default function Footer({ border = false }: { border?: boolean }) {
  return (
    <footer className="bg-white" aria-label="Site Footer">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        {/* Main footer content */}
        <div className={`grid gap-8 py-8 sm:grid-cols-12 md:py-12 ${
          border ? "border-t border-gray-200" : ""
        }`}>
          {/* Brand Section */}
          <div className="sm:col-span-12 lg:col-span-4">
            <div className="flex flex-col space-y-4">
              <div className="flex flex-col">
                <div className="flex items-center gap-2 mb-2">
                  <Logo />
                  <span className="text-xl font-semibold text-gray-900">
                    SitBlinkSip
                  </span>
                </div>
                <span className="text-sm text-gray-600 max-w-sm pl-10 leading-relaxed">
                  An open-source wellness companion for developers and digital
                  workers. Sit better, blink more, and sip regularly — with
                  timely reminders while you work.
                </span>
              </div>
            </div>
          </div>

          {/* Resources */}
          <div className="sm:col-span-6 md:col-span-4 lg:col-span-3">
            <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wider mb-4">Resources</h3>
            <ul className="space-y-3">
              {resourceLinks.map((link) => (
                <li key={link.href}>
                  {link.external ? (
                    <a
                      href={link.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-gray-600 hover:text-blue-500 transition-colors duration-200 flex items-center gap-2"
                      aria-label={link.label}
                    >
                      <link.icon className="w-4 h-4" />
                      <span>{link.label}</span>
                    </a>
                  ) : (
                    <Link
                      href={link.href}
                      className="text-gray-600 hover:text-blue-500 transition-colors duration-200 flex items-center gap-2"
                      aria-label={link.label}
                    >
                      <link.icon className="w-4 h-4" />
                      <span>{link.label}</span>
                    </Link>
                  )}
                </li>
              ))}
            </ul>
          </div>

          {/* Social Links */}
          <div className="sm:col-span-6 md:col-span-5 lg:col-span-5">
            <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wider mb-4">Connect With Us</h3>
            <div className="flex space-x-4">
              {socialLinks.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  className="text-gray-400 hover:text-blue-500 transition-colors duration-200 p-2 rounded-full hover:bg-blue-50"
                  aria-label={link.label}
                  target={link.href.startsWith("mailto:") ? undefined : "_blank"}
                  rel={link.href.startsWith("mailto:") ? undefined : "noopener noreferrer"}
                >
                  <link.icon className="w-6 h-6" />
                </a>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="py-4 border-t border-gray-200">
          <div className="md:flex md:items-center md:justify-between text-sm">
            <div className="flex flex-wrap items-center justify-center md:justify-start gap-x-2 gap-y-1 text-gray-600">
              <span>&copy; {new Date().getFullYear()} SitBlinkSip.</span>
              <span className="flex items-center gap-1">
                Built with <Heart className="w-4 h-4 text-red-500" /> by Ishwor Subedi for healthier work sessions.
              </span>
            </div>
            <div className="mt-4 md:mt-0 flex items-center justify-center md:justify-end gap-6 text-gray-600">
              {legalLinks.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className="hover:text-blue-500 transition-colors duration-200"
                  aria-label={link.label}
                >
                  {link.label}
                </Link>
              ))}
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}

// Link configurations
const resourceLinks = [
  { label: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { label: 'Preview', href: '/preview', icon: PresentationIcon },
  { label: 'Blog', href: '/blog', icon: MessageCircle },
  { label: 'FAQ', href: '/faq', icon: HelpCircle },
  { label: 'GitHub', href: 'https://github.com/ishworrsubedii/SitBlinkSip', icon: Github, external: true },
];

const socialLinks = [
  { label: 'GitHub', href: 'https://github.com/ishworrsubedii', icon: Github },
  { label: 'X / Twitter', href: 'https://x.com/ishworr_', icon: Twitter },
  { label: 'LinkedIn', href: 'https://www.linkedin.com/in/ishworrsubedii/', icon: Linkedin },
];

const legalLinks = [
  { label: 'Cookies', href: '/cookies' },
];
