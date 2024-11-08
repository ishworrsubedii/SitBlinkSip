import Link from "next/link";
import Logo from "./logo";
import { 
  Mail, 
  MessageCircle, 
  Twitter, 
  Github, 
  Linkedin, 
  Heart,
  BookOpen,
  HelpCircle,
  Shield,
  Users,
  Sparkles,
  Bot,
  MapPin,
  Phone
} from "lucide-react";

export default function Footer({ border = false }: { border?: boolean }) {
  return (
    <footer className="bg-white" aria-label="Site Footer">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        {/* Main footer content */}
        <div className={`grid gap-8 py-8 sm:grid-cols-12 md:py-12 ${
          border ? "border-t border-gray-200" : ""
        }`}>
          {/* Brand Section - Updated with inline logo and text */}
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
                  Your digital wellness companion. We help developers and professionals maintain better posture, 
                  prevent eye strain, and stay hydrated throughout their workday. Join to improve your health and 
                  productivity with our AI-powered monitoring solutions.
                </span>
                
              </div>
            </div>
          </div>

          {/* Quick Links */}
          <div className="sm:col-span-6 md:col-span-3 lg:col-span-2">
            <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wider mb-4">Features</h3>
            <ul className="space-y-3">
              {quickLinks.map((link) => (
                <li key={link.href}>
                  <Link 
                    href={link.href}
                    className="text-gray-600 hover:text-blue-500 transition-colors duration-200 flex items-center gap-2"
                    aria-label={link.label}
                  >
                    <link.icon className="w-4 h-4" />
                    <span>{link.label}</span>
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources */}
          <div className="sm:col-span-6 md:col-span-3 lg:col-span-2">
            <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wider mb-4">Resources</h3>
            <ul className="space-y-3">
              {resourceLinks.map((link) => (
                <li key={link.href}>
                  <Link 
                    href={link.href}
                    className="text-gray-600 hover:text-blue-500 transition-colors duration-200 flex items-center gap-2"
                    aria-label={link.label}
                  >
                    <link.icon className="w-4 h-4" />
                    <span>{link.label}</span>
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Social Links */}
          <div className="sm:col-span-6 md:col-span-3 lg:col-span-4">
            <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wider mb-4">Connect With Us</h3>
            <div className="flex space-x-4">
              {socialLinks.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  className="text-gray-400 hover:text-blue-500 transition-colors duration-200 p-2 rounded-full hover:bg-blue-50"
                  aria-label={link.label}
                  target="_blank"
                  rel="noopener noreferrer"
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
            <div className="flex items-center justify-center md:justify-start gap-2 text-gray-600">
              <span>&copy; {new Date().getFullYear()} SitBlinkSip.</span>
              <span className="flex items-center gap-1">
                Made with <Heart className="w-4 h-4 text-red-500" /> for developers
              </span>
            </div>
            <div className="mt-4 md:mt-0">
              <div className="flex justify-center md:justify-end space-x-6 text-gray-600">
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
      </div>
    </footer>
  );
}

// Link configurations
const quickLinks = [
  { label: 'Eye Care', href: '/features#eye-care', icon: BookOpen },
  { label: 'Posture Guardian', href: '/features#posture', icon: Shield },
  { label: 'Health Analytics', href: '/features#analytics', icon: Users },
];

const resourceLinks = [
  { label: 'Help Center', href: '/help', icon: HelpCircle },
  { label: 'Documentation', href: '/docs', icon: BookOpen },
  { label: 'Blog', href: '/blog', icon: MessageCircle },
];

const socialLinks = [
  { label: 'Twitter', href: 'https://twitter.com/sitblinksip', icon: Twitter },
  { label: 'GitHub', href: 'https://github.com/sitblinksip', icon: Github },
  { label: 'LinkedIn', href: 'https://linkedin.com/company/sitblinksip', icon: Linkedin },
];

const legalLinks = [
  { label: 'Privacy', href: '/privacy' },
  { label: 'Terms', href: '/terms' },
  { label: 'Cookies', href: '/cookies' },
];
