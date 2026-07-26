import { Github, Linkedin, Twitter, Globe, Mail } from "lucide-react";

const links = [
  {
    label: "GitHub",
    href: "https://github.com/ishworrsubedii",
    icon: Github,
  },
  {
    label: "LinkedIn",
    href: "https://www.linkedin.com/in/ishworrsubedii/",
    icon: Linkedin,
  },
  {
    label: "X / Twitter",
    href: "https://x.com/ishworr_",
    icon: Twitter,
  },
  {
    label: "Portfolio",
    href: "https://ishwor-subedi.com.np/",
    icon: Globe,
  },
  {
    label: "Email",
    href: "mailto:ishworr.subedi@gmail.com",
    icon: Mail,
  },
];

export default function DeveloperCredit() {
  return (
    <section className="relative">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="py-12 md:py-16">
          <div
            className="mx-auto flex max-w-3xl flex-col items-center gap-5 rounded-2xl border border-gray-100 bg-white p-8 text-center shadow-sm sm:p-10"
            data-aos="fade-up"
          >
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-gradient-to-br from-blue-600 to-blue-400 text-xl font-bold text-white">
              IS
            </div>
            <div>
              <h3 className="text-xl font-bold text-slate-900">Ishwor Subedi</h3>
              <p className="mt-1 text-sm font-medium text-blue-600">
                Machine Learning Engineer
              </p>
              <p className="text-sm text-slate-500">
                Developer &amp; Creator of SitBlinkSip
              </p>
            </div>
            <p className="max-w-md text-sm leading-relaxed text-slate-600">
              Built solo, out of a habit of forgetting to blink during long
              coding sessions — now shared as an open-source tool for anyone
              who spends too much time at a screen.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-2">
              {links.map((link) => (
                <a
                  key={link.label}
                  href={link.href}
                  target={link.href.startsWith("mailto:") ? undefined : "_blank"}
                  rel={link.href.startsWith("mailto:") ? undefined : "noopener noreferrer"}
                  aria-label={link.label}
                  title={link.label}
                  className="flex h-10 w-10 items-center justify-center rounded-full border border-gray-200 text-slate-500 transition-all duration-200 hover:-translate-y-0.5 hover:border-blue-200 hover:bg-blue-50 hover:text-blue-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                >
                  <link.icon className="h-4 w-4" />
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
