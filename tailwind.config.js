/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      // Maps the CSS custom properties defined in app/globals.css (--primary,
      // --border, --input, etc.) into actual Tailwind color utilities. Without
      // this, classes like bg-primary/border-input/bg-background used throughout
      // components/ui/* resolve to nothing — the cause of buttons, switches, and
      // dialog inputs rendering with no visible fill or border.
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        chart: {
          "1": "hsl(var(--chart-1))",
          "2": "hsl(var(--chart-2))",
          "3": "hsl(var(--chart-3))",
          "4": "hsl(var(--chart-4))",
          "5": "hsl(var(--chart-5))",
        },
        sidebar: {
          DEFAULT: "hsl(var(--sidebar-background))",
          foreground: "hsl(var(--sidebar-foreground))",
          primary: "hsl(var(--sidebar-primary))",
          "primary-foreground": "hsl(var(--sidebar-primary-foreground))",
          accent: "hsl(var(--sidebar-accent))",
          "accent-foreground": "hsl(var(--sidebar-accent-foreground))",
          border: "hsl(var(--sidebar-border))",
          ring: "hsl(var(--sidebar-ring))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      fontFamily: {
        inter: ["Inter", "sans-serif"],
        mono: ["Roboto Mono", "monospace"],
      },
      fontSize: {
        xs: ["0.75rem", { lineHeight: "1.5" }],
        sm: ["0.875rem", { lineHeight: "1.5715" }],
        base: ["1rem", { lineHeight: "1.5", letterSpacing: "-0.017em" }],
        lg: ["1.125rem", { lineHeight: "1.5", letterSpacing: "-0.017em" }],
        xl: ["1.25rem", { lineHeight: "1.5", letterSpacing: "-0.017em" }],
        "2xl": ["1.5rem", { lineHeight: "1.415", letterSpacing: "-0.037em" }],
        "3xl": [
          "1.875rem",
          { lineHeight: "1.3333", letterSpacing: "-0.037em" },
        ],
        "4xl": ["2.25rem", { lineHeight: "1.2777", letterSpacing: "-0.037em" }],
        "5xl": ["3rem", { lineHeight: "1", letterSpacing: "-0.037em" }],
        "6xl": ["4rem", { lineHeight: "1", letterSpacing: "-0.037em" }],
        "7xl": ["4.5rem", { lineHeight: "1", letterSpacing: "-0.037em" }],
      },
      scale: {
        '102': '1.02',
      },
      keyframes: {
        'code-1': { '0%': { opacity: '0' }, '3%, 100%': { opacity: '1' } },
        'code-2': { '0%, 10%': { opacity: '0' }, '13%, 100%': { opacity: '1' } },
        'code-3': { '0%, 20%': { opacity: '0' }, '23%, 100%': { opacity: '1' } },
        'code-4': { '0%, 30%': { opacity: '0' }, '33%, 100%': { opacity: '1' } },
        'code-5': { '0%, 40%': { opacity: '0' }, '43%, 100%': { opacity: '1' } },
        'code-6': { '0%, 50%': { opacity: '0' }, '53%, 100%': { opacity: '1' } },
        'code-7': { '0%, 60%': { opacity: '0' }, '63%, 100%': { opacity: '1' } },
        'code-8': { '0%, 70%': { opacity: '0' }, '73%, 100%': { opacity: '1' } },
        'code-9': { '0%, 80%': { opacity: '0' }, '83%, 100%': { opacity: '1' } },
        'code-10': { '0%, 90%': { opacity: '0' }, '93%, 100%': { opacity: '1' } },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        wiggle: {
          '0%, 100%': { transform: 'rotate(-3deg)' },
          '50%': { transform: 'rotate(3deg)' },
        },
      },
      animation: {
        wiggle: 'wiggle 1s ease-in-out infinite',
        'float': 'float 3s ease-in-out infinite',
        'code-1': 'code-1 5s ease-in-out forwards',
        'code-2': 'code-2 5s ease-in-out forwards',
        'code-3': 'code-3 5s ease-in-out forwards',
        'code-4': 'code-4 5s ease-in-out forwards',
        'code-5': 'code-5 5s ease-in-out forwards',
        'code-6': 'code-6 5s ease-in-out forwards',
        'code-7': 'code-7 5s ease-in-out forwards',
        'code-8': 'code-8 5s ease-in-out forwards',
        'code-9': 'code-9 5s ease-in-out forwards',
        'code-10': 'code-10 5s ease-in-out forwards',
      },
      aspectRatio: {
        'video': '16 / 9',
      },
    },
  },
  plugins: [require("@tailwindcss/forms"), require("tailwindcss-animate")],
};
