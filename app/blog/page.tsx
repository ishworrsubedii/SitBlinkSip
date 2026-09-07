import React from 'react';
import { ArrowRight, Monitor, EyeIcon, Droplet } from 'lucide-react';
import PageIllustration from '@/components/page-illustration';
import Header from '@/components/ui/header';
import Footer from '@/components/ui/footer';
import { Badge } from "@/components/ui/badge";
import Link from 'next/link';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Blog - Sit Right, Blink Bright, Sip Well | SitBlinkSip',
  description: 'Discover evidence-based articles about posture, eye care, and workplace wellness. Learn how to Sit Right, Blink Bright, and Sip Well for better health.',
  openGraph: {
    title: 'Blog - Health & Wellness Tips | SitBlinkSip',
    description: 'Discover evidence-based articles about posture, eye care, and workplace wellness. Read the latest research and tips for maintaining health while working.',
    url: 'https://sitblinksip.tech/blog',
    siteName: 'SitBlinkSip',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Blog - Health & Wellness Tips | SitBlinkSip',
    description: 'Discover evidence-based articles about posture, eye care, and workplace wellness.',
  },
  robots: {
    index: true,
    follow: true,
  }
};

const BlogPage = () => {
  const blogs = [
    {
      slug: 'maintaining-proper-posture',
      icon: <Monitor className="w-8 h-8 text-blue-500" />,
      title: "Maintaining Proper Posture",
      category: "Ergonomics",
      description: "Learn evidence-based techniques for maintaining optimal posture during long work sessions. Discover how small adjustments can lead to significant improvements in comfort and productivity.",
      readTime: "5 min read",
      date: "Apr 15, 2024"
    },
    {
      slug: 'eye-care-digital-age',
      icon: <EyeIcon className="w-8 h-8 text-blue-500" />,
      title: "Eye Care in Digital Age",
      category: "Health",
      description: "Understanding the importance of regular eye movement and blinking patterns while working with screens. Get practical tips for reducing eye strain and maintaining eye health.",
      readTime: "4 min read",
      date: "Apr 14, 2024"
    },
    {
      slug: 'hydration-workplace-wellness',
      icon: <Droplet className="w-8 h-8 text-blue-500" />,
      title: "Hydration and Workplace Wellness",
      category: "Wellness",
      description: "Explore the critical role of hydration in maintaining cognitive function and physical well-being during work hours. Learn strategies to ensure optimal hydration.",
      readTime: "6 min read",
      date: "Apr 13, 2024"
    }
  ];

  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="grow">
        <div className="relative">
          <PageIllustration />
          
          <div className="pt-32 pb-12 md:pt-40 md:pb-20">
            {/* Page header */}
            <div className="text-center pb-12 md:pb-16 max-w-3xl mx-auto px-4 sm:px-6">
              <Badge 
                variant="outline" 
                className="px-4 py-1 border-blue-200 text-blue-700 bg-blue-50 mb-4"
              >
                Our Blog
              </Badge>
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold text-gray-900 mb-4 font-playfair-display">
                Latest <span className="text-blue-600">Insights</span> and Tips
              </h1>
              <p className="text-xl text-slate-600">
                Discover the latest research and best practices for maintaining your health while working
              </p>
            </div>

            {/* Cards grid */}
            <div className="flex justify-center">
              <div className="w-full max-w-6xl mx-auto grid gap-8 md:grid-cols-2 px-4 sm:px-6">
                {blogs.map((blog, index) => (
                  <BlogCard key={index} {...blog} />
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

const BlogCard = ({ 
  slug,
  icon, 
  title, 
  category, 
  description, 
  readTime, 
  date 
}: {
  slug: string;
  icon: React.ReactNode;
  title: string;
  category: string;
  description: string;
  readTime: string;
  date: string;
}) => (
  <div className="relative group">
    <div className="absolute inset-0 bg-blue-100/50 rounded-3xl -rotate-2 transform-gpu transition-transform group-hover:rotate-0 group-hover:scale-105" />
    <div className="relative bg-white rounded-2xl shadow-sm transition-all duration-300 hover:shadow-xl p-6">
      <div className="flex flex-col h-full">
        {/* Icon and category */}
        <div className="flex items-start justify-between mb-4">
          <div className="w-14 h-14 rounded-2xl bg-blue-50 flex items-center justify-center">
            {icon}
          </div>
          <Badge variant="secondary" className="bg-slate-100 text-slate-600">
            {category}
          </Badge>
        </div>

        {/* Content */}
        <div className="grow">
          <div className="flex items-center space-x-2 mb-2">
            <p className="text-sm text-slate-500">{date}</p>
            <span className="text-slate-300">•</span>
            <p className="text-sm text-slate-500">{readTime}</p>
          </div>
          <h3 className="text-xl font-bold text-slate-900 mb-3">{title}</h3>
          <p className="text-slate-600 mb-4 line-clamp-3">{description}</p>
        </div>

        {/* Footer */}
        <div className="pt-4 border-t border-slate-100">
          <Link href={`/blog/${slug}`} className="inline-flex items-center text-blue-600 font-semibold hover:text-blue-800 transition-colors">
            Read Article
            <ArrowRight className="w-4 h-4 ml-2" />
          </Link>
        </div>
      </div>
    </div>
  </div>
);

export default BlogPage;