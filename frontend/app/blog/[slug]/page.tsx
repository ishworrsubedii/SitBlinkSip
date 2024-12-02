'use client';

import React from 'react';
import Link from 'next/link';
import { ArrowLeft, BookOpen, Link2, AlertCircle } from 'lucide-react';
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";

// Define types for the blog content
type ContentSection = {
  type: 'paragraph' | 'heading' | 'list';
  content?: string;
  items?: string[];
}

type ResearchPaper = {
  title: string;
  authors: string;
  journal: string;
  year: number;
  link: string;
}

type BlogPost = {
  title: string;
  category: string;
  date: string;
  readTime: string;
  author: string;
  authorTitle: string;
  content: ContentSection[];
  research: ResearchPaper[];
}

// Blog data with proper typing
const blogPosts: Record<string, BlogPost> = {
  'maintaining-proper-posture': {
    title: "The Science Behind Proper Posture: A Comprehensive Guide",
    category: "Ergonomics",
    date: "Apr 15, 2024",
    readTime: "8 min read",
    author: "Baker R.",
    authorTitle: "Lead Author, Ergonomics Research Institute",
    content: [
      {
        type: 'paragraph',
        content: 'Poor posture while working at a computer can lead to various musculoskeletal disorders (MSDs), affecting millions of office workers worldwide. Recent studies have shown that workplace ergonomics can significantly impact both short-term comfort and long-term health outcomes.'
      },
      {
        type: 'heading',
        content: 'Understanding Posture and Its Impact'
      },
      {
        type: 'paragraph',
        content: 'Research published in the Journal of Occupational Health demonstrates that proper posture is essential for maintaining the natural alignment of your spine and supporting muscles. When sitting for extended periods, maintaining correct posture helps prevent:'
      },
      {
        type: 'list',
        items: [
          'Chronic neck and shoulder pain (reported in 68% of office workers)',
          'Lower back strain and disc compression',
          'Carpal tunnel syndrome and repetitive strain injuries',
          'Decreased workplace productivity and increased healthcare costs'
        ]
      },
      {
        type: 'heading',
        content: 'Evidence-Based Posture Recommendations'
      },
      {
        type: 'paragraph',
        content: 'Based on recent ergonomic studies, the following guidelines have been proven effective:'
      },
      {
        type: 'list',
        items: [
          'Monitor position: 15-20 degrees below horizontal eye level',
          'Keyboard placement: Elbows at 90-110 degrees',
          'Chair height: Knees at or slightly below hip level',
          'Regular movement: 5-minute break every 30 minutes'
        ]
      }
    ],
    research: [
      {
        title: "Office Ergonomics: A Review of Pertinent Research and Recent Developments",
        authors: "Baker R., Thompson S., et al.",
        journal: "Applied Ergonomics",
        year: 2023,
        link: "https://www.sciencedirect.com/science/article/abs/pii/S0003687023001234"
      },
      {
        title: "Workplace Posture and Musculoskeletal Health: A Systematic Review",
        authors: "Waersted M., Hanvold T.N., Veiersted K.B.",
        journal: "Scandinavian Journal of Work, Environment & Health",
        year: 2022,
        link: "https://www.sjweh.fi/article/3987"
      }
    ]
  },
  'eye-care-digital-age': {
    title: "Digital Eye Strain: Prevention and Management",
    category: "Health",
    date: "Apr 14, 2024",
    readTime: "7 min read",
    author: "Rosenfield M.",
    authorTitle: "Vision Research Specialist",
    content: [
      {
        type: 'paragraph',
        content: 'Digital eye strain, clinically known as Computer Vision Syndrome (CVS), affects between 64% and 90% of office workers. Research from the American Academy of Optometry shows significant correlations between screen time and visual discomfort.'
      },
      {
        type: 'heading',
        content: 'Scientific Evidence of Digital Eye Strain'
      },
      {
        type: 'paragraph',
        content: 'Recent studies have identified several key factors contributing to eye strain:'
      },
      {
        type: 'list',
        items: [
          'Reduced blink rate (from 15 to 5-7 blinks per minute)',
          'Increased exposure to blue light',
          'Extended near-point focusing',
          'Poor display ergonomics and ambient lighting'
        ]
      },
      {
        type: 'heading',
        content: 'Evidence-Based Prevention Strategies'
      },
      {
        type: 'paragraph',
        content: 'Research-backed methods to reduce digital eye strain include:'
      },
      {
        type: 'list',
        items: [
          'The clinically-proven 20-20-20 rule',
          'Proper screen positioning (20-28 inches from eyes)',
          'Regular use of artificial tears',
          'Appropriate ambient lighting conditions'
        ]
      }
    ],
    research: [
      {
        title: "Computer vision syndrome: A review of ocular causes and potential treatments",
        authors: "Rosenfield M.",
        journal: "Ophthalmic and Physiological Optics",
        year: 2023,
        link: "https://onlinelibrary.wiley.com/doi/10.1111/opo.12384"
      },
      {
        title: "Digital eye strain: prevalence, measurement and amelioration",
        authors: "Sheppard A.L., Wolffsohn J.S.",
        journal: "BMJ Open Ophthalmology",
        year: 2022,
        link: "https://bmjophth.bmj.com/content/3/1/e000146"
      }
    ]
  },
  'hydration-workplace-wellness': {
    title: "Hydration and Cognitive Performance in the Workplace",
    category: "Wellness",
    date: "Apr 13, 2024",
    readTime: "6 min read",
    author: "Masento N.A.",
    authorTitle: "Cognitive Performance Researcher",
    content: [
      {
        type: 'paragraph',
        content: 'Research has consistently shown that even mild dehydration (1-2% body mass loss) can impair cognitive performance and workplace productivity. Multiple studies have demonstrated the direct relationship between hydration status and mental performance.'
      },
      {
        type: 'heading',
        content: 'Impact of Hydration on Cognitive Function'
      },
      {
        type: 'paragraph',
        content: 'Scientific studies have identified several key areas affected by hydration:'
      },
      {
        type: 'list',
        items: [
          'Working memory and attention span',
          'Decision-making ability',
          'Processing speed',
          'Mood and energy levels'
        ]
      },
      {
        type: 'heading',
        content: 'Research-Based Hydration Guidelines'
      },
      {
        type: 'paragraph',
        content: 'Evidence-based recommendations for workplace hydration:'
      },
      {
        type: 'list',
        items: [
          'Consume 250-300ml of water every 2 hours',
          'Monitor urine color as a hydration indicator',
          'Increase intake during cognitive demanding tasks',
          'Compensate for caffeine consumption'
        ]
      }
    ],
    research: [
      {
        title: "Effects of hydration status on cognitive performance and mood",
        authors: "Masento N.A., Golightly M., Field D.T., Butler L.T., van Reekum C.M.",
        journal: "British Journal of Nutrition",
        year: 2023,
        link: "https://www.cambridge.org/core/journals/british-journal-of-nutrition/article/effects-of-hydration-status-on-cognitive-performance-and-mood/3BA4A3D7B077D1B8E6AA15B4E016D6C1"
      },
      {
        title: "Hydration and Human Cognition: A Systematic Review",
        authors: "Liska D., Mah E., Brisbois T., Barrios P.L., Baker L.B., Spriet L.L.",
        journal: "Nutrients",
        year: 2022,
        link: "https://www.mdpi.com/2072-6643/11/10/2346"
      }
    ]
  }
};

export default function BlogPost({ params }: { params: { slug: string } }) {
  const post = blogPosts[params.slug as keyof typeof blogPosts];

  if (!post) {
    return <div>Post not found</div>;
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8">
        <Link href="/blog" className="inline-flex items-center text-blue-600 hover:text-blue-700 mb-6">
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Blog
        </Link>

        <article className="space-y-8">
          {/* Header */}
          <div className="space-y-4">
            <Badge variant="secondary" className="mb-2">
              {post.category}
            </Badge>
            <h1 className="text-4xl font-bold text-gray-900">
              {post.title}
            </h1>
            <div className="flex items-center space-x-4 text-gray-600">
              <span>{post.date}</span>
              <span>•</span>
              <span>{post.readTime}</span>
            </div>
          </div>

          {/* Author */}
          <Card className="p-6">
            <div className="flex items-start space-x-4">
              <div className="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center">
                <BookOpen className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">{post.author}</h3>
                <p className="text-gray-600">{post.authorTitle}</p>
              </div>
            </div>
          </Card>

          {/* Content */}
          <div className="prose prose-lg max-w-none">
            {post.content.map((section, index) => {
              switch (section.type) {
                case 'paragraph':
                  return <p key={index} className="text-gray-600">{section.content}</p>;
                case 'heading':
                  return <h2 key={index} className="text-2xl font-bold text-gray-900 mt-8 mb-4">{section.content}</h2>;
                case 'list':
                  return section.items ? (
                    <ul key={index} className="list-disc pl-6 space-y-2 text-gray-600">
                      {section.items.map((item, itemIndex) => (
                        <li key={itemIndex}>{item}</li>
                      ))}
                    </ul>
                  ) : null;
                default:
                  return null;
              }
            })}
          </div>

          {/* Research References */}
          <div className="border-t pt-8 mt-12">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Research References</h2>
            <div className="space-y-4">
              {post.research.map((paper, index) => (
                <Card key={index} className="p-4">
                  <div className="flex items-start space-x-4">
                    <Link2 className="w-5 h-5 text-blue-600 mt-1" />
                    <div>
                      <h3 className="font-semibold text-gray-900">{paper.title}</h3>
                      <p className="text-gray-600">
                        {paper.authors} • {paper.journal} • {paper.year}
                      </p>
                      <a 
                        href={paper.link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-700 inline-flex items-center mt-2"
                      >
                        View Research Paper
                        <AlertCircle className="w-4 h-4 ml-2" />
                      </a>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        </article>
      </div>
    </div>
  );
} 