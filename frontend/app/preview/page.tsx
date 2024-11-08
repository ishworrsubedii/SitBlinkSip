"use client";

import Image from 'next/image';
import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Eye, Monitor, Armchair, Droplets, Clock, 
  ArrowRight, Check, AlertCircle, Brain, 
  Activity, Glasses, Droplet, Focus
} from 'lucide-react';
import Header from "@/components/ui/header";
import Footer from "@/components/ui/footer";

// Add this interface above the VideoCard component
interface VideoCardProps {
  title: string;
  description: string;
  videoUrl: string;
  thumbnailUrl: string;
}

// Video Card Component
const VideoCard = ({ title, description, videoUrl, thumbnailUrl }: VideoCardProps) => {
  const [isPlaying, setIsPlaying] = useState(false);

  return (
    <motion.div 
      className="rounded-xl overflow-hidden shadow-lg bg-white"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
    >
      <div className="relative aspect-video">
        <video
          className="w-full h-full object-cover"
          controls
          poster={thumbnailUrl}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
        >
          <source src={videoUrl} type="video/mp4" />
          Your browser does not support video playback.
        </video>
        {!isPlaying && (
          <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
            <button 
              className="bg-white/90 p-4 rounded-full hover:bg-white transition-all transform hover:scale-110"
              onClick={() => {
                const video = document.querySelector('video');
                video?.play();
              }}
            >
              <motion.div
                whileHover={{ scale: 1.1 }}
                className="w-8 h-8 text-blue-500"
              >
                ▶️
              </motion.div>
            </button>
          </div>
        )}
      </div>
      <div className="p-6">
        <h3 className="text-xl font-bold mb-2">{title}</h3>
        <p className="text-gray-600">{description}</p>
      </div>
    </motion.div>
  );
};

// Add interface for FeatureCard props
interface FeatureCardProps {
  icon: React.ElementType;
  title: string;
  description: string;
  benefits: string[];
}

// Feature Card Component
const FeatureCard = ({ icon: Icon, title, description, benefits }: FeatureCardProps) => (
  <motion.div 
    className="bg-white rounded-xl p-6 shadow-sm hover:shadow-md transition-all duration-300"
    whileHover={{ y: -5 }}
  >
    <div className="flex items-center gap-4 mb-4">
      <div className="p-3 bg-blue-50 rounded-lg">
        <Icon className="w-6 h-6 text-blue-500" />
      </div>
      <h3 className="text-lg font-semibold">{title}</h3>
    </div>
    <p className="text-gray-600 mb-4">{description}</p>
    <ul className="space-y-2">
      {benefits.map((benefit, index) => (
        <li key={index} className="flex items-center gap-2 text-sm text-gray-500">
          <Check className="w-4 h-4 text-green-500" />
          {benefit}
        </li>
      ))}
    </ul>
  </motion.div>
);

export default function PreviewPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-gray-50">
      <Header />
      {/* Hero Section */}
      <section className="pt-24 pb-12 px-4 md:pt-32 max-w-6xl mx-auto">
        <motion.div 
          className="text-center mb-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h1 className="text-4xl md:text-5xl font-bold mb-6">
            <span className="bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
              AI-Powered Wellness Assistant
            </span>
          </h1>
          <p className="text-gray-600 text-lg max-w-2xl mx-auto">
            Maintain perfect posture, prevent eye strain, and stay hydrated with our intelligent monitoring system
          </p>
        </motion.div>

        {/* Video Demonstrations */}
        <div className="grid md:grid-cols-2 gap-8 mb-16">
          <VideoCard 
            title="Posture Detection Demo"
            description="See how our AI detects and corrects your posture in real-time"
            videoUrl="/videos/posture-demo.mp4"
            thumbnailUrl="/thumbnails/posture-thumb.jpg"
          />
          <VideoCard 
            title="Eye Care System Demo"
            description="Watch how we monitor and protect your eye health"
            videoUrl="/videos/eye-demo.mp4"
            thumbnailUrl="/thumbnails/eye-thumb.jpg"
          />
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-6 mb-16">
          <FeatureCard 
            icon={Armchair}
            title="Smart Posture Detection"
            description="AI-powered real-time posture monitoring"
            benefits={[
              "Prevents neck and back strain",
              "Customized ergonomic guidance",
              "Real-time correction alerts"
            ]}
          />
          <FeatureCard 
            icon={Eye}
            title="Eye Strain Prevention"
            description="Comprehensive eye care system"
            benefits={[
              "Blink rate monitoring",
              "20-20-20 rule reminders",
              "Screen distance optimization"
            ]}
          />
          <FeatureCard 
            icon={Droplets}
            title="Hydration Management"
            description="Smart hydration tracking system"
            benefits={[
              "Personalized intake goals",
              "Timely reminders",
              "Progress visualization"
            ]}
          />
        </div>
      </section>

      <Footer border={true} />

    </div>
  );
}
