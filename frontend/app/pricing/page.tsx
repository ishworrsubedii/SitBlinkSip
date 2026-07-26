'use client';

import React, { useState, useRef } from 'react';
import { toast } from 'sonner';
import { Loader2, CheckCircle2, Sparkles, ArrowRight, Check } from 'lucide-react';
import PageIllustration from '@/components/page-illustration';
import Header from '@/components/ui/header';
import Footer from '@/components/ui/footer';
import PricingSection from '@/components/PricingSection';
import { Badge } from '@/components/ui/badge';
import { apiClient, ApiError } from '@/lib/api-client';

export default function PricingPage() {
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const formRef = useRef<HTMLFormElement>(null);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setSuccess(false);

    const formData = new FormData(e.currentTarget);
    const email = formData.get('email') as string;
    const fullName = formData.get('fullName') as string;
    const profession = formData.get('profession') as string;

    try {
      await apiClient.fetch('waitlist', {
        method: 'POST',
        body: { email, full_name: fullName, profession },
      });

      setSuccess(true);
      toast.success('Successfully joined the waitlist!');
      formRef.current?.reset();
    } catch (error) {
      setSuccess(false);
      if (error instanceof ApiError && error.status === 409) {
        toast.error('This email is already registered in our waitlist.');
      } else {
        toast.error(error instanceof Error ? error.message : 'Failed to join waitlist');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Header />
      <main className="relative min-h-screen overflow-x-hidden">
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <PageIllustration />
        </div>

        <div className="relative">
          <div className="pt-32 md:pt-40">
            <PricingSection standalone={false} />
          </div>

          <div id="join" className="mx-auto max-w-6xl px-4 sm:px-6 py-20 scroll-mt-24">
            <div className="max-w-3xl mx-auto text-center pb-12 md:pb-16">
              <Badge
                variant="outline"
                className="px-4 py-1 border-blue-200 text-blue-700 bg-blue-50 mb-4"
              >
                Waitlist
              </Badge>
              <h2 className="text-3xl md:text-4xl font-bold mb-2">
                <span className="bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
                  Join Our Exclusive Waitlist
                </span>
              </h2>
              <p className="text-gray-600 text-lg max-w-2xl mx-auto">
                Be among the first to experience our revolutionary AI-powered wellness platform.
                Pick the plan above and reserve your spot below.
              </p>
            </div>

            <div className="max-w-xl mx-auto">
              <form ref={formRef} onSubmit={handleSubmit} className="bg-white/10 backdrop-blur-sm shadow-lg rounded-xl p-6 space-y-5 border border-gray-200/20">
                <div>
                  <label htmlFor="fullName" className="block text-sm font-medium text-gray-700 mb-1">
                    Full Name
                  </label>
                  <input
                    id="fullName"
                    name="fullName"
                    type="text"
                    required
                    className="w-full rounded-lg border-0 px-4 py-3 text-gray-900 ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-blue-600 focus:outline-none"
                    placeholder="John Doe"
                  />
                </div>

                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
                    Email Address
                  </label>
                  <input
                    id="email"
                    name="email"
                    type="email"
                    required
                    className="w-full rounded-lg border-0 px-4 py-3 text-gray-900 ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-blue-600 focus:outline-none"
                    placeholder="john@example.com"
                    pattern="[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$"
                    title="Please enter a valid email address."
                  />
                </div>

                <div>
                  <label htmlFor="profession" className="block text-sm font-medium text-gray-700 mb-1">
                    Your Role
                  </label>
                  <select
                    id="profession"
                    name="profession"
                    required
                    className="w-full rounded-lg border-0 px-4 py-3 text-gray-900 ring-1 ring-inset ring-gray-300 focus:ring-2 focus:ring-blue-600 focus:outline-none"
                  >
                    <option value="">Select your role</option>
                    <option value="developer">Developer</option>
                    <option value="student">Student</option>
                    <option value="designer">Designer</option>
                    <option value="manager">Manager</option>
                    <option value="other">Other</option>
                  </select>
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full inline-flex justify-center items-center rounded-lg bg-blue-600 px-6 py-3 text-white transition duration-150 ease-in-out hover:bg-blue-700 disabled:opacity-50"
                >
                  {loading ? (
                    <Loader2 className="animate-spin mr-2" size={20} />
                  ) : success ? (
                    <CheckCircle2 className="mr-2" size={20} />
                  ) : (
                    <Sparkles className="mr-2" size={20} />
                  )}
                  {loading ? 'Joining...' : success ? 'Joined Successfully' : 'Join Waitlist'}
                  {!loading && !success && <ArrowRight className="ml-2" size={20} />}
                </button>

                <div className="mt-6 flex items-center justify-center gap-x-8 text-sm text-gray-500">
                  <div className="flex items-center gap-2">
                    <Check className="h-4 w-4" />
                    <span>No spam</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Check className="h-4 w-4" />
                    <span>Cancel anytime</span>
                  </div>
                </div>
              </form>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </>
  );
}
