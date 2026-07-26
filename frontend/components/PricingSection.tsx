import { Check } from 'lucide-react';
import { Card, CardContent } from "@/components/ui/card";
import Header from "@/components/ui/header";
import Footer from "@/components/ui/footer";
import { Badge } from "@/components/ui/badge";

import Link from 'next/link';

interface PricingPlan {
  name: string;
  price: number;
  features: string[];
  buttonText: string;
  isPopular?: boolean;
}

const pricingPlans: PricingPlan[] = [
  {
    name: 'Basic Care',
    price: 10,
    features: [
      'Basic posture monitoring',
      'Eye blink tracking',
      'Water break reminders',
      '1 week health records',
      'Basic break scheduling',
      'Simple health stats',
      'Daily health tips',
      'Basic chat support',
    ],
    buttonText: 'Get Started'
  },
  {
    name: 'Pro Health',
    price: 25,
    features: [
      'Advanced posture analysis',
      'Comprehensive eye health tracking',
      'Smart break scheduling',
      'Water & exercise reminders',
      'Unlimited health records',
      'Weekly health insights',
      'Customizable alerts',
      'Personal AI health assistant',
      'Data-driven recommendations',
      'Health trend analysis'
    ],
    buttonText: 'Get Started',
    isPopular: true
  },
  {
    name: 'Enterprise',
    price: 49,
    features: [
      'Everything in Pro',
      'Team analytics dashboard',
      'Custom integration options',
      'Priority support',
      'Team health reports',
      'API access',
      'Advanced AI insights',
      'Custom AI training'
    ],
    buttonText: 'Contact Sales'
  }
];

export interface PricingSectionProps {
  className?: string;
  hideTitle?: boolean;
  standalone?: boolean;
}

export default function PricingSection({ 
  className = '', 
  hideTitle = false,
  standalone = true 
}: PricingSectionProps) {
  const content = (
    <div className={`mx-auto max-w-6xl px-4 sm:px-6 ${className}`}>
      {!hideTitle && (
        <div className="text-center mb-12">
        
          <h2 className="text-4xl md:text-5xl font-bold mb-4">Simple, Transparent <span className="text-blue-600">Pricing</span></h2>
          <p className="text-lg text-gray-600">Choose the plan that's right for you</p>
        </div>
      )}
      <Card className="bg-white/95 backdrop-blur-sm border-blue-100">
        <CardContent className="p-8 md:p-12">
          <div className="grid gap-8 md:grid-cols-3">
            {pricingPlans.map((plan) => (
              <Card
                key={plan.name}
                className={`relative transition-all duration-300 ${
                  plan.isPopular 
                    ? 'scale-105 md:-mt-4 border-blue-200 shadow-lg' 
                    : 'hover:scale-102 border-gray-100'
                }`}
              >
                <CardContent className="p-6">
                  {plan.isPopular && (
                    <div className="absolute -top-4 left-1/2 -translate-x-1/2 rounded-full bg-gradient-to-r from-blue-600 to-blue-700 px-4 py-1 text-sm font-semibold text-white">
                      Most Popular
                    </div>
                  )}
                  <div className="space-y-6">
                    <h3 className="text-xl font-bold">{plan.name}</h3>
                    <div className="flex items-baseline">
                      <span className="text-4xl font-bold">${plan.price}</span>
                      <span className="text-gray-600 ml-2">/month</span>
                    </div>
                    <ul className="space-y-3">
                      {plan.features.map((feature) => (
                        <li key={feature} className="flex items-center">
                          <Check className="h-5 w-5 text-blue-500 mr-2" />
                          <span>{feature}</span>
                        </li>
                      ))}
                    </ul>
                    <Link href="/pricing#join" passHref>
                      <button className={`w-full py-3 px-4 rounded-lg font-medium mt-4 ${
                        plan.isPopular
                          ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white hover:from-blue-700 hover:to-blue-800'
                          : 'bg-gray-100 text-gray-900 hover:bg-gray-200'
                      }`}>
                        {plan.buttonText}
                      </button>
                    </Link>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );

  if (standalone) {
    return (
      <>
        <Header />
        <main className="relative">{content}</main>
        <div className="h-60"></div>
        <Footer />
      </>
    );
  }

  return content;
}
