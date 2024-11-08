import { Check} from 'lucide-react';
import { Card, CardContent } from "@/components/ui/card";
import Header from "@/components/ui/header";
import Footer from "@/components/ui/footer";

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
      'Basic break reminders'
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
      'Workspace optimization tips',
      'Weekly health insights',
      'Priority support'
    ],
    buttonText: 'Get Started',
    isPopular: true
  },
  {
    name: 'Enterprise',
    price: 30,
    features: [
      'Everything in Pro',
      'Team analytics dashboard',
      'Custom integration options',
      'Dedicated support'
    ],
    buttonText: 'Contact Sales'
  }
];

interface PricingSectionProps {
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
    <div className={`mx-auto max-w-6xl px-4 sm:px-6 pt-24 md:pt-32 relative ${className}`}>
      {/* Decorative elements */}
      <div className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80" aria-hidden="true">
        <div className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[#ff80b5] to-[#9089fc] opacity-20 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]" />
      </div>
      
      <div className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80" aria-hidden="true">
        <div className="relative left-[calc(50%+3rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 bg-gradient-to-tr from-[#ff80b5] to-[#9089fc] opacity-20 sm:left-[calc(50%+36rem)] sm:w-[72.1875rem]" />
      </div>

      <div className="absolute inset-x-0 top-[calc(100%-13rem)] -z-10 transform-gpu overflow-hidden blur-3xl sm:top-[calc(100%-30rem)]" aria-hidden="true">
        <div className="relative left-[calc(50%-3rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 bg-gradient-to-tr from-[#ff80b5] to-[#9089fc] opacity-20 sm:left-[calc(50%-36rem)] sm:w-[72.1875rem]" />
      </div>
      
      <div className="text-center max-w-3xl mx-auto mb-16">
        
        <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold text-gray-900 mb-4 font-playfair-display">
          Choose Your <span className="text-blue-600">Health Journey</span>
        </h1>
        <p className="text-lg text-slate-600 mb-8">
          Select the perfect plan to enhance your workspace wellness and productivity. 
          Our AI-powered health monitoring solutions are designed for every professional.
        </p>
        
        {standalone && (
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className="flex items-center gap-2 text-slate-700">
              <Check className="w-5 h-5 text-green-500" />
              <span>30-day money-back guarantee</span>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-8 text-sm text-slate-600">
              <span>Trusted by developers from</span>
              <div className="flex gap-8">
                <span className="font-semibold">TCP</span>
                <span className="font-semibold">BrentInnovate</span>
              </div>
            </div>
          </div>
        )}
      </div>

      <Card className="bg-white/95 backdrop-blur-sm border-blue-100">
        <CardContent className="p-8 md:p-12">
          <div className="grid gap-8 md:grid-cols-3 relative">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-purple-500/5 to-blue-500/5 blur-3xl -z-10" />
            {pricingPlans.map((plan) => (
              <Card
                key={plan.name}
                className={`relative group transition-all duration-300 ${
                  plan.isPopular 
                    ? 'scale-105 md:-mt-4 border-blue-200 shadow-lg shadow-blue-100' 
                    : 'hover:scale-102 border-gray-100 hover:border-blue-100'
                } backdrop-blur-sm bg-white/95`}
              >
                <CardContent className={`p-8 rounded-2xl h-full ${
                  plan.isPopular 
                    ? 'bg-gradient-to-b from-blue-50/50 to-white' 
                    : 'hover:bg-blue-50/10'
                }`}>
                  {plan.isPopular && (
                    <div className="absolute -top-4 left-1/2 -translate-x-1/2 rounded-full bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-1.5 text-sm font-semibold text-white shadow-lg">
                      Most Popular
                    </div>
                  )}
                  
                  <div className="relative space-y-8">
                    <h3 className="text-2xl font-bold text-slate-900">{plan.name}</h3>
                    <div className="flex items-baseline gap-3">
                      <span className="text-5xl font-bold text-slate-900">${plan.price}</span>
                      <span className="text-slate-600">/month</span>
                    </div>
                    
                    <ul className="space-y-4 mb-8">
                      {plan.features.map((feature) => (
                        <li key={feature} className="flex items-center text-slate-700 gap-3">
                          <Check className="w-5 h-5 text-blue-600 shrink-0" />
                          <span>{feature}</span>
                        </li>
                      ))}
                    </ul>
                    
                    <button 
                      className={`w-full rounded-lg px-6 py-3.5 font-semibold transition-all duration-300 ${
                        plan.isPopular
                          ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white hover:from-blue-700 hover:to-blue-800 shadow-md shadow-blue-200'
                          : 'bg-slate-100 text-slate-700 hover:bg-blue-600 hover:text-white'
                      }`}
                    >
                      {plan.buttonText}
                    </button>
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
        <main className="relative min-h-screen bg-gradient-to-b from-white to-gray-50 pt-20 md:pt-15">
          {content}
        </main>
        <Footer />
      </>
    );
  }

  return content;
}