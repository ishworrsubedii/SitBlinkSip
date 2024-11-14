import PricingSection from '@/components/PricingSection';
import PageIllustration from '@/components/page-illustration';
import { Badge } from "@/components/ui/badge";


export default function PricingPage() {
  return (
    <main className="grow">
    <div className="relative">
      <PageIllustration />
     
      
      <div className="pt-32 pb-12 md:pt-20 md:pb-20">
        
      
        
        {/* Page header */}
        <div className="text-center pb-12 md:pb-16 max-w-3xl mx-auto px-4 sm:px-6">
          
        </div>
        
        <PricingSection standalone={true} />
      </div>
   
      </div>
     
    </main>
  );
}