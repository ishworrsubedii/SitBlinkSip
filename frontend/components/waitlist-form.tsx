// 'use client';

// import { useState } from 'react';
// import { toast } from 'sonner';
// import { Loader2 } from 'lucide-react';

// export default function WaitlistForm() {
//   const [email, setEmail] = useState('');
//   const [isLoading, setIsLoading] = useState(false);

//   const handleSubmit = async (e: React.FormEvent) => {
//     e.preventDefault();
//     setIsLoading(true);

//     try {
//       const response = await fetch('/api/waitlist', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ email }),
//       });

//       const data = await response.json();

//       if (!response.ok) {
//         throw new Error(data.error || 'Something went wrong');
//       }

//       toast.success('Successfully joined the waitlist!');
//       setEmail('');
//     } catch (error) {
//       toast.error(error instanceof Error ? error.message : 'Failed to join waitlist');
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   return (
//     <form onSubmit={handleSubmit} className="max-w-md w-full mx-auto">
//       <div className="flex flex-col sm:flex-row gap-3">
//         <div className="relative flex-grow">
//           <input 
//             type="email" 
//             value={email}
//             onChange={(e) => setEmail(e.target.value)}
//             placeholder="Enter your email" 
//             className="w-full rounded-lg border-0 bg-white/10 px-4 py-3 text-gray-900 placeholder-gray-500 backdrop-blur-sm ring-1 ring-inset ring-gray-300 focus:ring-2 focus:ring-blue-500 focus:outline-none"
//             required
//           />
//         </div>
//         <button 
//           type="submit"
//           disabled={isLoading}
//           className="group relative inline-flex items-center justify-center rounded-lg bg-blue-600 px-6 py-3 text-white transition-all hover:bg-blue-700 disabled:opacity-70"
//         >
//           {isLoading ? (
//             <Loader2 className="w-5 h-5 animate-spin" />
//           ) : (
//             'Join Waitlist'
//           )}
//         </button>
//       </div>
//     </form>
//   );
// } 