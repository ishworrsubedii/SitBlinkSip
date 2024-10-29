import Link from "next/link";

export default function Logo() {
  return (
    <Link href="/" className="inline-flex" aria-label="Sitblink">
      <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">
        <circle 
          cx="16" 
          cy="16" 
          r="15" 
          className="fill-none stroke-blue-500" 
          strokeWidth="2" 
        />
        
        <path
          className="fill-blue-400"
          d="M16 8c-6 0-11 4-11 8s5 8 11 8 11-4 11-8-5-8-11-8zm0 12c-2.2 0-4-1.8-4-4s1.8-4 4-4 4 1.8 4 4-1.8 4-4 4z"
        />
        
        <circle 
          cx="16" 
          cy="16" 
          r="2" 
          className="fill-blue-600" 
        />
        
       
      </svg>
    </Link>
  );
}
