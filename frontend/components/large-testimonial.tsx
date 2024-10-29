import Image from "next/image";
import TestimonialImg from "@/public/images/faham-profile.jpg";

export default function LargeTestimonial() {
  return (
    <section>
      <div className="mx-auto max-w-2xl px-4 sm:px-6">
        <div className="py-12 md:py-20">
          <div className="space-y-3 text-center">
            <div className="relative inline-flex">
              <Image
                className="rounded-full"
                src={TestimonialImg}
                width={48}
                height={48}
                alt="Faham Rahman"
              />
            </div>
            <p className="text-xl text-gray-900">
              "SitBlinkSip has helped me maintain good posture and eye health during long coding sessions."
            </p>
            <div className="text-sm text-gray-500">
              <span>Faham </span>{" "}
              <span className="text-gray-400">/</span>{" "}
              <span className="text-blue-500">
                Frontend Developer at TCP
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
