'use client';

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const healthTips = [
  "Remember to maintain good posture! 🧘",
  "Time for a water break? 💧",
  "Rest your eyes every 20 minutes 👀",
  "Stretch your neck and shoulders 💪",
  "Take a quick walk break 🚶",
  "Adjust your screen height 🖥️",
  "Stay hydrated for better focus 🌊",
  "Check your sitting position 🪑",
  "Blink more to prevent eye strain 👁️",
  "Keep your shoulders relaxed 😌"
];

interface FloatingText {
  id: number;
  text: string;
  x: number;
  y: number;
}

export default function AnimatedTextBackground() {
  const [floatingTexts, setFloatingTexts] = useState<FloatingText[]>([]);

  useEffect(() => {
    const createFloatingText = () => {
      const id = Date.now();
      const text = healthTips[Math.floor(Math.random() * healthTips.length)];
      const x = Math.random() * (window.innerWidth - 200);
      const y = Math.random() * (window.innerHeight - 50);

      setFloatingTexts(prev => [...prev, { id, text, x, y }]);

      // Remove the text after animation
      setTimeout(() => {
        setFloatingTexts(prev => prev.filter(t => t.id !== id));
      }, 5000);
    };

    const interval = setInterval(createFloatingText, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-blue-50 to-white" />
      <AnimatePresence>
        {floatingTexts.map(({ id, text, x, y }) => (
          <motion.div
            key={id}
            initial={{ opacity: 0, scale: 0.8, x, y }}
            animate={{ opacity: 0.3, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 2 }}
            className="absolute text-sm text-blue-600/50 whitespace-nowrap"
          >
            {text}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
} 