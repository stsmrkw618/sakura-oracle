"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import SakuraParticles from "@/components/SakuraParticles";
import RaceSelector from "@/components/RaceSelector";
import { useRace } from "@/context/RaceContext";

export default function Home() {
  const { predictions } = useRace();
  const raceInfo = predictions.race_info;

  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden bg-navy-dark">
      <SakuraParticles />

      <div className="relative z-10 flex flex-col items-center text-center px-6">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="mb-8"
        >
          <span className="text-6xl mb-4 block">🌸</span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="text-4xl font-bold text-sakura-pink tracking-wider mb-3"
        >
          SAKURA ORACLE
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="text-xl text-white font-medium mb-2"
        >
          {raceInfo.name} AI予測
        </motion.p>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
          className="text-sm text-muted-foreground mb-8"
        >
          {raceInfo.date} {raceInfo.course}
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.9 }}
          className="w-full max-w-[280px] mb-4"
        >
          <RaceSelector />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.0 }}
        >
          <Link href="/prediction">
            <button className="relative px-10 py-4 rounded-full font-bold text-lg text-white bg-gradient-to-r from-sakura-pink to-sakura-dark shadow-lg shadow-sakura-pink/30 transition-all duration-300 hover:shadow-xl hover:shadow-sakura-pink/50 hover:scale-105 active:scale-95">
              予測を見る →
              <span className="absolute inset-0 rounded-full bg-white/0 hover:bg-white/10 transition-colors duration-300" />
            </button>
          </Link>
        </motion.div>
      </div>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1, delay: 1.5 }}
        className="absolute bottom-8 text-xs text-muted-foreground z-10"
      >
        Powered by AI × LightGBM
      </motion.p>
    </div>
  );
}
