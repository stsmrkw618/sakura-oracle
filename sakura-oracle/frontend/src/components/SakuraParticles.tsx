"use client";

import { useEffect, useRef } from "react";

interface Petal {
  x: number;
  y: number;
  size: number;
  speedX: number;
  speedY: number;
  rotation: number;
  rotationSpeed: number;
  opacity: number;
  color: string;
}

const PETAL_COLORS = [
  "#E8879C",
  "#F4B8C5",
  "#FFB7C5",
  "#E8A0B0",
  "#D4708A",
];

export default function SakuraParticles() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    const petals: Petal[] = [];
    const PETAL_COUNT = 25;

    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };

    const createPetal = (): Petal => ({
      x: Math.random() * canvas.width,
      y: -20 - Math.random() * 100,
      size: 6 + Math.random() * 8,
      speedX: -0.5 + Math.random() * 1.5,
      speedY: 0.8 + Math.random() * 1.2,
      rotation: Math.random() * Math.PI * 2,
      rotationSpeed: 0.01 + Math.random() * 0.03,
      opacity: 0.4 + Math.random() * 0.5,
      color: PETAL_COLORS[Math.floor(Math.random() * PETAL_COLORS.length)],
    });

    const drawPetal = (petal: Petal) => {
      ctx.save();
      ctx.translate(petal.x, petal.y);
      ctx.rotate(petal.rotation);
      ctx.globalAlpha = petal.opacity;

      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.bezierCurveTo(
        petal.size * 0.5, -petal.size * 0.3,
        petal.size, -petal.size * 0.1,
        petal.size, 0
      );
      ctx.bezierCurveTo(
        petal.size, petal.size * 0.1,
        petal.size * 0.5, petal.size * 0.3,
        0, 0
      );
      ctx.fillStyle = petal.color;
      ctx.fill();

      ctx.restore();
    };

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (const petal of petals) {
        petal.x += petal.speedX + Math.sin(petal.y * 0.01) * 0.3;
        petal.y += petal.speedY;
        petal.rotation += petal.rotationSpeed;

        if (petal.y > canvas.height + 20) {
          petal.y = -20;
          petal.x = Math.random() * canvas.width;
        }

        drawPetal(petal);
      }

      animationId = requestAnimationFrame(animate);
    };

    resize();
    for (let i = 0; i < PETAL_COUNT; i++) {
      const petal = createPetal();
      petal.y = Math.random() * canvas.height;
      petals.push(petal);
    }

    animate();
    window.addEventListener("resize", resize);

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none z-0"
    />
  );
}
