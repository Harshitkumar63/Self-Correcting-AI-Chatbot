import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Geist_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Self-Improving LLM Pipeline | Automated Evaluation & Feedback Loop",
  description:
    "An automated feedback loop system where LLM outputs are evaluated using semantic similarity and low-quality responses are collected for iterative LoRA fine-tuning.",
  keywords: [
    "LLM",
    "fine-tuning",
    "LoRA",
    "PEFT",
    "machine learning",
    "evaluation",
    "self-improving",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${geistMono.variable} h-full antialiased dark`}
    >
      <body className="min-h-full flex flex-col bg-background">
        {children}
      </body>
    </html>
  );
}
