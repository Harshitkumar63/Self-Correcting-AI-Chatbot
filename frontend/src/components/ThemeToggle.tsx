"use client";

import { useEffect, useState } from "react";

export default function ThemeToggle() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    const stored = localStorage.getItem("theme") as "dark" | "light" | null;
    const initial = stored || "dark";
    setTheme(initial);
    document.documentElement.classList.toggle("dark", initial === "dark");
    document.documentElement.classList.toggle("light", initial === "light");
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("theme", next);
    document.documentElement.classList.toggle("dark", next === "dark");
    document.documentElement.classList.toggle("light", next === "light");
  };

  return (
    <button
      id="theme-toggle-btn"
      onClick={toggleTheme}
      className="flex h-9 w-9 items-center justify-center rounded-xl border border-border/30 bg-secondary/40 text-sm transition-all hover:bg-secondary/70 hover:border-border/50"
      title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
    >
      <span
        className="transition-transform duration-300"
        style={{ transform: theme === "dark" ? "rotate(0deg)" : "rotate(180deg)" }}
      >
        {theme === "dark" ? "🌙" : "☀️"}
      </span>
    </button>
  );
}
