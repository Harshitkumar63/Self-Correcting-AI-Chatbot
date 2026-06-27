"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { login } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!username.trim() || !password.trim()) return;

      setIsLoading(true);
      setError("");

      try {
        await login(username.trim(), password);
        router.push("/");
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Login failed. Please check your credentials."
        );
      } finally {
        setIsLoading(false);
      }
    },
    [username, password, router]
  );

  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <div className="w-full max-w-md animate-fade-in">
        {/* Header */}
        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl gradient-primary text-2xl shadow-lg shadow-primary/30">
            🧠
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground">
            Self-Improving LLM Pipeline
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Sign in to access the dashboard
          </p>
        </div>

        {/* Login Card */}
        <div className="gradient-card rounded-2xl border border-border/50 p-8 shadow-2xl">
          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Error Message */}
            {error && (
              <div className="rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3 text-sm text-red-400 animate-fade-in">
                {error}
              </div>
            )}

            {/* Username */}
            <div className="space-y-2">
              <label
                htmlFor="login-username"
                className="block text-sm font-medium text-foreground/80"
              >
                Username
              </label>
              <input
                id="login-username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
                className="w-full rounded-xl border border-border/50 bg-secondary/40 px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground/50 transition-all focus:border-primary/60 focus:outline-none focus:ring-2 focus:ring-primary/20"
                autoFocus
                autoComplete="username"
              />
            </div>

            {/* Password */}
            <div className="space-y-2">
              <label
                htmlFor="login-password"
                className="block text-sm font-medium text-foreground/80"
              >
                Password
              </label>
              <input
                id="login-password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                className="w-full rounded-xl border border-border/50 bg-secondary/40 px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground/50 transition-all focus:border-primary/60 focus:outline-none focus:ring-2 focus:ring-primary/20"
                autoComplete="current-password"
              />
            </div>

            {/* Submit */}
            <button
              id="login-submit-btn"
              type="submit"
              disabled={!username.trim() || !password.trim() || isLoading}
              className="w-full rounded-xl gradient-primary px-6 py-3 text-sm font-semibold text-primary-foreground shadow-lg shadow-primary/20 transition-all hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="animate-spin">⏳</span> Signing in...
                </span>
              ) : (
                "Sign In"
              )}
            </button>
          </form>

          {/* Default Credentials Hint */}
          <div className="mt-6 rounded-lg bg-secondary/30 border border-border/20 p-3">
            <p className="text-center text-xs text-muted-foreground">
              Default credentials: <code className="font-mono text-primary/80">admin</code> / <code className="font-mono text-primary/80">admin123</code>
            </p>
          </div>
        </div>

        {/* Footer */}
        <p className="mt-6 text-center text-xs text-muted-foreground">
          Qwen2.5-0.5B-Instruct • Hybrid Evaluator • PEFT/LoRA
        </p>
      </div>
    </div>
  );
}
