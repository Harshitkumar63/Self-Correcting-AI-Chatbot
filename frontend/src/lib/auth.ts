/**
 * Auth Library — Client-side authentication state management.
 *
 * Manages JWT token storage, user role checking, and auth state.
 */

import { loginUser, fetchCurrentUser, TokenResponse, UserResponse } from "./api";

const TOKEN_KEY = "auth_token";
const USER_KEY = "auth_user";

// ── Token Management ───────────────────────────────────────

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function removeToken(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

// ── User State ─────────────────────────────────────────────

export function getStoredUser(): { username: string; role: string } | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function storeUser(username: string, role: string): void {
  localStorage.setItem(USER_KEY, JSON.stringify({ username, role }));
}

// ── Auth Functions ─────────────────────────────────────────

export async function login(
  username: string,
  password: string
): Promise<TokenResponse> {
  const response = await loginUser(username, password);
  setToken(response.access_token);
  storeUser(response.username, response.role);
  return response;
}

export function logout(): void {
  removeToken();
  if (typeof window !== "undefined") {
    window.location.href = "/login";
  }
}

export function isAuthenticated(): boolean {
  return getToken() !== null;
}

export function getUserRole(): string | null {
  const user = getStoredUser();
  return user?.role || null;
}

export function isAdmin(): boolean {
  return getUserRole() === "admin";
}

export async function validateSession(): Promise<UserResponse | null> {
  const token = getToken();
  if (!token) return null;

  try {
    const user = await fetchCurrentUser();
    storeUser(user.username, user.role);
    return user;
  } catch {
    // Token expired or invalid
    removeToken();
    return null;
  }
}
