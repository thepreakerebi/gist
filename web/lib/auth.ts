/**
 * Signing in with an access code and out again.
 *
 * The session itself is an HttpOnly cookie set by the API, so nothing here
 * stores or reads a token: page scripts cannot see it, by design.
 */

import { apiFetch, apiUrl, errorDetail } from "@/lib/library";

export type Account = {
  label: string;
  is_admin: boolean;
  questions_used_today: number;
  /** null for admin codes, which have no daily cap. */
  daily_question_limit: number | null;
  questions_left_today: number | null;
  limit_resets_at: string;
};

export async function signIn(code: string): Promise<Account> {
  const response = await fetch(apiUrl("/v1/auth/login"), {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code }),
  });
  if (!response.ok) {
    let detail = "Could not sign in.";
    try {
      detail = errorDetail(await response.json()) ?? detail;
    } catch {
      // keep the generic message
    }
    throw new Error(detail);
  }
  return (await response.json()) as Account;
}

/** The signed-in code, or null when there is no session (the caller is redirected). */
export async function currentAccount(): Promise<Account | null> {
  try {
    const response = await apiFetch("/v1/auth/me", { cache: "no-store" });
    return response.ok ? ((await response.json()) as Account) : null;
  } catch {
    return null;
  }
}

/** Sign this code out on every device. The code still works to sign in again. */
export async function signOut(): Promise<void> {
  await fetch(apiUrl("/v1/auth/logout"), { method: "POST", credentials: "include" });
}
