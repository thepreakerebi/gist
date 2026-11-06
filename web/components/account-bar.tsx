"use client";

import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";

import type { Account } from "@/lib/auth";
import { currentAccount, signOut } from "@/lib/auth";
import { cn } from "@/lib/utils";

/**
 * Who is signed in, how many questions are left today, and "Log out".
 *
 * "Log out" ends the session on every device the code is signed in on. The
 * code keeps working, so signing in again restores the same library.
 */
export function AccountBar({ className }: { className?: string }) {
  const [account, setAccount] = useState<Account | null>(null);
  const [leaving, setLeaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const refresh = () => {
      currentAccount().then((value) => !cancelled && setAccount(value));
    };
    refresh();
    window.addEventListener("gist:account", refresh);
    return () => {
      cancelled = true;
      window.removeEventListener("gist:account", refresh);
    };
  }, []);

  if (!account) return null;

  async function logOut() {
    setLeaving(true);
    try {
      await signOut();
    } finally {
      window.location.assign("/login");
    }
  }

  const left = account.questions_left_today;

  return (
    <div className={cn("flex items-center gap-3 text-xs text-muted-foreground", className)}>
      <span className="tabular hidden min-w-0 truncate sm:inline">
        {account.label}
        {left !== null && (
          <>
            {" · "}
            <span className={cn(left === 0 && "font-medium text-destructive")}>
              {left} of {account.daily_question_limit} questions left today
            </span>
          </>
        )}
      </span>
      {left !== null && (
        <span className={cn("tabular sm:hidden", left === 0 && "font-medium text-destructive")}>
          {left}/{account.daily_question_limit} left
        </span>
      )}
      <button
        type="button"
        onClick={() => void logOut()}
        disabled={leaving}
        className={cn(
          "inline-flex shrink-0 items-center gap-1.5 rounded px-2 py-1.5 font-medium text-foreground transition-colors hover:bg-muted",
          "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring",
          "disabled:opacity-50",
        )}
      >
        {leaving && <Loader2 className="size-3.5 animate-spin" aria-hidden />}
        Log out
      </button>
    </div>
  );
}
