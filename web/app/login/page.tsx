"use client";

import { ArrowRight, Loader2 } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";

import { signIn } from "@/lib/auth";
import { cn } from "@/lib/utils";

/** Only same-site paths, so the redirect cannot be pointed at another website. */
function safeNext(value: string | null): string {
  return value && value.startsWith("/") && !value.startsWith("//") ? value : "/";
}

function SignInForm() {
  const router = useRouter();
  const next = safeNext(useSearchParams().get("next"));
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!code.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      await signIn(code.trim());
      router.replace(next);
    } catch (err) {
      setError(
        err instanceof Error && !/fetch|Failed/.test(err.message)
          ? err.message
          : "Can't reach the Gist API. Start it with: uvicorn gist.api.app:app --port 8000",
      );
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} noValidate className="mt-10">
      <label htmlFor="access-code" className="text-sm font-medium">
        Access code
      </label>
      <div
        className={cn(
          "mt-2 flex items-center gap-2 rounded-lg border bg-card pl-4 pr-2 transition-colors",
          "focus-within:border-ring/60 focus-within:ring-[3px] focus-within:ring-ring/15",
          error ? "border-destructive/50" : "border-border",
        )}
      >
        <input
          id="access-code"
          value={code}
          onChange={(event) => setCode(event.target.value)}
          placeholder="GIST-XXXX-XXXX-XXXX-XXXX"
          autoComplete="off"
          autoCapitalize="characters"
          spellCheck={false}
          autoFocus
          disabled={busy}
          aria-invalid={Boolean(error)}
          aria-describedby={error ? "access-code-error" : undefined}
          className="tabular h-12 min-w-0 flex-1 bg-transparent font-mono text-[15px] tracking-wide outline-none placeholder:text-muted-foreground/60 disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={busy || code.trim().length === 0}
          className={cn(
            "inline-flex h-9 items-center gap-1.5 rounded-md px-3 text-sm font-medium transition-all",
            "bg-primary text-primary-foreground hover:opacity-90",
            "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring",
            "disabled:pointer-events-none disabled:opacity-40",
          )}
        >
          {busy ? (
            <Loader2 className="size-4 animate-spin" aria-hidden />
          ) : (
            <ArrowRight className="size-4" aria-hidden />
          )}
          <span>{busy ? "Signing in" : "Sign in"}</span>
        </button>
      </div>
      {error && (
        <p id="access-code-error" role="alert" className="mt-2 text-sm text-destructive">
          {error}
        </p>
      )}
    </form>
  );
}

export default function LoginPage() {
  return (
    <main className="mx-auto w-full max-w-md px-6 pb-24 pt-16 sm:pt-24">
      <p className="text-sm font-semibold tracking-tight">Gist</p>
      <h1 className="mt-8 text-balance text-3xl font-semibold leading-[1.15] tracking-tight">
        Sign in
      </h1>
      <p className="mt-3 text-[15px] leading-relaxed text-muted-foreground">
        Enter the access code you were given. Your videos and questions stay with
        that code, so signing in with it again brings them back.
      </p>
      <Suspense>
        <SignInForm />
      </Suspense>
    </main>
  );
}
