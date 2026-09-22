"use client";

import { useCallback, useSyncExternalStore } from "react";

import type { AnswererId } from "@/lib/library";
import { ANSWERER_STORAGE_KEY, DEFAULT_ANSWERER, isAnswererId } from "@/lib/library";

/**
 * The chosen downstream model, remembered per browser.
 *
 * `useSyncExternalStore` rather than `useState` plus an effect: localStorage is
 * an external store that does not exist during server rendering, and seeding
 * state from it in an effect both desynchronises hydration and triggers a
 * cascading render. This gives React an explicit server snapshot instead.
 *
 * Every access is wrapped: private windows and blocked site data throw on read
 * as well as write. Losing the preference is survivable; failing the click is
 * not, so the default simply stands.
 */

const listeners = new Set<() => void>();

// Used only when localStorage refuses the write: without it the snapshot would
// keep reporting the old value and the selector would appear not to work.
let sessionChoice: AnswererId | null = null;

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  // Another tab changing the choice should be reflected here too.
  window.addEventListener("storage", listener);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", listener);
  };
}

function getSnapshot(): AnswererId {
  if (sessionChoice) return sessionChoice;
  try {
    const stored = window.localStorage.getItem(ANSWERER_STORAGE_KEY);
    return isAnswererId(stored) ? stored : DEFAULT_ANSWERER;
  } catch {
    return DEFAULT_ANSWERER;
  }
}

function getServerSnapshot(): AnswererId {
  return DEFAULT_ANSWERER;
}

export function useAnswerer(): [AnswererId, (id: AnswererId) => void] {
  const answerer = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const choose = useCallback((id: AnswererId) => {
    try {
      window.localStorage.setItem(ANSWERER_STORAGE_KEY, id);
      sessionChoice = null;
    } catch {
      // Not remembered across reloads, but it still takes effect now.
      sessionChoice = id;
    }
    listeners.forEach((listener) => listener());
  }, []);

  return [answerer, choose];
}
