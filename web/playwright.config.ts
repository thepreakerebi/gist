import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end functional tests for the web app.
 *
 * Two projects share one frontend, which always points at the API on port 8100:
 *
 *   offline  every API request is refused inside the browser, so the app runs on
 *            its cached-run snapshot. Needs nothing but the frontend.
 *   live     requests pass through to a real FastAPI server and Postgres. Runs
 *            only when GIST_E2E_LIVE=1; see e2e/README.md for the stack.
 *
 * Every test records a trace, a video and a final screenshot, which are the
 * pass/fail evidence the functional test table in docs/testing/ points to.
 */

export const API = "http://127.0.0.1:8100";
const PORT = 3100;
const live = process.env.GIST_E2E_LIVE === "1";

// Installed Google Chrome rather than Playwright's bundled Chromium: the bundled
// build has no H.264 decoder, and every evidence clip Gist cuts is H.264, so a
// clip test there fails on the browser, not on the app.
const CHROME = { ...devices["Desktop Chrome"], channel: "chrome", viewport: { width: 1440, height: 900 } };

export default defineConfig({
  testDir: "./e2e",
  outputDir: "./e2e-results/artifacts",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 60_000,
  reporter: [
    ["list"],
    ["json", { outputFile: "e2e-results/results.json" }],
    ["html", { outputFolder: "e2e-results/html", open: "never" }],
  ],
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "on",
    video: "on",
    screenshot: "on",
    viewport: { width: 1440, height: 900 },
  },
  projects: [
    { name: "offline", testMatch: /offline\.spec\.ts/, use: CHROME },
    ...(live
      ? [
          {
            name: "live",
            testMatch: /live\.spec\.ts/,
            timeout: 600_000,
            use: CHROME,
          },
        ]
      : []),
  ],
  webServer: {
    command: `bun run dev --port ${PORT}`,
    url: `http://localhost:${PORT}`,
    env: { NEXT_PUBLIC_GIST_API: API },
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
