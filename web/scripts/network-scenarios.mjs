// Time the main user flow under emulated network conditions.
//
// For each profile, three runs of: open the library, open the video, ask a
// question, wait for the evidence, open the first evidence clip and wait until
// it can play. Network conditions are applied through the Chrome DevTools
// Protocol for every request the page makes, including the API's event stream
// and the clip itself. Questions go to the extractive answerer (rewritten in
// flight), so the hosted model's latency is not part of the timing.
//
// Needs the live stack from e2e/README.md and `next start` on port 3200:
//
//   node scripts/network-scenarios.mjs <video-id> > ../results/web-performance/network.json

import { chromium } from "@playwright/test";

const BASE = "http://localhost:3200";
const API = "http://127.0.0.1:8100";
const VIDEO = process.argv[2];
if (!VIDEO) throw new Error("usage: node scripts/network-scenarios.mjs <video-id>");

// Throughput in bytes per second, latency in ms. The 4G and 3G figures are the
// presets Chrome DevTools ships; "Slow 4G" is Lighthouse's mobile profile.
const PROFILES = [
  { name: "unthrottled", latency: 0, down: -1, up: -1 },
  { name: "Fast 4G", latency: 60, down: (9 * 1024 * 1024) / 8, up: (1.5 * 1024 * 1024) / 8 },
  { name: "Slow 4G", latency: 150, down: (1.6 * 1024 * 1024) / 8, up: (750 * 1024) / 8 },
  { name: "Slow 3G", latency: 400, down: (400 * 1024) / 8, up: (400 * 1024) / 8 },
];
const QUESTIONS = [
  "Where do they watch the sunset?",
  "What does he say about the monkeys?",
  "Which island has no cars?",
];
const RUNS = 3;

async function clearConversation() {
  await fetch(`${API}/v1/library/videos/${VIDEO}/conversation`, { method: "DELETE" });
}

async function runOnce(browser, profile, question) {
  await clearConversation();
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  await cdp.send("Network.enable");
  await cdp.send("Network.setCacheDisabled", { cacheDisabled: true });
  await cdp.send("Network.emulateNetworkConditions", {
    offline: false,
    latency: profile.latency,
    downloadThroughput: profile.down,
    uploadThroughput: profile.up,
  });
  await page.route(`${API}/v1/library/videos/*/query`, async (route) => {
    const body = JSON.parse(route.request().postData() ?? "{}");
    await route.continue({ postData: JSON.stringify({ ...body, answerer: "extractive" }) });
  });

  const t = {};
  const timeout = 180_000;
  let start = Date.now();
  await page.goto(`${BASE}/`, { timeout });
  await page.getByRole("link").filter({ hasText: /BALI/i }).first().waitFor({ timeout });
  t.library_listed_s = (Date.now() - start) / 1000;

  start = Date.now();
  await page.goto(`${BASE}/v/${VIDEO}`, { timeout });
  await page.getByPlaceholder("Ask about this video").waitFor({ timeout });
  t.video_ready_s = (Date.now() - start) / 1000;

  await page.getByPlaceholder("Ask about this video").fill(question);
  start = Date.now();
  await page.getByRole("button", { name: "Ask" }).click();
  await page.getByText(/Kept \d+ of \d+ candidates/).first().waitFor({ timeout });
  t.evidence_shown_s = (Date.now() - start) / 1000;

  // Evidence items are the list items holding an expandable button. Clips are
  // cut for at most four of them, so take the first one that has a clip.
  const item = page.locator("li:has(> button[aria-expanded]):has(video)").first();
  start = Date.now();
  await item.locator("> button[aria-expanded]").click();
  const clip = item.locator("video");
  await clip.waitFor({ timeout });
  await page.waitForFunction(
    (el) => el && el.readyState >= 3,
    await clip.elementHandle(),
    { timeout },
  );
  t.clip_playable_s = (Date.now() - start) / 1000;

  await context.close();
  return t;
}

const browser = await chromium.launch({ channel: "chrome" });
const results = [];
for (const profile of PROFILES) {
  for (let run = 0; run < RUNS; run += 1) {
    const question = QUESTIONS[run % QUESTIONS.length];
    try {
      const timings = await runOnce(browser, profile, question);
      results.push({ profile: profile.name, run: run + 1, question, ok: true, ...timings });
    } catch (error) {
      results.push({ profile: profile.name, run: run + 1, question, ok: false, error: String(error).slice(0, 300) });
    }
    console.error(JSON.stringify(results.at(-1)));
  }
}
await browser.close();
await clearConversation();
console.log(JSON.stringify({ generated: new Date().toISOString(), video: VIDEO, results }, null, 2));
