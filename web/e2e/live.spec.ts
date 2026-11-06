import { execFileSync } from "node:child_process";
import path from "node:path";

import { expect, test, type BrowserContext, type Page } from "@playwright/test";

/**
 * Functional tests against the real stack: FastAPI, Postgres with pgvector, and
 * the CLIP, CLAP and Whisper scorers. Run with GIST_E2E_LIVE=1 once the stack in
 * e2e/README.md is up. The cases run in order on one short video, because each
 * depends on the state the previous one left.
 *
 * The answering model is switched to the extractive answerer in flight, so the
 * suite exercises Gist's own selection without spending TwelveLabs credits. The
 * browser still sends exactly the request the app builds; only the `answerer`
 * field is rewritten.
 *
 * Each run issues itself a fresh, ordinary (non-admin) access code with the
 * real `gist-access` command, so the library starts empty and the daily cap of
 * five questions applies exactly as it does to anyone else. Signing in once
 * through the API and handing the cookie to each test keeps the suite inside
 * the five-attempts-a-minute sign-in limit.
 *
 * Case IDs (FT-xx) match docs/testing/functional-tests.md.
 */

const API = "http://localhost:8100";
const DATABASE_URL =
  process.env.GIST_E2E_DATABASE_URL ?? "postgresql://postgres:gist@127.0.0.1:55432/gist";
const PYTHON = path.resolve(__dirname, "../../.venv/bin/python");

function issueCode(): string {
  const output = execFileSync(
    PYTHON,
    ["-m", "gist.access.cli", "create", "--label", `e2e ${new Date().toISOString()}`],
    { env: { ...process.env, DATABASE_URL }, encoding: "utf8" },
  );
  const code = output.match(/GIST(?:-[0-9A-Z]{4}){4}/)?.[0];
  if (!code) throw new Error(`gist-access printed no code:\n${output}`);
  return code;
}

async function useSession(context: BrowserContext, token: string) {
  await context.addCookies([{ name: "gist_session", value: token, domain: "localhost", path: "/" }]);
}

async function signInThroughTheForm(page: Page, code: string) {
  await page.goto("/login");
  await page.getByLabel("Access code").fill(code);
  await page.getByRole("button", { name: "Sign in" }).click();
}

async function sessionToken(context: BrowserContext): Promise<string> {
  const cookie = (await context.cookies()).find((item) => item.name === "gist_session");
  if (!cookie) throw new Error("no session cookie after signing in");
  return cookie.value;
}
const LINK = "https://www.youtube.com/watch?v=jNQXAC9IVRw"; // "Me at the zoo", 19 s
const QUESTION = "What does he say about the elephants?";

async function extractiveAnswers(page: Page) {
  await page.route(`${API}/v1/library/videos/*/query`, async (route) => {
    const body = JSON.parse(route.request().postData() ?? "{}");
    await route.continue({ postData: JSON.stringify({ ...body, answerer: "extractive" }) });
  });
}

async function addLink(page: Page, link: string) {
  await page.getByLabel("YouTube video URL").fill(link);
  await page.locator("#video-consent").check();
  await page.getByRole("button", { name: "Add" }).click();
}

test.describe.serial("Live stack", () => {
  let videoPath = "";
  let code = "";
  let token = "";

  // A new code has an empty library, so the suite can be re-run on the same database.
  test.beforeAll(async ({ request }) => {
    code = issueCode();
    const response = await request.post(`${API}/v1/auth/login`, { data: { code } });
    expect(response.status()).toBe(200);
    token = /gist_session=([^;]+)/.exec(response.headers()["set-cookie"] ?? "")?.[1] ?? "";
    expect(token).not.toBe("");
  });

  test.beforeEach(async ({ context }) => useSession(context, token));

  test("FT-11 a new video is ingested and becomes ready", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("Nothing here yet")).toBeVisible();
    await addLink(page, LINK);

    // Progress is streamed while ingestion runs.
    await expect(page.getByRole("progressbar", { name: /^Processing / })).toBeVisible({ timeout: 30_000 });

    const ready = page.getByRole("link", { name: /Me at the zoo/i }).first();
    await expect(ready).toBeVisible({ timeout: 480_000 });
    videoPath = (await ready.getAttribute("href")) ?? "";
    expect(videoPath).toMatch(/^\/v\//);
  });

  test("FT-12 adding the same link again does not ingest it twice", async ({ page }) => {
    await page.goto("/");
    await addLink(page, LINK);
    await expect(page.getByText("Already in your library.")).toBeVisible();
    await expect(page.getByRole("link", { name: /Me at the zoo/i })).toHaveCount(1);
  });

  test("FT-13 a question is answered from selected evidence, live", async ({ page }) => {
    await extractiveAnswers(page);
    await page.goto(videoPath);
    await page.getByPlaceholder("Ask about this video").fill(QUESTION);
    await page.getByRole("button", { name: "Ask" }).click();

    const turn = page.locator("li").filter({ has: page.getByRole("heading", { name: QUESTION }) }).last();
    await expect(turn.getByText(/Kept \d+ of \d+ candidates/)).toBeVisible({ timeout: 120_000 });
    await expect(turn.getByText("replayed from a cached run")).toHaveCount(0);

    const first = turn.locator("ol > li").first();
    await first.getByRole("button").click();
    await expect(first.getByText("Why it was kept:")).toBeVisible();
    const clip = first.locator("video");
    await expect(clip).toHaveAttribute("src", new RegExp(`^${API}/v1/library/videos/`));
    await expect
      .poll(() => clip.evaluate((el: HTMLVideoElement) => el.readyState), { timeout: 20_000 })
      .toBeGreaterThanOrEqual(1);
  });

  test("FT-14 the conversation survives a reload", async ({ page }) => {
    await page.goto(videoPath);
    await expect(page.getByRole("heading", { name: QUESTION })).toBeVisible();
  });

  test("FT-15 clearing the chat asks once, then removes the history", async ({ page }) => {
    await page.goto(videoPath);
    await expect(page.getByRole("heading", { name: QUESTION })).toBeVisible();
    const clear = page.getByRole("button", { name: "Clear chat history" });
    await clear.click();
    await expect(clear).toContainText("Clear?");
    await expect(page.getByRole("heading", { name: QUESTION })).toBeVisible(); // not yet
    await clear.click();
    await expect(page.getByRole("heading", { name: QUESTION })).toHaveCount(0);

    await page.reload();
    await expect(page.getByPlaceholder("Ask about this video")).toBeVisible();
    await expect(page.getByRole("heading", { name: QUESTION })).toHaveCount(0);
  });

  test("FT-16 a link the API cannot accept is explained to the user", async ({ page }) => {
    await page.goto("/");
    await addLink(page, "not a link");
    const alert = page.locator("form [role=alert]");
    await expect(alert).toBeVisible();
    // The message must say what is wrong, not only the HTTP status.
    await expect(alert).toContainText(/http|URL|link/i);
  });

  test("FT-18 without a session every page sends you to sign in", async ({ browser }) => {
    const stranger = await browser.newContext();
    const page = await stranger.newPage();
    await page.goto(videoPath);
    await expect(page).toHaveURL(/\/login\?next=/);
    await expect(page.getByRole("heading", { name: "Sign in" })).toBeVisible();

    // The API itself refuses, not only the page.
    const response = await page.request.get(`${API}/v1/library/videos`);
    expect(response.status()).toBe(401);
    await stranger.close();
  });

  test("FT-19 a wrong access code is refused with a clear message", async ({ browser }) => {
    const stranger = await browser.newContext();
    const page = await stranger.newPage();
    await signInThroughTheForm(page, "GIST-0000-0000-0000-0000");
    await expect(page.locator("form [role=alert]")).toHaveText("That code is not valid.");
    await expect(page).toHaveURL(/\/login/);
    await stranger.close();
  });

  test("FT-20 the code signs in and shows the questions left today", async ({ browser }) => {
    const fresh = await browser.newContext();
    const page = await fresh.newPage();
    await signInThroughTheForm(page, code);
    await expect(page).toHaveURL("http://localhost:3100/");
    await expect(page.getByRole("link", { name: /Me at the zoo/i })).toBeVisible();
    await expect(page.getByText("4 of 5 questions left today")).toBeVisible();
    await expect(page.getByRole("button", { name: "Log out" })).toBeVisible();
    await fresh.close();
  });

  test("FT-21 the sixth question of the day is refused", async ({ page }) => {
    await extractiveAnswers(page);
    await page.goto(videoPath);
    const box = page.getByPlaceholder("Ask about this video");
    for (let asked = 1; asked <= 5; asked++) {
      await box.fill(`${QUESTION} (${asked})`);
      await page.getByRole("button", { name: "Ask" }).click();
      if (asked < 5) {
        const turn = page.locator("li").filter({ has: page.getByRole("heading", { name: `${QUESTION} (${asked})` }) });
        await expect(turn.getByText(/Kept \d+ of \d+ candidates/)).toBeVisible({ timeout: 120_000 });
      }
    }
    const refusal = page.getByRole("alert").filter({ hasText: "used all 5 questions for today" });
    await expect(refusal).toBeVisible();
    await refusal.scrollIntoViewIfNeeded();
    await expect(page.getByText("0 of 5 questions left today")).toBeVisible();
  });

  test("FT-22 Log out signs out everywhere; the code brings the library back", async ({ page, browser }) => {
    const other = await browser.newContext();
    await useSession(other, token);
    const otherPage = await other.newPage();
    await otherPage.goto("/");
    await expect(otherPage.getByRole("link", { name: /Me at the zoo/i })).toBeVisible();

    await page.goto("/");
    await page.getByRole("button", { name: "Log out" }).click();
    await expect(page).toHaveURL(/\/login/);

    // The other device lost its session too.
    await otherPage.reload();
    await expect(otherPage).toHaveURL(/\/login/);
    await other.close();

    await signInThroughTheForm(page, code);
    await expect(page.getByRole("link", { name: /Me at the zoo/i })).toBeVisible();
    // Signing out and in again does not reset the day's count.
    await expect(page.getByText("0 of 5 questions left today")).toBeVisible();
    token = await sessionToken(page.context());
  });

  test("FT-17 removing a video deletes it from the library", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /^Remove / }).first().click();
    await expect(page.getByRole("link", { name: /Me at the zoo/i })).toHaveCount(0);
    await page.reload();
    await expect(page.getByText("Nothing here yet")).toBeVisible();
  });
});
