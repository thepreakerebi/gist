import { expect, test, type Page } from "@playwright/test";

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
 * Case IDs (FT-xx) match docs/testing/functional-tests.md.
 */

const API = "http://127.0.0.1:8100";
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

  // Start from an empty library so the suite can be re-run on the same database.
  test.beforeAll(async ({ request }) => {
    const { videos } = await (await request.get(`${API}/v1/library/videos`)).json();
    for (const video of videos as { id: string }[]) {
      await request.delete(`${API}/v1/library/videos/${video.id}`);
    }
  });

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

  test("FT-17 removing a video deletes it from the library", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /^Remove / }).first().click();
    await expect(page.getByRole("link", { name: /Me at the zoo/i })).toHaveCount(0);
    await page.reload();
    await expect(page.getByText("Nothing here yet")).toBeVisible();
  });
});
