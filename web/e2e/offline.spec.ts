import { expect, test, type Page } from "@playwright/test";

/**
 * Functional tests that need no backend. The API is made unreachable inside the
 * browser, which is the condition the cached-run snapshot exists for, and the
 * error-path tests replace individual responses with the failure under test.
 *
 * Case IDs (FT-xx) match docs/testing/functional-tests.md.
 */

const API = "http://127.0.0.1:8100";
const VIDEO_ID = "98c05bae-0303-488e-8bfc-db6a40f825b7"; // "Me at the zoo", baked snapshot
const CACHED_QUESTION = "What does he say is cool about the elephants?";

// Next adds its own role="alert" route announcer, so scope alerts to the form.
const formAlert = (page: Page) => page.locator("form [role=alert]");

async function apiUnreachable(page: Page) {
  await page.route(`${API}/**`, (route) => route.abort("connectionrefused"));
}

test.describe("API unreachable: cached-run snapshot", () => {
  test.beforeEach(async ({ page }) => apiUnreachable(page));

  test("FT-01 library falls back to the snapshot and says so", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("Offline.", { exact: false })).toBeVisible();
    await expect(page.getByRole("link", { name: /Me at the zoo/ }).first()).toBeVisible();
  });

  test("FT-02 opening a video shows its title and question box", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("link", { name: /Me at the zoo/ }).first().click();
    await expect(page).toHaveURL(new RegExp(`/v/${VIDEO_ID}`));
    await expect(page.getByPlaceholder("Ask about this video")).toBeVisible();
  });

  test("FT-03 a question is answered with evidence, labelled as cached", async ({ page }) => {
    await page.goto(`/v/${VIDEO_ID}`);
    await page.getByPlaceholder("Ask about this video").fill(CACHED_QUESTION);
    await page.getByRole("button", { name: "Ask" }).click();

    const turn = page.locator("li").filter({ has: page.getByRole("heading", { name: CACHED_QUESTION }) }).last();
    await expect(turn.getByText("replayed from a cached run")).toBeVisible({ timeout: 20_000 });
    await expect(turn.getByText(/long tr/)).toBeVisible();
    await expect(turn.getByText(/Kept \d+ of \d+ candidates/)).toBeVisible();
  });

  test("FT-04 an evidence item expands to its clip and the reason it was kept", async ({ page }) => {
    await page.goto(`/v/${VIDEO_ID}`);
    await page.getByPlaceholder("Ask about this video").fill(CACHED_QUESTION);
    await page.getByRole("button", { name: "Ask" }).click();

    const turn = page.locator("li").filter({ has: page.getByRole("heading", { name: CACHED_QUESTION }) }).last();
    await expect(turn.getByText("replayed from a cached run")).toBeVisible({ timeout: 20_000 });

    const item = turn.locator("ol > li").first();
    await item.getByRole("button").click();
    await expect(item.getByRole("button")).toHaveAttribute("aria-expanded", "true");
    await expect(item.getByText("Why it was kept:")).toBeVisible();

    const video = item.locator("video");
    await expect(video).toHaveAttribute("src", /\/cached-runs\/clips\//);
    // The clip must actually load, not just be referenced.
    await expect
      .poll(() => video.evaluate((el: HTMLVideoElement) => el.readyState), { timeout: 10_000 })
      .toBeGreaterThanOrEqual(1);
  });

  test("FT-05 a question with no recorded run still shows real evidence", async ({ page }) => {
    await page.goto(`/v/${VIDEO_ID}`);
    const unrehearsed = "What colour is his shirt?";
    await page.getByPlaceholder("Ask about this video").fill(unrehearsed);
    await page.getByRole("button", { name: "Ask" }).click();
    await expect(page.getByText("replayed from a cached run").last()).toBeVisible({ timeout: 20_000 });
  });

  test("FT-06 adding a video while the API is down gives a clear message", async ({ page }) => {
    await page.goto("/");
    await page.getByLabel("YouTube video URL").fill("https://www.youtube.com/watch?v=jNQXAC9IVRw");
    await page.locator("#video-consent").check();
    await page.getByRole("button", { name: "Add" }).click();
    await expect(formAlert(page)).toContainText("Can't reach the Gist API");
  });
});

test.describe("Input validation", () => {
  test.beforeEach(async ({ page }) => apiUnreachable(page));

  test("FT-07 Add stays disabled until there is a link and consent", async ({ page }) => {
    await page.goto("/");
    const add = page.getByRole("button", { name: "Add" });
    await expect(add).toBeDisabled();

    await page.getByLabel("YouTube video URL").fill("https://www.youtube.com/watch?v=jNQXAC9IVRw");
    await expect(add).toBeDisabled(); // link but no consent

    await page.locator("#video-consent").check();
    await expect(add).toBeEnabled();

    await page.getByLabel("YouTube video URL").fill("   ");
    await expect(add).toBeDisabled(); // consent but only whitespace
  });

  test("FT-08 the Ask button stays disabled for an empty question", async ({ page }) => {
    await page.goto(`/v/${VIDEO_ID}`);
    await expect(page.getByPlaceholder("Ask about this video")).toBeVisible();
    await expect(page.getByRole("button", { name: "Ask" })).toBeDisabled();
    await page.getByPlaceholder("Ask about this video").fill("   ");
    await expect(page.getByRole("button", { name: "Ask" })).toBeDisabled();
  });
});

test.describe("Server errors are shown, not hidden", () => {
  test("FT-09 the API's rejection of a link is shown to the user", async ({ page }) => {
    await page.route(`${API}/v1/library/videos`, (route) => {
      if (route.request().method() === "POST") {
        return route.fulfill({
          status: 422,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Only YouTube links are supported." }),
        });
      }
      return route.fulfill({ status: 200, contentType: "application/json", body: '{"videos": []}' });
    });
    await page.goto("/");
    await expect(page.getByText("Nothing here yet")).toBeVisible();
    await page.getByLabel("YouTube video URL").fill("https://example.com/not-a-video");
    await page.locator("#video-consent").check();
    await page.getByRole("button", { name: "Add" }).click();
    await expect(formAlert(page)).toContainText("Only YouTube links are supported.");
  });

  test("FT-10 a failed ingestion is shown with a retry", async ({ page }) => {
    const failed = {
      id: "00000000-0000-0000-0000-000000000001",
      url: "https://www.youtube.com/watch?v=xxxxxxxxxxx",
      youtube_id: "xxxxxxxxxxx",
      title: "A video that failed",
      duration_seconds: 60,
      thumbnail_url: null,
      status: "failed",
      status_detail: null,
      progress: 0.4,
      frame_count: 0,
      audio_window_count: 0,
      error: "download failed: video unavailable",
      created_at: null,
    };
    await page.route(`${API}/v1/library/videos`, (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ videos: [failed] }) }),
    );
    await page.goto("/");
    await expect(page.getByText(/video unavailable/)).toBeVisible();
    await expect(page.getByRole("button", { name: `Retry ${failed.title}` })).toBeVisible();
  });
});
