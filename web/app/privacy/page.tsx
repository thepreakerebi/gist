import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Privacy and terms of use · Gist",
  description: "What Gist stores, who else receives it, how to delete it, and what Gist may be used for.",
};

// Must match CONSENT_VERSION in src/gist/api/library.py: the API stores this
// value with every consent, so a stored agreement can be traced to this text.
const VERSION = "2026-10-25";

/**
 * The privacy notice and terms of use, in one page.
 *
 * Every statement here describes behaviour the code actually has, and the
 * comment beside each section names where. If the code changes, this page and
 * VERSION change with it.
 */
export default function PrivacyPage() {
  return (
    <main className="mx-auto w-full max-w-2xl px-6 pb-24 pt-16 sm:pt-24">
      <Link href="/" className="text-sm font-semibold tracking-tight">
        Gist
      </Link>
      <h1 className="mt-8 text-3xl font-semibold tracking-tight">Privacy and terms of use</h1>
      <p className="mt-2 text-sm text-muted-foreground">Version {VERSION}</p>

      <div className="mt-10 space-y-10 text-[15px] leading-relaxed [&_h2]:mb-3 [&_h2]:text-lg [&_h2]:font-semibold [&_li]:mt-1.5 [&_ul]:list-disc [&_ul]:pl-5">
        <section>
          <h2>What Gist is</h2>
          <p>
            Gist is a research prototype built by Elvis Preye Kerebi as a final-year
            capstone at African Leadership University. You give it a link to a public
            video and ask questions. It picks the few seconds of the video that answer
            each question and shows you those moments next to the answer.
          </p>
        </section>

        {/* src/gist/db/schema.sql; source files under .gist/library/<video id>/ */}
        <section>
          <h2>What Gist stores</h2>
          <ul>
            <li>The video link, its title, length and thumbnail link.</li>
            <li>A copy of the video file on the server, so clips can be cut and played back.</li>
            <li>
              What is said in the video (a transcript made by Whisper), text read from
              frames, and number fingerprints of frames and sound made by CLIP and CLAP.
            </li>
            <li>The questions you ask and the answers Gist gives, with the clips it used.</li>
            <li>The time you confirmed the box below the link field, and the version of this notice.</li>
          </ul>
          <p className="mt-3">
            Gist has no accounts. It does not ask for your name or email, sets no
            tracking cookies and runs no analytics. The database runs on Neon in the
            EU (Frankfurt).
          </p>
        </section>

        {/* src/gist/gateway/twelvelabs.py: only the concatenated selected spans */}
        <section>
          <h2>Who else receives your data</h2>
          <ul>
            <li>
              <strong>YouTube</strong> serves the video when Gist downloads it.
            </li>
            <li>
              <strong>TwelveLabs</strong> writes the answer. For each question, Gist
              sends it your question, a short text summary of the selected moments, and
              those moments joined into one short clip. It never receives the whole
              video. What TwelveLabs does with that data is covered by its own privacy
              policy.
            </li>
          </ul>
          <p className="mt-3">
            CLIP, CLAP and Whisper run on Gist&rsquo;s own server. Nothing is sold or
            shared for advertising.
          </p>
        </section>

        {/* DELETE /v1/library/videos/{id} and /v1/library/videos/{id}/conversation */}
        <section>
          <h2>Deleting your data</h2>
          <ul>
            <li>
              <strong>Remove</strong> (the bin icon on a video) deletes the video file,
              its frames and clips from the server, and its transcript, fingerprints and
              chat history from the database.
            </li>
            <li>
              <strong>Clear chat history</strong> inside a video deletes your questions
              and answers but keeps the processed video.
            </li>
          </ul>
          <p className="mt-3">
            Data stays until you remove it; there is no automatic expiry yet. Removing
            a video cannot recall clips TwelveLabs has already received.
          </p>
        </section>

        <section>
          <h2>What you agree to</h2>
          <ul>
            <li>
              Add only videos you have the right to process: your own, public domain,
              openly licensed, or public material used for study or research.
            </li>
            <li>
              Do not add private recordings of people who have not agreed to it,
              recordings of children, or medical, legal or other sensitive material.
            </li>
            <li>
              This is a shared demonstration library. Anyone using this copy of Gist can
              see the videos and chats in it.
            </li>
          </ul>
        </section>

        <section>
          <h2>Limits you should know about</h2>
          <ul>
            <li>
              Answers can be wrong. Gist shows the moments it used so you can check
              them; do not rely on it for medical, legal, financial or safety decisions.
            </li>
            <li>
              The speech and image models are less accurate on some accents, languages
              and older or low-quality footage, so Gist may miss the right moment there.
            </li>
          </ul>
        </section>

        <section>
          <h2>Contact</h2>
          <p>
            Questions or deletion requests:{" "}
            <a
              href="https://github.com/thepreakerebi/gist/issues"
              className="underline underline-offset-4"
            >
              github.com/thepreakerebi/gist/issues
            </a>
            . The source code is open under the Apache License 2.0.
          </p>
        </section>
      </div>
    </main>
  );
}
