import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Privacy and terms of use · Gist",
  description: "What Gist keeps, who else gets it, how to delete it, and what Gist may be used for.",
};

// Must match CONSENT_VERSION in src/gist/api/library.py: the API stores this
// value with every consent, so a stored agreement can be traced to this text.
const VERSION = "2026-10-25.2";

/**
 * The privacy notice and terms of use, in one page.
 *
 * Every statement here describes behaviour the code actually has, and the
 * comment beside each section names where. If the code changes, this page and
 * VERSION change with it.
 *
 * Written for readers with limited literacy or limited English: short
 * sentences, everyday words, and the three facts that matter most in a summary
 * before any detail. Keep it near a US grade 6 reading level (Flesch-Kincaid)
 * when editing; the previous version measured grade 8.
 */
export default function PrivacyPage() {
  return (
    <main className="mx-auto w-full max-w-2xl px-6 pb-24 pt-16 sm:pt-24">
      <Link href="/" className="text-sm font-semibold tracking-tight">
        Gist
      </Link>
      <h1 className="mt-8 text-3xl font-semibold tracking-tight">Privacy and terms of use</h1>
      <p className="mt-2 text-sm text-muted-foreground">Version {VERSION}</p>

      <section
        aria-labelledby="in-short"
        className="mt-8 rounded-lg border border-border bg-secondary/60 px-5 py-4 text-[15px] leading-relaxed"
      >
        <h2 id="in-short" className="mb-2 font-semibold">
          In short
        </h2>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>Gist keeps a copy of the video, the words spoken in it, and your questions.</li>
          <li>
            To answer you, Gist sends a few seconds of the video to another company, TwelveLabs.
            It never sends the whole video.
          </li>
          <li>You can delete a video, and everything Gist made from it, at any time.</li>
        </ul>
      </section>

      <div className="mt-10 space-y-10 text-[15px] leading-relaxed [&_h2]:mb-3 [&_h2]:text-lg [&_h2]:font-semibold [&_li]:mt-1.5 [&_ul]:list-disc [&_ul]:pl-5">
        <section>
          <h2>What Gist is</h2>
          <p>
            Gist is a student project by Elvis Preye Kerebi at African Leadership University. You
            give it a link to a video. Then you ask questions about it. Gist finds the few seconds
            that answer each question. It shows you those seconds next to the answer.
          </p>
        </section>

        {/* src/gist/db/schema.sql; source files under .gist/library/<video id>/ */}
        <section>
          <h2>What Gist keeps</h2>
          <ul>
            <li>The link, the name of the video, how long it is, and its picture.</li>
            <li>A copy of the video, so Gist can cut short clips from it.</li>
            <li>The words spoken in the video, and any words shown on screen.</li>
            <li>Number codes that describe each picture and sound. Gist uses them to search the video.</li>
            <li>Your questions, the answers, and the clips used.</li>
            <li>The time you ticked the box under the link, and which version of this page you agreed to.</li>
          </ul>
          <p className="mt-3">
            Gist has no accounts. It never asks for your name or email. It has no tracking cookies
            and no analytics. Its database is on Neon, in Frankfurt, Germany.
          </p>
        </section>

        {/* src/gist/gateway/twelvelabs.py: only the concatenated selected spans */}
        <section>
          <h2>Who else gets your data</h2>
          <ul>
            <li>
              <strong>YouTube.</strong> Gist downloads the video from YouTube.
            </li>
            <li>
              <strong>TwelveLabs</strong>, a company based in the United States, writes the
              answers. For each question, Gist sends it your question, a short written note about
              the chosen moments, and those moments joined into one short clip. It never gets the
              whole video. TwelveLabs has its own privacy policy.
            </li>
          </ul>
          <p className="mt-3">
            The tools that read the pictures and sound (CLIP, CLAP and Whisper) run on
            Gist&rsquo;s own computer. Gist does not sell your data or use it for ads.
          </p>
        </section>

        {/* DELETE /v1/library/videos/{id} and /v1/library/videos/{id}/conversation */}
        <section>
          <h2>Deleting your data</h2>
          <ul>
            <li>
              <strong>Remove</strong> (the bin icon next to a video) deletes the video, its pictures
              and clips, the words, the number codes and the chat.
            </li>
            <li>
              <strong>Clear chat history</strong> (the eraser icon in a chat) deletes your questions
              and answers. The video stays.
            </li>
          </ul>
          <p className="mt-3">
            Gist keeps your data until you delete it. It does not delete old data on its own yet.
            If you delete a video, Gist cannot take back clips it already sent to TwelveLabs.
          </p>
        </section>

        <section>
          <h2>What you agree to</h2>
          <ul>
            <li>
              Only add videos you are allowed to use. That means your own videos, videos anyone may
              use, or public videos for study or research.
            </li>
            <li>
              Do not add private videos of people who did not agree. Do not add videos of children,
              or medical, legal or other private material.
            </li>
            <li>
              Everyone using this copy of Gist shares one library. They can see your videos and
              chats.
            </li>
          </ul>
        </section>

        <section>
          <h2>Limits you should know about</h2>
          <ul>
            <li>
              Answers can be wrong. Check the clips Gist shows you. Do not use Gist for health,
              legal, money or safety decisions.
            </li>
            <li>
              Gist hears some accents and languages less well. It also does worse on old or blurry
              video. It may miss the right moment there.
            </li>
          </ul>
        </section>

        <section>
          <h2>Contact</h2>
          <p>
            Questions, or want your data deleted? Open an issue at{" "}
            <a
              href="https://github.com/thepreakerebi/gist/issues"
              className="underline underline-offset-4"
            >
              github.com/thepreakerebi/gist/issues
            </a>
            . The code is open under the Apache License 2.0.
          </p>
        </section>
      </div>
    </main>
  );
}
