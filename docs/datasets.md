# Evaluation data

The single source of truth for what Gist is evaluated on. The paper, the proposal
and the defense brief should quote this file rather than restate it, because the
three have drifted from each other before.

Two tiers, with different jobs. Neither replaces the other.

---

## Tier 1 — Video-MME, for comparability

A **curated audio-visual subset of Video-MME** (`lmms-lab/Video-MME`), built by
`scripts/prepare_videomme_subset.py`, which downloads the test parquet from
HuggingFace, filters by duration, and fetches videos at 360p.

Questions are filtered to those requiring audio **and** video. A method that
arbitrates a budget between two modalities cannot be tested on visual-only
questions, so the full benchmark is not the right instrument.

| Pool | Questions | Videos | Used by |
| :--- | --------: | -----: | :------ |
| `videomme_av6.json` | 18 | 6 | every existing result; fp16 run; efficiency run |
| `videomme_av_all.json` | 51 | 18 | the n=51 4-bit run behind Table II |
| `videomme_long.json` | 60 | 21 | long-split runs |

Pools live in `.gist/benchmark/` (gitignored). Raw per-question output for each
completed run is committed under `results/`.

### What Video-MME is, and its ceiling

900 videos across three duration bands: short (11 s–2 min, mean 80.8 s), medium
(4–15 min, mean 520.2 s), long (30–60 min, **mean 2471 s ≈ 41 min**).

**Video-MME's long split tops out at 60 minutes and averages 41.** It therefore
cannot supply the hour-plus conditions this project is about. That is the entire
reason Tier 2 exists.

### How to describe these numbers

Say "a curated audio-visual subset of Video-MME, n = 51". Never "the Video-MME
dataset" unqualified — a reader who knows the benchmark will read 51% as a
Video-MME leaderboard score and ask why it is so far from published figures. These
are not leaderboard scores and are not comparable to them.

---

## Tier 2 — curated corpus, for the hour-plus claim

Self-assembled recordings longer than one hour, covering conditions no published
audio-visual benchmark reaches. Manifest: `data/eval/long-video-sources.json`.

### Current corpus — 7 genuine recordings

| Recording | Length | Licence |
| :-------- | -----: | :------ |
| Night of the Living Dead (1968) | 95.8 min | Public domain (US) |
| Bio-Inspired Motor Control, Lecture 02 | 86.4 min | CC BY-NC-ND 3.0 |
| Bio-Inspired Motor Control, Lecture 01 | 75.0 min | CC BY-NC-ND 3.0 |
| Microsoft Kinect Keynote, Art && Code 2011 | 69.6 min | CC BY-SA 4.0 |
| Paul Graham / Y Combinator discussion | 67.4 min | Standard YouTube License |
| The Quiet One (1948) documentary | 63.9 min | **unverified** |
| NASA STS-115 post-flight briefing (2006) | 60.1 min | Public domain (US Gov) |

### Removed: the Tears of Steel loop

`tears_of_steel_61min.webm` was never a 61-minute recording. It was the
12.2-minute Blender open movie concatenated exactly five times: 5 x 734.07 s =
3670.38 s, with byte-identical frames at t=734.07 and t=1468.15. Verified and
removed on 2026-09-22, along with its run artifacts and every evaluation case that
ran against it.

It was removed rather than relabelled because a looped video is the most
favourable possible input for a selector that removes redundancy -- four fifths of
the content is literally duplicated -- and because it is the exact case Pillar 3's
temporal gate exists to handle. Any figure computed on it measures the loop, not
long video.

Two consequences:

- **The "61.17 minutes, 99.98% token reduction" smoke test is withdrawn.** It is
  not the project's longest end-to-end run and must not be quoted as one. The
  longest genuine run is now whatever the 7-recording corpus produces.
- **Nine evaluation cases were dropped** with it: 3 of 9 in
  `gist-acceptance.jsonl`, 2 of 7 in `local-regression.jsonl`, and 4 of 39 in
  `long-video-quality.jsonl`, which is the set the heuristics ablation ran on. See
  "Results that need recomputing" below.

The genuine 12.2-minute original (`tears_of_steel_720p.webm`, CC BY 3.0, Blender
Foundation) is kept. It is a real short video and the cases that use it stand.

### Results that need recomputing

`long-video-quality.jsonl` went from 39 to 35 cases. Two committed results were
computed on the 39-case version and each drew 4 of their cases from the loop:

| Result | Status |
| :----- | :----- |
| `results/heuristics-ablation/` | computed at n=39; needs re-running at n=35 |
| `results/split-budget-ablation/` | same |

The committed JSON and Markdown under `results/` are left exactly as they are.
They are the experimental record of what was run on the day, and rewriting them
would be falsifying that record. They must be **re-run** and the new numbers
published alongside, with a note that the earlier version included four cases from
a duplicated video. Anything in the paper that cites either ablation is provisional
until that happens.

One silver lining: dropping four robotics-heavy cases slightly reduces the corpus
skew the next section describes.

### Three problems to fix before this tier is defensible

1. **Question skew.** 20 of the 39 current cases come from two robotics lectures.
   This is the most attackable feature of the evaluation and more video does not
   fix it — the *question* distribution is what is lumpy.
2. **Licences are now recorded, and two need care.** The Paul Graham recording is
   under YouTube's default Standard YouTube License, all rights reserved: analysis
   for research is ordinary use, but the file is never redistributed and only the
   manifest is published. The Quiet One (1948) remains unverified and must not
   appear in a published table until its rights statement is checked.
3. **Content monoculture.** The corpus is lecture and talking-head heavy, which
   flatters a method that leans on speech.

### Target shape — 12 recordings

- **3–4 questions per recording, hard cap.** Twelve recordings gives 36–48 cases
  with no single recording above roughly 8%.
- **Stratified across the six query-intent categories**, since RQ4 is defined by
  them and the per-intent heuristics ablation depends on them.
- **CC-BY, CC-BY-SA or public domain only.** Internet Archive conference talks,
  Blender open movies, NASA footage, openly licensed university lecture series.
  The Kinect keynote is the model to copy.
- **Diversify away from lectures.**
- The frozen 12-case held-out split stays grouped by recording, so no recording
  appears on both sides. It is run exactly once, at the end.

### Drafting questions, and what it actually yields

`scripts/draft_corpus_questions.py` drafts candidates from transcript plus frames
and then screens each one three ways — transcript only, frames only, both — keeping
only those where both modalities are needed and the ground truth holds up. The
screen exists because a model drafting from a transcript writes
transcript-answerable questions even when told not to.

**Measured yield, 2026-09-22, gpt-4.1-mini with 16 draft frames:**

| Recording | Drafted | Kept | Dominant rejection |
| :-------- | ------: | ---: | :----------------- |
| NASA STS-115 briefing | 3 | 0 | speech alone answers it |
| Night of the Living Dead | 8 | 1 | ground truth wrong (5 of 8) |

About one in ten survives, so budget 30–40 candidates per recording to land 3–4
keepers. That is cheap in API terms — four calls per candidate — but the failure
mode matters more than the rate.

**Five of eight film rejections were unreliable ground truth**, not modality
failures. Sixteen frames sampled across 96 minutes is too thin a view for a model
to write checkable questions about a specific moment; it confabulates details.
Two fixes worth trying before a full run: draft per segment rather than per
recording, with dense frames over a five to ten minute window, and use a stronger
drafting model. The screen catches these either way, which is the point of having
it, but a higher yield means less human review per keeper.

**The screen does not replace human verification.** It removes questions that are
clearly broken. Every survivor still needs a person to confirm the answer and the
timestamp before it enters the dataset.

### Where to source the seven replacements

Automated search of the Internet Archive was attempted on 2026-09-22 and abandoned:
the scrape API returned identical result counts for materially different queries, so
nothing it produced can be trusted as a survey. Source candidates by hand instead,
and verify duration and licence on the item page before adding any of them.

Source families worth checking, roughly in order of fit:

1. **Public-domain feature films** (`collection:feature_films` on the Internet
   Archive). The best fit nobody has used yet: 70–100 minutes, and genuinely
   audio-visual — dialogue, score, sound effects, scene changes, and questions whose
   answers depend on both channels. Verify each title's rights statement
   individually; the collection mixes rights.
2. **NASA mission and ISS coverage** — US Government work, public domain, routinely
   hours long, with continuous radio comms over changing visuals.
3. **Blender Foundation open movies** (CC BY) — short individually, but the licence
   is unambiguous and one is already in the corpus.
4. **Openly licensed university lecture series** — only if the lecture share of the
   corpus is already falling. Two are in there now and they are the skew.

Selection rule: prefer material where the answer to a plausible question requires
both channels. A recording where the speech transcript alone answers everything
tests the Whisper path, not Gist.

Authoring one hour-video question with verified ground truth and timestamps takes
15–20 minutes. Forty is 10–13 hours of work, and that — not GPU time — is what
gates this tier.

---

## How sampling responds to duration

From `src/gist/media/longform.py`. Mode is resolved from duration unless forced.

| Mode | Duration | Frames | Audio window | Context windows |
| :--- | :------- | -----: | -----------: | --------------: |
| SHORT | ≤ 10 min | 128 | 2 s | 1 |
| MEDIUM | ≤ 60 min | 256 | 10 s | 1 |
| LONG | > 60 min | 512 | 30 s | 0 |

In LONG mode the audio window widens to at least `ceil(duration / 240)` so window
count stays capped at 240.

### Measured across both tiers

Computed from `plan_ingestion` on 2026-09-22, not estimated:

| Case | Minutes | Mode | Frames | Window | Windows | Candidates |
| :--- | ------: | :--- | -----: | -----: | ------: | ---------: |
| Video-MME long, mean | 41.2 | medium | 256 | 10 s | 248 | 504 |
| Video-MME long, ceiling | 60.0 | medium | 256 | 10 s | 360 | 616 |
| Paul Graham | 67.4 | long | 512 | 30 s | 135 | 647 |
| Kinect keynote | 69.5 | long | 512 | 30 s | 140 | 652 |
| Bio-Inspired L01 | 75.0 | long | 512 | 30 s | 150 | 662 |
| Bio-Inspired L02 | 86.4 | long | 512 | 30 s | 173 | 685 |

Two consequences, and the first is better news than expected:

1. **Total candidate count is near-flat across the tier boundary** — 504 to 685
   across 41 to 86 minutes — because LONG doubles the frame budget while widening
   the audio window. Visual sampling density is also near-constant, about one frame
   per 10 seconds on both sides. Token-reduction percentages are therefore
   comparable across tiers without adjustment.
2. **Audio resolution is not comparable.** The window widens from 10 s to 30 s at
   the boundary, so Tier 2 audio evidence is three times coarser in time. Any table
   mixing the tiers must say so, and any claim about timestamp precision on audio
   evidence has to be made per tier.

---

## Datasets considered and not used

**AVQA** (Yang et al., ACM MM 2022) — 57,015 videos, 57,335 QA pairs, 158 hours
total, so roughly **10 seconds per video**.

**MUSIC-AVQA** — 9,288 videos, 45,867 QA pairs, **60 seconds each**, ~150 hours.

**WorldSense** (Hong et al., 2025) — omni-modal benchmark; not run.

All three appear in the proposal's Table 6 as planned sources. None was used, for
one reason: **neither AVQA nor MUSIC-AVQA contains a video longer than about a
minute**, so neither can test compression on long video. On a 10-second clip Gist
would prune 128 candidates down to 3 — a number you can compute and cannot
interpret, because the premise being tested is that long videos produce more
candidates than the encoders should have to process.

If the audio pathway ever needs evidence of generalization beyond Video-MME,
**MUSIC-AVQA is the one worth running** — 60-second clips with genuine sound-event
reasoning — at roughly $3 of GPU time. It would need a new loader and a new
exact-match scoring path, since it is answer-vocabulary classification rather than
multiple choice. AVQA adds little that MUSIC-AVQA does not cover.

The proposal should say these were scoped and not reached, with the reason above.
Leaving them listed as data sources implies runs that do not exist.
