# Measured efficiency: wall clock and peak GPU memory

**Status: not yet run.** The runner is written and tested; it needs an NVIDIA GPU,
which the development MacBook does not have. Everything below is copy and paste
onto a RunPod pod.

## Why this run exists

Every efficiency number the project currently quotes is analytic. `gist.eval.efficiency`
derives encoder GFLOPs from the encoder configs, which is defensible arithmetic and
clearly labelled as such, but it is not a measurement, and the citation audit already
had to walk back the word "measured" once. Two things follow:

1. Peak memory cannot be derived from a FLOP count at all. Half the pre-encoder
   argument has therefore never been evidenced.
2. A reviewer reading an efficiency paper whose efficiency is computed rather than
   observed will stop there.

This run closes both. It is the cheapest high-value experiment left.

## Cost

The OmniZip head to head took well under an hour on an RTX 4090. This run answers
each question up to three times instead of twice and adds no new checkpoint, so
budget about an hour, plus the Qwen2.5-Omni-7B pull (~16 GB).

At Community Cloud RTX 4090 pricing (~$0.34/hr) that is well under **$2**. Allow
$5 for a re-run.

## 1. Pod

- GPU: **RTX 4090 24 GB** — deliberately the same card as the existing runs, because
  the `dense` arm OOMing on 24 GB is one of the results.
- Template: PyTorch CUDA image
- Volume: **100 GB** — one 7B checkpoint plus six videos

## 2. Setup

```bash
cd /root
git clone https://github.com/thepreakerebi/gist.git
cd gist
pip install -e '.[vision,audio,sound,sota]'
pip install -U qwen-omni-utils av soundfile librosa

python -c "import torch; print('cuda', torch.cuda.is_available())"
python -c "from gist.eval.profiling import DeviceInfo; print(DeviceInfo.capture().describe())"
```

That second line should print the card, not `cpu (no CUDA device...)`. If it prints
the CPU message the run will still complete and every memory column will be empty,
which is the one failure mode worth catching before the model downloads.

## 3. Data

The same 18 questions and 6 videos as the OmniZip run:

```bash
scp .gist/benchmark/videomme_av6.json root@<pod>:/root/videomme_av6.json
# fetch sources on the pod rather than copying them up; the question file carries the URL
python - <<'PY'
import json, subprocess, pathlib
rows = json.loads(pathlib.Path("/root/videomme_av6.json").read_text())
out = pathlib.Path("/content/vids"); out.mkdir(parents=True, exist_ok=True)
fmt = "bv*[height<=360][vcodec^=avc1]+ba[acodec^=mp4a]/bv*[height<=360]+ba/b[height<=360]/best"
for vid, url in {r["videoID"]: r["url"] for r in rows}.items():
    dest = out / f"videomme-{vid}.mp4"
    if not dest.exists():
        subprocess.run(["yt-dlp", "-f", fmt, "--merge-output-format", "mp4", "-o", str(dest), url])
PY
```

## 4. Run

```bash
cd /root/gist
python results/efficiency-measured/efficiency_bench.py /root/videomme_av6.json 2>&1 | tee /tmp/eff.log
```

Cheap first pass without the OOM arm, to confirm the plumbing before spending the
full hour:

```bash
GIST_SKIP_DENSE=1 python results/efficiency-measured/efficiency_bench.py /root/videomme_av6.json
```

Conditions:

| Arm | Input | Purpose |
| :-- | :---- | :------ |
| `dense` | all 64 candidate frames + every 30 s window | what a system with no compression hands the encoders; expected to OOM on 24 GB |
| `full` | uniform 8 frames + 4 windows | the accuracy-matched baseline the existing results use |
| `gist` | Gist's selection | the method |

Stage `select` measures Gist's own scoring cost (CLIP + CLAP + Whisper). Report it.
If selection costs more than the encoding it saves, the claim collapses, and finding
that out here is far better than finding it out in review. Note that `select` is per
question while the candidate decode and transcription behind it are per video and
cached, so a library amortises it further than one row suggests.

## 5. Report

```bash
python -m gist.eval.efficiency \
  --measurements /tmp/efficiency_measurements.jsonl \
  --measured-baseline full \
  --measured-markdown results/efficiency-measured/measured.md \
  --measured-json results/efficiency-measured/measured.json \
  --markdown results/efficiency-measured/analytic.md \
  --json results/efficiency-measured/analytic.json
```

Then copy back `/tmp/efficiency_measurements.jsonl` and `/tmp/efficiency_answers.jsonl`
and commit all of it. `reports/` is gitignored; anything that has to survive a fresh
clone belongs under `results/`.

## What to write down afterwards, before the pod is destroyed

- The exact card and driver — `DeviceInfo` records this in the log header automatically.
- Whether `dense` OOMed, at how many frames and windows, and how much it had reserved.
- The `select` cost beside the `answer` saving, as one sentence, either way it lands.
- The resting footprint printed after the model loads, so activation deltas are readable
  on their own.

## Caveats to carry into the paper

- **n=18 on six videos.** This measures cost, not accuracy, so the sample size matters
  far less here than it does for the accuracy tables — cost per question has much lower
  variance than correctness. Say that explicitly rather than letting a reviewer assume
  the n=18 caveat transfers unchanged.
- **One card, one batch size, one precision.** fp16, batch 1, RTX 4090. Latency ratios
  do not transfer across hardware, so quote the ratio and name the card.
- **`dense` is not a published baseline.** It is the no-compression arm of this harness,
  not any competitor's configuration. Comparisons to OmniZip live in
  `results/runpod-omnizip-h2h/`.
- **Conditions run in a fixed order** (dense, full, gist) rather than shuffled, so
  condition and position are confounded in principle. The allocator cache is cleared
  between arms so fragmentation does not carry, but if the timings come out close
  enough that ordering could explain them, re-run with the order reversed before
  quoting a ratio.
- **Accuracy is a by-product here, not a result.** This run reports per-condition
  accuracy only as a sanity check that the arms are wired correctly. The accuracy
  numbers of record stay in `results/runpod-fp16/`; do not quote these instead.
