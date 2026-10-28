# One pod session, both outstanding runs, $10 budget

Two GPU runs are left, and both fit in one RTX 4090 session:

1. `results/efficiency-measured/`: measured wall clock and peak memory for
   `dense`, `full` and `gist`. It replaces the analytic FLOP figures.
2. `results/omnipack-replication/`: the uniform baseline against our
   reimplementation of OmniPack, with a full sweep of the four paper ratios. That
   is n=48, not 51: video `ZkhAbUGoD18` has been removed from YouTube (checked
   2026-10-28), which takes 3 questions with it. The n=18 set is complete.

Everything here was exercised on CPU before any money was spent. The OmniPack
runner ran end to end against a tiny random-weight Qwen2.5-Omni built from the real
config (`omnipack-replication/smoke_cpu.py`), and its `full` arm reproduced stock
`generate` token for token. The efficiency runner ran all three arms the same way.
Those tests caught four bugs that would otherwise have surfaced on the paid clock:

- PyAV 14+ breaks the torchvision video reader.
- librosa 1.0 cannot read the audio handle `qwen-omni-utils` passes it.
- Gist's CLAP call only works on transformers 5.x.
- `max_new_tokens=8` is silently ignored by the Omni `generate`, which then
  produces up to 1024 tokens per answer.

What a CPU test cannot prove: CUDA-specific failures, fp16 behaviour, and real
memory. The control check at the start of the OmniPack run covers the second.

## Budget

| Step | Pod time | At ~$0.34/hr (Community) | At ~$0.69/hr (Secure) |
| :--- | -------: | -----: | -----: |
| Setup: installs, 16 GB model, checks | ~15 min | $0.09 | $0.17 |
| Plumbing pass, one video | ~5 min | $0.03 | $0.06 |
| Efficiency, n=18, three arms | ~60 min | $0.34 | $0.69 |
| OmniPack, n=51, full + four ratios | ~45 min | $0.26 | $0.52 |
| Slack for one re-run | ~60 min | $0.34 | $0.69 |
| **Total** | **~3 h** | **~$1.10** | **~$2.20** |

Check the live price before choosing. Idle time costs exactly the same as compute.

## 1. Create the pod

- **GPU:** 1x RTX 4090, Community Cloud (Secure Cloud if Community has no stock).
- **Template:** any RunPod PyTorch image with CUDA 12.4 or newer.
- **Container disk:** 80 GB. **Volume disk: 0 GB.** With no volume, nothing keeps
  billing after the pod is terminated.

Open the web terminal (Connect → Start Web Terminal).

## 2. Start setup (on the pod)

```bash
mkdir -p /root/logs
git clone https://github.com/thepreakerebi/gist.git /root/gist
cd /root/gist
nohup bash results/pod_session.sh setup > /root/logs/setup.log 2>&1 &
tail -f /root/logs/setup.log      # Ctrl-C stops watching, not the job
```

Setup takes about 10–15 minutes. It installs both environments, checks CUDA,
checks that faster-whisper loads on the GPU, and pulls the 16 GB model. It finishes
by printing `SETUP_DONE`. Send the data (step 3) while it runs.

## 3. Send the questions and videos (Mac → pod)

The question files are gitignored, so they never arrive with the clone. The 18
videos were already fetched on the Mac, so the pod does not depend on YouTube
accepting a datacenter address.

There are two bundles, so the efficiency run never waits on the larger one:

| Bundle | Contents | Size | Needed by |
| :----- | :------- | ---: | :-------- |
| `pod_bundle_1.tgz` | both question files and the 6 videos of the n=18 set | 1.1 GB | plumbing pass, efficiency run |
| `pod_bundle_2.tgz` | the other 11 videos | 1.6 GB | OmniPack run (n=48) |

On the Mac, send the first one:

```bash
runpodctl send ~/Documents/ALU/gist/.gist/pod_bundle_1.tgz
```

It prints a one-time code. On the pod, in a second terminal tab:

```bash
cd /root && runpodctl receive <code>
tar xzf pod_bundle_1.tgz -C /        # unpacks to /root/data and /content/vids
```

Send `pod_bundle_2.tgz` the same way once the efficiency run has started. It
uploads while the GPU works.

If the upload is too slow, the fallback is `bash results/pod_session.sh fetch`.
That downloads the same videos from YouTube on the pod.

## 4. Plumbing pass, then the efficiency run

```bash
cd /root/gist
bash results/pod_session.sh plumb      # ~5 min, one video, both runners
```

The plumbing pass runs both runners down their CUDA paths on one video, writing
to separate files. It ends by printing the lines that matter. You want a
`control: full arm reproduces...` line and no `FATAL`, `ERR` or `OOM`. If that is
what you see:

```bash
nohup bash results/pod_session.sh efficiency > /root/logs/eff.out 2>&1 &
tail -f /root/logs/efficiency.log
```

**Decision rule, agreed before seeing numbers:** read the `select` stage first. It is
Gist's own CLIP + CLAP + Whisper cost. If it exceeds the encoding it saves, the
contribution needs rephrasing before anything else is run.

## 5. OmniPack run

```bash
nohup bash results/pod_session.sh omnipack > /root/logs/op.out 2>&1 &
tail -f /root/logs/omnipack.log
```

The first thing to look for is one line:

```
control: full arm reproduces stock generate exactly
```

or the fp16 variant, "tokens differ but the answer matches". If the run instead
stops with `FATAL: the full arm does not reproduce stock Qwen`, nothing it would
have produced means anything. Stop and debug before spending more.

If stage 2 misbehaves (errors, or answers collapsing to one letter), re-run stage 1
only. That is still publishable, captioned as such:

```bash
OMNIPACK_STAGE2=0 nohup bash results/pod_session.sh omnipack > /root/logs/op1.out 2>&1 &
```

## 6. Bring the results home, then terminate

```bash
bash results/pod_session.sh collect
runpodctl send /root/results.tgz
```

On the Mac, run `runpodctl receive <code>` inside the repo. Then **terminate** the
pod (Stop is not enough). Commit everything under `results/` from the Mac. Commits
go out as thepreakerebi, with no attribution trailer.

## Outcome, 2026-10-28

One RTX 4090 Secure Cloud pod at $0.89/hr, about 1 h 40 min, **$1.00 total**.

**Efficiency, n=18** (`efficiency-measured/measured.md`):

| Condition | Median answer s | Activation GB (median) |
| :-- | --: | --: |
| `full`, uniform 8 frames + 4 windows | 1.20 | 1.722 |
| `gist` | 0.88 (1.37x) | 1.216 (29.4% less) |

- `dense` (64 frames, 107–115 windows) OOMed on all 6 questions attempted on 24 GB.
  Those rows are in `efficiency-measured/dense-oom/`.
- That arm was dropped from the run of record. After a dense OOM, the following
  `full` and `gist` arms also OOMed, so their numbers were contaminated.
- One `gist` answer (869-2, 1 frame) OOMed in the run of record. That run kept
  CLIP, CLAP and Whisper resident beside the 7B model.

**Decision rule: `select` exceeds the saving.** Gist's selection costs a median
12.2 s per question (6.9–66 s). The first question on a video pays Whisper over the
whole track. It saves 0.32 s of answer time against the uniform baseline. Per query,
against a cheap uniform baseline, Gist is not a wall-clock saving. What the
measurement supports is memory, and fitting where dense does not. The selection
cost is largely per video and cached, so it amortises across queries on a library,
but this run does not measure by how much.

**OmniPack reimplementation, n=48** (`omnipack-replication/omnipack_h2h_results.jsonl`):

| Arm | Correct | Relative to `full` |
| :-- | --: | --: |
| `full` | 22/48 (46%) | 100% |
| ratio 0.25 | 22/48 (46%) | 100% |
| ratio 0.20 | 22/48 (46%) | 100% |
| ratio 0.15 | 24/48 (50%) | 109% |
| ratio 0.10 | 20/48 (42%) | 91% |

- The control arm reproduced stock generate token for token.
- Retention sits near the paper's 98%, within the noise of 48 questions. This is
  a reimplementation under seven assumptions, not OmniPack.

**Found only on the GPU, all fixed before the runs of record:**
- torchvision decoded whole videos into RAM, so the pod switched to decord.
- CLIP frame scoring did not truncate long questions.
- Single-call vision and audio encoding OOMed. Encoding is now chunk-aligned,
  which is exact (max diff 0.0 on CPU).

Logs are in `pod-logs-2026-10-28/`.
