# Running the OmniSIFT head to head

**Status: not yet run.** The runner is written and ready; it needs an NVIDIA GPU,
which the development MacBook does not have. Everything below is copy and paste
onto a RunPod pod.

Cost estimate: the OmniZip head to head took well under an hour on an RTX 4090 at
$0.69/hr. This one downloads a second 7B checkpoint (OmniSIFT-7B, roughly 16 GB),
so budget about an hour including the pull.

## 1. Pod

Same shape as the OmniZip run, one size up on disk for the extra checkpoint:

- GPU: **RTX 4090 24 GB** (A100 40 GB if you want to try the paper's 256 frame config)
- Template: PyTorch CUDA image
- Volume: **150 GB** — two 7B checkpoints plus videos

## 2. Setup

```bash
cd /root
git clone https://github.com/dingyue772/OmniSIFT.git
cd OmniSIFT

# OmniSIFT defers to the official Qwen2.5-Omni environment.
pip install -U "transformers" accelerate qwen-omni-utils av soundfile librosa
pip install -e .

python -c "import omnisift, torch; print('omnisift ok, cuda', torch.cuda.is_available())"
```

If `pip install -e .` fails, the package can also be used by running from inside
the cloned directory, since `omnisift/` is importable as a local package.

## 3. Data

The same 18 questions and 6 videos the OmniZip run used:

Six long YouTube videos, three questions each. Fetch them on the pod rather than
copying, since the question file carries the source URL.

```bash
scp .gist/benchmark/videomme_av6.json root@<pod>:/root/videomme_av6.json
scp results/runpod-omnisift-h2h/fetch_videos.py root@<pod>:/root/
```

Then on the pod:

```bash
pip install -U yt-dlp
python /root/fetch_videos.py /root/videomme_av6.json /content/vids
ls -la /content/vids    # expect 6 files
```

If a video is unavailable the runner prints `SKIP <videoID>` and carries on, and
the denominator in the printed result drops with it. Note which were skipped: the
comparison against OmniZip and Gist only holds on questions all three answered.

## 4. Run

```bash
scp results/runpod-omnisift-h2h/omnisift_h2h.py root@<pod>:/root/

cd /root/OmniSIFT
python /root/omnisift_h2h.py /root/videomme_av6.json /content/vids 2>&1 | tee /root/omnisift.log
```

Knobs, all optional:

| Variable | Default | Why you would change it |
| :--- | :--- | :--- |
| `MAXFRAMES` | 32 | Matches the OmniZip run. Their paper uses 256, which will not fit on 24 GB. |
| `MAXPIX` | `128*28*28` | Matches the OmniZip run. Their paper uses `256*28*28`. |
| `RHO_VIDEO` | 0.7 | Fraction of video tokens removed. From their quick start, not confirmed as the paper's benchmark value. |
| `RHO_AUDIO` | 0.3 | Fraction of audio tokens removed. Same caveat. |

## 5. Collect

```bash
scp root@<pod>:/tmp/omnisift_h2h_results.jsonl results/runpod-omnisift-h2h/
scp root@<pod>:/root/omnisift.log            results/runpod-omnisift-h2h/
```

Then, locally:

```bash
uv run python -m gist.eval.bootstrap \
  --predictions results/runpod-omnisift-h2h/omnisift_h2h_results.jsonl \
  --baseline full --conditions full,omnisift \
  --title "OmniSIFT vs uncompressed, 18 Video-MME AV questions" \
  --markdown reports/bootstrap-omnisift.md
```

## What the result can and cannot say

**Can:** whether OmniSIFT beats Gist on accuracy at our frame budget, on our
questions, on one 24 GB card. That is the comparison a reviewer wants.

**Cannot:** reproduce the paper's reported numbers. Their evaluation runs eight
times our frame count at four times the pixels. If OmniSIFT scores below its
published figures here, that is our budget, not a failure of their method, and the
write up has to say so.

**Watch for:** whether it fits at all. OmniZip needed inputs trimmed to 90 seconds
on this card. If OmniSIFT also needs trimming, that is itself a result worth
reporting, and it goes in the cost column beside the accuracy.

## Why the uncompressed arm is not the earlier full context number

OmniSIFT-7B is a fine tuned checkpoint, so "compression off" here is *that* model
without compression, not stock Qwen2.5-Omni-7B. It is the correct internal control
for OmniSIFT and it is not interchangeable with the 6/18 from the OmniZip run.
Report both, and label them.
