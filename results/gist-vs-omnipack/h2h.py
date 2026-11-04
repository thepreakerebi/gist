"""Gist against our OmniPack rebuild, on the same input, model, prompt and GPU.

Earlier runs could not be compared directly. OmniPack's rebuild fed Qwen 32
frames and the audio track through `omnipack_h2h.py`; Gist's runs fed it JPEG
frames and WAV windows through stock `generate` with another prompt. This runner
puts both methods, and the uncompressed model, through one process and one
answer path, so the only thing that differs between arms is which encoder tokens
reach the language model and what it cost to decide.

The common ground
-----------------

Every arm starts from the same decoded input: the 32 frames and the audio track
that `qwen_omni_utils` gives Qwen for a video (`load_media`). Qwen encodes frames
in pairs (one temporal patch) and the harness encodes audio in fixed segments
aligned to the audio tower's attention chunks. Both are exact units: attention
never crosses them, so encoding a subset gives the same tokens as encoding the
whole and slicing. That makes the units a shared currency:

  full       every unit encoded, every token to the language model.
  omnipack   every unit encoded, then stage 1 and stage 2 discard tokens.
  gist       CLIP, CLAP and Whisper score the frames and segments *before* any
             Qwen encoder runs; only the units holding the chosen frames and
             windows are encoded, and only their tokens are kept.
  uniform    the same number of units as Gist chose for that question, evenly
             spaced. The control for whether Gist's choice beats chance.

Kept tokens keep their original rope positions in every arm (assumption A7 of
the rebuild), and every arm answers through the same hand-written prefill and
greedy decode that was checked token for token against stock `generate`.

What is measured, per stage, on the same GPU: wall time, peak memory and GPU
energy (NVML counter, covering CTranslate2's Whisper too). FLOPs are counted in
a separate pass (`H2H_COUNT_FLOPS=1`), because the counter slows the code it
counts. Stages are tagged `per_video` or `per_question`, which is what the
break-even analysis needs: a post-encoder method pays its encoders once per
video and reuses the tokens for every question; Gist pays its scorers once per
video and the Qwen encoders on a few units per question.

Usage on the pod (side env, transformers 4.52.3):

    python results/gist-vs-omnipack/h2h.py /root/data/videomme_av_all.json

    H2H_RATIOS=0.25,0.20,0.15,0.10   OmniPack retention ratios
    H2H_COUNT_FLOPS=1                FLOP-counting pass; its timings are not used
    H2H_MAX_VIDEOS=2                 stop after this many videos
    H2H_PRODUCT_WHISPER=1            also time the product's Whisper setting
"""

from __future__ import annotations

import json
import os
import sys
import wave
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "omnipack-replication"))
sys.path.insert(0, str(HERE.parent / "efficiency-measured"))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import omnipack_h2h as op  # noqa: E402
from omnipack import Stage1Result, stage1_compress  # noqa: E402

from gist.eval.profiling import DeviceInfo, MeasurementLog, idle_power_w  # noqa: E402

QFILE = sys.argv[1] if len(sys.argv) > 1 else "/root/data/videomme_av_all.json"
VIDDIR = os.environ.get("GIST_VIDEOS", "/content/vids")
OUT = Path(os.environ.get("H2H_OUT", "/root/h2h"))
WORK = Path(os.environ.get("H2H_WORK", "/root/h2h_work"))
RATIOS = [float(r) for r in os.environ.get("H2H_RATIOS", "0.25,0.20,0.15,0.10").split(",")]
COUNT_FLOPS = os.environ.get("H2H_COUNT_FLOPS") == "1"
MAX_VIDEOS = int(os.environ.get("H2H_MAX_VIDEOS", "0")) or None
PRODUCT_WHISPER = os.environ.get("H2H_PRODUCT_WHISPER", "1") == "1"
VERIFY_CONTROL = os.environ.get("H2H_VERIFY", "1") == "1"
SCORER_DEVICE = os.environ.get("H2H_SCORER_DEVICE", op.DEVICE)
# Long-input protocol (RUN.md, "Follow-up"). Qwen's feature extractor stops at 300 s
# of audio; a positive cap raises that so every arm can choose from the whole track.
AUDIO_CAP_SECONDS = int(os.environ.get("H2H_AUDIO_CAP_SECONDS", "0"))
# ASSUMPTION A8: OmniPack's stage 1 builds N x N matrices over a modality, which an
# hour of audio (~87k tokens) cannot fit on a 24 GB card. A positive value applies
# stage 1 to consecutive chunks of this many seconds of audio, at the same ratio.
STAGE1_CHUNK_SECONDS = int(os.environ.get("H2H_STAGE1_CHUNK_SECONDS", "0"))
# Arms whose prefill would exceed the language model's context are recorded, not run.
MAX_CONTEXT_OVERRIDE = int(os.environ.get("H2H_MAX_CONTEXT", "0"))

# Gist's selection exactly as in every earlier Gist result (efficiency_bench.py).
FRAME_BUDGET, AUDIO_BUDGET = 8, 4
# The Whisper setting behind Gist's accuracy results, and the product's.
EVAL_WHISPER = ("tiny", 1)
PRODUCT_WHISPER_SETTING = ("base", 3)
MEL_HOP = 160  # samples per Whisper mel frame at 16 kHz


def log_line(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Units: temporal patches and audio segments
# ---------------------------------------------------------------------------


def audio_units(thinker, feature_length: int) -> list[tuple[int, int, int]]:
    """(start_frame, end_frame, tokens) per segment, as `op.encode_audio` cuts them."""
    step = op.audio_segment_frames(thinker)
    units = []
    for start in range(0, feature_length, step):
        end = min(start + step, feature_length)
        _, out_len = thinker.audio_tower._get_feat_extract_output_lengths(
            torch.tensor([end - start])
        )
        units.append((start, end, int(out_len)))
    return units


def encode_units(thinker, inputs, patches: list[int], segments: list[int], audio_plan):
    """Run Qwen's encoders on the chosen units only. Exact: see the module docstring."""
    device = thinker.device
    grid = inputs["video_grid_thw"].to(device)
    pixels = inputs["pixel_values_videos"].to(device)
    features = inputs["input_features"].to(device, thinker.audio_tower.dtype)
    _, h, w = (int(v) for v in grid[0])
    per_grid = h * w
    visual, audio = [], []
    with torch.no_grad():
        for p in patches:
            part = pixels[p * per_grid : (p + 1) * per_grid]
            part_grid = torch.tensor([[1, h, w]], device=device, dtype=grid.dtype)
            visual.append(thinker.get_video_features(part, part_grid))
        for s in segments:
            start, end, _ = audio_plan[s]
            segment = features[0, :, start:end].unsqueeze(0)
            mask = torch.ones(1, end - start, dtype=torch.long, device=device)
            audio.append(thinker.get_audio_features(segment, feature_attention_mask=mask))
    hidden = thinker.get_input_embeddings().weight.shape[1]
    empty = torch.empty(0, hidden, device=device, dtype=op.DTYPE)
    return (torch.cat(visual) if visual else empty), (torch.cat(audio) if audio else empty)


def stage1_audio(audio, centrality, ratio: float, chunk_tokens: int):
    """OmniPack stage 1 on the audio tokens, whole or chunk by chunk (A8)."""
    if chunk_tokens <= 0 or audio.shape[0] <= chunk_tokens:
        return stage1_compress(audio.float(), centrality, modality="audio", retention=ratio)
    embeds, kept, before, after = [], [], 0, 0
    for start in range(0, audio.shape[0], chunk_tokens):
        out = stage1_compress(
            audio[start : start + chunk_tokens].float(),
            centrality[start : start + chunk_tokens],
            modality="audio",
            retention=ratio,
        )
        embeds.append(out.embeds)
        kept.append(out.kept_index + start)
        before += out.n_before
        after += out.n_after
    return Stage1Result(torch.cat(embeds), torch.cat(kept), before, after)


def keep_index(units: list[int], sizes: list[int]) -> torch.Tensor:
    """Token indices, in order, of the chosen units within a modality's token list."""
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    idx = [np.arange(offsets[u], offsets[u + 1]) for u in sorted(units)]
    flat = np.concatenate(idx) if idx else np.zeros(0, dtype=np.int64)
    return torch.as_tensor(flat, dtype=torch.long)


def evenly(n_units: int, k: int) -> list[int]:
    if k <= 0:
        return []
    if k >= n_units:
        return list(range(n_units))
    return sorted(
        {round(i * (n_units - 1) / (k - 1)) for i in range(k)} if k > 1 else {n_units // 2}
    )


# ---------------------------------------------------------------------------
# Gist: once per video, then per question
# ---------------------------------------------------------------------------


class GistSide:
    """Gist's scorers, loaded once, run on the media Qwen was given."""

    def __init__(self) -> None:
        from cached_selection import raw_clap_text

        from gist.audio.clap import HuggingFaceClapAudioScorer
        from gist.audio.dispatcher import _AMBIENT_PROBE, _SPEECH_PROBE, SentenceTransformerEmbedder
        from gist.audio.whisper import FasterWhisperTranscriber
        from gist.vision.clip import HuggingFaceClipFrameScorer

        cuda = SCORER_DEVICE == "cuda"
        self.clip = HuggingFaceClipFrameScorer(device=SCORER_DEVICE)
        self.clap = HuggingFaceClapAudioScorer(device=SCORER_DEVICE)
        compute = "float16" if cuda else "int8"
        self.whisper = FasterWhisperTranscriber(
            model_size=EVAL_WHISPER[0],
            device=SCORER_DEVICE,
            compute_type=compute,
            beam_size=EVAL_WHISPER[1],
        )
        self.product_whisper = (
            FasterWhisperTranscriber(
                model_size=PRODUCT_WHISPER_SETTING[0],
                device=SCORER_DEVICE,
                compute_type=compute,
                beam_size=PRODUCT_WHISPER_SETTING[1],
            )
            if PRODUCT_WHISPER
            else None
        )
        self.embedder = SentenceTransformerEmbedder()
        self.speech_threshold = 0.0  # SpeechSoundDispatcher's default
        self.probes = raw_clap_text(self.clap, [_SPEECH_PROBE, _AMBIENT_PROBE])
        # Load every model now, so no stage below pays a first-call load.
        self.clip._load()
        self.clip.embed_text("warm up")
        self.clap.embed_text("warm up")
        self.embedder.encode(["warm up"])
        from gist.media.models import AudioWindow

        WORK.mkdir(parents=True, exist_ok=True)
        silence = WORK / "warmup.wav"
        with wave.open(str(silence), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(16000)
            fh.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
        warm = [AudioWindow(index=0, start_seconds=0.0, duration_seconds=1.0, path=silence)]
        for transcriber in (self.whisper, self.product_whisper):
            if transcriber is not None:
                transcriber.transcribe_windows(warm)

    def ingest(self, log, vid, media, inputs, audio_plan, frame_times, count_flops):
        """Everything Gist does once per video. Each step is its own measured stage."""
        from PIL import Image

        from gist.media.models import AudioWindow, ExtractedFrame

        vdir = WORK / vid
        vdir.mkdir(parents=True, exist_ok=True)
        tag = dict(method="gist", scope="per_video", video=vid)

        with log.measure("gist_write_media", **tag):
            frames = []
            video = media["videos"][0]
            for i in range(video.shape[0]):
                arr = video[i].permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
                path = vdir / f"f{i:03d}.jpg"
                Image.fromarray(arr).save(path, quality=95)
                frames.append(ExtractedFrame(index=i, timestamp_seconds=frame_times[i], path=path))
            track = np.asarray(media["audios"][0], dtype=np.float32)
            windows = []
            for j, (start, end, _tokens) in enumerate(audio_plan):
                path = vdir / f"a{j:03d}.wav"
                clip = np.clip(track[start * MEL_HOP : end * MEL_HOP], -1.0, 1.0)
                with wave.open(str(path), "wb") as fh:
                    fh.setnchannels(1)
                    fh.setsampwidth(2)
                    fh.setframerate(16000)
                    fh.writeframes((clip * 32767).astype(np.int16).tobytes())
                windows.append(
                    AudioWindow(
                        index=j,
                        start_seconds=start / 100.0,
                        duration_seconds=(end - start) / 100.0,
                        path=path,
                    )
                )

        with log.measure("gist_clip_frames", count_flops=count_flops, **tag) as rec:
            F = np.array([e.vector for e in self.clip.embed_frames(frames)])
            rec.extra["items"] = len(frames)
        with log.measure("gist_clap_windows", count_flops=count_flops, **tag) as rec:
            embedded = self.clap.embed_windows(windows)
            A = np.array([embedded[w.path] for w in windows])
            rec.extra["items"] = len(windows)
        sp = A @ self.probes.T
        speech = [
            w for w, row in zip(windows, sp, strict=True) if row[0] - row[1] > self.speech_threshold
        ]
        with log.measure("gist_whisper", **tag) as rec:
            transcripts = self.whisper.transcribe_windows(speech)
            rec.extra.update(
                items=len(speech),
                seconds=sum(w.duration_seconds for w in speech),
                model=EVAL_WHISPER[0],
                beam=EVAL_WHISPER[1],
            )
        texts = [(transcripts.get(w.path) or "").strip() for w in speech]
        rec.extra["transcript_words"] = sum(len(t.split()) for t in texts)
        indexed = [(w, t) for w, t in zip(speech, texts, strict=True) if t]
        with log.measure("gist_minilm", count_flops=count_flops, **tag) as rec:
            T = np.array(self.embedder.encode([t for _w, t in indexed])) if indexed else None
            rec.extra["items"] = len(indexed)
        if self.product_whisper is not None and not count_flops:
            # Sensitivity only: the product transcribes every window with base,
            # beam 3. Its output is not used for selection.
            with log.measure(
                "gist_whisper_product", method="gist_product", scope="per_video", video=vid
            ) as rec:
                product = self.product_whisper.transcribe_windows(windows)
                rec.extra["transcript_words"] = sum(
                    len((t or "").split()) for t in product.values()
                )
                rec.extra.update(
                    items=len(windows),
                    seconds=sum(w.duration_seconds for w in windows),
                    model=PRODUCT_WHISPER_SETTING[0],
                    beam=PRODUCT_WHISPER_SETTING[1],
                )
        return {
            "frames": frames,
            "windows": windows,
            "F": F,
            "A": A,
            "indexed": indexed,
            "T": T,
            "speech": len(speech),
            "duration": float(frame_times[-1]) if frame_times else 0.0,
        }

    def select(self, index, query: str):
        """Cached scoring and Gist's compressor, as in cached_selection.py."""
        from gist.core.compressor import GistCompressor
        from gist.core.presets import CompressionPreset
        from gist.core.schemas import Candidate, CompressionRequest, Modality
        from gist.core.token_estimation import TokenEstimatorProfile

        frames, windows = index["frames"], index["windows"]
        qv = np.array(self.clip.embed_text(query))
        vscores = {f.path: float(v) for f, v in zip(frames, index["F"] @ qv, strict=True)}
        qa = np.array(self.clap.embed_text(query))
        ascores = {w.path: float(v) for w, v in zip(windows, index["A"] @ qa, strict=True)}
        if index["indexed"]:
            qt = np.array(self.embedder.encode([query.strip()]))[0]
            for (w, _t), v in zip(index["indexed"], index["T"] @ qt, strict=True):
                ascores[w.path] = float(v)
        vc = [
            Candidate(
                id=f"v{f.index}",
                timestamp_seconds=f.timestamp_seconds,
                saliency_score=vscores[f.path],
                asset_path=f.path,
            )
            for f in frames
        ]
        ac = [
            Candidate(
                id=f"a{w.index}",
                timestamp_seconds=w.start_seconds + w.duration_seconds / 2,
                saliency_score=ascores[w.path],
                asset_path=w.path,
            )
            for w in windows
        ]
        resp = GistCompressor().compress(
            CompressionRequest(
                video_id="v",
                query=query,
                duration_seconds=max(index["duration"], 1.0),
                preset=CompressionPreset.BALANCED,
                adaptive_budget=True,
                decompose_query=True,
                token_estimator=TokenEstimatorProfile.GENERIC,
                task_aware_selection=True,
                visual_candidates=vc,
                audio_candidates=ac,
            )
        )
        chosen_frames = [int(s.id[1:]) for s in resp.selected if s.modality == Modality.VISUAL][
            :FRAME_BUDGET
        ]
        chosen_windows = [int(s.id[1:]) for s in resp.selected if s.modality == Modality.AUDIO][
            :AUDIO_BUDGET
        ]
        return chosen_frames, chosen_windows


def frame_times_for(media, n_frames: int) -> list[float]:
    """Timestamps of the frames qwen-vl-utils sampled: evenly spaced over the video."""
    import subprocess

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            media["path"],
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    duration = float(probe) if probe else float(n_frames)
    if n_frames == 1:
        return [0.0]
    return [duration * i / (n_frames - 1) for i in range(n_frames)]


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def main() -> int:
    if op.DEVICE == "cuda" and not torch.cuda.is_available():
        op.die("no CUDA device; this runner is for the pod. Use smoke.py on a laptop.")
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    suffix = "_flops" if COUNT_FLOPS else ""
    meas_path, res_path = OUT / f"measurements{suffix}.jsonl", OUT / f"answers{suffix}.jsonl"
    res_path.write_text("")

    rows = json.loads(Path(QFILE).read_text())
    by_vid: dict[str, list] = {}
    for row in rows:
        by_vid.setdefault(row["videoID"], []).append(row)
    log_line(
        f"{len(rows)} questions across {len(by_vid)} videos; ratios={RATIOS}; flops={COUNT_FLOPS}"
    )

    device = DeviceInfo.capture()
    log_line(f"device: {device.describe()}")
    log = MeasurementLog(meas_path, device=device)
    idle = idle_power_w(10.0)
    log.note("idle_power", watts=idle)
    log_line(f"idle power: {idle} W")

    with log.measure("load_qwen", scope="setup"):
        model, processor = op.load_model()
    if AUDIO_CAP_SECONDS:
        fe = processor.feature_extractor
        fe.chunk_length = AUDIO_CAP_SECONDS
        fe.n_samples = AUDIO_CAP_SECONDS * fe.sampling_rate
        fe.nb_max_frames = fe.n_samples // fe.hop_length
        log.note("audio_cap", seconds=AUDIO_CAP_SECONDS)
    text_cfg = getattr(model.thinker.config, "text_config", model.thinker.config)
    max_context = MAX_CONTEXT_OVERRIDE or int(getattr(text_cfg, "max_position_embeddings", 32768))
    log.note("max_context", tokens=max_context, stage1_chunk_seconds=STAGE1_CHUNK_SECONDS)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        log.note("resident_qwen", gb=round(torch.cuda.memory_allocated() / 1e9, 3))
    layers = op.decoder_layers(model)
    if len(layers) <= op.INNER_LAYER:
        op.die(f"insertion layer {op.INNER_LAYER} beyond {len(layers)} decoder layers")
    with log.measure("load_gist_scorers", scope="setup"):
        gist = GistSide()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        log.note("resident", gb=round(torch.cuda.memory_allocated() / 1e9, 3))
    thinker = model.thinker

    arms = ["full"] + [f"omnipack_r{r:g}" for r in RATIOS] + ["gist", "uniform"]
    verified = not VERIFY_CONTROL
    exactness_checked = False
    warmed = False
    done_videos = 0

    for vid, questions in by_vid.items():
        if MAX_VIDEOS and done_videos >= MAX_VIDEOS:
            break
        source = f"{VIDDIR}/videomme-{vid}.mp4"
        if not Path(source).exists():
            log_line(f"SKIP {vid} (no video)")
            continue
        done_videos += 1
        shared = dict(scope="per_video", video=vid)
        try:
            with log.measure("load_media", method="shared", **shared):
                media = op.load_media(source)
                first = op.build_inputs(processor, media, questions[0]["question"])
            n_patches = int(first["video_grid_thw"][0, 0])
            feature_length = int(first["feature_attention_mask"][0].sum())
            audio_plan = audio_units(thinker, feature_length)
            merge = thinker.visual.spatial_merge_size
            per_patch = (int(first["video_grid_thw"][0, 1]) // merge) * (
                int(first["video_grid_thw"][0, 2]) // merge
            )
            visual_sizes = [per_patch] * n_patches
            audio_sizes = [u[2] for u in audio_plan]
            frame_times = frame_times_for(media, int(media["videos"][0].shape[0]))
            log.note(
                "video",
                video=vid,
                frames=len(frame_times),
                patches=n_patches,
                patch_tokens=per_patch,
                segments=len(audio_plan),
                audio_tokens=sum(audio_sizes),
                audio_seconds=feature_length / 100.0,
                video_seconds=frame_times[-1],
            )

            if not warmed:
                # Unmeasured: first kernel launches and allocator growth would
                # otherwise land on whichever stage happens to run first.
                encode_units(thinker, first, [0], [0], audio_plan)
            with log.measure(
                "encode", method="post_encoder", count_flops=COUNT_FLOPS, **shared
            ) as rec:
                visual, audio, info = op.encode(model, first)
                rec.extra.update(
                    visual_tokens=int(visual.shape[0]),
                    audio_tokens=int(audio.shape[0]),
                    patches=n_patches,
                    mel_frames=feature_length,
                )
            if rec.oom:
                log_line(f"{vid}: OOM in encode; video skipped")
                continue
            if visual.shape[0] != sum(visual_sizes) or audio.shape[0] != sum(audio_sizes):
                op.die(
                    f"{vid}: unit plan {sum(visual_sizes)}/{sum(audio_sizes)} != encoded "
                    f"{visual.shape[0]}/{audio.shape[0]}; the token accounting is wrong"
                )

            index = gist.ingest(log, vid, media, first, audio_plan, frame_times, COUNT_FLOPS)
        except Exception as exc:  # noqa: BLE001 - logged, then the video is skipped
            log_line(f"{vid} prepare: ERR {type(exc).__name__}: {exc}"[:400])
            continue

        tokens_per_second = sum(audio_sizes) / max(feature_length / 100.0, 1e-9)
        chunk_tokens = int(round(STAGE1_CHUNK_SECONDS * tokens_per_second))
        compressed = {}
        for ratio in RATIOS:
            with log.measure(
                "omnipack_stage1", method=f"omnipack_r{ratio:g}", count_flops=COUNT_FLOPS, **shared
            ) as rec:
                v_out = stage1_compress(
                    visual.float(),
                    info["visual_centrality"],
                    modality="visual",
                    retention=ratio,
                    frames=info["frames"],
                    patches=info["patches"],
                    grid_w=info["grid_w"],
                )
                a_out = stage1_audio(audio, info["audio_centrality"], ratio, chunk_tokens)
                rec.extra.update(visual_after=v_out.n_after, audio_after=a_out.n_after)
            compressed[ratio] = (v_out, a_out)

        for item in questions:
            qid = item["question_id"]
            gold = str(item["answer"]).strip()[:1].upper()
            prompt = (
                "Select the best answer to the following multiple-choice question "
                "based on the video. Respond with only the letter (A, B, C, or D) of "
                "the correct option. Question: "
                + item["question"]
                + "\n"
                + " ".join(item["options"])
                + "\nThe best answer is:"
            )
            inputs = op.build_inputs(processor, media, prompt)
            layout = op.locate(model, processor, inputs)
            qtag = dict(scope="per_question", qid=qid, video=vid)

            with log.measure("gist_select", method="gist", count_flops=COUNT_FLOPS, **qtag) as rec:
                chosen_frames, chosen_windows = gist.select(index, item["question"])
                patches = sorted({f // 2 for f in chosen_frames})
                segments = sorted(set(chosen_windows))
                rec.extra.update(
                    frames=chosen_frames, windows=chosen_windows, patches=patches, segments=segments
                )
            uni_patches = evenly(n_patches, len(patches))
            uni_segments = evenly(len(audio_plan), len(segments))

            unit_choice = {"gist": (patches, segments), "uniform": (uni_patches, uni_segments)}
            unit_tokens = {}
            for arm, (pp, ss) in unit_choice.items():
                with log.measure(
                    "encode_units", method=arm, count_flops=COUNT_FLOPS, **qtag
                ) as rec:
                    v_sel, a_sel = encode_units(thinker, inputs, pp, ss, audio_plan)
                    rec.extra.update(
                        patches=len(pp),
                        segments=len(ss),
                        mel_frames=sum(audio_plan[i][1] - audio_plan[i][0] for i in ss),
                        visual_tokens=int(v_sel.shape[0]),
                        audio_tokens=int(a_sel.shape[0]),
                    )
                unit_tokens[arm] = (
                    v_sel,
                    a_sel,
                    keep_index(pp, visual_sizes),
                    keep_index(ss, audio_sizes),
                )

            for arm, (pp, ss) in unit_choice.items():
                if exactness_checked or not (pp and ss):
                    continue
                v_sel, a_sel, vk, ak = unit_tokens[arm]
                vdiff = float((v_sel.float() - visual[vk.to(visual.device)].float()).abs().max())
                adiff = float((a_sel.float() - audio[ak.to(audio.device)].float()).abs().max())
                log.note(
                    "exactness",
                    video=vid,
                    qid=qid,
                    arm=arm,
                    patches=len(pp),
                    segments=len(ss),
                    visual_max_abs_diff=vdiff,
                    audio_max_abs_diff=adiff,
                )
                log_line(
                    f"exactness ({arm}): unit vs full encoding, max |diff| "
                    f"visual={vdiff:.3g} audio={adiff:.3g}"
                )
                exactness_checked = True

            if not warmed:
                v_sel, a_sel, vk, ak = unit_tokens["uniform"]
                op.generate(
                    model,
                    processor,
                    op.assemble(
                        model,
                        layout,
                        v_sel,
                        a_sel,
                        vk.to(layout.video.device),
                        ak.to(layout.audio.device),
                    ),
                    None,
                )
                warmed = True

            answers: dict[str, str] = {}
            for arm in arms:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    stats = None
                    if arm == "full":
                        seq = op.assemble(model, layout, visual, audio)
                    elif arm.startswith("omnipack"):
                        v_out, a_out = compressed[float(arm.split("_r")[1])]
                        seq = op.assemble(
                            model,
                            layout,
                            v_out.embeds,
                            a_out.embeds,
                            v_out.kept_index,
                            a_out.kept_index,
                        )
                        stats = op.Stage2Stats() if op.USE_STAGE2 else None
                    else:
                        v_sel, a_sel, vk, ak = unit_tokens[arm]
                        seq = op.assemble(
                            model,
                            layout,
                            v_sel,
                            a_sel,
                            vk.to(layout.video.device),
                            ak.to(layout.audio.device),
                        )
                    if seq.embeds.shape[1] > max_context:
                        log.note(
                            "over_context",
                            qid=qid,
                            arm=arm,
                            tokens=int(seq.embeds.shape[1]),
                            max_context=max_context,
                        )
                        log_line(
                            f"{qid} {arm}: {seq.embeds.shape[1]} tokens > {max_context}; not run"
                        )
                        del seq
                        continue
                    with log.measure("answer", method=arm, count_flops=COUNT_FLOPS, **qtag) as rec:
                        text, tokens = op.generate(model, processor, seq, stats)
                        answer = op.letter(text)
                        rec.extra.update(
                            answer=answer,
                            gold=gold,
                            correct=answer == gold,
                            prefill_tokens=int(seq.embeds.shape[1]),
                            media_tokens=int(seq.video.numel() + seq.audio.numel()),
                        )
                        if stats is not None:
                            rec.extra["stage2"] = vars(stats)
                    if rec.oom:
                        log_line(f"{qid} {arm}: OOM in answer")
                        continue
                    if arm == "full" and not verified:
                        op.check_control(model, processor, inputs, seq, tokens, answer)
                        verified = True
                except Exception as exc:  # noqa: BLE001 - logged, run continues
                    log_line(f"{qid} {arm}: ERR {type(exc).__name__}: {exc}"[:400])
                    continue
                answers[arm] = answer

            with res_path.open("a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "qid": qid,
                            "video": vid,
                            "gold": gold,
                            **answers,
                            "gist_units": [patches, segments],
                        }
                    )
                    + "\n"
                )
            log_line(
                f"{qid} gold={gold} "
                + " ".join(f"{a}={x}{'Y' if x == gold else 'n'}" for a, x in answers.items())
            )

        del visual, audio, info, compressed, index
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log_line(f"MEASUREMENTS: {meas_path}")
    log_line("H2H_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
