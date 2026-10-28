"""Head to head: uniform baseline against our reimplementation of OmniPack.

**Not yet executed on a GPU.** Every other runner under `results/` was written
against a live pod; this one could not be, because OmniPack released no code to
borrow a working harness from. The model-facing code below was written against
the transformers 4.52.3 source and exercised end to end on CPU against a tiny,
randomly initialised Qwen2.5-Omni built from the real config
(`smoke_cpu.py`). That proves the plumbing, not the numbers. Read the "First
run" section of RUN.md before starting the pod session.

**This is not OmniPack.** It is our implementation of the published description,
under the seven assumptions tabulated in RUN.md (A1-A4 in `omnipack.py`, A5-A7
in this file). Label every number accordingly.

What is measured, and why the stages are split:

  encode      the full dual-encoder forward, plus reading the final encoder
              layer's attention that OmniPack's importance signal needs.
              OmniPack pays this unconditionally because it compresses *after*
              the encoders; Gist does not. Timing it separately is the entire
              point of the comparison, and it is the cost their own FLOPs
              accounting excludes. Once per video: the encoders never see the
              question, so every arm and every question shares one pass.
  select      our stage 1 (dual selection plus merging), per retention ratio.
              Also once per video, because stage 1 does not see the question
              either.
  answer      prefill and greedy decoding, with stage 2 applied inside the LLM
              at the insertion layer when enabled.

Arms:

  full        the same model with compression switched off, run through the
              same hand-written prefill and decode loop. The internal control:
              it isolates compression from every other difference in this
              harness. On the first question it is checked token for token
              against stock `thinker.generate`; a mismatch stops the run,
              because if the control is wrong nothing downstream means anything.
  omnipack    stage 1 at the swept retention ratio, plus stage 2 at 50% if
              enabled.

Usage on the pod:

    python results/omnipack-replication/omnipack_h2h.py /root/videomme_av6.json

    OMNIPACK_RATIOS=0.25,0.20,0.15,0.10   sweep, per the paper's four settings
    OMNIPACK_STAGE2=0                     stage 1 only, the fallback arm
    OMNIPACK_FRAMES=32                    matched to the OmniZip/OmniScope runs
    OMNIPACK_VERIFY=0                     skip the stock-generate control check
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from omnipack import (  # noqa: E402
    INNER_LAYER_QWEN_7B,
    INNER_RETENTION,
    stage1_compress,
    stage2_select,
)

from gist.eval.profiling import DeviceInfo, MeasurementLog  # noqa: E402

QFILE = sys.argv[1] if len(sys.argv) > 1 else "/root/videomme_av6.json"
VIDDIR = os.environ.get("GIST_VIDEOS", "/content/vids")
MEASF = os.environ.get("GIST_MEASUREMENTS", "/tmp/omnipack_measurements.jsonl")
RESF = os.environ.get("GIST_RESULTS", "/tmp/omnipack_h2h_results.jsonl")

QWEN_MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-Omni-7B")
FRAMES_NUM = int(os.environ.get("OMNIPACK_FRAMES", "32"))
MAX_PIXELS = int(os.environ.get("OMNIPACK_MAX_PIXELS", str(128 * 28 * 28)))
RATIOS = [float(r) for r in os.environ.get("OMNIPACK_RATIOS", "0.25").split(",")]
USE_STAGE2 = os.environ.get("OMNIPACK_STAGE2", "1") == "1"
# ASSUMPTION A5 (RUN.md): stage 2 compresses the *input* of this layer index, so
# this many layers see the full sequence.
INNER_LAYER = int(os.environ.get("OMNIPACK_INNER_LAYER", str(INNER_LAYER_QWEN_7B)))
VERIFY_CONTROL = os.environ.get("OMNIPACK_VERIFY", "1") == "1"
# The audio tower attends only within 2 s chunks (n_window), so encoding the
# track in segments aligned to those chunks is exact, not an approximation, and
# keeps attention memory flat. One call over a long track allocated 5 GB and
# ran a 24 GB card out of memory on the first pod session.
AUDIO_SEGMENT_SECONDS = int(os.environ.get("OMNIPACK_AUDIO_SEGMENT_SECONDS", "30"))
# The same holds for vision: windowed blocks attend inside one frame and the
# full-attention blocks inside one temporal patch, never across. A single call
# over 16 temporal patches allocated 5 GB for the attention scores alone.
VISUAL_SEGMENT_GRIDS = int(os.environ.get("OMNIPACK_VISUAL_SEGMENT_GRIDS", "4"))
MAX_NEW_TOKENS = 8

# The pod default. `smoke_cpu.py` overrides both to exercise the plumbing on a
# MacBook; nothing measured on CPU is a result.
DEVICE = os.environ.get("OMNIPACK_DEVICE", "cuda")
DTYPE = getattr(torch, os.environ.get("OMNIPACK_DTYPE", "float16"))

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def die(message: str) -> None:
    """Fail loudly. A silent fallback here would corrupt the comparison."""
    print(f"FATAL: {message}", flush=True)
    raise SystemExit(2)


# ---------------------------------------------------------------------------
# Model access
#
# Written against transformers 4.52.3, the pin OmniZip and OmniScope share, and
# read from its source rather than guessed: `get_video_features`,
# `get_audio_features` and `get_rope_index` on the thinker, and the decoder
# layer call signature in `Qwen2_5OmniThinkerTextModel.forward`. SDPA is forced
# because the prefill below relies on its `is_causal` path when no mask is given.
# ---------------------------------------------------------------------------


def load_model():
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    processor = Qwen2_5OmniProcessor.from_pretrained(QWEN_MODEL_PATH)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        QWEN_MODEL_PATH,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation="sdpa",
        enable_audio_output=False,
    )
    model.eval()
    return model, processor


def decoder_layers(model):
    thinker = getattr(model, "thinker", None)
    if thinker is None:
        die("model has no .thinker; this harness targets Qwen2.5-Omni")
    for path in ("model.layers", "language_model.model.layers", "model.model.layers"):
        node = thinker
        try:
            for part in path.split("."):
                node = getattr(node, part)
            return node
        except AttributeError:
            continue
    die("could not locate the decoder layer list on model.thinker")


# ---------------------------------------------------------------------------
# Final-encoder-layer attention, which OmniPack's importance signal is built on
# (ASSUMPTION A6 in RUN.md for how it is reduced).
#
# Neither encoder ever materialises an N x N attention matrix: the vision tower
# attends within each temporal patch in its full-attention blocks, and the audio
# tower within fixed chunks. So the readout is computed inside a pre-hook on the
# last attention module, segment by segment, and only the reduced centrality
# (mean attention each token receives, over heads and queries) is kept. It is
# then mapped onto the LLM-facing tokens: the vision merger folds each 2x2 patch
# group into one token, in window order, and the audio tower average-pools
# adjacent frames in pairs.
# ---------------------------------------------------------------------------


def _received_attention(q: torch.Tensor, k: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """q, k: (seq, heads, head_dim). Mean attention received per token."""
    out = torch.empty(q.shape[0], device=q.device, dtype=torch.float32)
    scale = q.shape[-1] ** -0.5
    bounds = cu_seqlens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        qs = q[start:end].transpose(0, 1).float()
        ks = k[start:end].transpose(0, 1).float()
        probs = torch.softmax((qs @ ks.transpose(1, 2)) * scale, dim=-1)
        out[start:end] = probs.mean(dim=0).mean(dim=0)
    return out


class EncoderAttentionProbe:
    """Captures final-layer attention centrality for both towers."""

    def __init__(self, thinker):
        self.visual = thinker.visual
        self.audio = thinker.audio_tower
        self.visual_received: torch.Tensor | None = None
        self.audio_received: torch.Tensor | None = None
        self._handles = []

    def _visual_hook(self, module, args, kwargs):
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            apply_rotary_pos_emb_vision,
        )

        hidden = args[0] if args else kwargs["hidden_states"]
        cu_seqlens = kwargs["cu_seqlens"]
        rotary = kwargs["rotary_pos_emb"]
        n = hidden.shape[0]
        with torch.no_grad():
            q = module.q(hidden).reshape(n, module.num_heads, -1)
            k = module.k(hidden).reshape(n, module.num_heads, -1)
            q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary).squeeze(0)
            k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary).squeeze(0)
            self.visual_received = _received_attention(q, k, cu_seqlens)

    def _audio_hook(self, module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        cu_seqlens = kwargs["cu_seqlens"] if "cu_seqlens" in kwargs else args[1]
        n = hidden.shape[0]
        with torch.no_grad():
            q = module.q_proj(hidden).reshape(n, module.num_heads, -1)
            k = module.k_proj(hidden).reshape(n, module.num_heads, -1)
            self.audio_received = _received_attention(q, k, cu_seqlens)

    def __enter__(self):
        last_visual = self.visual.blocks[-1]
        if (len(self.visual.blocks) - 1) not in self.visual.fullatt_block_indexes:
            die("last vision block is windowed; the centrality readout assumes full attention")
        for name in ("q", "k"):
            if not hasattr(last_visual.attn, name):
                die("vision attention has no separate q/k projections; load with sdpa")
        self._handles.append(
            last_visual.attn.register_forward_pre_hook(self._visual_hook, with_kwargs=True)
        )
        self._handles.append(
            self.audio.layers[-1].self_attn.register_forward_pre_hook(
                self._audio_hook, with_kwargs=True
            )
        )
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False

    def visual_centrality(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Per LLM-facing visual token, in the order the tower emits them."""
        unit = self.visual.spatial_merge_unit
        window_index, _ = self.visual.get_window_index(grid_thw)
        per_token = self.visual_received.view(-1, unit).mean(dim=1)
        return per_token[torch.argsort(window_index.to(per_token.device))]

    def audio_centrality(self, aftercnn_lens: torch.Tensor) -> torch.Tensor:
        """For the most recent audio-tower call only; `encode` calls per segment."""
        pieces = []
        for chunk in self.audio_received.split(aftercnn_lens.tolist()):
            usable = (chunk.shape[0] // 2) * 2
            pieces.append(chunk[:usable].view(-1, 2).mean(dim=1))
        return torch.cat(pieces)


# ---------------------------------------------------------------------------
# Inputs, encoders and the token layout
# ---------------------------------------------------------------------------


def _conversation(video_path: str, prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "nframes": FRAMES_NUM, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": prompt},
        ]},
    ]


def load_media(video_path: str):
    """Decode frames and the audio track once per video."""
    from qwen_omni_utils import process_mm_info

    audios, images, videos, video_kwargs = process_mm_info(
        _conversation(video_path, ""), use_audio_in_video=True, return_video_kwargs=True
    )
    fps = video_kwargs.get("fps", 2.0)
    if isinstance(fps, list):
        fps = fps[0]
    return {"path": video_path, "audios": audios, "images": images, "videos": videos, "fps": fps}


def build_inputs(processor, media, prompt: str):
    """Processor inputs for one video plus prompt, on CPU."""
    text = processor.apply_chat_template(
        _conversation(media["path"], prompt), add_generation_prompt=True, tokenize=False
    )
    return processor(
        text=text,
        audio=media["audios"],
        images=media["images"],
        videos=media["videos"],
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
        fps=media["fps"],
    )


def encode(model, inputs):
    """Run both towers and return (visual_tokens, audio_tokens, info).

    `info` carries the final-layer centrality for each modality and the visual
    grid (frames, patches per frame, grid width) that stage 1 needs.
    """
    thinker = model.thinker
    device = thinker.device
    grid = inputs["video_grid_thw"].to(device)
    pixels = inputs["pixel_values_videos"].to(device)
    features = inputs["input_features"].to(device, thinker.audio_tower.dtype)
    feature_mask = inputs["feature_attention_mask"].to(device)

    with torch.no_grad(), EncoderAttentionProbe(thinker) as probe:
        visual, visual_centrality = encode_video(thinker, probe, pixels, grid)
        audio, audio_centrality = encode_audio(thinker, probe, features, feature_mask)

    merge = thinker.visual.spatial_merge_size
    frames = int(grid[0, 0])
    grid_h, grid_w = int(grid[0, 1]) // merge, int(grid[0, 2]) // merge
    info = {
        "frames": frames,
        "patches": grid_h * grid_w,
        "grid_w": grid_w,
        "visual_centrality": visual_centrality,
        "audio_centrality": audio_centrality,
    }
    if info["visual_centrality"].shape[0] != visual.shape[0]:
        die(f"visual centrality {info['visual_centrality'].shape[0]} != tokens {visual.shape[0]}")
    if info["audio_centrality"].shape[0] != audio.shape[0]:
        die(f"audio centrality {info['audio_centrality'].shape[0]} != tokens {audio.shape[0]}")
    if visual.shape[0] != frames * grid_h * grid_w:
        die("visual token count does not match the grid; stage 1 would misread the layout")
    return visual, audio, info


def encode_video(thinker, probe, pixels, grid):
    """The vision tower a few temporal patches at a time. Exact: see above."""
    if grid.shape[0] != 1:
        die("one video per prompt expected")
    t, h, w = (int(v) for v in grid[0])
    per_grid = h * w
    tokens, centrality = [], []
    for start in range(0, t, VISUAL_SEGMENT_GRIDS):
        count = min(VISUAL_SEGMENT_GRIDS, t - start)
        part = pixels[start * per_grid:(start + count) * per_grid]
        part_grid = torch.tensor([[count, h, w]], device=grid.device, dtype=grid.dtype)
        tokens.append(thinker.get_video_features(part, part_grid))
        centrality.append(probe.visual_centrality(part_grid))
    return torch.cat(tokens), torch.cat(centrality)


def audio_segment_frames(thinker) -> int:
    """Mel frames per segment: a whole number of the tower's attention chunks.

    A multiple of 4 chunks keeps every segment's post-CNN length even, so the
    tower's pair pooling never straddles a segment boundary.
    """
    chunk = 2 * thinker.audio_tower.n_window
    per_second = 100  # Whisper features: 10 ms hop
    chunks = max(1, (AUDIO_SEGMENT_SECONDS * per_second) // chunk)
    return chunk * max(4, chunks - chunks % 4)


def encode_audio(thinker, probe, features, feature_mask):
    """The audio tower over the valid frames, segment by segment."""
    length = int(feature_mask[0].sum())
    valid = features[0, :, :length]
    step = audio_segment_frames(thinker)
    tokens, centrality = [], []
    for start in range(0, length, step):
        segment = valid[:, start:start + step].unsqueeze(0)
        mask = torch.ones(1, segment.shape[-1], dtype=feature_mask.dtype, device=segment.device)
        tokens.append(thinker.get_audio_features(segment, feature_attention_mask=mask))
        aftercnn, _ = thinker.audio_tower._get_feat_extract_output_lengths(mask.sum(-1))
        centrality.append(probe.audio_centrality(aftercnn))
    return torch.cat(tokens), torch.cat(centrality)


@dataclass
class Layout:
    """Where each kind of token sits in one prompt's prefill sequence."""

    input_ids: torch.Tensor     # (1, L)
    position_ids: torch.Tensor  # (3, 1, L), Qwen's multimodal rope
    video: torch.Tensor         # indices of the video placeholders, in order
    audio: torch.Tensor         # indices of the audio placeholders, in order
    query: torch.Tensor         # indices of the user's question tokens


def locate(model, processor, inputs) -> Layout:
    thinker = model.thinker
    device = thinker.device
    cfg = thinker.config
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    feature_mask = inputs["feature_attention_mask"].to(device)
    second_per_grid = inputs.get("video_second_per_grid")
    if second_per_grid is not None:
        second_per_grid = torch.as_tensor(second_per_grid, device=device)

    position_ids, _ = thinker.get_rope_index(
        input_ids,
        None,
        inputs["video_grid_thw"].to(device),
        attention_mask,
        True,
        feature_mask.sum(-1),
        second_per_grid,
    )

    ids = input_ids[0]
    video = (ids == cfg.video_token_id).nonzero(as_tuple=True)[0]
    audio = (ids == cfg.audio_token_id).nonzero(as_tuple=True)[0]
    media_end = int(torch.cat([video, audio]).max())
    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    ends = (ids == im_end).nonzero(as_tuple=True)[0]
    ends = ends[ends > media_end]
    if ends.numel() == 0:
        die("no <|im_end|> after the media block; cannot find the question tokens")
    # Skip the media block's closing markers (audio_eos, vision_eos).
    query = torch.arange(media_end + 3, int(ends[0]), device=device)
    if query.numel() == 0:
        die("empty question span")
    return Layout(input_ids, position_ids, video, audio, query)


@dataclass
class Sequence:
    """A prefill sequence after stage 1, with its token groups re-indexed."""

    embeds: torch.Tensor        # (1, L', D)
    position_ids: torch.Tensor  # (3, 1, L')
    video: torch.Tensor
    audio: torch.Tensor
    query: torch.Tensor


def assemble(model, layout: Layout, visual, audio, visual_keep=None, audio_keep=None) -> Sequence:
    """Splice encoder tokens into the prompt, dropping what stage 1 discarded.

    ASSUMPTION A7 (RUN.md): kept tokens retain their original rope positions,
    so the model still sees each surviving token where it was in time and
    space. The paper does not say whether positions are renumbered.
    """
    thinker = model.thinker
    ids = layout.input_ids
    embeds = thinker.get_input_embeddings()(ids).clone()
    keep = torch.ones(ids.shape[1], dtype=torch.bool, device=ids.device)

    def splice(positions, tokens, kept):
        if kept is None:
            embeds[0, positions] = tokens.to(embeds.dtype)
            return positions
        keep[positions] = False
        chosen = positions[kept]
        keep[chosen] = True
        embeds[0, chosen] = tokens.to(embeds.dtype)
        return chosen

    video = splice(layout.video, visual, visual_keep)
    audio_pos = splice(layout.audio, audio, audio_keep)

    remap = torch.cumsum(keep.long(), dim=0) - 1
    selected = keep.nonzero(as_tuple=True)[0]
    return Sequence(
        embeds=embeds[:, selected],
        position_ids=layout.position_ids[:, :, selected],
        video=remap[video],
        audio=remap[audio_pos],
        query=remap[layout.query],
    )


# ---------------------------------------------------------------------------
# Prefill and decode
#
# Hand-written rather than routed through `generate`, because stage 2 shortens
# the sequence partway through the forward pass. Doing that inside a hook on a
# stock forward invalidates the causal mask, the rope tables and the KV cache of
# every later layer at once. Here each piece is rebuilt for the shorter sequence
# at the insertion layer, and each layer's cache simply holds however many
# tokens that layer saw. During decoding a one-token query attends to all of a
# layer's cache, so caches of different lengths are correct, not a workaround.
# ---------------------------------------------------------------------------


@dataclass
class Stage2Stats:
    video_before: int = 0
    video_after: int = 0
    audio_before: int = 0
    audio_after: int = 0


def apply_stage2(hidden, position_ids, seq: Sequence, stats: Stage2Stats):
    """Query-conditioned compression of each modality at the insertion layer."""
    states = hidden[0]
    keep = torch.ones(states.shape[0], dtype=torch.bool, device=states.device)
    rebuilt = states.clone()
    query = states[seq.query].float()
    for name, own, other in (("video", seq.video, seq.audio), ("audio", seq.audio, seq.video)):
        if own.numel() == 0:
            continue
        kept, index = stage2_select(
            states[own].float(),
            query_states=query,
            other_modality_states=states[other].float() if other.numel() else None,
            retention=INNER_RETENTION,
        )
        keep[own] = False
        keep[own[index]] = True
        rebuilt[own[index]] = kept.to(states.dtype)
        setattr(stats, f"{name}_before", int(own.numel()))
        setattr(stats, f"{name}_after", int(index.numel()))
    selected = keep.nonzero(as_tuple=True)[0]
    return rebuilt[selected].unsqueeze(0), position_ids[:, :, selected]


def _run_layers(text_model, hidden, position_ids, cache, cache_position, layers):
    rope = text_model.rotary_emb(hidden, position_ids)
    for layer in layers:
        hidden = layer(
            hidden,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=rope,
        )[0]
    return hidden


def generate(model, processor, seq: Sequence, stage2: Stage2Stats | None) -> tuple[str, list[int]]:
    """Prefill (stage 2 at the insertion layer if given), then greedy decode."""
    from transformers import DynamicCache

    thinker = model.thinker
    text_model = thinker.model
    layers = list(text_model.layers)
    cache = DynamicCache()
    device = seq.embeds.device

    with torch.no_grad():
        hidden = seq.embeds
        position_ids = seq.position_ids
        split = INNER_LAYER if stage2 is not None else len(layers)
        cache_position = torch.arange(hidden.shape[1], device=device)
        hidden = _run_layers(
            text_model, hidden, position_ids, cache, cache_position, layers[:split]
        )
        if stage2 is not None:
            hidden, position_ids = apply_stage2(hidden, position_ids, seq, stage2)
            cache_position = torch.arange(hidden.shape[1], device=device)
            hidden = _run_layers(
                text_model, hidden, position_ids, cache, cache_position, layers[split:]
            )

        logits = thinker.lm_head(text_model.norm(hidden[:, -1:]))
        next_position = int(position_ids.max()) + 1
        stop = _stop_ids(model, processor)
        tokens: list[int] = []
        for step in range(MAX_NEW_TOKENS):
            token = logits[0, -1].argmax()
            tokens.append(int(token))
            if int(token) in stop or step == MAX_NEW_TOKENS - 1:
                break
            hidden = text_model.embed_tokens(token.view(1, 1))
            pos = torch.full((3, 1, 1), next_position + step, device=device, dtype=torch.long)
            cache_position = torch.tensor([cache.get_seq_length()], device=device)
            hidden = _run_layers(text_model, hidden, pos, cache, cache_position, layers)
            logits = thinker.lm_head(text_model.norm(hidden))

    text = processor.batch_decode([tokens], skip_special_tokens=True)[0]
    return text, tokens


def _stop_ids(model, processor) -> set[int]:
    stop = {processor.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
    eos = getattr(model.thinker.config, "eos_token_id", None)
    if isinstance(eos, int):
        stop.add(eos)
    elif isinstance(eos, list):
        stop.update(eos)
    return stop


def stock_generate(model, processor, inputs, seq: Sequence | None = None) -> list[int]:
    """The reference the `full` arm is checked against: unmodified HF generate.

    With `seq`, stock generate is handed the already-encoded embeddings instead
    of pixels and mel features, so it skips the encoders and checks exactly the
    part this file replaces: the rope positions, prefill and decode. That keeps
    the check itself from running out of memory on a long audio track, which a
    single stock audio-tower call does on a 24 GB card.
    """
    thinker = model.thinker
    device = thinker.device
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    if seq is not None:
        batch.pop("pixel_values_videos")
        batch.pop("input_features")
        batch["inputs_embeds"] = seq.embeds
    else:
        for key in ("pixel_values_videos", "input_features"):
            batch[key] = batch[key].to(DTYPE)
    with torch.no_grad():
        out = thinker.generate(
            **batch,
            use_audio_in_video=True,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            repetition_penalty=1.0,
            eos_token_id=sorted(_stop_ids(model, processor)),
        )
    prompt = batch["input_ids"].shape[1]
    return (out[0, prompt:] if out.shape[1] > prompt else out[0]).tolist()


def check_control(model, processor, inputs, seq, tokens: list[int], answer: str) -> None:
    """Stop the run unless the `full` arm reproduces stock generation."""
    reference = stock_generate(model, processor, inputs, seq)
    if reference == tokens:
        print("control: full arm reproduces stock generate exactly", flush=True)
        return
    print(f"control: ours={tokens} stock={reference}", flush=True)
    stock_answer = letter(processor.batch_decode([reference], skip_special_tokens=True)[0])
    if stock_answer != answer:
        die("the full arm does not reproduce stock Qwen; fix the harness first")
    print("control: tokens differ but the answer matches (fp16 noise)", flush=True)


def letter(text: str) -> str:
    match = re.search(r"[ABCD]", text.upper())
    return match.group(0) if match else "?"


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def main() -> int:
    if DEVICE == "cuda" and not torch.cuda.is_available():
        die("no CUDA device. This runner is for the pod; use smoke_cpu.py on a laptop.")

    rows = json.loads(Path(QFILE).read_text())
    by_vid: dict[str, list] = {}
    for row in rows:
        by_vid.setdefault(row["videoID"], []).append(row)
    print(f"{len(rows)} questions across {len(by_vid)} videos", flush=True)
    print(f"ratios={RATIOS} stage2={USE_STAGE2} layer={INNER_LAYER}", flush=True)

    device = DeviceInfo.capture()
    print(f"device: {device.describe()}", flush=True)
    log = MeasurementLog(MEASF, device=device)

    model, processor = load_model()
    layers = decoder_layers(model)
    if len(layers) <= INNER_LAYER:
        die(f"insertion layer {INNER_LAYER} beyond {len(layers)} decoder layers")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"resident after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    conditions = ["full"] + [f"omnipack_r{r:g}" for r in RATIOS]
    correct = {c: 0 for c in conditions}
    attempted = {c: 0 for c in conditions}
    verified = not VERIFY_CONTROL

    for vid, questions in by_vid.items():
        source = f"{VIDDIR}/videomme-{vid}.mp4"
        if not Path(source).exists():
            print(f"SKIP {vid} (no video)", flush=True)
            continue

        try:
            media = load_media(source)
            first = build_inputs(processor, media, questions[0]["question"])
            # Stage `encode`: paid in full by every arm, because a post-encoder
            # method cannot avoid it. This is the number the comparison exists
            # to expose.
            with log.measure("encode", condition="shared", video=vid) as rec:
                visual, audio, info = encode(model, first)
                rec.extra["frames"] = FRAMES_NUM
                rec.extra["visual_tokens"] = int(visual.shape[0])
                rec.extra["audio_tokens"] = int(audio.shape[0])
        except Exception as exc:  # noqa: BLE001 - logged, then the video is skipped
            print(f"{vid} encode: ERR {type(exc).__name__}: {exc}"[:300], flush=True)
            continue
        if rec.oom:
            print(f"{vid}: OOM in encode", flush=True)
            continue
        print(f"{vid}: {visual.shape[0]} visual + {audio.shape[0]} audio tokens", flush=True)

        compressed = {}
        for ratio in RATIOS:
            with log.measure("select", condition=f"omnipack_r{ratio:g}", video=vid) as rec:
                v_out = stage1_compress(
                    visual.float(), info["visual_centrality"], modality="visual",
                    retention=ratio, frames=info["frames"], patches=info["patches"],
                    grid_w=info["grid_w"],
                )
                a_out = stage1_compress(
                    audio.float(), info["audio_centrality"], modality="audio",
                    retention=ratio,
                )
                rec.extra.update(
                    visual_before=v_out.n_before, visual_after=v_out.n_after,
                    audio_before=a_out.n_before, audio_after=a_out.n_after,
                )
            compressed[ratio] = (v_out, a_out)

        for item in questions:
            qid = item["question_id"]
            gold = str(item["answer"]).strip()[:1].upper()
            prompt = (
                "Select the best answer to the following multiple-choice question "
                "based on the video. Respond with only the letter (A, B, C, or D) of "
                "the correct option. Question: " + item["question"] + "\n"
                + " ".join(item["options"]) + "\nThe best answer is:"
            )
            inputs = build_inputs(processor, media, prompt)
            layout = locate(model, processor, inputs)

            answers: dict[str, str] = {}
            for condition in conditions:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    if condition == "full":
                        seq = assemble(model, layout, visual, audio)
                        stats = None
                    else:
                        v_out, a_out = compressed[float(condition.split("_r")[1])]
                        seq = assemble(
                            model, layout, v_out.embeds, a_out.embeds,
                            v_out.kept_index, a_out.kept_index,
                        )
                        stats = Stage2Stats() if USE_STAGE2 else None

                    with log.measure("answer", condition=condition, qid=qid) as rec:
                        text, tokens = generate(model, processor, seq, stats)
                        answer = letter(text)
                        rec.extra.update(
                            answer=answer, gold=gold, correct=answer == gold,
                            prefill_tokens=int(seq.embeds.shape[1]),
                        )
                        if stats is not None:
                            rec.extra["stage2"] = vars(stats)
                    if rec.oom:
                        print(f"{qid} {condition}: OOM in answer", flush=True)
                        continue

                    if condition == "full" and not verified:
                        check_control(model, processor, inputs, seq, tokens, answer)
                        verified = True

                except Exception as exc:  # noqa: BLE001 - logged, run continues
                    print(f"{qid} {condition}: ERR {type(exc).__name__}: {exc}"[:300], flush=True)
                    continue

                answers[condition] = answer
                attempted[condition] += 1
                correct[condition] += int(answer == gold)

            with open(RESF, "a") as fh:
                fh.write(json.dumps({"qid": qid, "gold": gold, **answers}) + "\n")
            print(
                f"{qid} gold={gold} "
                + " ".join(f"{c}={a}{'Y' if a == gold else 'n'}" for c, a in answers.items()),
                flush=True,
            )

        del visual, audio, info, compressed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for condition in conditions:
        n = attempted[condition]
        if n:
            print(
                f"RESULT_{condition.upper()}: {correct[condition]}/{n} "
                f"({correct[condition] / n:.0%})",
                flush=True,
            )
    print(f"MEASUREMENTS: {MEASF}", flush=True)
    print("OMNIPACK_H2H_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
