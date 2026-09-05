"""TwelveLabs (Pegasus) answerer — the compressed video, answered natively.

Why this gateway is different from the OpenAI/Claude ones, and why that matters:

Those gateways receive Gist's selected evidence as a montage of still frames
plus transcript text, because that is all a general-purpose chat model accepts.
For an *audio-visual* compression method that is a lossy hand-off — the audio
Gist worked to select is flattened into text, and motion is gone entirely.

Pegasus is video-native, so this gateway hands it the thing Gist actually
produced: the selected evidence spans, cut from the source and concatenated
into one short video. An hour of input becomes perhaps thirty seconds of
compressed video, with its audio intact, and the model answers from that.

That framing is also what keeps the demo honest. Sending TwelveLabs the *whole*
video would let its own indexing do the work and reduce Gist to decoration —
the answer would be evidence of TwelveLabs' capability, not of compression. By
construction, this gateway can only ever see what the selector kept.

The integrity boundary is unchanged: this is a hosted model used for demo
reliability. Every measured efficiency claim in the capstone still comes from
Qwen2.5-Omni-7B run offline, because a closed API cannot report encoder FLOPs.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from gist.gateway.context import render_evidence_context
from gist.gateway.schemas import GatewayRequest, GatewayResponse

API_URL = "https://api.twelvelabs.io/v1.3/analyze"
DEFAULT_MODEL = "pegasus1.5"

# Pegasus rejects anything under 4 seconds, so a single short clip is padded by
# concatenation or, failing that, skipped rather than sent and refused.
MIN_DURATION_SECONDS = 4.0

# The documented ceiling on a base64 payload. Clips are cut from a 360p source,
# so this is generous, but a long selection is trimmed rather than rejected.
MAX_PAYLOAD_BYTES = 30 * 1024 * 1024

# Pegasus rejects anything under 360x360. Sources are often smaller than that —
# the archive footage this is most useful on predates HD entirely (the first
# test video is 320x240) — so the evidence is upscaled to clear the floor rather
# than refused. Upscaling adds no information, but the alternative is being
# unable to answer questions about older material at all.
MIN_DIMENSION = 360
_SCALE_FILTER = (
    f"scale='if(lt(iw,ih),{MIN_DIMENSION},-2)':'if(lt(iw,ih),-2,{MIN_DIMENSION})',"
    f"pad='max(iw,{MIN_DIMENSION})':'max(ih,{MIN_DIMENSION})':'(ow-iw)/2':'(oh-ih)/2'"
)


class TwelveLabsError(RuntimeError):
    """Raised when the TwelveLabs API is unreachable, unauthorized, or refuses."""


@dataclass(frozen=True, slots=True)
class ClipSource:
    path: Path
    start_seconds: float
    end_seconds: float

    @property
    def duration(self) -> float:
        return max(self.end_seconds - self.start_seconds, 0.0)


def load_api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    key = os.environ.get("TWELVELABS_API_KEY", "").strip()
    if key:
        return key
    # Match the fallback the other gateways use for local development.
    legacy = Path(".gist/.twelvelabs_key")
    if legacy.exists():
        return legacy.read_text().strip() or None
    return None


def concatenate_clips(clips: list[ClipSource], output_path: Path) -> Path | None:
    """Join the selected evidence spans into a single short video.

    This *is* the compressed video — the artifact the whole method exists to
    produce. Re-encoded rather than stream-copied because the clips come from
    one source but may not share keyframe alignment, and a concat of copied
    streams can desync audio.
    """

    usable = [clip for clip in clips if clip.path.exists()]
    if not usable:
        return None

    if len(usable) == 1:
        source = usable[0]
        # Always re-encoded, never passed through: the clip must be normalized
        # to Pegasus's resolution floor, and a cut that came in under the
        # duration floor has to be extended.
        hold = source.duration < MIN_DURATION_SECONDS
        return _normalize(source.path, output_path, extend=hold)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as manifest:
        for clip in usable:
            escaped = str(clip.path.resolve()).replace("'", r"'\''")
            manifest.write(f"file '{escaped}'\n")
        manifest_path = Path(manifest.name)

    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest_path),
                "-vf",
                _SCALE_FILTER,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "28",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )
    finally:
        manifest_path.unlink(missing_ok=True)

    if completed.returncode != 0 or not output_path.exists():
        return None

    total = sum(clip.duration for clip in usable)
    if total < MIN_DURATION_SECONDS:
        return _normalize(
            output_path,
            output_path.with_name(f"held-{output_path.name}"),
            extend=True,
        )
    return output_path


def _normalize(source: Path, output_path: Path, *, extend: bool) -> Path | None:
    """Re-encode a clip to clear Pegasus's resolution and duration floors.

    ``extend`` holds the final frame out to the 4 s minimum for a cut that came
    in shorter, which is common when the selector lands on a single tight moment.
    """

    video_filter = _SCALE_FILTER
    command = ["ffmpeg", "-nostdin", "-y", "-i", str(source)]
    if extend:
        video_filter = (
            f"{_SCALE_FILTER},tpad=stop_mode=clone:stop_duration={MIN_DURATION_SECONDS}"
        )
    command += ["-vf", video_filter]
    if extend:
        command += ["-t", f"{MIN_DURATION_SECONDS + 0.5:.2f}"]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    completed = subprocess.run(command, capture_output=True, text=True)
    return output_path if completed.returncode == 0 and output_path.exists() else None


def answer_with_twelvelabs(
    request: GatewayRequest,
    *,
    clips: list[ClipSource],
    api_key: str | None = None,
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = 512,
    timeout_seconds: float = 120.0,
    work_dir: Path | None = None,
) -> GatewayResponse:
    """Answer a query from Gist's selected evidence, as video."""

    key = load_api_key(api_key)
    if not key:
        raise TwelveLabsError(
            "TWELVELABS_API_KEY is not set. Add it to .env to use the TwelveLabs answerer."
        )
    if not clips:
        raise TwelveLabsError("no evidence clips were produced for this query")

    directory = work_dir or Path(tempfile.mkdtemp(prefix="gist-tl-"))
    directory.mkdir(parents=True, exist_ok=True)
    compressed = concatenate_clips(clips, directory / "compressed-evidence.mp4")
    if compressed is None:
        raise TwelveLabsError("could not assemble the evidence clips into a video")

    payload_bytes = compressed.read_bytes()
    if len(payload_bytes) > MAX_PAYLOAD_BYTES:
        raise TwelveLabsError(
            f"compressed evidence is {len(payload_bytes) / 1e6:.1f} MB, over the 30 MB API limit"
        )

    context = render_evidence_context(request.compression)
    body = json.dumps(
        {
            "model_name": model_name,
            "video": {
                "type": "base64_string",
                "base64_string": base64.b64encode(payload_bytes).decode("ascii"),
            },
            "prompt_v2": {"input_text": _prompt(request.query, context)},
            "stream": False,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")

    http_request = urllib.request.Request(
        API_URL,
        data=body,
        headers={"x-api-key": key, "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(http_request, timeout=timeout_seconds) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise TwelveLabsError(f"TwelveLabs returned {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise TwelveLabsError(f"could not reach TwelveLabs: {exc.reason}") from exc

    answer = _extract_answer(parsed)
    if not answer:
        raise TwelveLabsError("TwelveLabs returned no text")

    return GatewayResponse(answer=answer, context=context, provider=f"twelvelabs:{model_name}")


def _prompt(query: str, context: str) -> str:
    # The model sees only Gist's selected spans, so it is told that explicitly:
    # without it, Pegasus narrates the clip rather than answering the question,
    # and may apologise for missing context it was never meant to have.
    return (
        "You are answering a question about a longer video. The video attached is "
        "not the whole recording: it is only the few moments a compression system "
        "selected as the evidence most likely to answer this question, cut out and "
        "joined together. Cuts between moments are expected.\n\n"
        f"Question: {query}\n\n"
        "Answer the question directly from what you see and hear. If the evidence "
        "does not contain the answer, say so plainly rather than guessing. Do not "
        "describe the editing or the cuts.\n\n"
        f"Transcript and on-screen text for the selected moments:\n{context}"
    )


def _extract_answer(payload: object) -> str:
    """Pull the generated text out, tolerating the documented shape variants."""

    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, list):
        return "".join(_extract_answer(item) for item in payload).strip()
    if isinstance(payload, dict):
        for field in ("data", "text", "answer", "message"):
            value = payload.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, (list, dict)):
                nested = _extract_answer(value)
                if nested:
                    return nested
    return ""
