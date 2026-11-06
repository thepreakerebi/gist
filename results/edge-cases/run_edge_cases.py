"""Edge-case inputs through the real library path: add a video by URL, ingest, ask.

Each case is a video or a question that sits outside what the evaluation corpus
covers: no sound, no audio stream, sound that is not speech, speech that is not
English, a clip of a few seconds, a picture that never changes, a question the
video cannot answer, and queries in unusual forms. The point is not accuracy but
behaviour: does the system finish, fail with a message a person can act on, or
fail silently?

Synthetic videos are generated here with ffmpeg and the macOS `say` voices, then
served from a local HTTP server so the API ingests them exactly as it ingests a
link: through yt-dlp, frame and window extraction, the scorers and Postgres.
Questions use the extractive answerer, so no hosted model is involved and the
evidence shown is Gist's own selection.

Needs the stack in web/e2e/README.md (API on port 8100) and, for the
question-only cases, an ingested long video passed as --video.

    python results/edge-cases/run_edge_cases.py --video <ready video id>
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
API = "http://127.0.0.1:8100"
PORT = 8765
INGEST_TIMEOUT_S = 900

FRENCH = (
    "Bonjour. Aujourd'hui il fait très beau à Kigali. "
    "Le marché ouvre à neuf heures et ferme à dix-huit heures. "
    "Les bananes coûtent cinq cents francs le kilo. "
    "Demain, il va pleuvoir sur le lac Kivu."
)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args], check=True)


def build_media(work: Path) -> dict[str, Path]:
    """Generate every synthetic video; return name -> file."""

    pattern = ["-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25"]
    h264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    aac = ["-c:a", "aac", "-shortest"]
    files: dict[str, Path] = {}

    files["silent"] = work / "silent.mp4"
    ffmpeg(
        *pattern,
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=stereo",
        "-t",
        "60",
        *h264,
        *aac,
        str(files["silent"]),
    )

    files["no_audio_stream"] = work / "no_audio_stream.mp4"
    ffmpeg(*pattern, "-t", "60", *h264, "-an", str(files["no_audio_stream"]))

    # A simple melody: the pitch steps every 2 s, so there is structure to find
    # but no speech for Whisper to transcribe. Quoted because the commas inside
    # the expression would otherwise split ffmpeg's filter arguments.
    files["music_only"] = work / "music_only.mp4"
    melody = "0.3*sin(2*PI*(220*pow(2,floor(mod(t,16)/2)/12))*t)"
    ffmpeg(
        *pattern,
        "-f",
        "lavfi",
        "-i",
        f"aevalsrc='{melody}':s=44100",
        "-t",
        "60",
        *h264,
        *aac,
        str(files["music_only"]),
    )

    speech = work / "french.aiff"
    subprocess.run(["say", "-v", "Thomas", "-o", str(speech), FRENCH], check=True)
    files["french_speech"] = work / "french_speech.mp4"
    ffmpeg(*pattern, "-i", str(speech), "-t", "60", *h264, *aac, str(files["french_speech"]))

    files["three_seconds"] = work / "three_seconds.mp4"
    ffmpeg(
        *pattern,
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100",
        "-t",
        "3",
        *h264,
        *aac,
        str(files["three_seconds"]),
    )

    still = work / "still.png"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360", "-frames:v", "1", str(still))
    files["static_ten_minutes"] = work / "static_ten_minutes.mp4"
    ffmpeg(
        "-loop",
        "1",
        "-framerate",
        "1",
        "-i",
        str(still),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=stereo",
        "-t",
        "600",
        *h264,
        "-r",
        "1",
        *aac,
        str(files["static_ten_minutes"]),
    )
    return files


def serve(directory: Path) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def request(method: str, path: str, body: dict | None = None) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{API}{path}", data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read().decode()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        raw = error.read().decode()
        try:
            return error.code, json.loads(raw)
        except json.JSONDecodeError:
            return error.code, raw


def ingest(url: str) -> dict:
    status, body = request("POST", "/v1/library/videos", {"url": url, "consent": True})
    if status >= 400:
        return {"status": "rejected", "http_status": status, "error": body}
    video_id = body["video"]["id"]
    deadline = time.time() + INGEST_TIMEOUT_S
    started = time.perf_counter()
    while time.time() < deadline:
        _, detail = request("GET", f"/v1/library/videos/{video_id}")
        video = detail["video"]
        if video["status"] in {"ready", "failed"}:
            return {
                "video_id": video_id,
                "status": video["status"],
                "error": video["error"],
                "duration_seconds": video["duration_seconds"],
                "frames": video["frame_count"],
                "audio_windows": video["audio_window_count"],
                "ingest_seconds": round(time.perf_counter() - started, 1),
            }
        time.sleep(3)
    return {"video_id": video_id, "status": "timeout"}


def ask(video_id: str, query: str) -> dict:
    """Post a question and read the whole event stream."""

    req = urllib.request.Request(
        f"{API}/v1/library/videos/{video_id}/query",
        data=json.dumps({"query": query, "answerer": "extractive"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    events: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            event = None
            for raw in response:
                line = raw.decode().strip()
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:") and event:
                    events[event] = json.loads(line[5:])
    except urllib.error.HTTPError as error:
        return {"http_status": error.code, "error": error.read().decode()[:300]}

    out: dict[str, Any] = {
        "seconds": round(time.perf_counter() - started, 2),
        "completed": "done" in events,
        "error": events.get("error", {}).get("message"),
    }
    scored = events.get("scored", {}).get("candidates", [])
    out["candidates"] = len(scored)
    out["unscored_candidates"] = sum(1 for c in scored if c.get("score") is None)
    selected = events.get("selected", {}).get("selected", [])
    out["selected"] = [
        {
            "modality": item.get("modality"),
            "timestamp_seconds": item.get("timestamp_seconds"),
            "text": (item.get("text") or "")[:160],
        }
        for item in selected
    ]
    done = events.get("done", {})
    out["answer"] = (done.get("answer") or "")[:300]
    out["clips"] = len(done.get("clips") or [])
    return out


VIDEO_CASES = [
    ("silent", "Silent audio track", ["What does the speaker say?", "What is shown on screen?"]),
    ("no_audio_stream", "No audio stream", ["What is shown on screen?"]),
    (
        "music_only",
        "Music, no speech",
        ["When does the music change?", "What does the speaker say?"],
    ),
    (
        "french_speech",
        "French speech",
        [
            "What does the speaker say about the market?",
            "Que dit-il sur le marché ?",
            "How much do the bananas cost?",
        ],
    ),
    ("three_seconds", "Three-second clip", ["What is shown?"]),
    ("static_ten_minutes", "One still image for ten minutes", ["What is shown on screen?"]),
]

QUESTION_CASES = [
    ("unanswerable", "Question the video cannot answer", "What is the capital of Canada?"),
    ("non_latin", "Question in Chinese", "他们在哪里看日落？"),
    ("emoji_only", "Question made only of emoji", "🐢🐢🌅"),
    ("very_long", "Question of about 2,000 characters", "Where do they go snorkelling? " * 70),
    ("whitespace", "Question of spaces only", "     "),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="id of an ingested long video")
    args = parser.parse_args()

    results: dict[str, Any] = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "cases": []}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        files = build_media(work)
        server = serve(work)
        try:
            for key, label, questions in VIDEO_CASES:
                url = f"http://127.0.0.1:{PORT}/{files[key].name}"
                print(f"== {label}", flush=True)
                ingestion = ingest(url)
                answers = []
                if ingestion.get("status") == "ready":
                    answers = [{"query": q, **ask(ingestion["video_id"], q)} for q in questions]
                    request("DELETE", f"/v1/library/videos/{ingestion['video_id']}")
                elif ingestion.get("video_id"):
                    request("DELETE", f"/v1/library/videos/{ingestion['video_id']}")
                results["cases"].append(
                    {"case": key, "label": label, "ingestion": ingestion, "answers": answers}
                )
                print(json.dumps(results["cases"][-1])[:400], flush=True)
        finally:
            server.shutdown()

    request("DELETE", f"/v1/library/videos/{args.video}/conversation")
    for key, label, query in QUESTION_CASES:
        print(f"== {label}", flush=True)
        answer = {"query": query[:80] + ("…" if len(query) > 80 else ""), **ask(args.video, query)}
        results["cases"].append({"case": key, "label": label, "answers": [answer]})
        print(json.dumps(results["cases"][-1])[:400], flush=True)
    request("DELETE", f"/v1/library/videos/{args.video}/conversation")

    (HERE / "edge_cases.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
