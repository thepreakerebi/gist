import json
import subprocess
from pathlib import Path
import threading
from typing import Any

from gist.gateway.context import render_evidence_context
from gist.gateway.schemas import GatewayRequest, GatewayResponse


class SubprocessGatewayError(RuntimeError):
    """Raised when an external Video-LLM command fails."""


class SubprocessVideoLlmGateway:
    provider = "subprocess"

    def __init__(
        self,
        command: list[str],
        timeout_seconds: float = 120.0,
        provider: str | None = None,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self.command = command
        self.timeout_seconds = timeout_seconds
        self.provider = provider or self.provider

    def answer(self, request: GatewayRequest) -> GatewayResponse:
        context, payload = build_gateway_payload(request)
        completed = self._run(payload)
        answer, returned_context, provider = _parse_gateway_stdout(
            stdout=completed.stdout,
            fallback_context=context,
            fallback_provider=self.provider,
        )
        return GatewayResponse(
            answer=answer,
            context=returned_context,
            provider=provider,
        )

    def _run(self, payload: dict[str, Any]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                self.command,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise SubprocessGatewayError(
                f"Video-LLM command timed out after {self.timeout_seconds:.1f}s"
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""
            raise SubprocessGatewayError(
                f"Video-LLM command failed with exit code {exc.returncode}: {stderr}"
            ) from exc


class PersistentSubprocessVideoLlmGateway:
    provider = "persistent-subprocess"

    def __init__(
        self,
        command: list[str],
        timeout_seconds: float = 120.0,
        provider: str | None = None,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self.command = command
        self.timeout_seconds = timeout_seconds
        self.provider = provider or self.provider
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def answer(self, request: GatewayRequest) -> GatewayResponse:
        context, payload = build_gateway_payload(request)
        with self._lock:
            process = self._start()
            assert process.stdin is not None
            assert process.stdout is not None
            try:
                process.stdin.write(json.dumps(payload) + "\n")
                process.stdin.flush()
                stdout = process.stdout.readline()
            except BrokenPipeError as exc:
                raise SubprocessGatewayError("persistent Video-LLM command closed stdin") from exc

        if not stdout:
            raise SubprocessGatewayError("persistent Video-LLM command exited before responding")

        answer, returned_context, provider = _parse_gateway_stdout(
            stdout=stdout,
            fallback_context=context,
            fallback_provider=self.provider,
        )
        return GatewayResponse(answer=answer, context=returned_context, provider=provider)

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        if process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                process.stdin.flush()
        except BrokenPipeError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def __enter__(self) -> "PersistentSubprocessVideoLlmGateway":
        self._start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _start(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process

        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        return self._process


def build_gateway_payload(request: GatewayRequest) -> tuple[str, dict[str, Any]]:
    context = render_evidence_context(request.compression)
    return context, {
        "query": request.query,
        "context": context,
        "evidence": [
            {
                "id": item.id,
                "modality": item.modality.value,
                "timestamp_seconds": item.timestamp_seconds,
                "text": item.text,
                "clip_path": _path_to_string(item.clip_path),
                "asset_path": _path_to_string(item.asset_path),
                "spatial_mask_path": _path_to_string(item.spatial_mask_path),
                "clip_start_seconds": item.clip_start_seconds,
                "clip_end_seconds": item.clip_end_seconds,
                "segment_id": item.segment_id,
                "scene_start_seconds": item.scene_start_seconds,
                "scene_end_seconds": item.scene_end_seconds,
            }
            for item in request.compression.selected
        ],
        "compression": request.compression.model_dump(mode="json"),
    }


def _parse_gateway_stdout(
    stdout: str,
    fallback_context: str,
    fallback_provider: str,
) -> tuple[str, str, str]:
    stripped = stdout.strip()
    if not stripped:
        raise SubprocessGatewayError("Video-LLM command returned empty stdout")

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped, fallback_context, fallback_provider

    if not isinstance(payload, dict):
        raise SubprocessGatewayError("Video-LLM command JSON stdout must be an object")

    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise SubprocessGatewayError("Video-LLM command JSON stdout must include answer")

    context = payload.get("context")
    provider = payload.get("provider")
    return (
        answer.strip(),
        context if isinstance(context, str) and context.strip() else fallback_context,
        provider if isinstance(provider, str) and provider.strip() else fallback_provider,
    )


def _path_to_string(path: Path | None) -> str | None:
    return str(path) if path is not None else None
