import json
from urllib import error, request

from gist.gateway.evidence_package import build_evidence_prompt
from gist.gateway.schemas import GatewayRequest, GatewayResponse


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"


class OllamaGatewayError(RuntimeError):
    """Raised when the local Ollama server cannot answer from evidence."""


class OllamaTextGateway:
    provider = "ollama"

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout_seconds: float = 120.0,
        num_ctx: int | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if num_ctx is not None and num_ctx <= 0:
            raise ValueError("num_ctx must be greater than zero")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.num_ctx = num_ctx

    def answer(self, request_payload: GatewayRequest) -> GatewayResponse:
        prompt = build_evidence_prompt(request_payload.compression)
        options: dict[str, object] = {"temperature": 0.0}
        if self.num_ctx is not None:
            # Ollama truncates prompts to num_ctx (default ~2048); set explicitly
            # so a full-transcript context is actually seen, not silently cut.
            options["num_ctx"] = self.num_ctx
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        response_payload = self._post_json("/api/generate", payload)
        answer = response_payload.get("response")
        if not isinstance(answer, str) or not answer.strip():
            raise OllamaGatewayError("Ollama response did not include a non-empty response")
        return GatewayResponse(
            answer=answer.strip(),
            context=prompt,
            provider=f"{self.provider}:{self.model}",
        )

    def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(  # noqa: S310 - localhost/user-configured model endpoint.
                http_request,
                timeout=self.timeout_seconds,
            ) as response:
                decoded = response.read().decode("utf-8")
        except error.URLError as exc:
            raise OllamaGatewayError(
                f"Could not reach Ollama at {self.base_url}. "
                "Start Ollama locally or use --answer-with extractive/local-text."
            ) from exc

        try:
            parsed = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise OllamaGatewayError("Ollama returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise OllamaGatewayError("Ollama JSON response must be an object")
        return parsed
