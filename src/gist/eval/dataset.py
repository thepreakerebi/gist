import json
from pathlib import Path

from gist.eval.schemas import EvalExample


def load_jsonl_dataset(path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        examples.append(EvalExample.model_validate(payload))
    return examples

