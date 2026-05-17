from gist.core.schemas import CompressionResponse


def render_evidence_context(compression: CompressionResponse) -> str:
    lines = [
        f"Query: {compression.query}",
        "Selected evidence:",
    ]
    for item in compression.selected:
        lines.append(
            f"- [{item.modality.value} @ {item.timestamp_seconds:.2f}s] "
            f"{item.text} (reason: {item.reason})"
        )
    return "\n".join(lines)

