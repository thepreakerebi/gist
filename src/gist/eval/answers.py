import re


def answer_score(predicted: str | None, expected: str | None, choices: list[str]) -> float | None:
    if expected is None or not expected.strip():
        return None
    if predicted is None or not predicted.strip():
        return 0.0

    normalized_expected = _normalize_answer(expected)
    normalized_predicted = _normalize_answer(predicted)
    if normalized_predicted == normalized_expected:
        return 1.0

    if choices:
        expected_choice = _resolve_choice(expected, choices)
        predicted_choice = _resolve_choice(predicted, choices)
        if expected_choice is not None:
            return 1.0 if predicted_choice == expected_choice else 0.0

    return 1.0 if normalized_expected in normalized_predicted else 0.0


def _resolve_choice(value: str, choices: list[str]) -> int | None:
    normalized_value = _normalize_answer(value)
    letter_match = re.fullmatch(r"[a-z]", normalized_value)
    if letter_match:
        index = ord(normalized_value) - ord("a")
        if 0 <= index < len(choices):
            return index

    prefixed_letter = re.search(
        r"\b(?:answer|option|choice|letter)(?:\s+is)?\s+([a-z])\b",
        normalized_value,
    )
    if prefixed_letter:
        index = ord(prefixed_letter.group(1)) - ord("a")
        if 0 <= index < len(choices):
            return index

    for index, choice in enumerate(choices):
        normalized_choice = _normalize_answer(choice)
        normalized_choice_text = _strip_choice_prefix(normalized_choice)
        if (
            normalized_value == normalized_choice
            or normalized_choice in normalized_value
            or (
                normalized_choice_text
                and (
                    normalized_value == normalized_choice_text
                    or normalized_choice_text in normalized_value
                    or normalized_value in normalized_choice_text
                )
            )
        ):
            return index

    best_index: int | None = None
    best_overlap = 0.0
    for index, choice in enumerate(choices):
        overlap = _token_overlap(
            _content_tokens(normalized_value),
            _content_tokens(_strip_choice_prefix(_normalize_answer(choice))),
        )
        if overlap > best_overlap:
            best_index = index
            best_overlap = overlap
    if best_index is not None and best_overlap >= 0.6:
        return best_index
    return None


def _normalize_answer(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _strip_choice_prefix(normalized_choice: str) -> str:
    return re.sub(r"^[a-z]\s+", "", normalized_choice, count=1)


def _content_tokens(normalized_value: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "because",
        "is",
        "it",
        "of",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
    return {token for token in normalized_value.split() if token not in stopwords}


def _token_overlap(predicted_tokens: set[str], choice_tokens: set[str]) -> float:
    if not predicted_tokens or not choice_tokens:
        return 0.0
    return len(predicted_tokens & choice_tokens) / len(choice_tokens)
