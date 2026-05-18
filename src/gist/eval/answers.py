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
        if expected_choice is not None and predicted_choice is not None:
            return 1.0 if predicted_choice == expected_choice else 0.0

    return 1.0 if normalized_expected in normalized_predicted else 0.0


def _resolve_choice(value: str, choices: list[str]) -> int | None:
    normalized_value = _normalize_answer(value)
    letter_match = re.fullmatch(r"[a-z]", normalized_value)
    if letter_match:
        index = ord(normalized_value) - ord("a")
        if 0 <= index < len(choices):
            return index

    for index, choice in enumerate(choices):
        normalized_choice = _normalize_answer(choice)
        if normalized_value == normalized_choice or normalized_choice in normalized_value:
            return index
    return None


def _normalize_answer(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))
