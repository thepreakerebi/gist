"""The privacy notice must stay readable for people with limited literacy.

The notice is written for readers who are not confident with long English text,
so its reading level is pinned here rather than left to whoever edits it next.
Flesch-Kincaid grade is a rough measure, but a rise above grade 6 reliably means
sentences or words have grown long again. The version before this test existed
measured grade 8.1.
"""

import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "web" / "app" / "privacy" / "page.tsx"
MAX_GRADE = 6.0


def _visible_text(source: str) -> str:
    body = source[source.index("<h1") :]
    body = re.sub(r"\{/\*.*?\*/\}", "", body, flags=re.S)  # JSX comments
    body = body.replace('{" "}', " ").replace("&rsquo;", "'")
    body = re.sub(r"<[^>]+>", " ", body)
    return " ".join(body.split())


def _syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1 and not word.endswith("le"):
        count -= 1
    return max(count, 1)


def flesch_kincaid_grade(text: str) -> float:
    sentences = max(1, len(re.findall(r"[.!?](\s|$)", text)))
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    syllables = sum(_syllables(word) for word in words)
    return 0.39 * len(words) / sentences + 11.8 * syllables / len(words) - 15.59


def test_privacy_notice_reads_at_or_below_grade_six() -> None:
    grade = flesch_kincaid_grade(_visible_text(PAGE.read_text()))

    assert grade <= MAX_GRADE, f"privacy notice reads at grade {grade:.1f}; simplify it"


def test_privacy_notice_starts_with_a_summary() -> None:
    # The three facts that matter most come first, so a reader who stops early
    # still learns what is kept, what leaves the server, and how to delete it.
    text = _visible_text(PAGE.read_text())

    assert text.index("In short") < text.index("What Gist is")
