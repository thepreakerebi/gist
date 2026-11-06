"""Generating, normalising and hashing access codes and session tokens.

A code looks like ``GIST-7K3M-QX9P-R2TA-H8WD``: sixteen characters from
Crockford's base32 alphabet, which leaves out I, L, O and U so a code read
aloud or copied by hand cannot be mistaken for another. Sixteen characters at
five bits each is 80 bits of randomness, far beyond what can be guessed through
a login endpoint limited to a few attempts a minute.

Only SHA-256 digests are stored. A slow password hash (bcrypt, Argon2) exists to
make guessing *low-entropy human* input expensive; for a long random string a
fast hash is already infeasible to invert, so the extra dependency buys nothing.
"""

from __future__ import annotations

import hashlib
import secrets

ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
PREFIX = "GIST"
GROUPS = 4
GROUP_LENGTH = 4
CODE_LENGTH = GROUPS * GROUP_LENGTH

# What a person might type in place of the intended character.
_CONFUSABLE = str.maketrans({"O": "0", "I": "1", "L": "1"})


def generate_code() -> str:
    """Return a new random code in its display form."""

    raw = "".join(secrets.choice(ALPHABET) for _ in range(CODE_LENGTH))
    groups = [raw[index : index + GROUP_LENGTH] for index in range(0, CODE_LENGTH, GROUP_LENGTH)]
    return "-".join([PREFIX, *groups])


def normalize_code(value: str) -> str | None:
    """Reduce typed input to the sixteen code characters, or None if it cannot be one.

    Case, spaces, dashes and the optional ``GIST`` prefix are ignored, and the
    letters O, I and L are read as the digits they resemble.
    """

    cleaned = "".join(ch for ch in value.upper() if ch.isalnum())
    if cleaned.startswith(PREFIX) and len(cleaned) == len(PREFIX) + CODE_LENGTH:
        cleaned = cleaned[len(PREFIX) :]
    cleaned = cleaned.translate(_CONFUSABLE)
    if len(cleaned) != CODE_LENGTH or any(ch not in ALPHABET for ch in cleaned):
        return None
    return cleaned


def hash_secret(value: str) -> str:
    """SHA-256 hex digest of a normalised code or a session token."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def generate_session_token() -> str:
    """Return a new session token: 32 random bytes, URL-safe."""

    return secrets.token_urlsafe(32)
