"""
Rave Atlas, text normalisation.

A single place that enforces the house typography rules on any text that
reaches a reader: no em or en dashes, no non-breaking hyphens, straight quotes
instead of curly ones, and no stray control characters.

Why this exists as a runtime step and not just a prompt instruction: models
ignore "do not use em dashes" often enough that relying on the prompt alone
leaks dashes into live output (Telegram briefings, in-app digests, chat
answers). Cleaning the text after generation guarantees the rule regardless of
which model wrote it.

humanize() is intentionally conservative. It only rewrites punctuation and
quote glyphs; it never touches words, numbers, or links.
"""

from __future__ import annotations

import re

# Glyph -> replacement. Order does not matter; each is a straight swap.
_GLYPHS: dict[str, str] = {
    "—": ", ",   # em dash
    "–": ", ",   # en dash
    "‒": "-",    # figure dash
    "―": ", ",   # horizontal bar
    "‑": "-",    # non-breaking hyphen
    "‘": "'",    # left single quote
    "’": "'",    # right single quote / apostrophe
    "“": '"',    # left double quote
    "”": '"',    # right double quote
    "…": "...",  # ellipsis
    " ": " ",    # non-breaking space
    "​": "",     # zero-width space
}

_GLYPH_RE = re.compile("|".join(re.escape(g) for g in _GLYPHS))


# AI slop openers stripped from the start of any agent response.
# These add zero information and mark the text as machine-generated.
_SLOP_OPENERS = re.compile(
    r"^(Certainly|Absolutely|Sure|Of course|Great question|That's a great question"
    r"|Happy to help|I'd be happy to|I'm happy to|I'll be happy to"
    r"|Of course,|Sure,|Certainly,|Absolutely,)[!,.]?\s*",
    re.IGNORECASE,
)


def humanize(text: str) -> str:
    """
    Return text with house typography and tone enforced.

    - Em, en, figure dashes and horizontal bars become a comma plus space.
    - A dash between two digits (range like "130-150") keeps a plain hyphen.
    - Non-breaking hyphens and spaces, curly quotes, ellipses, zero-width
      characters normalised to plain ASCII equivalents.
    - AI slop openers (Certainly!, Absolutely!, Great question!) stripped.
    - Any double spaces or " ," artefacts tidied up.
    """
    if not text:
        return text

    # Numeric ranges first: "130–150" / "130—150" -> "130-150".
    text = re.sub(r"(?<=\d)\s*[‒–—―]\s*(?=\d)", "-", text)

    text = _GLYPH_RE.sub(lambda m: _GLYPHS[m.group()], text)

    # Strip sycophantic openers.
    text = _SLOP_OPENERS.sub("", text)
    # Capitalise the first letter after stripping (opener may have left lowercase).
    if text:
        text = text[0].upper() + text[1:]

    # Tidy artefacts the comma swap can leave behind.
    text = text.replace(" ,", ",").replace(",,", ",")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

    samples = [
        "Tresor is iconic — and the sound is bass-heavy.",
        "Techno sits at 130–150 BPM.",
        "It’s a “queer” night, low‑key and warm…",
    ]
    for s in samples:
        print(f"  in : {s!r}")
        print(f"  out: {humanize(s)!r}")
    assert humanize("130–150") == "130-150", "numeric range must keep a hyphen"
    assert "—" not in humanize("a — b"), "em dash must be gone"
    assert "’" not in humanize("it’s"), "curly apostrophe must be gone"
    print("textfmt OK")
