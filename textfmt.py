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


def humanize(text: str) -> str:
    """
    Return text with house typography enforced.

    - Em, en, figure dashes and horizontal bars become a comma plus space when
      they were used as separators, so "techno, dark and long" reads naturally.
    - A dash sitting directly between two digits (a range like "130-150") keeps
      a plain hyphen, since a comma there would be wrong.
    - Non-breaking hyphens and spaces, curly quotes, ellipses, and zero-width
      characters are normalised to their plain ASCII equivalents.
    - Any double spaces or " ," artefacts left by the swaps are tidied up.
    """
    if not text:
        return text

    # Numeric ranges first: "130–150" / "130—150" -> "130-150".
    text = re.sub(r"(?<=\d)\s*[‒–—―]\s*(?=\d)", "-", text)

    text = _GLYPH_RE.sub(lambda m: _GLYPHS[m.group()], text)

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
