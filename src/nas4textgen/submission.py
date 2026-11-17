"""The pieces an Elsevier submission form asks for, separately from the manuscript.

The editorial system does not take the paper and read the abstract out of it. It asks for
the abstract, the keywords and the highlights one at a time, pasted into fields, and what
goes in has to be plain text: a field that receives ``**Keywords:**`` prints those asterisks
on the article page.

So each piece is extracted from the manuscript rather than retyped beside it. Retyping is
how an abstract comes to differ between the PDF and the record that indexes it, and the
record is the one most readers meet first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Elsevier journals cap the abstract, commonly between 150 and 250 words. The exact figure
#: is the journal's to state, so this is a line to check against rather than a rule to
#: enforce: over it, the submission form truncates or the editor asks.
ABSTRACT_WORDS = 250

#: Highlights: three to five, each at most this many characters including spaces.
HIGHLIGHT_CHARACTERS = 85
MAX_HIGHLIGHTS = 5


class SectionMissing(RuntimeError):
    """Raised when the manuscript has no section of the requested name."""


@dataclass(frozen=True)
class Piece:
    """One field of the submission form."""

    name: str
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def characters(self) -> int:
        return len(self.text)


def plain(markdown: str) -> str:
    """Markdown reduced to what a form field should receive.

    Emphasis, code ticks and links are formatting instructions for a renderer; a text field
    has no renderer and prints them. Non-ASCII punctuation is kept — an em dash is a
    character, not markup, and the form accepts UTF-8.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", markdown, flags=re.S)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\1", text)
    text = re.sub(r"`([^`]+?)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{[^}]*\}", "", text)
    return text


def section(markdown: str, heading: str) -> str:
    """The body of one ``##`` section, without its heading or the rule that follows it."""
    lines = markdown.splitlines()
    wanted = f"## {heading}".lower()
    try:
        start = next(i for i, line in enumerate(lines)
                     if line.strip().lower() == wanted)
    except StopIteration as exc:
        raise SectionMissing(f"no section named {heading!r}") from exc
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("## ")), len(lines))
    body = "\n".join(lines[start + 1:end])
    return body.strip().removesuffix("---").strip()


def abstract(markdown: str) -> Piece:
    """The abstract as one paragraph of plain text.

    Joined into a single paragraph because the field is one, and because an abstract pasted
    with its line breaks arrives in the record with them in it. The keyword line is cut: it
    is a separate field, and leaving it attached is how `Keywords:` ends up inside an
    abstract in a database.
    """
    body = section(markdown, "Abstract")
    body = re.split(r"\n\s*\*\*Keywords", body)[0]
    paragraphs = [" ".join(block.split())
                  for block in re.split(r"\n\s*\n", plain(body)) if block.strip()]
    return Piece(name="abstract", text=" ".join(paragraphs))


def keywords(markdown: str) -> Piece:
    """The keyword list, comma-separated, as the form wants it."""
    body = section(markdown, "Abstract")
    match = re.search(r"\*\*Keywords:\*\*(.+)", body, flags=re.S)
    if not match:
        raise SectionMissing("the abstract carries no keyword line")
    return Piece(name="keywords", text=" ".join(plain(match.group(1)).split()))


def highlights(markdown: str) -> list[Piece]:
    """One piece per highlight, so each can be checked against the length limit."""
    body = section(markdown, "Highlights")
    found = [plain(line[2:]).strip() for line in body.splitlines()
             if line.strip().startswith("- ")]
    return [Piece(name=f"highlight {n}", text=text)
            for n, text in enumerate(found, start=1)]


#: The statements the form collects, as they are named under "Submission statements" in the
#: manuscript and as the file each is written to. They are not sections of the article: the
#: editorial system has a field for each, so printing them in the PDF duplicates what the
#: platform already holds.
STATEMENTS = {
    "Data and software availability": "data-availability",
    "Declaration of competing interest": "competing-interest",
    "CRediT authorship contribution statement": "credit",
}


def subsection(markdown: str, heading: str) -> str:
    """The body of one ``###`` subsection, without its heading.

    The statements sit under a ``##`` section that is not typeset, so each is a level
    deeper than the abstract and the highlights.
    """
    lines = markdown.splitlines()
    wanted = f"### {heading}".lower()
    try:
        start = next(i for i, line in enumerate(lines)
                     if line.strip().lower() == wanted)
    except StopIteration as exc:
        raise SectionMissing(f"no subsection named {heading!r}") from exc
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith(("## ", "### "))), len(lines))
    body = "\n".join(lines[start + 1:end])
    return body.strip().removesuffix("---").strip()


def statements(markdown: str) -> list[Piece]:
    """Each form statement as plain text, ready to paste.

    A statement missing from the manuscript is reported rather than skipped. The one that
    gets forgotten is the one nobody notices is absent until the submission is refused.
    """
    found: list[Piece] = []
    for heading, name in STATEMENTS.items():
        body = plain(subsection(markdown, heading))
        paragraphs = [" ".join(block.split())
                      for block in re.split(r"\n\s*\n", body) if block.strip()]
        found.append(Piece(name=name, text="\n\n".join(paragraphs)))
    return found


def report(markdown: str) -> dict[str, object]:
    """Every piece with the limit it has to clear."""
    summary = abstract(markdown)
    words = keywords(markdown)
    bullets = highlights(markdown)
    declarations = statements(markdown)
    return {
        "statements": [
            {"name": piece.name, "text": piece.text, "words": piece.words}
            for piece in declarations
        ],
        "abstract": {"text": summary.text, "words": summary.words,
                     "limit": ABSTRACT_WORDS,
                     "within_limit": summary.words <= ABSTRACT_WORDS},
        "keywords": {"text": words.text,
                     "count": len([k for k in words.text.split(",") if k.strip()])},
        "highlights": [
            {"text": piece.text, "characters": piece.characters,
             "limit": HIGHLIGHT_CHARACTERS,
             "within_limit": piece.characters <= HIGHLIGHT_CHARACTERS}
            for piece in bullets
        ],
        "highlights_within_count": len(bullets) <= MAX_HIGHLIGHTS,
        "why_extracted": (
            "The submission form asks for each of these separately and takes plain text. "
            "Retyping them beside the manuscript is how an abstract comes to differ "
            "between the PDF and the record that indexes it."
        ),
    }
