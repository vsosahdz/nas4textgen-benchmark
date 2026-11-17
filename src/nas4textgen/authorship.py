"""Who the citation metadata names, and whether it is complete enough to publish.

Task 16.10 asked that authorship name only human researchers across commits, contributor
files and citation metadata. The commit side was wired into CI; the citation file was
checked by reading it, which is a check that stops working the moment nobody reads it again.
This is that check as code.

**Two questions, and they fail differently.**

The first is whether anything non-human is listed. That is a hard refusal with no
conditions: this project does not name AI tools as authors or contributors anywhere, and a
check that could be argued with is not the right shape for a rule that cannot.

The second is whether every author is complete — surname, affiliation, ORCID. Incomplete
metadata is not wrong, it is unfinished, and the distinction matters because the remedy is
different: one is a deletion, the other is an email to a colleague. So the marker is
explicit text rather than an omission, and it fails a release rather than a commit. An
author's name is not a detail to approximate, and a placeholder that looked plausible would
be worse than one that shouts.

**The corresponding author is data, not a convention.** Author order carries seniority and
contribution; it does not say who answers a reviewer's email. CFF has a `contact` field for
exactly that, and a contact who is not among the authors is a mistake worth catching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: The marker left where a value is owed rather than absent. Loud on purpose: a reader
#: skimming the file should not be able to mistake it for a name.
PLACEHOLDER = "TO BE CONFIRMED"

#: Fields every author needs before a release. ORCID is on the list because a citable
#: release without one leaves the author ambiguous in every index that ingests it, which is
#: the thing a DOI exists to prevent.
REQUIRED = ("given-names", "family-names", "affiliation", "orcid")

#: What an author writes when they hold no ORCID. Not every researcher has one, and a rule
#: that cannot tell "none" from "nobody filled this in" either blocks a release forever or
#: is switched off — and switched off it stops catching the blank it exists for. The value
#: has to be written deliberately, which is the whole of the distinction.
DECLARED_ABSENT = "NONE"


class OrcidInvalid(ValueError):
    """Raised when an ORCID's check digit does not match its body."""


def check_digit(orcid: str) -> str:
    """The ISO 7064 MOD 11-2 digit an ORCID's first fifteen imply.

    Computed rather than trusted. An ORCID with one digit wrong is a well-formed identifier
    belonging to somebody else, and every index that ingests the release will attribute the
    work to them; the checksum is what the standard provides to catch exactly that, and it
    costs four lines.
    """
    digits = "".join(c for c in orcid if c.isdigit() or c in "Xx")
    if len(digits) != 16:
        raise OrcidInvalid(f"{orcid!r} does not have sixteen characters")
    total = 0
    for character in digits[:-1]:
        total = (total + int(character)) * 2
    result = (12 - total % 11) % 11
    return "X" if result == 10 else str(result)


def require_orcid(orcid: str) -> None:
    """Refuse an ORCID whose check digit does not match."""
    if orcid.strip() in (DECLARED_ABSENT, PLACEHOLDER, ""):
        return
    digits = "".join(c for c in orcid if c.isdigit() or c in "Xx")
    expected = check_digit(orcid)
    if digits[-1].upper() != expected:
        raise OrcidInvalid(
            f"{orcid}: check digit is {digits[-1]} and the first fifteen imply "
            f"{expected}. A mistyped ORCID is a valid identifier belonging to "
            "another researcher."
        )

#: Names that must never appear. Matched case-insensitively against the whole entry, since
#: an assistant could be listed as a family name, an affiliation, or a note.
NON_HUMAN = (
    "claude", "anthropic", "openai", "gpt", "copilot", "chatgpt", "gemini",
    "llm", "assistant", "ai-generated", "artificial intelligence",
)


class NonHumanAuthor(RuntimeError):
    """Raised when citation metadata names something that is not a person."""


class AuthorshipIncomplete(RuntimeError):
    """Raised when a release would go out on metadata nobody has confirmed."""


@dataclass(frozen=True)
class Author:
    """One entry of the citation file, and what it is still missing."""

    position: int
    fields: dict[str, str]

    @property
    def display(self) -> str:
        given = self.fields.get("given-names", "?")
        family = self.fields.get("family-names", "")
        return f"{given} {family}".strip() if PLACEHOLDER not in family else given

    @property
    def missing(self) -> list[str]:
        return [
            field for field in REQUIRED
            if field not in self.fields
            or PLACEHOLDER in self.fields[field]
            or not self.fields[field].strip()
        ]

    @property
    def without_orcid(self) -> bool:
        """Whether this author has declared they hold no ORCID."""
        return self.fields.get("orcid", "").strip() == DECLARED_ABSENT

    @property
    def complete(self) -> bool:
        return not self.missing


def _parse_block(text: str, key: str) -> list[dict[str, str]]:
    """Entries under a top-level list key, flattened to field maps.

    Hand-rolled rather than pulled from a YAML library, because this check runs in the
    verify tier where the dependency set is the analysis lock and adding a parser to it for
    one file would be a strange trade. The file's shape is fixed and small.
    """
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.rstrip() == f"{key}:")
    except StopIteration:
        return []

    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in lines[start + 1:]:
        if line and not line[0].isspace():
            break  # next top-level key
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            if current:
                entries.append(current)
            current = {}
            stripped = stripped[2:]
        if ":" in stripped:
            name, _, value = stripped.partition(":")
            value = value.strip().strip('"').strip("'")
            if value in (">-", "|", ">"):
                value = ""  # folded block; the continuation lines fill it in
            current[name.strip()] = value
        elif current:
            # Continuation of a folded scalar, such as a wrapped affiliation.
            last = list(current)[-1]
            current[last] = (current[last] + " " + stripped).strip()
    if current:
        entries.append(current)
    return entries


def authors(path: Path | None = None) -> list[Author]:
    path = path or Path(__file__).resolve().parents[2] / "CITATION.cff"
    text = path.read_text(encoding="utf-8")
    return [Author(index, fields)
            for index, fields in enumerate(_parse_block(text, "authors"), start=1)]


def contacts(path: Path | None = None) -> list[dict[str, str]]:
    path = path or Path(__file__).resolve().parents[2] / "CITATION.cff"
    return _parse_block(path.read_text(encoding="utf-8"), "contact")


def require_human(path: Path | None = None) -> None:
    """Refuse citation metadata naming anything that is not a person.

    Unconditional. Every other check in this project states the condition that would change
    its verdict; this one has none, which is why it is separate from the completeness check
    rather than one more item in its list.
    """
    path = path or Path(__file__).resolve().parents[2] / "CITATION.cff"
    text = path.read_text(encoding="utf-8")
    # Comments are excluded: the file explains the rule, and the explanation names the
    # things the rule forbids.
    body = "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))
    found = [name for name in NON_HUMAN
             if re.search(rf"\b{re.escape(name)}\b", body, re.IGNORECASE)]
    if found:
        raise NonHumanAuthor(
            f"citation metadata names {', '.join(found)}. This project does not list AI "
            "tools as authors or contributors anywhere — see openspec/config.yaml."
        )


def require_complete(path: Path | None = None) -> None:
    """Refuse a release while any author is unconfirmed.

    Called before publishing, not on every commit. Unfinished metadata is a normal state
    during the work and an unacceptable one at the moment a DOI is minted, because the DOI
    is what makes it permanent.
    """
    listed = authors(path)
    if not listed:
        raise AuthorshipIncomplete("the citation file lists no authors")

    incomplete = [a for a in listed if not a.complete]
    if incomplete:
        lines = [
            f"    {a.position}. {a.display}: missing {', '.join(a.missing)}"
            for a in incomplete
        ]
        raise AuthorshipIncomplete(
            f"{len(incomplete)} of {len(listed)} authors are not confirmed:\n"
            + "\n".join(lines)
            + "\nAn author's name is not a detail to approximate, so these are marked "
              "rather than guessed. Supply them before the release that mints a DOI."
        )

    named = {(a.fields.get("given-names"), a.fields.get("family-names")) for a in listed}
    for contact in contacts(path):
        key = (contact.get("given-names"), contact.get("family-names"))
        if key not in named:
            raise AuthorshipIncomplete(
                f"the corresponding author {key[0]} {key[1]} is not among the authors. "
                "Author order says who contributed; the contact says who answers a "
                "reviewer, and one cannot be someone the paper does not list."
            )


def report(path: Path | None = None) -> dict[str, object]:
    listed = authors(path)
    return {
        "authors": [
            {"position": a.position, "name": a.display,
             "complete": a.complete, "missing": a.missing}
            for a in listed
        ],
        "corresponding": [
            f"{c.get('given-names', '?')} {c.get('family-names', '')}".strip()
            for c in contacts(path)
        ],
        "complete": all(a.complete for a in listed),
        "why_placeholders_are_loud": (
            "A placeholder that looked like a name would be worse than one that shouts: it "
            "could reach a release. The marker fails a check rather than sitting in a "
            "comment nobody reads."
        ),
    }
