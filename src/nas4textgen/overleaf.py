"""The Overleaf project, generated from the manuscript and the artifacts.

Two rules decide the shape of this.

**One source for the prose.** The manuscript lives in ``docs/manuscript/paper.md`` and the
LaTeX is converted from it. Writing the paper twice — once readable, once for the compiler —
means two files that drift, and the drift is invisible until a reviewer reads the version
nobody edited. So ``main.tex`` is an output, not a file to edit: edit the Markdown.

**One source for the numbers.** Tables are emitted from the artifact directory as ``.tex``
and pulled in with ``\\input``, so a figure in the compiled paper is the figure the pipeline
measured. A LaTeX table typed once and adjusted later disagrees with its source silently,
which is the failure this project exists to avoid.

The bibliography is the exception, and deliberately not solved by machine. Reviewer R3 found
three verified errors in the prior reference list — a citation to DistilBERT where DistilGPT2
was meant among them — and a list with three verified errors has an unknown number of
unverified ones. The entries are carried over marked unverified, and the build refuses to
call itself submission-ready while any marker remains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Venue:
    """A target journal's LaTeX conventions.

    Parameterised rather than hard-coded because the venue decision and the build should not
    be coupled: the document class was `acmsmall` while ACM TELO was the target, and
    switching to an Elsevier journal meant editing the generator rather than passing a flag.
    Three things differ and nothing else does — the class, where the title and authors go,
    and the bibliography style.
    """

    key: str
    name: str
    document_class: str
    class_options: str
    bibliography_style: str
    #: Whether title, authors and abstract are wrapped in a ``frontmatter`` environment, as
    #: ``elsarticle`` requires and ``acmart`` does not.
    frontmatter: bool
    #: ``adjustbox`` is here for the tables: several of them carry six or seven columns
    #: whose headers are phrases, and at natural width those run past the right margin. Set
    #: to ``max width=\linewidth`` the box shrinks only the tables that need it and leaves
    #: the rest at the document's type size.
    #: ``microtype`` is the remedy for the handful of lines the paragraph breaker cannot
    #: fit: without it a line carrying a long compound such as ``surrogate-assisted`` runs
    #: a centimetre past the margin rather than being set a shade tighter.
    #: ``amsmath`` for the numbered equations and ``algorithm`` with ``algpseudocode`` for
    #: the two procedures. The manuscript writes both as raw LaTeX, so the classes have to
    #: carry the environments they use.
    #: ``amssymb`` comes with ``amsmath`` because the Pareto set is written with
    #: ``\nexists``, which lives in the symbol package rather than the maths one.
    #: ``array`` is what lets a paragraph column carry ``\raggedright``, which the tier
    #: table needs: justified text in a column that narrow opens word spaces wide enough to
    #: read as gaps in the table.
    packages: tuple[str, ...] = ("amsmath", "amssymb", "array", "booktabs", "graphicx",
                                 "adjustbox", "microtype", "algorithm", "algpseudocode")


NEUROCOMPUTING = Venue(
    key="neurocomputing",
    name="Neurocomputing (Elsevier)",
    document_class="elsarticle",
    class_options="preprint,12pt",
    bibliography_style="elsarticle-num",
    frontmatter=True,
)

ACM_TELO = Venue(
    key="telo",
    name="ACM Transactions on Evolutionary Learning and Optimization",
    document_class="acmart",
    class_options="acmsmall",
    bibliography_style="ACM-Reference-Format",
    frontmatter=False,
)

VENUES = {venue.key: venue for venue in (NEUROCOMPUTING, ACM_TELO)}

#: The target. Both are Scopus Q1, so the institutional constraint does not decide between
#: them; Neurocomputing was chosen on SJR and on the authors' prior history with it.
DEFAULT_VENUE = NEUROCOMPUTING

#: Marker on a reference nobody has checked against its source yet.
UNVERIFIED = "UNVERIFIED"


class ManuscriptMissing(RuntimeError):
    """Raised when there is no draft to convert."""


#: What an entry of each type needs before it can be called complete. A citation missing its
#: volume and pages is one a reader cannot follow to the page, and 125 of the prior list's
#: 152 journal entries are missing both.
#: A DOI is not on any of these lists. It is the single most useful field a reference can
#: carry and a great many works simply do not have one — NeurIPS before 2019, JMLR,
#: OpenReview, older proceedings — so requiring it marks correct entries as unverified
#: forever and teaches a reader to ignore the marker. What completeness means here is that
#: the entry identifies the work well enough to find it; the DOI is added wherever a
#: registry supplies one and its absence is recorded rather than treated as a defect.
REQUIRED_FIELDS = {
    "article": ("author", "title", "journal", "year"),
    "inproceedings": ("author", "title", "booktitle", "year"),
    "incollection": ("author", "title", "booktitle", "year"),
    "book": ("author", "title", "year", "publisher"),
    "phdthesis": ("author", "title", "year", "school"),
    "mastersthesis": ("author", "title", "year", "school"),
    "misc": ("author", "title", "year"),
}


@dataclass(frozen=True)
class Reference:
    """One entry carried over from the prior submission, and its status.

    Two provenances. An entry parsed from the prior article's printed reference list has
    only ``raw`` — a line of text, with no field structure to check. An entry imported from
    the prior project's ``.bib`` has its real type and fields, which is strictly better: the
    verification pass becomes correcting fields rather than reconstructing them, and the
    fields a type requires and does not have can be named rather than guessed at.
    """

    key: str
    number: int
    raw: str
    verified: bool = False
    entry_type: str = "misc"
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def missing(self) -> tuple[str, ...]:
        """Fields this entry's type requires and this entry does not have."""
        if not self.fields:
            return ()
        required = REQUIRED_FIELDS.get(self.entry_type, REQUIRED_FIELDS["misc"])
        return tuple(name for name in required if not self.fields.get(name, "").strip())

    def as_bibtex(self) -> str:
        if not self.fields:
            note = "" if self.verified else (
                "\n  note         = {" + UNVERIFIED + ": carried from the prior "
                "submission's reference list and not yet checked against its source},"
            )
            return (
                f"@misc{{{self.key},\n"
                f"  title        = {{{self.raw}}},{note}\n"
                f"  year         = {{}},\n"
                "}\n"
            )

        lines = [f"@{self.entry_type}{{{self.key},"]
        for name, value in self.fields.items():
            lines.append(f"  {name:<12} = {{{_escape_field(value)}}},")
        if not self.verified:
            gap = (" Missing: " + ", ".join(self.missing) + "." if self.missing else "")
            lines.append(
                f"  {'note':<12} = {{{UNVERIFIED}: imported from the prior submission's "
                f"bibliography and not yet checked against its source.{gap}}},")
        lines.append("}\n")
        return "\n".join(lines)


def _fields(block: str) -> dict[str, str]:
    """Field values from one BibTeX entry, counting braces rather than matching them.

    A regular expression cannot do this. Nearly every title in the prior project's file
    protects its capitalisation with nested braces — ``{Neural} {Architecture} {Search}`` —
    and a non-greedy ``\\{(.*?)\\}`` stops at the first inner close, returning a value with
    unbalanced braces. Written as a regex first, and it produced a bibliography that would
    not compile while looking correct in a summary.
    """
    fields: dict[str, str] = {}
    for match in re.finditer(r"(\w+)\s*=\s*\{", block):
        name = match.group(1).lower()
        depth, index = 1, match.end()
        while index < len(block) and depth:
            if block[index] == "{":
                depth += 1
            elif block[index] == "}":
                depth -= 1
            index += 1
        if depth:
            continue  # unterminated: skip rather than truncate
        fields[name] = " ".join(block[match.end():index - 1].split())
    return fields


def _escape_field(value: str) -> str:
    """Make a BibTeX field value safe for LaTeX without disturbing what is already safe.

    DOIs are the reason. A DOI may contain an underscore — `10.1162/tacl_a_00321` is one
    the paper cites — and an unescaped underscore halts the compile in math mode, from
    inside the generated bibliography where it is hardest to attribute. Only characters
    that are not already escaped are touched, so an already-escaped ampersand stays put.
    """
    out = []
    for index, character in enumerate(value):
        escaped = index > 0 and value[index - 1] == "\\"
        if character in "_%#&" and not escaped:
            out.append("\\" + character)
        else:
            out.append(character)
    return "".join(out)


def parse_bibtex(text: str) -> dict[str, Reference]:
    """Entries from a BibTeX file, by citation key.

    Deliberately small: it reads the shape the prior project's file actually has — one
    ``field = {value},`` per line — and does not attempt the whole grammar. A parser that
    silently mishandles a construct it does not understand would corrupt a reference list
    whose errors are the thing being corrected.
    """
    found: dict[str, Reference] = {}
    for block in re.split(r"\n(?=@)", text):
        header = re.match(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", block.strip())
        if not header:
            continue
        entry_type, key = header.group(1).lower(), header.group(2)
        fields = _fields(block)
        found[key] = Reference(
            key=key,
            number=len(found) + 1,
            raw=fields.get("title", key),
            entry_type=entry_type,
            fields=fields,
            # The marker in the file is what says whether an entry was settled. Reading a
            # frozen bibliography without it reported every entry as unverified, including
            # the ones a registry had already confirmed.
            verified=UNVERIFIED not in fields.get("note", ""),
        )
    return found


def cited_keys(latex: str) -> list[str]:
    """Citation keys the manuscript actually uses, in order of first appearance.

    The prior project's bibliography holds 165 entries and the manuscript cites 59. Carrying
    all 165 into a new submission would present 106 references nobody cited as if they had
    been read.
    """
    seen: list[str] = []
    for match in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", latex):
        for key in match.group(1).split(","):
            key = key.strip()
            if key and key not in seen:
                seen.append(key)
    return seen


def import_cited(bib_text: str, latex: str) -> tuple[list[Reference], list[str]]:
    """The cited subset of a BibTeX file, renumbered. Returns the entries and any key
    cited by the manuscript that the bibliography does not define."""
    entries = parse_bibtex(bib_text)
    references, absent = [], []
    for number, key in enumerate(cited_keys(latex), start=1):
        entry = entries.get(key)
        if entry is None:
            absent.append(key)
            continue
        references.append(
            Reference(key=key, number=number, raw=entry.raw,
                      entry_type=entry.entry_type, fields=entry.fields))
    return references, absent


def parse_references(text: str) -> list[Reference]:
    """Numbered entries from the prior article's reference list."""
    if "\nReferences" not in text:
        return []
    tail = text[text.index("\nReferences"):]
    found: list[Reference] = []
    current: list[str] = []
    number: int | None = None
    for line in tail.splitlines():
        match = re.match(r"^\[(\d+)\]\s*(.*)", line)
        if match:
            if current and number is not None:
                found.append(_reference(number, " ".join(current)))
            number, current = int(match.group(1)), [match.group(2)]
        elif current and line.strip() and not line.strip().isdigit():
            current.append(line.strip())
    if current and number is not None:
        found.append(_reference(number, " ".join(current)))
    return found


def _reference(number: int, body: str) -> Reference:
    """Clean one entry and give it a citation key from the first author's surname.

    Split on commas only. Splitting on periods as well takes "A." from "A. Thessen" as the
    first field, and every key becomes a single initial — `a1`, `o2` — which collides as
    soon as two first authors share one.
    """
    body = re.sub(r"\s*\|\s*", " ", body)
    body = re.sub(r"\s{2,}", " ", body).strip(" -|")
    first = body.split(",")[0].strip()
    words = [w for w in first.split() if len(w.rstrip(".")) > 1]
    surname = (words[-1] if words else "ref").lower()
    key = re.sub(r"[^a-z0-9]", "", surname) or "ref"
    return Reference(key=f"{key}{number}", number=number, raw=body)


#: Sections the manuscript holds as source and the article does not print. The submission
#: form takes the highlights as a field of its own, so they belong in the file that feeds
#: that field, not between the keywords and the introduction where a reader meets them.
#:
#: The submission statements join them. The data availability statement, the competing
#: interest declaration and the CRediT roles are fields the editorial system collects
#: through its own form, so printing them in the article duplicates what the platform
#: already holds. They stay in the manuscript because a declaration retyped into a form
#: months later is a declaration that can differ from what the authors agreed to.
NOT_TYPESET = ("highlights", "submission statements")

#: Sections that are printed but not numbered. These are the declarations Elsevier requires
#: after the conclusions, and they are back matter rather than part of the argument: numbered
#: alongside the chapters they make the paper look as though it reasons its way to a
#: competing-interest statement, and they push the last real section number away from the
#: count of sections the introduction promises.
UNNUMBERED = (
    "data and software availability",
    "declaration of competing interest",
    "credit authorship contribution statement",
    "acknowledgements",
    "acknowledgments",
)


def drop_sections(markdown: str, names: tuple[str, ...] = NOT_TYPESET) -> str:
    """Remove whole ``##`` sections by name, leaving the rest in order."""
    lines = markdown.splitlines()
    kept, skipping = [], False
    for line in lines:
        if line.startswith("## "):
            skipping = line[3:].strip().lower() in names
        if not skipping:
            kept.append(line)
    return "\n".join(kept)



#: The manual number a heading carries in the Markdown, such as `4.9` in `### 4.9 Transfer
#: across corpora`. It is the key the prose refers to the section by, and the key the label
#: is built from, so that `Section 4.9` in a sentence and the heading it points at cannot
#: drift apart.
HEADING_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+")


def _section_label(heading: str) -> str:
    """The `\\label` that lets the prose point at this heading.

    Returns the empty string for a heading with no number, which is what the unnumbered
    back matter has: a label on a section LaTeX does not number would resolve to nothing.
    """
    match = HEADING_NUMBER.match(heading.strip())
    return r"\label{sec:" + match.group(1) + "}" if match else ""


def section_labels(markdown: str) -> dict[str, str]:
    """Every `Section N` the prose may write, mapped to the label that resolves it.

    Built from the headings rather than from the sentences, so a reference to a section
    that does not exist is left as written and shows up in the compile as an undefined
    reference, rather than being silently invented.
    """
    found: dict[str, str] = {}
    for line in markdown.splitlines():
        if not line.startswith(("## ", "### ")):
            continue
        heading = line.split(" ", 1)[1]
        if heading.strip().lower() in UNNUMBERED:
            continue
        match = HEADING_NUMBER.match(heading.strip())
        if match:
            found[f"Section {match.group(1)}"] = f"sec:{match.group(1)}"
    return found


def rewrite_references(body: str, references: dict[str, str]) -> str:
    """Turn "Table 9" in the prose into a cross-reference LaTeX resolves.

    A float's printed number is the order in which it appears, and once floats are placed
    beside the paragraph that introduces them that order is the order of first mention. The
    numbers the prose was written with need not agree: the transfer table printed as Table 7
    while three sentences called it Table 9. Replacing the literal with a `\ref` makes the
    two agree by construction instead of by inspection.
    """
    for literal, label in sorted(references.items(), key=lambda kv: -len(kv[0])):
        kind = literal.split()[0]
        # A function rather than a replacement string: `re.sub` reads escapes in the
        # replacement, so the `\r` of `\ref` became a carriage return and every reference
        # printed as "Figure eﬀig:search-space".
        target = kind + "~\\ref{" + label + "}"
        body = re.sub(rf"\b{re.escape(literal)}\b(?!\d)",
                      lambda _, t=target: t, body)
    return body


def place_floats(body: str, floats: dict[str, str]) -> tuple[str, list[str]]:
    """Put each table and figure after the paragraph that first refers to it.

    Emitting every float after the whole body, which is what this did, leaves LaTeX to
    place them and LaTeX places them at the end: the five figures landed on pages 33 to 35
    while the text cited them on page 7. A reader working through the method finds no
    diagram anywhere near the prose that needs one, and reasonably concludes there is none.

    ``floats`` maps a label as the prose writes it, such as ``Figure 1`` or ``Table 3``, to
    the LaTeX that typesets it. Anything never referred to is returned so the caller can
    decide what to do with it rather than have it silently vanish.
    """
    paragraphs = body.split("\n\n")
    placed: set[str] = set()
    out: list[str] = []
    for paragraph in paragraphs:
        out.append(paragraph)
        # A paragraph may introduce more than one float; they follow in label order so a
        # reader meets Table 2 before Table 3 even when one sentence names both.
        here = sorted(label for label in floats
                      if label not in placed and _refers_to(paragraph, label))
        for label in here:
            out.append(floats[label])
            placed.add(label)
    return "\n\n".join(out), [label for label in floats if label not in placed]


def _refers_to(paragraph: str, label: str) -> bool:
    """Whether a paragraph names a float, without matching a longer number.

    `Table 1` must not match the `Table 11` of another sentence, which would place the
    eleventh table beside the first one's discussion.
    """
    return re.search(rf"\b{re.escape(label)}\b(?!\d)", paragraph) is not None


def drop_front_block(markdown: str) -> str:
    """Remove everything before the first section heading.

    The title, the author list and the affiliations are built into the preamble from the
    citation metadata, so leaving them in the body prints them a second time under the
    real ones. Anything else sitting there — a status note for the authors, a highlights
    block that the submission form takes as a separate field — is not part of the article
    either, and a reader of the PDF should not meet it between the keywords and the
    introduction.
    """
    lines = markdown.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("## ")), 0)
    return "\n".join(lines[start:])


def join_wrapped_items(markdown: str) -> str:
    """Fold a list item's continuation lines back onto the item.

    Markdown lets a bullet wrap across lines with an indent. The converter emits one
    one ``item`` per line beginning with `- ` and then wraps runs of consecutive items in an
    `itemize`, so a wrapped bullet closed the list after its first line and dropped the
    rest of the sentence outside it. The contributions list lost two thirds of every
    bullet that way, and the result reads as text cut off mid-clause.
    """
    out: list[str] = []
    for line in markdown.splitlines():
        continuation = (out and out[-1].startswith("- ")
                        and line.startswith(("  ", "\t")) and line.strip())
        if continuation:
            out[-1] = out[-1].rstrip() + " " + line.strip()
        else:
            out.append(line)
    return "\n".join(out)


#: Opens a block of LaTeX the converter copies out untouched. Equations and algorithms are
#: written as LaTeX because there is no Markdown for them, and because an approximation of
#: an equation is a different equation. Everything between the fences is the author's
#: responsibility: the converter neither escapes it nor checks it.
RAW_FENCE = "```latex"


def to_latex(markdown: str) -> str:
    """Convert the manuscript's Markdown to the subset of LaTeX it needs.

    Narrow on purpose. It handles what ``paper.md`` actually contains — headings, emphasis,
    lists, block quotes, tables, inline code and fenced blocks of raw LaTeX — and leaves
    anything else alone rather than guessing. A converter that silently mangles a construct
    it does not understand is worse than one that passes it through for a human to notice.
    """
    lines: list[str] = []
    in_table = False
    in_raw = False

    for raw in join_wrapped_items(markdown).splitlines():
        line = raw.rstrip()

        # Raw LaTeX, checked before anything else: inside a fence a line starting with `-`
        # is a minus sign and a line starting with `#` is a comment, and neither is a list
        # or a heading.
        if in_raw:
            if line.strip() == "```":
                in_raw = False
            else:
                lines.append(line)
            continue
        if line.strip() == RAW_FENCE:
            in_raw = True
            continue

        # Tables in the prose are rewritten by hand into the generated .tex files, so a
        # Markdown table here marks a place the author has not moved yet.
        if line.startswith("|"):
            if not in_table:
                lines.append(r"% TODO: this table is inline Markdown. Either move it to a")
                lines.append(r"% generated table under tables/ or typeset it here.")
                in_table = True
            lines.append("% " + line)
            continue
        in_table = False

        if line.startswith("### "):
            lines.append(r"\subsection{" + _inline(_strip_number(line[4:])) + "}"
                         + _section_label(line[4:]))
        elif line.startswith("## "):
            heading = _strip_number(line[3:])
            star = "*" if heading.strip().lower() in UNNUMBERED else ""
            lines.append(r"\section" + star + "{" + _inline(heading) + "}"
                         + ("" if star else _section_label(line[3:])))
        elif line.startswith("# "):
            continue  # the title comes from the preamble
        elif line.startswith("> "):
            lines.append(r"\begin{quote}" + _inline(line[2:]) + r"\end{quote}")
        elif line.startswith("- "):
            lines.append(r"\item " + _inline(line[2:]))
        elif line.strip() == "---":
            lines.append("")
        else:
            lines.append(_inline(line))

    body = "\n".join(lines)
    # Wrap runs of \item in an itemize, which is easier to get right after the fact than
    # while streaming.
    body = re.sub(
        r"((?:^\\item .*\n?)+)",
        lambda m: "\\begin{itemize}\n" + m.group(1) + "\\end{itemize}\n",
        body,
        flags=re.MULTILINE,
    )
    return re.sub(r"\n{3,}", "\n\n", one_paragraph_per_line(body))


#: Environments whose body is a program rather than prose. Everything between the two is
#: copied out line for line: a step of an algorithm joined to the step above it is a
#: different algorithm, and a row of a table moved onto the previous row is a broken table.
#: ``abstract``, ``quote`` and ``itemize`` are deliberately absent, because their contents
#: are paragraphs and are reflowed like any others.
_VERBATIM = r"(equation|align|algorithm|algorithmic|tabular|split|cases)"
VERBATIM_BEGIN = re.compile(rf"^\s*\\begin\{{{_VERBATIM}")
VERBATIM_END = re.compile(rf"^\s*\\end\{{{_VERBATIM}")

#: Lines that are structure rather than prose, and are left exactly as they are. Reflowing
#: a `\State` would move a step of an algorithm onto the line above it.
STRUCTURAL = re.compile(
    r"^\s*(?:\\(?:section|subsection|begin|end|item|input|includegraphics|caption"
    r"|label|State|Require|Ensure|ForAll|EndFor|Return|Comment|documentclass"
    r"|usepackage|providecommand|setlength|author|ead|affiliation|cortext|title"
    r"|maketitle|bibliography|bibliographystyle|adjustbox|centering|toprule|midrule"
    r"|bottomrule|smallskip|footnotesize|Description|providecommand)\b|%|$)"
)


def one_paragraph_per_line(body: str) -> str:
    """Reflow prose so that each paragraph occupies a single line.

    The manuscript is wrapped at a column width, which puts a line break wherever the
    measure happened to run out. That is fine to read and poor to edit: changing a word
    rewraps the paragraph, so a one-word change shows up as a five-line diff, and a
    co-author rewriting a sentence has to repair the wrapping of everything after it.

    A paragraph on one line removes the problem entirely. The editor wraps it on screen,
    the paragraph is selected and moved as the unit it is, and a blank line still marks
    where one paragraph ends and the next begins. Nothing changes on the page: LaTeX
    treats any run of whitespace inside a paragraph as a single space.

    Structural lines and anything inside a verbatim-ish environment are left alone, because
    a line of an algorithm is not prose and moving it changes the program.
    """
    out: list[str] = []
    paragraph: list[str] = []
    depth = 0

    def flush() -> None:
        if paragraph:
            out.append(" ".join(" ".join(paragraph).split()))
            paragraph.clear()

    for line in body.splitlines():
        if VERBATIM_BEGIN.match(line):
            depth += 1
        if depth or STRUCTURAL.match(line):
            flush()
            out.append(line)
            if VERBATIM_END.match(line):
                depth = max(0, depth - 1)
            continue
        paragraph.append(line)
    flush()
    return "\n".join(out)


def _strip_number(heading: str) -> str:
    """Remove a manual section number, which LaTeX supplies itself.

    The pattern allows the trailing period the manuscript writes — `1. Introduction` and
    `4.11 What the measurement cost` are both numbered headings — because the earlier
    pattern required a digit after every dot and therefore matched neither, printing
    "1 1. Introduction" on the page.
    """
    return re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", heading)


#: LaTeX the manuscript is allowed to write directly. Everything else is escaped, because
#: a stray backslash in prose breaks the compile; but escaping these turned every citation
#: into the literal text `\textbackslash{}cite{...}`, which typesets as nonsense and cites
#: nothing. The paper had 19 of them and the bibliography printed empty.
#: ``mbox`` is here so a string that must stay transcribable can be kept off a line
#: break. The grant number in the acknowledgements broke after its last hyphen, which
#: leaves a reader copying it to guess whether the hyphen belongs to the number.
PASSTHROUGH = re.compile(
    r"\\(?:cite[tp]?\*?|ref|label|eqref|mbox)(?:\[[^\]]*\])*\{[^}]*\}")

#: Every ``$...$`` span on a line, paired left to right.
#:
#: Pairing is the whole point, and getting it wrong is subtle. A pattern that also required
#: the span to look like maths could begin matching at a *closing* delimiter: in
#: ``$P(a)$ ... and $k_{v} \mid h$`` it paired the close of the first span with the open of
#: the second, held the prose between them and escaped the real formula, which printed as
#: ``k_v\midh``. Matching every span unconditionally pairs them the way the author wrote
#: them, and whether a span is maths is then decided one span at a time.
MATHS = re.compile(r"\$[^$\n]+\$")

#: What makes a ``$...$`` span maths rather than two sums of money with words between them.
#: A control sequence, a subscript, a superscript or a brace group appears in formulas and
#: not in ``$77 and the total was $590``, and those are exactly the characters whose
#: escaping would destroy a formula.
LOOKS_LIKE_MATHS = re.compile(r"[\\_^{]")


def _inline(text: str) -> str:
    # Protected before escaping and restored after, so a command survives intact while the
    # prose around it is still made safe.
    kept: list[str] = []

    def hold(match: re.Match) -> str:
        kept.append(match.group(0))
        return f"\x00{len(kept) - 1}\x00"

    def hold_if_maths(match: re.Match) -> str:
        span = match.group(0)
        return hold(match) if LOOKS_LIKE_MATHS.search(span) else span

    # Commands first, so that a `\ref` in the prose cannot be mistaken for the contents of
    # a maths span once the delimiters are being paired.
    text = PASSTHROUGH.sub(hold, text)
    text = MATHS.sub(hold_if_maths, text)
    text = text.replace("\\", r"\textbackslash{}")
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\emph{\1}", text)
    text = re.sub(r"`([^`]+?)`", r"\\texttt{\1}", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Every character the manuscript uses that the document font cannot set. Written out
    # rather than discovered one compile at a time: each missing entry costs a run of
    # XeTeX and a warning that is easy to read past, and a superscript dropped silently
    # turns an exponent into a different number.
    for character, replacement in (
        ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
        ("—", "---"), ("–", "--"),
        ("≤", r"$\leq$"), ("≥", r"$\geq$"), ("≈", r"$\approx$"),
        ("×", r"$\times$"), ("−", "--"), ("τ", r"$\tau$"),
        ("⁻", r"$^-$"),
        ("⁰", r"$^0$"), ("¹", r"$^1$"), ("²", r"$^2$"), ("³", r"$^3$"),
        ("⁴", r"$^4$"), ("⁵", r"$^5$"), ("⁶", r"$^6$"), ("⁷", r"$^7$"),
        ("⁸", r"$^8$"), ("⁹", r"$^9$"),
    ):
        text = text.replace(character, replacement)
    # Underscores are escaped everywhere, inside `\texttt` as well as outside. They were
    # left alone there on the belief that the monospace font made them literal; it does
    # not. `_` is the subscript character in any mode LaTeX is not told otherwise about,
    # and `\texttt{test_paper.py}` halts the compile with "Missing $ inserted". The paper
    # had five of them and none of this was visible until it was compiled.
    text = text.replace("_", r"\_")
    return re.sub(r"\x00(\d+)\x00", lambda m: kept[int(m.group(1))], text)


def split_abstract(markdown: str) -> tuple[str, str]:
    """Lift the abstract out of the body, for a class that wants it in the frontmatter.

    Returns ``(abstract, rest)``. ``elsarticle`` puts the abstract inside ``frontmatter``;
    leaving it as a numbered section produces a paper whose abstract appears after the
    author list as ordinary prose, which is a submission defect rather than a style choice.
    """
    lines = markdown.splitlines()
    try:
        start = next(i for i, line in enumerate(lines)
                     if line.strip().lower() == "## abstract")
    except StopIteration:
        return "", markdown
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("## ")), len(lines))
    abstract = "\n".join(lines[start + 1:end]).strip()
    rest = "\n".join(lines[:start] + lines[end:])
    return abstract, rest


def split_keywords(abstract: str) -> tuple[str, list[str]]:
    """Separate the keyword line from the abstract text.

    The Markdown carries the keywords as a bold line at the end of the abstract, which is
    how a reader of the Markdown wants them. A journal class does not: ``elsarticle`` has a
    ``keyword`` environment that production reads, and a bold line inside the abstract is
    both the wrong place and, set in bold, the wrong face. The trailing horizontal rule that
    separates sections in the Markdown is dropped here as well, since a rule inside an
    abstract sets as a dash.
    """
    lines = abstract.splitlines()
    try:
        start = next(i for i, line in enumerate(lines)
                     if re.match(r"\s*\*\*keywords:?\*\*", line, flags=re.I))
    except StopIteration:
        body = "\n".join(line for line in lines if line.strip() != "---").strip()
        return body, []
    tail = "\n".join(lines[start:])
    tail = re.sub(r"^\s*\*\*keywords:?\*\*", "", tail, flags=re.I)
    words = [word.strip() for word in tail.replace("---", " ").split(",")]
    body = "\n".join(lines[:start]).strip()
    return body, [word for word in words if word]


def _author_blocks(venue: Venue, authors: list[dict[str, str]],
                   corresponding: str) -> list[str]:
    """Author, affiliation and e-mail, in the form the class expects."""
    blocks: list[str] = []
    if not venue.frontmatter:
        for author in authors:
            name = f"{author.get('given-names', '')} {author.get('family-names', '')}"
            mark = r" \thanks{Corresponding author.}" if name == corresponding else ""
            blocks.append("\n".join([
                r"\author{" + name + mark + "}",
                r"\affiliation{%",
                r"  \institution{" + author.get("affiliation", "") + "}",
                r"  \country{Mexico}}",
                r"\email{" + author.get("email", "") + "}",
            ]))
        return blocks

    # elsarticle: affiliations are numbered and shared, so each distinct one is declared
    # once and referred to by label. Repeating an affiliation per author would print the
    # institution four times.
    affiliations: list[str] = []
    for author in authors:
        affiliation = author.get("affiliation", "")
        if affiliation and affiliation not in affiliations:
            affiliations.append(affiliation)

    for author in authors:
        name = f"{author.get('given-names', '')} {author.get('family-names', '')}"
        label = (f"[{affiliations.index(author.get('affiliation', '')) + 1}]"
                 if author.get("affiliation", "") in affiliations else "")
        mark = r"\corref{cor1}" if name == corresponding else ""
        blocks.append("\n".join([
            r"\author" + label + "{" + name + mark + "}",
            r"\ead{" + author.get("email", "") + "}",
        ]))

    for index, affiliation in enumerate(affiliations, start=1):
        blocks.append(
            r"\affiliation[" + str(index) + "]{organization={" + affiliation
            + "}, country={Mexico}}")
    if corresponding:
        blocks.append(r"\cortext[cor1]{Corresponding author.}")
    return blocks


def preamble(title: str, authors: list[dict[str, str]], corresponding: str,
             venue: Venue = DEFAULT_VENUE, abstract: str = "") -> str:
    """The document header for one venue, with the author block from the citation metadata."""
    parts = [
        f"\\documentclass[{venue.class_options}]{{{venue.document_class}}}",
        "",
        "% Generated by `python -m nas4textgen.cli overleaf`. Edit the Markdown in",
        "% docs/manuscript/paper.md and the artifacts, not this file: it is an output,",
        "% and an edit here is lost on the next regeneration.",
        f"% Target venue: {venue.name}",
        "",
        *(r"\usepackage{" + package + "}" for package in venue.packages),
        "",
        # The generated tables carry `\Description`, which acmart defines for accessibility
        # and elsarticle does not. Declared here as a no-op so one set of tables compiles
        # under either class: the alternative is a table renderer that knows about venues,
        # and the tables would then differ between them for no reason a reader can see.
        r"\providecommand{\Description}[1]{}",
        "",
        # A last pass the paragraph breaker is allowed when every other attempt has failed.
        # Without it a paragraph containing a long compound and a bracketed citation sets
        # one line a centimetre into the right margin rather than loosening the word spacing
        # of the paragraph it belongs to.
        r"\setlength{\emergencystretch}{3em}",
        "",
        r"\begin{document}",
        "",
    ]
    if venue.frontmatter:
        parts += [r"\begin{frontmatter}", ""]
    parts += [r"\title{" + title + "}", ""]
    parts += _author_blocks(venue, authors, corresponding)
    body, words = split_keywords(abstract)
    if venue.frontmatter:
        parts += [""]
        if body:
            parts += [r"\begin{abstract}", one_paragraph_per_line(_inline(body)),
                      r"\end{abstract}", ""]
        if words:
            # elsarticle separates keywords with \sep, not commas: a comma-separated line
            # inside the environment is typeset as one long keyword.
            parts += [r"\begin{keyword}",
                      " \\sep ".join(_inline(word) for word in words),
                      r"\end{keyword}", ""]
        parts += [r"\end{frontmatter}", ""]
    elif words:
        parts += [r"\keywords{" + ", ".join(_inline(word) for word in words) + "}", ""]
    return "\n".join(parts)


#: Where finished versions of the document are kept, one numbered folder each.
VERSIONS = "docs/manuscript/versions"


def next_version(root: Path) -> str:
    """The next version label, as ``v01``, ``v02`` and so on.

    Numbered rather than dated. A version is a thing the authors circulated and referred to
    by name — "the one we sent in v02" — and a date tells you when a folder was written,
    which is not the same question.
    """
    import re

    target = root / VERSIONS
    if not target.exists():
        return "v01"
    used = [int(m.group(1)) for d in target.iterdir()
            if d.is_dir() and (m := re.fullmatch(r"v(\d+)", d.name))]
    return f"v{max(used, default=0) + 1:02d}"


#: Build by-products that a snapshot does not keep. They are rewritten by every compile and
#: say nothing about the version; the PDF is kept, because the PDF is the thing the authors
#: circulate and then refer to by name.
TRANSIENT = ("*.log", "*.blg", "*.aux", "*.out", "*.bbl", "*.toc")


def snapshot(root: Path, built: Path, label: str | None = None,
             note: str = "") -> Path:
    """Copy a built project to a numbered version folder.

    A copy rather than a tag: the build directory is regenerated in place, so the only way
    to still have v01 after building v02 is to have kept it.
    """
    import shutil

    label = label or next_version(root)
    target = root / VERSIONS / label
    if target.exists():
        raise FileExistsError(f"{target} already exists; pass a different label")
    shutil.copytree(built, target, ignore=shutil.ignore_patterns(*TRANSIENT))
    (target / "VERSION.md").write_text(_version_note(label, target, note),
                                       encoding="utf-8")
    return target


def _version_note(label: str, target: Path, note: str) -> str:
    """What this version is, written beside it.

    A folder named `v03` records that there was a third version and nothing about what
    changed in it. The question asked of a version months later is which one the reviewer
    saw, and that is answered by a date and a revision rather than by a number.
    """
    import subprocess
    from datetime import datetime, timezone

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            cwd=target, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = "unknown"

    pdf = target / "main.pdf"
    lines = [
        f"# {label}",
        "",
        f"- kept: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- revision: {revision}",
        f"- compiled: {'yes' if pdf.exists() else 'no, main.pdf was not in the build'}",
    ]
    if note:
        lines += ["", note.strip()]
    lines += [
        "",
        "Generated by `python -m nas4textgen.cli version`. The folder is a copy of the",
        "build at the moment it was kept, without the log files a compile rewrites.",
    ]
    return "\n".join(lines) + "\n"
