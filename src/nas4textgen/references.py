"""Checking the carried-over reference list against the record, entry by entry.

Reviewer R3 found three outright errors in the prior submission's bibliography, including a
citation to DistilBERT where DistilGPT2 was meant. A list with three verified errors has an
unknown number of unverified ones, so every entry is checked rather than the three that were
caught. The structural gaps are larger than the factual ones: of the 59 entries the
manuscript cites, 58 carry no DOI, 44 no volume and 43 no pages.

**Crossref rather than a search engine.** The registry is where publishers deposit the
metadata, so it answers the question being asked — what the record says — instead of what a
page about the work says. It also returns structured fields, which makes the comparison a
diff rather than a reading.

**A proposal is not a correction.** Nothing here edits the bibliography. Each entry gets a
candidate and a similarity, and anything below the confident threshold is reported for a
human to settle. A wrong DOI inserted automatically is worse than a missing one: the missing
one is visible and the wrong one resolves to somebody else's paper.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from difflib import SequenceMatcher

CROSSREF = "https://api.crossref.org/works"
ARXIV = "http://export.arxiv.org/api/query"
OPENALEX = "https://api.openalex.org/works"

#: Above this, the titles agree closely enough that the record is taken to be the same work
#: and its fields are proposed. Set high: the cost of a wrong match is a citation that
#: resolves to another paper, which is worse than the gap it would have filled.
CONFIDENT = 0.92

#: Between this and `CONFIDENT`, a candidate is reported for a human to confirm or reject.
#: Below it, the search is treated as having found nothing.
PLAUSIBLE = 0.75

#: Fields worth proposing when the record has them and the entry does not.
PROPOSABLE = ("doi", "volume", "pages", "journal", "year", "publisher")

#: Reported alongside, never proposed as an addition: the registries give full
#: names where the list uses initials, so a "difference" here is usually format.
COMPARED_ONLY = ("author",)

#: Entries whose "journal" names a preprint server rather than a journal. Crossref does not
#: hold most arXiv preprints, so these are looked up there instead.
PREPRINT_MARKERS = ("arxiv", "preprint", "corr")


class LookupUnavailable(RuntimeError):
    """Raised when the registry cannot be reached at all."""


@dataclass(frozen=True)
class Candidate:
    """What a registry returned for one entry."""

    source: str
    title: str
    similarity: float
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def confident(self) -> bool:
        return self.similarity >= CONFIDENT

    @property
    def plausible(self) -> bool:
        return PLAUSIBLE <= self.similarity < CONFIDENT

    def as_dict(self) -> dict[str, object]:
        return {"source": self.source, "title": self.title,
                "similarity": round(self.similarity, 4),
                "fields": dict(self.fields)}


@dataclass(frozen=True)
class Finding:
    """One entry, what the registry says, and what differs."""

    key: str
    title: str
    status: str
    candidate: Candidate | None = None
    #: Fields the record has and the entry does not, as {field: proposed value}.
    additions: dict[str, str] = field(default_factory=dict)
    #: Fields both have and disagree on, as {field: (ours, theirs)}.
    conflicts: dict[str, tuple[str, str]] = field(default_factory=dict)
    missing_after: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "title": self.title,
            "status": self.status,
            "candidate": self.candidate.as_dict() if self.candidate else None,
            "additions": dict(self.additions),
            "conflicts": {k: {"ours": a, "registry": b}
                          for k, (a, b) in self.conflicts.items()},
            "still_missing": list(self.missing_after),
        }


def normalise(text: str) -> str:
    """A title reduced to what a comparison should see.

    Brace protection, punctuation and case are all formatting. Leaving them in makes
    ``{Neural} {Architecture} {Search}`` a poor match for ``Neural architecture search``,
    which is the same title typeset by someone else.
    """
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalise(left), normalise(right)).ratio()


def _get(url: str, timeout: int = 30, attempts: int = 3) -> bytes:
    """One request, retried on a transient failure.

    Without this a single dropped connection marks an entry unverifiable, and an entry
    reported as unverifiable for a network reason is indistinguishable in the output from
    one the registry genuinely does not hold.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "nas4textgen-reference-check/1.0"})
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise LookupUnavailable(f"{type(last).__name__}: {last}") from last


def _authors(entry_authors: str) -> str:
    """First author's surname, for narrowing a title query."""
    first = entry_authors.split(" and ")[0].strip()
    if "," in first:
        return first.split(",")[0].strip()
    parts = [p for p in first.split() if len(p.rstrip(".")) > 1]
    return parts[-1] if parts else ""


def search_crossref(title: str, author: str = "") -> Candidate | None:
    """The best Crossref match for a title, or None if nothing comes close."""
    query = {
        "rows": "5",
        "select": "DOI,title,container-title,volume,page,author,issued,publisher,type",
        "query.bibliographic": normalise(title),
    }
    if author:
        query["query.author"] = author
    payload = json.loads(_get(f"{CROSSREF}?{urllib.parse.urlencode(query)}"))

    best: Candidate | None = None
    for item in payload.get("message", {}).get("items", []):
        candidate_title = (item.get("title") or [""])[0]
        if not candidate_title:
            continue
        score = similarity(title, candidate_title)
        if best is not None and score <= best.similarity:
            continue
        issued = item.get("issued", {}).get("date-parts", [[None]])[0]
        # Authors were requested in `select` and never read, so every entry resolved
        # through Crossref arrived without one. A reference with no author is not a
        # reference, and six of the fourteen additions were written that way before this.
        people = []
        for person in item.get("author") or []:
            given, family = person.get("given", ""), person.get("family", "")
            if family:
                people.append(f"{family}, {given}".strip().rstrip(","))
            elif person.get("name"):
                people.append(person["name"])
        fields = {
            "author": " and ".join(people),
            "doi": item.get("DOI", ""),
            "journal": (item.get("container-title") or [""])[0],
            "volume": item.get("volume", "") or "",
            "pages": (item.get("page", "") or "").replace("-", "--"),
            "year": str(issued[0]) if issued and issued[0] else "",
            "publisher": item.get("publisher", "") or "",
        }
        best = Candidate(source="crossref", title=candidate_title, similarity=score,
                         fields={k: v for k, v in fields.items() if v})
    return best


def search_arxiv(title: str) -> Candidate | None:
    """arXiv, for the preprints Crossref does not hold."""
    query = urllib.parse.urlencode({
        "search_query": f'ti:"{normalise(title)}"', "max_results": "3"})
    text = _get(f"{ARXIV}?{query}").decode("utf-8", errors="replace")

    best: Candidate | None = None
    for block in re.findall(r"<entry>(.*?)</entry>", text, flags=re.S):
        found = re.search(r"<title>(.*?)</title>", block, flags=re.S)
        identifier = re.search(r"<id>(.*?)</id>", block, flags=re.S)
        published = re.search(r"<published>(\d{4})", block)
        if not found:
            continue
        candidate_title = " ".join(found.group(1).split())
        score = similarity(title, candidate_title)
        if best is not None and score <= best.similarity:
            continue
        fields = {}
        if identifier:
            eprint = identifier.group(1).rsplit("/", 1)[-1]
            fields["eprint"] = eprint
            fields["archiveprefix"] = "arXiv"
        if published:
            fields["year"] = published.group(1)
        best = Candidate(source="arxiv", title=candidate_title, similarity=score,
                         fields=fields)
    return best


def search_openalex(title: str) -> Candidate | None:
    """OpenAlex, which holds what Crossref does not.

    Added after a first pass reported ten entries as having no record. They had records:
    NeurIPS 1991, JMLR and OpenReview do not deposit DOIs with Crossref, so a check that
    stops there reports a classic paper as unfindable. "Not in this registry" and "no record
    exists" are different claims, and only the second would justify dropping a citation.
    """
    query = urllib.parse.urlencode({"per-page": "5", "search": normalise(title)})
    payload = json.loads(_get(f"{OPENALEX}?{query}"))

    best: Candidate | None = None
    for work in payload.get("results", []):
        candidate_title = work.get("display_name") or ""
        if not candidate_title:
            continue
        score = similarity(title, candidate_title)
        if best is not None and score <= best.similarity:
            continue
        source = (work.get("primary_location") or {}).get("source") or {}
        biblio = work.get("biblio") or {}
        first, last = biblio.get("first_page"), biblio.get("last_page")
        doi = (work.get("doi") or "").replace("https://doi.org/", "")
        fields = {
            "doi": doi,
            "journal": source.get("display_name", "") or "",
            "volume": biblio.get("volume", "") or "",
            "pages": f"{first}--{last}" if first and last else "",
            "year": str(work.get("publication_year") or ""),
            "author": " and ".join(
                a["author"]["display_name"] for a in (work.get("authorships") or [])
                if a.get("author", {}).get("display_name")),
        }
        best = Candidate(source="openalex", title=candidate_title, similarity=score,
                         fields={k: v for k, v in fields.items() if v})
    return best


def is_preprint(entry) -> bool:
    venue = (entry.fields.get("journal", "") + " "
             + entry.fields.get("booktitle", "")).lower()
    return any(marker in venue for marker in PREPRINT_MARKERS)


def check(entry, pause: float = 0.4) -> Finding:
    """One entry against the registry."""
    title = entry.fields.get("title", entry.raw)
    author = _authors(entry.fields.get("author", ""))

    # Each registry is tried and allowed to fail on its own. Wrapping the sequence in one
    # `try` meant a Crossref outage reported the entry as unreachable without ever asking
    # the others, and six entries OpenAlex matches at 1.00 came back unverifiable for a
    # reason that had nothing to do with them. "Unreachable" now means every registry
    # failed, which is the only form of it that says anything about the entry.
    #
    # Ordered by which is likeliest to hold the work: a preprint's own server first, a
    # published paper's registry of record first, and OpenAlex last because it holds both
    # and is the broadest.
    order = ([(search_arxiv, None), (search_crossref, author)] if is_preprint(entry)
             else [(search_crossref, author), (search_arxiv, None)])
    order.append((search_openalex, None))

    candidate, failures = None, 0
    for lookup, extra in order:
        try:
            found = lookup(title, extra) if extra else lookup(title)
        except LookupUnavailable:
            failures += 1
            continue
        finally:
            time.sleep(pause)
        if found and (candidate is None or found.similarity > candidate.similarity):
            candidate = found
        if candidate is not None and candidate.similarity >= CONFIDENT:
            break

    if candidate is None and failures == len(order):
        return Finding(key=entry.key, title=title, status="unreachable")

    if candidate is None or candidate.similarity < PLAUSIBLE:
        return Finding(key=entry.key, title=title, status="not found",
                       missing_after=entry.missing)
    if candidate.plausible:
        return Finding(key=entry.key, title=title, status="needs a human",
                       candidate=candidate, missing_after=entry.missing)

    additions, conflicts = {}, {}
    for name in PROPOSABLE:
        proposed = candidate.fields.get(name, "").strip()
        if not proposed:
            continue
        current = entry.fields.get(name, "").strip()
        if not current:
            additions[name] = proposed
        elif normalise(current) != normalise(proposed):
            conflicts[name] = (current, proposed)

    still = tuple(name for name in entry.missing if name not in additions)
    return Finding(key=entry.key, title=title,
                   status="conflict" if conflicts else "matched",
                   candidate=candidate, additions=additions,
                   conflicts=conflicts, missing_after=still)


def check_all(entries, pause: float = 0.4) -> list[Finding]:
    return [check(entry, pause=pause) for entry in entries]


def report(findings: list[Finding]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.status] = counts.get(finding.status, 0) + 1
    return {
        "entries": len(findings),
        "by_status": counts,
        "confident_threshold": CONFIDENT,
        "plausible_threshold": PLAUSIBLE,
        "findings": [f.as_dict() for f in findings],
        "what_this_is_not": (
            "Nothing here edits the bibliography. Each entry carries a proposal and the "
            "similarity it rests on; a wrong DOI applied automatically resolves to another "
            "paper, which is worse than the gap it would have filled."
        ),
    }


#: A venue naming one of these is a conference, whatever the entry calls itself. Thirteen
#: of the carried entries are `@article` with a proceedings in the `journal` field, which
#: `elsarticle-num` renders as a journal article with a missing volume.
CONFERENCE_MARKERS = (
    "proceedings", "conference", "advances in neural information processing",
    "workshop", "symposium", "neurips", "nips",
)


def repair_authors(entries):
    """Put back the comma between a surname and an initial.

    BibTeX reads `Parmar N.` as a single given name with no surname, and prints it in the
    reference list where the family name should be. Two of the carried entries are written
    that way, including the transformer paper; the fix is a comma and the cost of missing
    it is a reference a reader cannot look up by author.
    """
    from dataclasses import replace

    pattern = re.compile(r"^([A-Z][\w'\u2019-]+) ([A-Z]\.)$")
    repaired = []
    for entry in entries:
        authors = entry.fields.get("author", "")
        if not authors:
            repaired.append(entry)
            continue
        parts = [pattern.sub(r"\1, \2", person.strip())
                 for person in authors.split(" and ")]
        fixed = " and ".join(parts)
        if fixed == authors:
            repaired.append(entry)
        else:
            repaired.append(replace(entry, fields={**entry.fields, "author": fixed}))
    return repaired


def retype(entries):
    """Give conference papers the entry type they have.

    A citation's type decides how the style renders it, so an `@article` whose journal is
    `Proceedings of the AAAI Conference` prints as a journal article that has mislaid its
    volume. The venue moves from `journal` to `booktitle` with it, because that is the field
    the type reads.
    """
    from dataclasses import replace

    retyped = []
    for entry in entries:
        venue = entry.fields.get("journal", "")
        if entry.entry_type != "article" or not any(
                marker in venue.lower() for marker in CONFERENCE_MARKERS):
            retyped.append(entry)
            continue
        fields = {k: v for k, v in entry.fields.items() if k != "journal"}
        fields["booktitle"] = venue
        retyped.append(replace(entry, entry_type="inproceedings", fields=fields))
    return retyped


def apply(entries, findings, accept_conflicts: bool = False):
    """Fold the confident proposals back into the entries.

    Additions only, by default. A field the entry lacks and the registry supplies is a gap
    being filled; a field they disagree on is a question about which source is right, and
    answering it by overwriting would delete the evidence that there was a disagreement —
    which is how the prior list came to carry a Communications of the ACM paper under
    another journal's name without anyone noticing.

    Entries keep the ``UNVERIFIED`` marker unless every field their type requires is now
    present and nothing is in conflict. Complete is not the same as correct, but incomplete
    is certainly not verified.
    """
    from dataclasses import replace

    by_key = {f.key: f for f in findings}

    # Three steps, and the order is the whole of it. Merge first, because a venue only
    # becomes recognisable as a conference once the registry's value has replaced the
    # publisher's name that was sitting in the journal field. Retype second, since the
    # fields a type requires depend on it. Judge last, or an entry carries a verdict
    # reached under the type it used to have — which left two conference papers typed as
    # journal articles with a proceedings in their journal field.
    merged = []
    for entry in entries:
        finding = by_key.get(entry.key)
        if finding is None or finding.status not in ("matched", "conflict"):
            merged.append(entry)
            continue
        fields = dict(entry.fields)
        fields.update(finding.additions)
        if accept_conflicts:
            for name, (_ours, theirs) in finding.conflicts.items():
                fields[name] = theirs
        merged.append(replace(entry, fields=fields))

    updated = []
    for entry in retype(repair_authors(merged)):
        finding = by_key.get(entry.key)
        touched = finding is not None and finding.status in ("matched", "conflict")
        settled = touched and not entry.missing and not (
            finding.conflicts and not accept_conflicts)
        updated.append(replace(entry, verified=settled))
    return updated


def render_bibliography(entries, header: str = "") -> str:
    """The .bib file, entries in citation order."""
    parts = [header] if header else []
    parts.extend(entry.as_bibtex() for entry in entries)
    return "\n".join(parts)
