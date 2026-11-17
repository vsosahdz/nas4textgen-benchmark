"""Scan tracked files for credentials, and name the high-entropy strings that are not.

GitGuardian flagged a "generic high entropy secret" in this repository. It was a Terraform
provider checksum — ``h1:`` followed by base64, which is what a 32-byte key also looks like.
The alert was correct to fire and wrong about the cause, and the useful response is not to
dismiss it but to make the distinction checkable here, where it can be checked before a push
rather than reported after one.

Two lists, and the second is the one that does the work.

**Patterns that are credentials.** Azure account keys, SAS tokens, connection strings, PEM
blocks, client secrets, subscription GUIDs. Any match fails, with no allowlist — a real key
does not become acceptable by being expected.

**Formats that are high-entropy and public by design.** Terraform's ``h1:``/``zh:`` provider
checksums, pip's ``--hash=sha256:`` pins, container image digests, and the content hashes
this project writes into its own manifests. Each is listed with why it is not a secret, so
the allowlist is an argument rather than a hole. Anything high-entropy that matches none of
them is reported for a human to classify, because an unrecognised 44-character base64 string
is exactly what a leaked key looks like.

    python -m nas4textgen.audit_secrets
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: Things that are credentials wherever they appear. No exceptions list.
CREDENTIAL_PATTERNS: dict[str, str] = {
    # Each pattern carries the allow marker because these lines ARE the detector. Without
    # it the scanner reports itself and the CI job fails on every run.
    r"AccountKey\s*=\s*[A-Za-z0-9+/]{80,}":  # audit-secrets: allow — detector pattern
        "Azure storage account key",
    r"DefaultEndpointsProtocol=.*AccountKey=":  # audit-secrets: allow — detector pattern
        "Azure storage connection string",
    r"\bsig=[A-Za-z0-9%+/]{40,}":  # audit-secrets: allow — detector pattern
        "shared access signature",
    r"SharedAccessSignature=":  # audit-secrets: allow — detector pattern
        "shared access signature in a URL",
    r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----":  # audit-secrets: allow — detector
        "private key",
    r"\bclient_secret\s*[:=]\s*['\"][^'\"]{8,}":  # audit-secrets: allow — detector pattern
        "service principal secret",
    r"\bAZURE_CLIENT_SECRET\s*=\s*\S+":  # audit-secrets: allow — detector pattern
        "service principal secret in the environment",
    r"\bgh[pousr]_[A-Za-z0-9]{36,}":  # audit-secrets: allow — detector pattern
        "GitHub token",
    r"\bsk-[A-Za-z0-9]{32,}":  # audit-secrets: allow — detector pattern
        "API key",
    r"\bxox[baprs]-[A-Za-z0-9-]{10,}":  # audit-secrets: allow — detector pattern
        "Slack token",
    # The subscription this project runs in must never be committed; target.tfvars pins it
    # by display name for exactly that reason.
    # audit-secrets: allow — the GUID detector pattern itself
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b":
        "GUID, which for this project means a subscription id",
}

#: A GUID of all zeros is a documented placeholder, used in tests to stand in for a
#: subscription id without being one.
GUID_PLACEHOLDERS = {"00000000-0000-0000-0000-000000000000"}

#: High-entropy strings that are public integrity data. Each entry says why.
BENIGN_ENTROPY: dict[str, str] = {
    r"/mnt/azureml/cr/j/[0-9a-f]{32}/[A-Za-z0-9_\-/.]+":
        "Azure ML mount path from a run manifest. The hex segment is the job's own "
        "container id, which the platform prints in its logs and which identifies nothing "
        "outside the job. It reads as entropy because it is a GUID in a path, and the "
        "manifests carry 5,716 of them — one per run that mounted the corpus cache.",
    r"\b[0-9a-f]{40}\b":
        "Git revision. Every artifact records the commit it was produced at, which is what "
        "lets a number be traced to the code that made it; publishing it is the point.",
    r"Linux-[0-9][\w.\-]+-azure-x86_64-with-glibc[\d.]+":
        "Kernel and libc of the compute node, recorded as provenance. A platform string, "
        "not an identifier.",
    r"\b[A-Z][A-Za-z]{19,}(?:(?:19|20)\d{2})?\b":
        "BibTeX citation key: the cited authors' surnames concatenated, sometimes ending "
        "in the publication year. Long and mixed-case enough to read as entropy, and "
        "carrying no information that is not printed in the reference list beside it. "
        "What keeps this from matching a credential is the letters-only body: a key or a "
        "token mixes digits, and base64 adds `+`, `/` and `=`. The year was required here "
        "at first, which left a nine-author key with no year flagged.",
    r'"h1:[A-Za-z0-9+/]+={0,2}"':
        "Terraform provider checksum: base64 SHA256 of a public provider archive. "
        "Committing it is what pins provider integrity, and HashiCorp recommends it. "
        "This is the string GitGuardian flagged.",
    r'"zh:[0-9a-f]{64}"':
        "Terraform provider zip hash, hex SHA256 of a public artefact.",
    r"--hash=sha256:[0-9a-f]{64}":
        "pip hash pin, SHA256 of a public wheel; removing it would weaken the lock.",
    r"sha256:[0-9a-f]{64}":
        "container image digest or content hash. Public by construction, and the whole "
        "point is that it can be quoted to identify what ran.",
    r"\b[0-9a-f]{64}\b":
        "SHA256 in hex. This project hashes corpora, locks and manifests on purpose, and "
        "a content hash of public data reveals nothing about a credential.",
    r"^\s*(?:class|def)\s+[A-Za-z_][A-Za-z0-9_]*":
        "a Python class or function name on its declaration line. Long descriptive test "
        "names read as high-entropy tokens — it flagged "
        # audit-secrets: allow — the two names this rule exists for
        "TestAnAllocationFailureFromAnotherLibrary, then "
        # audit-secrets: allow — the two names this rule exists for
        "TestSeedingCannotMaskAStageThatWroteNothing — and renaming them to placate "
        "a credential scanner would trade a readable name for a false positive. An "
        "identifier the interpreter has to resolve cannot be a secret: a credential "
        "declared as a class would have to be a valid identifier and would still not be "
        "read by anything. Narrow on purpose: it matches the declaration line only, so a "
        "secret assigned to a variable, passed as an argument or sitting in a string is "
        "still caught.",
}

#: Files whose entire contents are integrity data, scanned only for credential patterns.
ENTROPY_EXEMPT_FILES = {
    "infra/.terraform.lock.hcl",
    "requirements/analysis.txt",
    "requirements/azure.txt",
    "requirements/dev.txt",
    "requirements/train-linux-x86_64-cu121.txt",
}

#: A line carrying this marker is skipped, and must say why on the same line.
#:
#: Needed because the scanner detects itself: its own pattern definitions and its own test
#: fixtures contain the shapes it looks for, so without this the CI job fails on every run —
#: and a check that always fails is a check nobody reads. Exempting the two files by path
#: would have been simpler and would have put a blind spot exactly where a real credential
#: would be most embarrassing, so the suppression is per line and carries an argument.
#:
#: The count of suppressions is printed, so they stay visible rather than accumulating
#: quietly.
#:
#: A marker at the end of a code line suppresses that line. A marker alone on its own line
#: suppresses the next one, which is what makes it usable at all: appending a reason to an
#: already long line puts it over the line-length limit, and choosing between two lint rules
#: is not a choice anyone should have to make to document a decision.
ALLOW_MARKER = re.compile(r"#\s*audit-secrets:\s*allow\s*[-—:]\s*(?P<reason>.{15,})")


def _suppressed_lines(text: str) -> set[int]:
    """Line numbers the allow marker covers, one-indexed."""
    covered: set[int] = set()
    lines = text.splitlines()
    for index, line in enumerate(lines, start=1):
        if not ALLOW_MARKER.search(line):
            continue
        covered.add(index)
        if line.lstrip().startswith("#"):
            # A marker on its own line covers the next line that carries code, skipping
            # any further comment lines. The reason often needs more than one line, and an
            # earlier version stopped at the first non-empty line — so a two-line
            # explanation covered its own second line and left the string exposed. A rule
            # that makes a short reason easier to write than a good one is the wrong rule.
            for following in range(index + 1, len(lines) + 1):
                candidate = lines[following - 1]
                if not candidate.strip() or candidate.lstrip().startswith("#"):
                    covered.add(following)
                    continue
                covered.add(following)
                break
    return covered

#: Long enough that a random string is unlikely and a real key is likely.
ENTROPY_MIN_LENGTH = 40
_ENTROPY = re.compile(rf"[A-Za-z0-9+/]{{{ENTROPY_MIN_LENGTH},}}={{0,2}}")

#: Files that must never be tracked at all, whatever they contain.
#: Files where the entropy heuristic is not applied, and why. Credential patterns — cloud
#: keys, bearer tokens, private key headers — still run here; only the "this looks
#: high-entropy" heuristic is suspended.
#:
#: `docs/prior/` holds the group's earlier article and thesis, extracted from PDF. The
#: extraction lost word boundaries, so sentences became strings like a fifty-character
#: title with the spaces removed, and model names put digits in them — `DistilGPT2` alone
#: accounts for dozens. The heuristic fires 108 times on two documents that contain prose.
#:
#: The alternatives were worse. Listing each fragment as a benign format would bury the
#: real entries under prose. Loosening the entropy rule enough to admit them would weaken
#: it in the code and configuration where it does the work. Editing the documents to
#: placate a scanner would corrupt the record they are kept for — they are the source for
#: the reference list and for citing the thesis as prior work.
#:
#: What it gives up: a credential pasted into those two documents, in a format no explicit
#: pattern matches, would not be caught. They were read in full when extracted and contain
#: none, and they do not change: they are a fixed record of what was submitted.
ENTROPY_EXEMPT: dict[str, str] = {
    "submission-neurocomputing/references.bib": (
        "generated from `docs/prior/`, so it inherits the same lost word boundaries in "
        "titles carried over verbatim. Regenerating it cannot fix them — the titles are "
        "what the prior submission printed — and they are being replaced one at a time as "
        "each reference is checked against its source."
    ),
    "docs/prior/": (
        "verbatim record of the prior submission, extracted from PDF. Word boundaries were "
        "lost in extraction, so prose reads as high-entropy strings; credential patterns "
        "still apply to these files."
    ),
}

FORBIDDEN_PATHS = (
    re.compile(r"(^|/)\.env$"),
    re.compile(r"(^|/)\.env\.(?!example)"),
    re.compile(r"\.tfstate(\.|$)"),
    re.compile(r"\.(pem|pfx|p12)$"),
    re.compile(r"(^|/)id_(rsa|dsa|ecdsa|ed25519)$"),
)


@dataclass
class Finding:
    path: str
    line: int
    kind: str
    detail: str
    excerpt: str

    def render(self) -> str:
        return f"  {self.kind:<22}{self.path}:{self.line}\n    {self.detail}\n    {self.excerpt}"


def tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        capture_output=True, text=True, check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _mask(text: str) -> str:
    """Show enough to locate a finding without reprinting a credential into a log."""
    stripped = text.strip()
    if len(stripped) <= 24:
        return stripped
    return f"{stripped[:12]}...{stripped[-6:]} ({len(stripped)} chars)"


def scan(root: Path, suppressed: list[str] | None = None) -> list[Finding]:
    findings: list[Finding] = []
    suppressed = suppressed if suppressed is not None else []
    credential = {re.compile(p): why for p, why in CREDENTIAL_PATTERNS.items()}
    benign = [re.compile(p) for p in BENIGN_ENTROPY]

    for relative in tracked_files(root):
        for forbidden in FORBIDDEN_PATHS:
            if forbidden.search(relative):
                findings.append(Finding(
                    relative, 0, "forbidden file",
                    "this path must never be tracked, whatever it contains", relative,
                ))

        path = root / relative
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        exempt = next((why for prefix, why in ENTROPY_EXEMPT.items()
                       if relative.startswith(prefix)), None)
        covered = _suppressed_lines(text)
        for number, line in enumerate(text.splitlines(), start=1):
            if number in covered:
                if ALLOW_MARKER.search(line):
                    suppressed.append(f"{relative}:{number}")
                continue

            for pattern, why in credential.items():
                for match in pattern.finditer(line):
                    if match.group(0) in GUID_PLACEHOLDERS:
                        continue
                    findings.append(Finding(
                        relative, number, "credential", why, _mask(match.group(0)),
                    ))

            if relative in ENTROPY_EXEMPT_FILES:
                continue
            if exempt is not None:
                continue  # credential patterns above still ran; only the heuristic stops
            for match in _ENTROPY.finditer(line):
                token = match.group(0)
                if any(b.search(line) for b in benign):
                    continue
                findings.append(Finding(
                    relative, number, "unclassified entropy",
                    "high-entropy string matching no known-benign format. Classify it: "
                    "either add its format to BENIGN_ENTROPY with a reason, or remove it.",
                    _mask(token),
                ))
    return findings


def main(argv: list[str] | None = None) -> int:
    root = Path(argv[0]) if argv else Path(__file__).resolve().parents[2]
    suppressed: list[str] = []
    findings = scan(root, suppressed)

    credentials = [f for f in findings if f.kind != "unclassified entropy"]
    unclassified = [f for f in findings if f.kind == "unclassified entropy"]

    print()
    print(f"  scanned {len(tracked_files(root))} tracked files")
    print(f"  credential patterns    {len(CREDENTIAL_PATTERNS)}")
    print(f"  benign entropy formats {len(BENIGN_ENTROPY)}")
    print(f"  suppressed lines       {len(suppressed)}")
    if suppressed:
        for location in suppressed:
            print(f"    {location}")
    print()

    for finding in credentials + unclassified:
        print(finding.render())

    if credentials:
        print(f"\n  {len(credentials)} credential finding(s). Rotate first, then remove from "
              "history — a key that has been pushed is a key that has been published.")
        return 1
    if unclassified:
        print(f"\n  {len(unclassified)} unclassified high-entropy string(s). None matched a "
              "credential pattern, but an unrecognised 44-character base64 string is what a "
              "leaked key looks like, so each needs a human decision.")
        return 1

    print("  No credentials, and every high-entropy string matches a format documented as "
          "public integrity data.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
