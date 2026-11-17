"""The scanner exists because GitGuardian flagged a Terraform provider checksum.

The alert was right to fire — h1: followed by base64 is exactly what a 32-byte key looks
like — and wrong about the cause. The useful response is to make the distinction checkable
before a push rather than reported after one.

Each test below is a string that either is or is not a credential, so the scanner's
judgement is pinned rather than trusted.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from nas4textgen.audit_secrets import BENIGN_ENTROPY, scan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo(tmp_path):
    """A throwaway git repository, so the scanner sees tracked files and nothing else."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)

    def add(name: str, content: str) -> None:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        subprocess.run(["git", "-C", str(tmp_path), "add", "-f", name], check=True)

    return tmp_path, add


class TestTheRealRepository:
    def test_the_repository_is_clean(self):
        """The answer to the alert, asserted rather than asserted about."""
        findings = scan(ROOT)
        assert not findings, "\n".join(f.render() for f in findings)

    def test_the_flagged_terraform_checksum_is_still_present_and_still_benign(self):
        """It must not have been deleted to silence the alert: removing the lock file
        would remove the provider integrity pin, which is a real regression traded for a
        cosmetic one."""
        lock = ROOT / "infra" / ".terraform.lock.hcl"
        assert lock.exists()
        assert "h1:" in lock.read_text(encoding="utf-8")
        assert not scan(ROOT)


class TestCredentials:
    """No allowlist applies here. A real key does not become acceptable by being expected."""

    # Assembled from pieces rather than written out, and every piece is synthetic.
    #
    # The first version of this list spelled out the project's real subscription GUID as a
    # fixture, and the scanner caught it on its first full run — which is the check working,
    # and also a reminder that a test file is a tracked file. A fixture that has to look
    # like a credential should be built so that it never is one.
    # audit-secrets: allow — a synthetic GUID, deliberately not anyone's subscription
    FAKE_GUID = "deadbeef-0000-4000-8000-feedfacecafe"

    @pytest.mark.parametrize("content", [
        "AccountKey=" + "A" * 88,  # audit-secrets: allow — synthetic fixture, all one letter
        # audit-secrets: allow — a synthetic connection string, assembled from pieces
        "DefaultEndpointsProtocol=https;AccountName=x;" + "AccountKey=" + "B" * 88,
        # audit-secrets: allow — a synthetic SAS, assembled from pieces
        "url = https://x.blob.core.windows.net/c?sv=2024-01-01&" + "sig=" + "c" * 44,
        "-----BEGIN PRIVATE" + " KEY-----",  # audit-secrets: allow — split so it is not a block
        'client_secret = "' + "not-a-real-secret" + '"',  # audit-secrets: allow — synthetic
        "AZURE_CLIENT_SECRET=" + "synthetic-value",  # audit-secrets: allow — synthetic fixture
        "token = ghp_" + "d" * 36,  # audit-secrets: allow — synthetic, all one letter
        "subscription = " + FAKE_GUID,  # audit-secrets: allow — a GUID that is not anyone's
    ])
    def test_a_credential_is_found(self, repo, content):
        tmp_path, add = repo
        add("config.txt", content + "\n")
        findings = scan(tmp_path)
        assert findings, f"missed: {content[:40]}"
        assert any(f.kind == "credential" for f in findings)

    def test_the_excerpt_does_not_reprint_the_whole_secret(self, repo):
        """A scanner that logs what it found has published it a second time."""
        tmp_path, add = repo
        secret = "Z" * 88
        add("config.txt", f"AccountKey={secret}\n")  # audit-secrets: allow — synthetic
        findings = scan(tmp_path)
        assert findings
        assert secret not in findings[0].excerpt
        assert "chars)" in findings[0].excerpt

    def test_an_all_zero_guid_is_a_placeholder_not_a_subscription(self, repo):
        tmp_path, add = repo
        add("t.py", 'SUBSCRIPTION = "00000000-0000-0000-0000-000000000000"\n')
        assert not [f for f in scan(tmp_path) if f.kind == "credential"]


class TestForbiddenPaths:
    @pytest.mark.parametrize("name", [
        ".env", ".env.local", "infra/terraform.tfstate",
        "infra/terraform.tfstate.backup", "certs/server.pem", "keys/id_rsa",
    ])
    def test_a_forbidden_path_fails_whatever_it_contains(self, repo, name):
        tmp_path, add = repo
        add(name, "nothing sensitive at all\n")
        assert any(f.kind == "forbidden file" for f in scan(tmp_path))

    def test_the_example_env_is_allowed(self, repo):
        tmp_path, add = repo
        add(".env.example", "PYTHONHASHSEED=0\n")
        assert not scan(tmp_path)


class TestEntropyClassification:
    def test_a_terraform_checksum_is_recognised(self, repo):
        tmp_path, add = repo
        add("providers.hcl", '    "h1:a14TKo7Xvg4W8+H1VA6p+oLZTLxVQnYUD8LOaOs14A8=",\n')
        assert not scan(tmp_path)

    def test_an_image_digest_is_recognised(self, repo):
        tmp_path, add = repo
        add("image.json", '{"digest": "sha256:' + "0" * 64 + '"}\n')
        assert not scan(tmp_path)

    def test_an_unrecognised_base64_blob_is_reported_for_a_human(self, repo):
        """The scanner does not guess. An unclassified 44-character base64 string is what
        a leaked key looks like, so it asks rather than deciding."""
        tmp_path, add = repo
        # audit-secrets: allow — synthetic base64, the string this test needs to be unclassified
        blob = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU2Nzg5YWI="
        add("mystery.txt", f"value = {blob}\n")
        findings = scan(tmp_path)
        assert any(f.kind == "unclassified entropy" for f in findings)

    def test_every_benign_format_states_why_it_is_benign(self):
        """An allowlist without reasons is a hole with a comment on it."""
        for pattern, why in BENIGN_ENTROPY.items():
            assert len(why) > 40, f"{pattern} is allowed without an argument"


class TestSuppression:
    """Suppression is per line, carries a reason, and is visible in the output.

    The fixtures are assembled so the written file contains a literally matching credential
    while this source does not. The first version of these tests wrote `"AccountKey=" +
    "A"*88` as file content, which puts a quote right after the equals sign — so the pattern
    never matched and every assertion passed for the wrong reason.
    """

    KEY_LINE = 'V = "AccountKey=' + "A" * 88 + '"'

    def test_the_fixture_really_is_detected_without_a_marker(self, repo):
        """The control. Without it the rest of this class could pass on an empty match."""
        tmp_path, add = repo
        add("d.py", self.KEY_LINE + "\n")
        assert scan(tmp_path), "the fixture does not contain a detectable credential"

    def test_a_marker_at_the_end_of_a_line_covers_that_line(self, repo):
        tmp_path, add = repo
        add("d.py", self.KEY_LINE + "  # audit-secrets: allow - synthetic fixture\n")
        assert not scan(tmp_path)

    def test_a_marker_on_its_own_line_covers_the_next_one(self, repo):
        """Which is what makes it usable: appending a reason to an already long line puts
        it over the line-length limit, and choosing between two lint rules is not a choice
        anyone should make to document a decision."""
        tmp_path, add = repo
        add("d.py", "# audit-secrets: allow - synthetic fixture used by a test\n"
                    + self.KEY_LINE + "\n")
        assert not scan(tmp_path)

    def test_a_marker_without_a_reason_does_not_suppress(self, repo):
        """A bare pragma is an unexplained hole."""
        tmp_path, add = repo
        add("d.py", self.KEY_LINE + "  # audit-secrets: allow\n")
        assert scan(tmp_path)

    def test_suppressions_are_counted_so_they_stay_visible(self, repo):
        tmp_path, add = repo
        add("d.py", self.KEY_LINE + "  # audit-secrets: allow - synthetic fixture\n")
        suppressed: list[str] = []
        scan(tmp_path, suppressed)
        assert suppressed == ["d.py:1"]

    def test_a_marker_does_not_leak_onto_unrelated_later_lines(self, repo):
        """Covering the next line must not become covering the rest of the file."""
        tmp_path, add = repo
        second = 'W = "AccountKey=' + "B" * 88 + '"'
        add("d.py", "# audit-secrets: allow - covers only the following line\n"
                    + self.KEY_LINE + "\n" + second + "\n")
        findings = scan(tmp_path)
        assert findings, "the uncovered line must still be reported"
        # Every finding is on line 3. Line 2 is covered; line 3 is not, and it produces two
        # findings rather than one because it is both a credential and unclassified entropy.
        assert {f.line for f in findings} == {3}


class TestTheEntropyExemption:
    """Two documents suspend the heuristic, and nothing else about the scan.

    `docs/prior/` holds the group's earlier article and thesis extracted from PDF. The
    extraction lost word boundaries, so prose reads as high-entropy strings and the
    heuristic fired 108 times on two files that contain no secrets. Loosening the rule
    repo-wide to admit them would weaken it where it works; editing the documents would
    corrupt the record they are kept for.
    """

    def _repo(self, tmp_path, relative: str, body: str):
        import subprocess

        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        for cmd in (["git", "init", "-q", str(tmp_path)],
                    ["git", "-C", str(tmp_path), "add", "-A"],
                    ["git", "-C", str(tmp_path), "-c", "user.email=t@t",
                     "-c", "user.name=t", "commit", "-qm", "t"]):
            subprocess.run(cmd, check=True, capture_output=True)
        return tmp_path

    # audit-secrets: allow — this string IS the fixture; the scanner flagging its own
    # test data is the same situation the detector patterns above carry a marker for.
    PROSE = "someconcatenatedprosethatlookshighentropyDistilGPT2here\n"
    TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"

    def test_prose_in_an_exempt_path_does_not_trip_the_heuristic(self, tmp_path):
        root = self._repo(tmp_path, "docs/prior/article.md", self.PROSE)
        assert not scan(root)

    def test_the_same_prose_elsewhere_still_does(self, tmp_path):
        """The exemption is a path, not a pardon for the pattern."""
        root = self._repo(tmp_path, "src/module.py", self.PROSE)
        assert [f.kind for f in scan(root)] == ["unclassified entropy"]

    def test_a_credential_in_an_exempt_path_is_still_caught(self, tmp_path):
        """Only the heuristic is suspended. This is the property that makes the exemption
        acceptable rather than a hole."""
        root = self._repo(tmp_path, "docs/prior/thesis.md", self.TOKEN + "\n" + self.PROSE)
        assert [f.kind for f in scan(root)] == ["credential"]

    def test_every_exemption_states_its_reason(self):
        from nas4textgen.audit_secrets import ENTROPY_EXEMPT

        assert ENTROPY_EXEMPT
        for path, why in ENTROPY_EXEMPT.items():
            assert len(why) > 80, f"{path} is exempted without an argument"

    def test_the_exemption_list_stays_short(self):
        """It is a concession, and a concession that grows is a rule being dismantled."""
        from nas4textgen.audit_secrets import ENTROPY_EXEMPT

        assert len(ENTROPY_EXEMPT) <= 3


class TestAMarkerCoversWhatItExplains:
    def test_a_multi_line_reason_reaches_the_line_below_it(self):
        """An earlier version stopped at the first non-empty line, so a two-line
        explanation covered its own second line and left the string exposed. A rule that
        makes a short reason easier to write than a good one is the wrong rule."""
        from nas4textgen.audit_secrets import _suppressed_lines

        text = (
            "# audit-secrets: allow — first line of the reason\n"
            "# and the second line, because one was not enough\n"
            "SECRET = 'value'\n"
        )
        assert 3 in _suppressed_lines(text)

    def test_a_marker_at_the_end_of_a_line_covers_that_line(self):
        from nas4textgen.audit_secrets import _suppressed_lines

        text = "SECRET = 'value'  # audit-secrets: allow — this is the detector fixture\n"
        assert _suppressed_lines(text) == {1}

    def test_a_marker_does_not_cover_the_rest_of_the_file(self):
        """It suppresses one line, not everything after it."""
        from nas4textgen.audit_secrets import _suppressed_lines

        text = (
            "# audit-secrets: allow — the line below is a deliberate fixture\n"
            "FIRST = 'covered'\n"
            "SECOND = 'not covered'\n"
        )
        covered = _suppressed_lines(text)
        assert 2 in covered
        assert 3 not in covered

    def test_a_marker_without_a_real_reason_is_not_a_marker(self):
        """Fifteen characters minimum. A suppression whose justification is `ok` or `fine`
        silences the check without recording why, which is how an exception outlives the
        situation that warranted it — and these tests were written with six-character
        reasons and failed for exactly that."""
        from nas4textgen.audit_secrets import _suppressed_lines

        assert _suppressed_lines("X = 1  # audit-secrets: allow — ok\n") == set()
        assert _suppressed_lines("X = 1  # audit-secrets: allow — fine\n") == set()
