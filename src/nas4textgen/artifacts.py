"""Writing artifacts, so that no published number is ever typed by hand.

Every stage writes its outputs here with a provenance header. The convention exists
because the prior work's manuscript contained numbers that could not be reproduced from
its own code, and the only defence against that is for the artifact to record what
produced it.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

#: Packages whose versions are recorded with every artifact. A number produced by a
#: different resolution of these is a different number.
TRACKED_PACKAGES = ("numpy", "scipy", "pandas", "pymoo", "scikit-learn")


def artifacts_root() -> Path:
    """Where artifacts land. Overridable so tests never write into the real tree."""
    return Path(os.environ.get("NAS4TG_ARTIFACTS", "./artifacts")).resolve()


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _git_is_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def provenance() -> dict[str, Any]:
    """What is needed to know whether two artifacts are comparable.

    ``git_dirty`` matters: an artifact produced from uncommitted work cannot be
    reproduced from the recorded revision, and recording the fact is better than
    discovering it later.
    """
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "absent"

    determinism = {
        var: os.environ.get(var)
        for var in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    }
    return {
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": _git_revision(),
        "git_dirty": _git_is_dirty(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "determinism_environment": determinism,
        "seed": os.environ.get("NAS4TG_SEED"),
        # The command, with the interpreter's absolute path reduced to the module it
        # ran. `sys.argv[0]` is where this checkout happens to live, which carries the
        # author's home directory into every artifact, differs between their machine and
        # a reader's by construction, and tells nobody anything about the measurement.
        # The project says as much about paths elsewhere and recorded them here anyway.
        "command": " ".join(["nas4textgen", *sys.argv[1:]]),
    }


def write_json(relative_path: str, payload: dict[str, Any]) -> Path:
    """Write a JSON artifact with its provenance header."""
    target = artifacts_root() / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {"provenance": provenance(), **payload}
    target.write_text(
        json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return target


def write_table(relative_path: str, rows: list[dict[str, Any]], columns: list[str]) -> Path:
    """Write a CSV artifact, with provenance alongside it as a sidecar.

    CSV rather than JSON for the tables meant to be consumed by others: the benchmark
    table should open in anything, including a spreadsheet, without a parser.
    """
    target = artifacts_root() / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in columns))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(relative_path + ".provenance.json", {"rows": len(rows), "columns": columns})
    return target
