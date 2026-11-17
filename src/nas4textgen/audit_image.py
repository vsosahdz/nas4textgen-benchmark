"""Check that the image's installed packages are the ones the committed lock names.

The image and the lock are two artefacts that are supposed to describe the same
environment, and nothing about building the image enforces that. A ``pip install`` that
resolved a different version, a layer restored from a stale cache, a hand-fixed package
that was never written back — each leaves an image that runs and produces numbers, and no
step downstream would notice. The manifest would even record the lock's hash, truthfully
describing a file that no longer describes the container.

That is worse than a broken image. A broken image fails; a drifted one publishes.

Run inside the container::

    python -m nas4textgen.audit_image

Exits non-zero on any mismatch. It compares what is installed against the lock the image
carries at /workspace/requirements.lock, and against the lock's recorded hash, so it also
catches an image that shipped a different lock file than the one it installed from.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

#: Where the Dockerfile places the lock and its hash.
IMAGE_LOCK = Path("/workspace/requirements.lock")
IMAGE_LOCK_HASH = Path("/workspace/requirements.lock.sha256")

#: Distributions installed by the image build itself rather than by the lock. Excluded
#: because their presence is expected and their versions are not pinned anywhere.
BUILD_TOOLING = {"pip", "setuptools", "wheel", "nas4textgen"}

_REQUIREMENT = re.compile(r"^([A-Za-z0-9._-]+)\s*==\s*([^\s;#]+)")


def parse_lock(text: str) -> dict[str, str]:
    """Pinned distributions from a compiled requirements file, normalised for comparison."""
    pinned: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-")):
            continue
        match = _REQUIREMENT.match(stripped)
        if match:
            name, version = match.groups()
            pinned[canonical(name)] = version
    return pinned


def canonical(name: str) -> str:
    """PEP 503 normalisation. ``Pillow``, ``pillow`` and ``PIL-low`` must not disagree."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _install_roots() -> tuple[str, ...]:
    """Where pip installs for *this* interpreter.

    Asking sysconfig rather than matching a path by name. The first attempt excluded
    anything under "dist-packages", which on Debian is where pip installs too — Debian
    patches the interpreter to use that name for both — and the check reported zero
    installed distributions while claiming the image matched nothing.

    The real split in the CUDA image is between /usr/lib/python3/dist-packages, holding
    nineteen packages apt pulled in as dependencies of the tooling that adds Python 3.12,
    and /usr/local/lib/python3.12/dist-packages, holding the 128 the lock installed. Only
    the second is purelib. Asking sysconfig also makes this work unchanged in a venv, on
    macOS and on Windows, where the names differ again.
    """
    import sysconfig

    paths = sysconfig.get_paths()
    return tuple({paths["purelib"], paths["platlib"]})


def installed_distributions() -> dict[str, str]:
    """Distributions pip put here, excluding the ones the operating system did.

    Operating-system packages are not pip's to pin and no lock will ever name them, so
    counting them as drift would leave this check permanently red and therefore ignored.
    """
    from importlib.metadata import distributions

    roots = _install_roots()
    found: dict[str, str] = {}
    for dist in distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        location = str(dist.locate_file(""))
        if not location.startswith(roots):
            continue
        found[canonical(name)] = dist.version
    return found


def audit(lock_text: str, installed: dict[str, str]) -> list[str]:
    """Return every disagreement between the lock and what is installed."""
    pinned = parse_lock(lock_text)
    problems: list[str] = []

    for name, version in sorted(pinned.items()):
        actual = installed.get(name)
        if actual is None:
            problems.append(f"{name} {version} is locked but not installed")
        elif actual != version:
            problems.append(f"{name} is {actual}, the lock pins {version}")

    # The other direction matters too. A package present but unlocked entered the image
    # some way the lock does not describe, so the lock no longer reproduces the image even
    # though every pinned version matches.
    for name, version in sorted(installed.items()):
        if name not in pinned and name not in BUILD_TOOLING:
            problems.append(f"{name} {version} is installed but absent from the lock")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nas4textgen.audit_image", description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        action="append",
        help=(
            "Repeatable. The image installs from exactly one lock, so one is the norm. An "
            "environment assembled from several — CI installs the analysis and dev tiers "
            "together — must name all of them, because a package pinned in a lock that was "
            "not given looks identical to a package that arrived from nowhere."
        ),
    )
    parser.add_argument("--lock-hash", type=Path, default=IMAGE_LOCK_HASH)
    args = parser.parse_args(argv)

    locks = args.lock or [IMAGE_LOCK]
    missing = [str(path) for path in locks if not path.exists()]
    if missing:
        print(
            f"no lock at {', '.join(missing)}. Inside the training image this defaults to "
            "the lock the Dockerfile installed from; elsewhere, name the locks the "
            "environment was built from.",
            file=sys.stderr,
        )
        return 2

    lock_text = "\n".join(path.read_text(encoding="utf-8") for path in locks)

    # The lock the image carries must be the lock the image was built from. Comparing the
    # hash the build recorded catches an image that shipped one file and installed another
    # — in which case every version below could match and still mean nothing.
    if len(locks) == 1 and args.lock_hash.exists():
        recorded = args.lock_hash.read_text(encoding="utf-8").strip()
        actual = hashlib.sha256(locks[0].read_bytes()).hexdigest()
        if recorded != actual:
            print(
                f"the lock in the image hashes to {actual[:12]} but the build recorded "
                f"{recorded[:12]}. The file was replaced after installation, so nothing "
                "below can be trusted.",
                file=sys.stderr,
            )
            return 1

    problems = audit(lock_text, installed_distributions())

    print()
    for path in locks:
        print(f"  lock       {path}")
    print(f"  pinned     {len(parse_lock(lock_text))} distributions")
    print(f"  installed  {len(installed_distributions())} distributions")
    print()
    if problems:
        for problem in problems:
            print(f"  DRIFT  {problem}")
        print()
        print(f"  {len(problems)} disagreement(s). The image does not match its lock, so the")
        print("  lock hash a manifest records describes a file, not this container.")
        return 1

    print("  The image's packages are exactly the ones the committed lock names.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
