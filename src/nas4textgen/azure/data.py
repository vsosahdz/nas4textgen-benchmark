"""Put the cached corpora where jobs can read them, and check they arrived intact.

A node has no network beyond Azure by design: the corpus it trains on must be the corpus
that was measured, not whatever the upstream dataset looks like today. So the cache built
locally is uploaded once and mounted read-only into every job.

The upload is verified against the same hashes the local cache carries. An upload that
silently truncated a half-gigabyte corpus would produce a sweep of perfectly consistent,
perfectly wrong numbers, and the only moment it is cheap to catch that is here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..corpora.cache import cache_root
from .workspace import Platform, resolve_platform

#: Datastore names jobs address. A job cannot be handed a plain https:// URL: Azure ML
#: reads that scheme as public storage, assumes it is read-only, and rejects the job before
#: allocating a node. Inputs and outputs must be azureml:// datastore paths.
CACHE_DATASTORE = "corpus_cache"
ARTIFACT_DATASTORE = "run_artifacts"


class UploadIncomplete(RuntimeError):
    """Raised when what landed in storage does not match what was cached locally."""


def ensure_datastores(platform: Platform | None = None) -> list[tuple[str, str]]:
    """Register both containers with the workspace, without a stored credential.

    Not in Terraform, and the reason is worth recording: the azurerm provider requires an
    account key or a SAS token on a blob datastore and offers no keyless option. Declaring
    them there would put a storage key into Terraform state and into the datastore itself,
    trading an identity that can be revoked for a secret that has to be stored and rotated.
    Passing no credentials here selects identity-based access, which the cluster role
    assignments in access.tf are what make work.
    """
    from azure.ai.ml.entities import AzureBlobDatastore

    from .workspace import ml_client

    platform = platform or resolve_platform()
    client = ml_client(platform)

    wanted = [
        (CACHE_DATASTORE, platform.cache_container,
         "The data of record, mounted read-only into every job. Nothing downstream reaches "
         "the network, so a run cannot train on content that moved since the sweep began."),
        (ARTIFACT_DATASTORE, platform.artifact_container,
         "Where every run writes its manifest and measurements, under its own fingerprint. "
         "Results must outlive the node, because spot nodes are reclaimed."),
    ]

    registered = []
    for name, container, description in wanted:
        client.datastores.create_or_update(
            AzureBlobDatastore(
                name=name,
                account_name=platform.storage_account,
                container_name=container,
                description=description,
                # No credentials: identity-based access.
                credentials=None,
            )
        )
        registered.append((name, container))
    return registered


@dataclass(frozen=True)
class UploadedSplit:
    corpus: str
    split: str
    blob: str
    bytes_uploaded: int
    sha256: str
    already_present: bool


def _blob_service(platform: Platform):
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    return BlobServiceClient(
        account_url=f"https://{platform.storage_account}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    )


#: Where run manifests are kept once fetched, one JSON object per line.
MANIFEST_CACHE = "manifests/runs.jsonl"


def sync_manifests(destination: Path, platform: Platform | None = None,
                   progress_every: int = 250) -> tuple[int, int]:
    """Mirror run manifests from the artifact container into a local file.

    Returns ``(fetched, total)``. Incremental: a manifest already in the cache is not
    downloaded again, so a sync after a new wave costs only the new runs.

    Written because every command that reads manifests was listing and downloading all of
    them from the container on each invocation. At 3,341 runs that is ten minutes per
    command, which is the kind of cost that quietly decides which analyses get done. It also
    makes the analysis reproducible without the subscription: the cache is a file, and a
    reader who is given it can re-derive every table that reads manifests.
    """
    platform = platform or resolve_platform()
    target = destination / MANIFEST_CACHE
    target.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    if target.exists():
        with target.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    seen.add(json.loads(line)["_blob"])
                except (json.JSONDecodeError, KeyError):
                    continue

    container = _blob_service(platform).get_container_client(
        platform.artifact_container)
    fetched = 0
    total = len(seen)
    with target.open("a", encoding="utf-8") as handle:
        for blob in container.list_blobs(name_starts_with="runs/"):
            if not blob.name.endswith("manifest.json") or blob.name in seen:
                continue
            try:
                payload = json.loads(
                    container.get_blob_client(blob.name).download_blob().readall())
            except Exception:  # noqa: BLE001
                # A manifest that cannot be read is a run that did not finish writing one.
                # Skipping it is correct; counting it as present would overstate coverage.
                continue
            payload["_blob"] = blob.name
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fetched += 1
            total += 1
            if progress_every and fetched % progress_every == 0:
                print(f"    {fetched:,} fetched", flush=True)
    return fetched, total


def load_manifests(destination: Path) -> list[dict]:
    """Every cached manifest. Raises if the cache has never been synced."""
    target = destination / MANIFEST_CACHE
    if not target.exists():
        raise FileNotFoundError(
            f"no manifest cache at {target}. Run: python -m nas4textgen.cli manifests"
        )
    manifests = []
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                manifests.append(json.loads(line))
    return manifests


def _cache_files() -> list[tuple[str, Path]]:
    """Every file in the local cache, with the blob name it takes in storage.

    A whole-directory mirror rather than a list of known split names. The cache now holds
    two kinds of file per split — the text and its pre-computed token ids — and enumerating
    the kinds here would mean this function needs editing every time the cache gains one,
    with the failure being a file that quietly never reaches the cluster.
    """
    root = cache_root()
    return sorted(
        (str(path.relative_to(root)).replace("\\", "/"), path)
        for path in root.rglob("*")
        if path.is_file()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_cache(
    platform: Platform | None = None,
    force: bool = False,
) -> list[UploadedSplit]:
    """Mirror the local cache into the cache container, skipping what is already there.

    Skipping is by content hash, stored as blob metadata when the file is written. A blob
    with the right name and size and the wrong content is the failure this guards against,
    and it is the only one that would not announce itself.
    """
    platform = platform or resolve_platform()
    container = _blob_service(platform).get_container_client(platform.cache_container)

    uploaded: list[UploadedSplit] = []
    for relative, path in _cache_files():
        blob_name = f"corpora/{relative}"
        blob = container.get_blob_client(blob_name)
        local_hash = _sha256(path)

        present = False
        if not force and blob.exists():
            properties = blob.get_blob_properties()
            present = (
                properties.metadata.get("sha256") == local_hash
                and properties.size == path.stat().st_size
            )

        if not present:
            with path.open("rb") as handle:
                container.upload_blob(
                    blob_name,
                    handle,
                    overwrite=True,
                    max_concurrency=4,
                    metadata={"sha256": local_hash},
                )

        corpus, _, name = relative.partition("/")
        uploaded.append(
            UploadedSplit(
                corpus=corpus,
                split=name,
                blob=blob_name,
                bytes_uploaded=path.stat().st_size,
                sha256=local_hash,
                already_present=present,
            )
        )
    return uploaded


def verify_uploaded(platform: Platform | None = None) -> list[str]:
    """Confirm every local file has a byte-identical remote one. Returns the mismatches."""
    platform = platform or resolve_platform()
    container = _blob_service(platform).get_container_client(platform.cache_container)

    problems: list[str] = []
    for relative, path in _cache_files():
        blob = container.get_blob_client(f"corpora/{relative}")
        if not blob.exists():
            problems.append(f"{relative}: not present in {platform.cache_container}")
            continue

        properties = blob.get_blob_properties()
        # Size and hash. The stored hash is a claim made at upload time; the size is what
        # the account actually holds, and a truncated upload keeps a correct-looking claim.
        if properties.size != path.stat().st_size:
            problems.append(
                f"{relative}: {properties.size:,} bytes stored against "
                f"{path.stat().st_size:,} local — the upload was truncated"
            )
        elif properties.metadata.get("sha256") != _sha256(path):
            problems.append(
                f"{relative}: content differs from the local file — the uploaded corpus "
                "is not the measured one"
            )
    return problems


def require_uploaded(platform: Platform | None = None) -> None:
    """Refuse to submit against a cache that does not match the one measured locally."""
    problems = verify_uploaded(platform)
    if problems:
        raise UploadIncomplete(
            "the remote corpus cache does not match the local one:\n  "
            + "\n  ".join(problems)
            + "\nRe-run the upload. A sweep on the wrong corpus is internally consistent "
            "and externally meaningless."
        )
