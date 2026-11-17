"""Resolve the workspace from Terraform's outputs, never from typed-in names.

The platform's identifiers already exist in exactly one authoritative place: the Terraform
state that created them. Re-declaring them in a config file or a flag creates a second
source of truth that is correct until the day it is not, and the failure mode — submitting
a wave into a workspace that is not the one under study — produces results that look fine
and belong to nothing.

No subscription identifier is committed. It is read from the active Azure CLI login at run
time, and asserted against the subscription the project declares by name.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

#: Where Terraform runs. Outputs are read from here rather than from a copied-out file, so
#: they cannot go stale relative to the infrastructure they describe.
INFRA_DIR = Path(__file__).resolve().parents[3] / "infra"


class PlatformNotProvisioned(RuntimeError):
    """Raised when Terraform has no state, so there is nothing to submit against."""


class WrongSubscription(RuntimeError):
    """Raised when the active Azure login is not the subscription this project declares."""


class ImageIsStale(RuntimeError):
    """Raised when the published image was built from a different source revision."""


#: Modules a training job actually loads, measured by running one rather than inferred.
#:
#: Inferring it from static imports is not enough: training.py imports nas4textgen.model
#: inside a function, so the module that defines the architecture never appears in the
#: import closure of train.py at load time. This list came from executing a smoke run and
#: reading sys.modules, and a test re-measures it so that it cannot drift.
TRAINING_MODULES = (
    "nas4textgen",
    "nas4textgen.artifacts",
    "nas4textgen.checkpoint",
    "nas4textgen.corpora",
    "nas4textgen.corpora.cache",
    "nas4textgen.corpora.registry",
    "nas4textgen.corpora.tokenise",
    "nas4textgen.evaluation",
    "nas4textgen.evaluation.perplexity",
    "nas4textgen.model",
    "nas4textgen.space",
    "nas4textgen.space.cost",
    "nas4textgen.space.genotype",
    "nas4textgen.train",
    "nas4textgen.training",
    # Not loaded by a job, but run by the Dockerfile at build time, so a change to either
    # changes what the image asserts about itself.
    "nas4textgen.audit_image",
    "nas4textgen.check_env",
)

#: Non-module inputs: the dependency set, the package metadata, and the recipe.
IMAGE_FILES = ("pyproject.toml", "requirements", "infra/Dockerfile")


def _module_path(root: Path, module: str) -> Path:
    parts = module.split(".")
    package = root / "src" / Path(*parts) / "__init__.py"
    return package if package.exists() else root / "src" / Path(*parts[:-1]) / f"{parts[-1]}.py"


def image_input_hash(root: Path | None = None) -> str:
    """Content hash of what the image actually runs.

    Compared instead of the repository revision, and narrower than all of src/. Two earlier
    versions were each wrong in the same direction — too broad — and a guard that fires on
    changes it does not care about is a guard that gets worked around.

    Comparing HEAD refused a sweep wave for commits touching only tests. Hashing all of
    src/ then refused two more for edits to the analysis modules — the surrogate slate, the
    characterisation — which no training job imports and which sit in the image only because
    the package is copied whole. What is hashed now is the measured runtime closure of a
    training run plus what the Dockerfile executes.
    """
    import hashlib

    root = root or Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()

    for module in sorted(TRAINING_MODULES):
        path = _module_path(root, module)
        if path.exists():
            digest.update(module.encode())
            digest.update(path.read_bytes())

    for entry in IMAGE_FILES:
        target = root / entry
        candidates = sorted(target.rglob("*")) if target.is_dir() else [target]
        for path in candidates:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())

    return digest.hexdigest()[:16]


def require_current_image(record: Path | None = None) -> str:
    """Refuse to submit an image whose contents differ from what is checked out.

    Sweep wave one went out on an image built before a fix to the memory probe, and the
    pre-flight passed: it verified that a digest was recorded, which is a different question
    from whether the digest contains the current code. The lock hash does not cover it
    either — dependencies are pinned, so two images built from different source share a tag
    and a lock hash and differ only in the part under test.

    What is compared is a hash of the image's inputs rather than the repository revision.
    Comparing revisions refused wave two for commits that touched only tests and task files,
    and a guard that fires on changes it does not care about gets bypassed.
    """
    import json as _json

    record = record or Path(
        os.environ.get("NAS4TG_IMAGE_RECORD", "artifacts/image/published.json")
    )
    if not record.exists():
        raise ImageIsStale(f"no published image record at {record}")

    published = _json.loads(record.read_text(encoding="utf-8"))
    recorded = published.get("input_hash")
    if not recorded:
        raise ImageIsStale(
            f"{record} records no hash of the image's inputs. Rebuild with "
            "./scripts/build_image.sh --push; an image whose contents nobody can name is "
            "an image whose results nobody can reproduce."
        )

    current = image_input_hash()
    if recorded != current:
        raise ImageIsStale(
            f"the image was built from inputs hashing to {recorded} and this checkout "
            f"hashes to {current}.\n"
            "A module the training job loads, or the dependency set, or the Dockerfile has "
            "changed since, so submitting would run code that is not the code here. "
            "Rebuild with ./scripts/build_image.sh --push."
        )
    return recorded


@dataclass(frozen=True)
class Platform:
    """Everything submission needs, as created."""

    workspace: str
    resource_group: str
    location: str
    sweep_cluster: str
    measure_cluster: str
    storage_account: str
    cache_container: str
    artifact_container: str
    registry: str
    subscription_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "workspace": self.workspace,
            "resource_group": self.resource_group,
            "location": self.location,
            "sweep_cluster": self.sweep_cluster,
            "measure_cluster": self.measure_cluster,
            "storage_account": self.storage_account,
            "cache_container": self.cache_container,
            "artifact_container": self.artifact_container,
            "registry": self.registry,
        }


def _terraform_output() -> dict:
    if not (INFRA_DIR / ".terraform").exists():
        raise PlatformNotProvisioned(
            f"{INFRA_DIR} has not been initialised. Run ./scripts/infra.sh apply first; "
            "submission reads the platform's identifiers from Terraform's outputs rather "
            "than from anything typed in twice."
        )
    result = subprocess.run(
        ["terraform", f"-chdir={INFRA_DIR}", "output", "-json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PlatformNotProvisioned(
            f"terraform output failed: {result.stderr.strip()[:400]}"
        )
    outputs = json.loads(result.stdout or "{}")
    if "workspace" not in outputs:
        raise PlatformNotProvisioned(
            "Terraform holds no workspace output, so the platform does not exist yet. "
            "Run ./scripts/infra.sh apply."
        )
    return outputs


def _active_subscription() -> tuple[str, str]:
    """The subscription the CLI is logged into, as (id, name)."""
    result = subprocess.run(
        ["az", "account", "show", "-o", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise WrongSubscription(
            "no active Azure login. Run `az login` and select the subscription this "
            "project declares in infra/target.tfvars."
        )
    account = json.loads(result.stdout)
    return account["id"], account["name"]


def _declared_subscription_name() -> str | None:
    """The subscription name pinned in target.tfvars. Names, never identifiers."""
    tfvars = INFRA_DIR / "target.tfvars"
    if not tfvars.exists():
        return None
    for line in tfvars.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("subscription_name"):
            return stripped.split("=", 1)[1].strip().strip('"')
    return None


@lru_cache(maxsize=1)
def resolve_platform() -> Platform:
    """Read the provisioned platform, asserting the login points at it.

    The subscription check is not ceremony. This project shares a tenant with unrelated
    subscriptions, and the default login is not the one that holds the grant paying for
    these GPUs. Submitting a wave into the wrong one spends someone else's budget and
    produces results in a workspace nobody is looking at.
    """
    outputs = _terraform_output()
    subscription_id, subscription_name = _active_subscription()

    declared = _declared_subscription_name()
    if declared and subscription_name != declared:
        raise WrongSubscription(
            f"the active subscription is {subscription_name!r}, but this project declares "
            f"{declared!r} in infra/target.tfvars.\n"
            f"Switch with: az account set --subscription {declared!r}"
        )

    workspace = outputs["workspace"]["value"]
    storage = outputs["storage"]["value"]
    return Platform(
        workspace=workspace["name"],
        resource_group=workspace["resource_group"],
        location=workspace["location"],
        sweep_cluster=workspace["sweep_cluster"],
        measure_cluster=workspace["measure_cluster"],
        storage_account=storage["account"],
        cache_container=storage["cache_container"],
        artifact_container=storage["artifact_container"],
        registry=outputs["container_registry"]["value"],
        subscription_id=subscription_id,
    )


def ml_client(platform: Platform | None = None):
    """An authenticated Azure ML client.

    Credentials come from the environment — the CLI login locally, a managed identity in
    CI. No secret is read from a file this repository can see, which is what keeps the
    replication package publishable.
    """
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    platform = platform or resolve_platform()
    return MLClient(
        credential=DefaultAzureCredential(exclude_interactive_browser_credential=False),
        subscription_id=platform.subscription_id,
        resource_group_name=platform.resource_group,
        workspace_name=platform.workspace,
    )


def image_reference() -> str:
    """The training image pinned by digest, from the record the build wrote.

    A tag is a moving pointer. Two jobs submitted a week apart against ``:latest`` — or
    against any tag someone re-pushed — ran different code, and no manifest would say so.
    The digest is what goes into every job and every manifest.
    """
    record = Path(os.environ.get("NAS4TG_IMAGE_RECORD", "artifacts/image/published.json"))
    if not record.exists():
        raise PlatformNotProvisioned(
            f"no published image record at {record}. Build and push it with "
            "./scripts/build_image.sh; jobs run a digest, never a tag."
        )
    published = json.loads(record.read_text(encoding="utf-8"))
    registry = published.get("registry")
    repository = published.get("repository")
    digest = published.get("digest")
    if not digest or digest == "unknown" or not registry or not repository:
        raise PlatformNotProvisioned(
            f"{record} records no digest, so the image was built but never pushed — or the "
            "push did not report one. Re-run ./scripts/build_image.sh --push."
        )
    return f"{registry}/{repository}@{digest}"
