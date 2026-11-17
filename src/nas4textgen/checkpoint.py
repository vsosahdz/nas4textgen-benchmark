"""Resume a run where it stopped, and produce the same row it would have produced.

Spot nodes are reclaimed. That is not a failure mode to tolerate but the normal condition
of the cheapest compute available, and without checkpoints every reclaim throws away the
whole run: the sweep's largest architectures take three hours, so a preemption at 2h50
costs three hours of GPU time and buys nothing.

The requirement is stronger than "start again from somewhere". A resumed run has to be the
same experiment as an uninterrupted one, or the benchmark is comparing rows produced under
different conditions — which is the class of defect this project exists to correct. So the
checkpoint captures everything that determines the trajectory, and the test suite asserts
that resuming from the middle reproduces the uninterrupted result exactly.

**What determines the trajectory.** Weights and optimiser moments are obvious. The loss
scaler is less so: under float16 it carries an adaptive scale that has usually settled by
mid-run, and restarting it at the default changes the gradients for the next several steps.
The random state matters for dropout. And the data position matters most of all — resuming
at step 900 must consume the same micro-batch the uninterrupted run would have.

**Position is addressable, not stateful.** The obvious way to feed data is a generator that
shuffles and cycles, and its position cannot be serialised — a generator's internal RNG
state is not picklable, and reconstructing it by replaying nine hundred draws is both slow
and easy to get subtly wrong. Instead the stream is a pure function of the micro-batch
index: the epoch and the offset within it are derived by division, and the epoch's
permutation is derived from the seed and the epoch number. Resuming needs one integer.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

#: Slots written alternately, so a process killed mid-write never destroys the only good
#: checkpoint. A pointer file names which slot is complete, and is written last.
SLOTS = ("a", "b")
POINTER = "checkpoint.json"


class CheckpointCorrupt(RuntimeError):
    """Raised when a checkpoint exists but does not describe the run being resumed."""


class AddressableStream:
    """Shuffled micro-batches, addressed by index rather than consumed in order.

    ``at(i)`` returns the same batch for the same ``i`` in every process, which is what
    makes resume exact. The epoch's permutation is cached because regenerating it per call
    would be a shuffle of every block on every micro-batch.
    """

    def __init__(self, blocks: list[list[int]], micro_batch: int, seed: int) -> None:
        if micro_batch <= 0:
            raise ValueError("micro-batch size must be positive")
        if len(blocks) < micro_batch:
            raise ValueError(
                f"{len(blocks)} blocks cannot fill a micro-batch of {micro_batch}"
            )
        self.blocks = blocks
        self.micro_batch = micro_batch
        self.seed = seed
        self.per_epoch = len(blocks) // micro_batch
        self._epoch: int | None = None
        self._order: list[int] = []

    def _permutation(self, epoch: int) -> list[int]:
        if self._epoch != epoch:
            order = list(range(len(self.blocks)))
            # Seeded from the epoch rather than advanced from a running generator, so any
            # epoch can be reconstructed without replaying the ones before it.
            #
            # An explicit integer rather than a tuple or a formatted string: tuples are not
            # accepted as seeds at all, and string seeding goes through a hash whose
            # internals are not something a reproducibility-critical path should depend on.
            # A large odd multiplier keeps consecutive epochs from sharing a stream.
            random.Random(self.seed * 1_000_003 + epoch).shuffle(order)
            self._epoch, self._order = epoch, order
        return self._order

    def at(self, index: int) -> list[list[int]]:
        epoch, position = divmod(index, self.per_epoch)
        order = self._permutation(epoch)
        start = position * self.micro_batch
        return [self.blocks[i] for i in order[start : start + self.micro_batch]]


@dataclass(frozen=True)
class Checkpoint:
    """Where a run had got to. Every field is something that changes what happens next."""

    step: int
    tokens: int
    micro_batch_index: int
    elapsed_seconds: float
    resumes: int
    #: Identifies the experiment. A checkpoint from a different architecture, corpus, seed
    #: or budget describes a different run and must never be loaded into this one.
    fingerprint: str

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "tokens": self.tokens,
            "micro_batch_index": self.micro_batch_index,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "resumes": self.resumes,
            "fingerprint": self.fingerprint,
        }


def run_fingerprint(genotype, corpus: str, config) -> str:
    """What makes two runs the same experiment. Mirrors the submitter's job fingerprint."""
    import hashlib

    from .space.genotype import GENES

    parts = [
        *(f"{gene}={getattr(genotype, gene)}" for gene in GENES),
        f"corpus={corpus}",
        f"seed={config.seed}",
        f"tokens={config.token_budget}",
        f"block={config.block_size}",
        f"batch_tokens={config.effective_batch_tokens}",
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def save(
    directory: Path,
    checkpoint: Checkpoint,
    model,
    optimiser,
    scaler,
) -> Path:
    """Write a checkpoint that a killed process cannot leave half-formed.

    Payload first, into whichever slot the current pointer is not using; pointer last. A
    process that dies between the two leaves the previous checkpoint still named and still
    complete, which is the only ordering that survives being killed at an arbitrary moment.
    """
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    current = _read_pointer(directory)
    slot = SLOTS[0] if (current is None or current["slot"] == SLOTS[1]) else SLOTS[1]

    payload = {
        "meta": checkpoint.as_dict(),
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        # The loss scale has usually settled by mid-run; restarting it at the default would
        # change the gradients for the next several steps.
        "scaler": scaler.state_dict() if scaler is not None else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "python_rng": random.getstate(),
    }
    torch.save(payload, directory / f"state.{slot}.pt")

    (directory / POINTER).write_text(
        json.dumps({"slot": slot, **checkpoint.as_dict()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return directory / f"state.{slot}.pt"


def _read_pointer(directory: Path) -> dict | None:
    path = Path(directory) / POINTER
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load(directory: Path, fingerprint: str, model, optimiser, scaler) -> Checkpoint | None:
    """Restore a run, or return None if there is nothing valid to restore.

    A checkpoint for a different experiment is refused rather than ignored. Loading one
    would produce a plausible result for the wrong architecture, which no downstream check
    could detect — the manifest would describe what was asked for, and the weights would be
    something else.
    """
    import torch

    pointer = _read_pointer(Path(directory))
    if pointer is None:
        return None

    if pointer.get("fingerprint") != fingerprint:
        raise CheckpointCorrupt(
            f"{directory} holds a checkpoint for {pointer.get('fingerprint')!r} and this "
            f"run is {fingerprint!r}. Loading it would train the wrong architecture and "
            "report the right one."
        )

    path = Path(directory) / f"state.{pointer['slot']}.pt"
    if not path.exists():
        # The pointer is written last, so this means the payload was removed after it —
        # not a partial write, which the ordering already prevents.
        raise CheckpointCorrupt(f"{path} named by the pointer does not exist")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimiser.load_state_dict(payload["optimiser"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])

    torch.set_rng_state(payload["torch_rng"].to(torch.uint8).cpu())
    if payload.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    random.setstate(payload["python_rng"])

    meta = payload["meta"]
    return Checkpoint(
        step=meta["step"],
        tokens=meta["tokens"],
        micro_batch_index=meta["micro_batch_index"],
        elapsed_seconds=meta["elapsed_seconds"],
        resumes=meta["resumes"] + 1,
        fingerprint=meta["fingerprint"],
    )


def clear(directory: Path) -> None:
    """Remove a completed run's checkpoints.

    Called on success. Leaving them would keep a gigabyte per architecture in the artifact
    container for a run whose result is already written, and across the sweep that is more
    storage than the benchmark itself.
    """
    directory = Path(directory)
    for name in (POINTER, *(f"state.{slot}.pt" for slot in SLOTS)):
        path = directory / name
        if path.exists():
            path.unlink()
