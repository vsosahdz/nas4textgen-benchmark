"""The architectures that gate every fan-out.

The canary exists because the expensive failures in a sweep are operational, not
statistical: an image that will not pull, a corpus path that is wrong, a manifest missing a
field, a node that never releases. Those fail identically on four jobs and on a thousand,
and only one of those costs a thousand jobs to discover.

The members are chosen by enumeration, not written by hand. The extremes of the space are
whatever the gene levels and the ceiling actually produce, and hand-writing the genotype
one believes to be largest is how a canary ends up not covering the case it exists for.

**Why four rather than three.** The obvious reading of "smallest, median, largest" takes
largest to mean largest parameter count, and in this space that is the wrong architecture
to bound anything with. Parameters here are dominated by the embedding matrix — 50,257
rows by e — so the parameter-largest model is simply the widest one, and width is cheap:
it has four layers, runs at 67% of the compute and holds 33% of the activation memory of
the heaviest architecture under the ceiling. A canary that set the job timeout and the
memory bound from it would pass, and the sweep would then meet a twenty-four-layer
architecture that runs half again as long and needs three times the activation memory.

So the parameter extreme is kept, because it is an extreme of an objective, and a fourth
member is added for the extreme that actually binds wall clock and memory. The two are
different architectures, and one measurement cannot stand for both.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..space.cost import (
    activation_width,
    forward_flops_per_token,
    kv_cache_bytes,
    parameter_count,
)
from ..space.enumerate import enumerate_with_ceiling
from ..space.genotype import GENES, Genotype


@dataclass(frozen=True)
class CanaryMember:
    role: str  # smallest | median | largest_params | heaviest
    genotype: Genotype
    params: int
    kv_bytes: int
    flops_per_token: int
    activation_width: int
    why: str
    bounds: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "genotype": {gene: getattr(self.genotype, gene) for gene in GENES},
            "e": self.genotype.e,
            "params": self.params,
            "kv_bytes": self.kv_bytes,
            "flops_per_token": self.flops_per_token,
            "activation_width": self.activation_width,
            "why": self.why,
            "bounds": list(self.bounds),
        }


def _member(role: str, g: Genotype, why: str, bounds: tuple[str, ...]) -> CanaryMember:
    return CanaryMember(
        role=role,
        genotype=g,
        params=parameter_count(g),
        kv_bytes=kv_cache_bytes(g),
        flops_per_token=forward_flops_per_token(g),
        activation_width=activation_width(g),
        why=why,
        bounds=bounds,
    )


def select_canary(ceiling: int | None = None) -> list[CanaryMember]:
    """Pick the canary set by enumerating the space and taking its real extremes."""
    enumeration = (
        enumerate_with_ceiling(ceiling=ceiling) if ceiling else enumerate_with_ceiling()
    )
    space = enumeration.genotypes
    by_params = sorted(space, key=parameter_count)

    return [
        _member(
            "smallest", by_params[0],
            "the cheapest run under the ceiling; proves the path end to end for the least money",
            ("none — this one is a smoke test of the platform, not a bound",),
        ),
        _member(
            "median", by_params[len(by_params) // 2],
            "the cost figure the sweep projection is revised against",
            ("projected sweep cost", "projected sweep wall clock"),
        ),
        _member(
            "largest_params", by_params[-1],
            "the extreme of the parameter objective, and the widest model the ceiling allows",
            ("parameter objective range", "optimiser and gradient memory"),
        ),
        _member(
            # Ties on FLOPs are broken by activation width, so when several architectures
            # cost the same to run, the canary takes the one that is hardest to fit.
            "heaviest", max(space, key=lambda g: (forward_flops_per_token(g), activation_width(g))),
            "most compute and widest activations under the ceiling; the only place the "
            "duration and memory bounds can break",
            ("per-job timeout", "peak activation memory", "preemption-loss bound"),
        ),
    ]


def timeout_source(members: list[CanaryMember]) -> CanaryMember:
    """The member whose measured wall clock sets the limit every sweep job carries.

    Named rather than assumed. Reading the timeout off whichever canary job happened to run
    longest would silently reintroduce the error this module exists to document, on any day
    the heaviest job was scheduled on a faster node.
    """
    return next(m for m in members if m.role == "heaviest")
