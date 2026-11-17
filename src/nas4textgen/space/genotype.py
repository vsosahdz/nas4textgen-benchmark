"""The architecture genotype: schema, validity, encoding, and repair.

The space is anchored on DistilGPT2 (design.md D1). Five genes, all categorical or
ordinal. The model dimension is **derived**, not searched::

    e = h * d_head

That single choice removes three defects of the prior encoding: divisibility of ``e``
by ``h`` holds by construction rather than through a repair operator, structural
genes cannot take fractional values, and the genotype-to-phenotype map is injective.
The prior work evaluated its cost objective on unrepaired continuous values and
reported architectures with ``e = 8.0017``; the 87-parameter difference between two
"distinct" architectures in the thesis is ``50257 * 0.0017``, float noise in the
objective rather than a design difference.

``d_head >= 32`` eliminates the degenerate regime the prior search collapsed into,
where ``e = 8`` with heads of dimension 1 and the token embedding was 98.6% of the
parameter count.

``ffn`` is an **integer** multiplier. A ratio of 8/3 was considered and rejected:
``int(8/3 * 768)`` is 2047 rather than 2048 in float64, and rounding the intermediate
dimension to a multiple of 64 makes 8/3 and 3 collide on the same value, which breaks
injectivity. Integer multipliers need no rounding and cannot collide. 8/3 becomes
meaningful only if a SwiGLU MLP is introduced, since it exists to parameter-match a
three-matrix MLP against a four-times two-matrix one.

``kv`` is the number of key/value heads (grouped-query attention). It must divide
``h``. It is the gene that makes KV-cache memory depend on the head configuration,
and it is the source of the objective conflict that justifies the multi-objective
formulation: at a matched parameter count, KV-cache memory varies by up to 192x.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

#: Fixed configuration. The vocabulary is not searched — measurement showed the
#: objective conflict is identical with or without it, so fixing it keeps perplexity
#: comparable across the space and avoids retraining tokenizers (design.md D3).
VOCAB_SIZE = 50_257

#: Size of the learned positional embedding table, from the GPT-2 block family.
MAX_POSITION = 1_024

#: Context length at which KV-cache figures are reported. Recorded with every
#: KV-cache number, because the quantity is meaningless without it.
REFERENCE_CONTEXT = 1_024

#: Allowed values per gene. Order is significant: encoding maps a gene to its index
#: in this tuple, so reordering a tuple changes every encoded vector.
LEVELS: dict[str, tuple[int, ...]] = {
    "h": (2, 4, 6, 8, 12, 16),
    "d_head": (32, 64, 128),
    "n": (2, 4, 6, 8, 12, 16, 24),
    "ffn": (2, 3, 4),
    # ``kv = 1`` is multi-query attention: a single key/value head shared by every
    # query head, and the extreme point of the memory axis. It is a real deployment
    # configuration, and excluding it truncated the front at an arbitrary place —
    # including it takes the space from 1,008 architectures to 1,386, doubles the
    # KV-cache spread within a parameter band from 96x to 192x, and improves the
    # parameters/memory rank correlation from +0.672 to +0.615.
    # ``kv = 3`` was measured and rejected: it grows the space to 1,512 and leaves the
    # spread at 192x, so it widens nothing beyond what MQA already provides.
    "kv": (1, 2, 4, 6, 8, 12, 16),
}

GENES: tuple[str, ...] = tuple(LEVELS)


def valid_kv(h: int) -> tuple[int, ...]:
    """KV head counts admissible for ``h`` query heads.

    Grouped-query attention requires that the query heads partition evenly into
    key/value groups, so ``kv`` must divide ``h`` and cannot exceed it. ``kv == h``
    is ordinary multi-head attention.
    """
    return tuple(k for k in LEVELS["kv"] if h % k == 0 and k <= h)


@dataclass(frozen=True, slots=True)
class Genotype:
    """One architecture. Immutable, hashable, and comparable by value."""

    h: int
    d_head: int
    n: int
    ffn: int
    kv: int

    @property
    def e(self) -> int:
        """Model dimension. Derived, never searched."""
        return self.h * self.d_head

    @property
    def intermediate(self) -> int:
        """Feed-forward inner dimension. Exact: ``ffn`` is an integer."""
        return self.ffn * self.e

    @property
    def kv_width(self) -> int:
        """Total width of the key and value projections under GQA."""
        return self.kv * self.d_head

    def is_valid(self) -> bool:
        return (
            all(getattr(self, g) in LEVELS[g] for g in GENES)
            and self.h % self.kv == 0
            and self.kv <= self.h
        )

    def as_tuple(self) -> tuple[int, ...]:
        return tuple(getattr(self, g) for g in GENES)

    def __str__(self) -> str:
        return (
            f"h={self.h} d_head={self.d_head} n={self.n} "
            f"ffn={self.ffn} kv={self.kv} (e={self.e})"
        )


#: The DistilGPT2 anchor, expressed in this space. Its parameter count must come out
#: at exactly 81,912,576 — the real checkpoint's count — which is the exactness test
#: for the cost model.
DISTILGPT2 = Genotype(h=12, d_head=64, n=6, ffn=4, kv=12)


def enumerate_space() -> Iterator[Genotype]:
    """Every valid genotype, in a deterministic order.

    The order is fixed by ``LEVELS`` so that enumeration is reproducible: the
    verification tier asserts that re-running this yields a bit-identical sequence.
    """
    for h, d_head, n, ffn in itertools.product(
        LEVELS["h"], LEVELS["d_head"], LEVELS["n"], LEVELS["ffn"]
    ):
        for kv in valid_kv(h):
            yield Genotype(h=h, d_head=d_head, n=n, ffn=ffn, kv=kv)


def encode(g: Genotype) -> np.ndarray:
    """Map a genotype to ``[0, 1]^5`` for continuous variation operators.

    Each gene becomes its level index scaled to the unit interval. Exactly
    invertible by :func:`decode` for any valid genotype.
    """
    out = np.empty(len(GENES), dtype=np.float64)
    for i, gene in enumerate(GENES):
        levels = LEVELS[gene]
        out[i] = levels.index(getattr(g, gene)) / (len(levels) - 1)
    return out


def decode(vector: np.ndarray) -> Genotype:
    """Map a point in ``[0, 1]^5`` back to a valid genotype.

    Values are clipped, snapped to the nearest level index, and then repaired so
    that ``kv`` divides ``h``. Repair is the only place a decoded genotype may differ
    from a naive nearest-level reading, and it is idempotent on valid genotypes —
    which is what makes ``decode(encode(g)) == g`` hold.
    """
    values: dict[str, int] = {}
    for i, gene in enumerate(GENES):
        levels = LEVELS[gene]
        idx = int(round(float(np.clip(vector[i], 0.0, 1.0)) * (len(levels) - 1)))
        values[gene] = levels[idx]
    return repair(Genotype(**values))


def repair(g: Genotype) -> Genotype:
    """Return the nearest valid genotype.

    Only ``kv`` can be invalid after decoding, because it is the one gene whose
    admissible set depends on another gene. It is snapped to the admissible divisor
    of ``h`` closest to the requested value, preferring the smaller on a tie so that
    repair never silently increases memory cost.
    """
    if g.is_valid():
        return g
    admissible = valid_kv(g.h)
    kv = min(admissible, key=lambda k: (abs(k - g.kv), k))
    return Genotype(h=g.h, d_head=g.d_head, n=g.n, ffn=g.ffn, kv=kv)
