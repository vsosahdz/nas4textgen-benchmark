"""Assemble the benchmark table from what the runs wrote, and take its exact front.

This is the artefact the paper publishes and the second paper searches over, so what it
refuses to include matters as much as what it contains.

**Only rows that met their budget.** A truncated run measured something real, and it is not
the same thing: it saw fewer tokens, so its perplexity answers a different question. Placing
it beside a complete row would reintroduce the prior work's central defect — unequal
training presented as an architecture comparison — in the one artefact everything else
depends on.

**Only rows measured alike.** Precision, corpus vintage, token budget and tokeniser have to
agree, and the check is not hypothetical: the canary produced three rows in float16 and one
in emulated bfloat16, all complete, all valid, with nothing in the perplexities to say so.

**The front is labelled measured.** With every architecture under the ceiling evaluated, the
Pareto front is exact rather than approximated, and that is the property that makes the
algorithm comparison in the second paper possible at all. Where coverage is partial the
front is still exact *for what was measured*, and the table says how much that was, because
a front over a third of the space is a different object from a front over all of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .space.cost import (
    activation_width,
    embedding_share,
    forward_flops_per_token,
    kv_cache_bytes,
    parameter_count,
)
from .space.genotype import GENES, Genotype

#: Columns of the published table, in order. Genotype first, then what is derived from it,
#: then what was measured, then how.
COLUMNS: tuple[str, ...] = (
    *GENES,
    "e",
    "intermediate",
    "params",
    "params_non_embedding",
    "embedding_share",
    "kv_cache_bytes",
    "flops_per_token",
    "activation_width",
    "perplexity",
    "bits_per_byte",
    "tokens_consumed",
    "wall_clock_seconds",
    "tokens_per_second",
    "achieved_tflops",
    "micro_batch_used",
    "resumes",
    "seed",
    "precision",
    "device",
    "data_vintage",
    "tokeniser_fingerprint",
    "on_measured_front",
)

VOCAB = 50257


class NoUsableRows(RuntimeError):
    """Raised when nothing in the artifact container can be placed in the table."""


@dataclass
class Assembly:
    """The table, the front, and an account of what was left out."""

    rows: list[dict] = field(default_factory=list)
    front_indices: list[int] = field(default_factory=list)
    #: Architectures on the front, which is what the front's size means. ``front_indices``
    #: holds every row belonging to one of them, so the two differ wherever a shape was
    #: measured more than once.
    front_architectures: int = 0
    #: Distinct architectures, as against rows. Coverage is measured in these: a row is a
    #: run, and comparing runs to the size of the space reported 1,103 of 1,099 — 100.4 per
    #: cent, which is the sort of figure that should stop a report rather than appear in it.
    architectures: int = 0
    excluded: dict[str, int] = field(default_factory=dict)
    held_constant: dict[str, str] = field(default_factory=dict)
    #: The configuration this table was assembled for, so a reader of the artifact knows
    #: which of several it is looking at.
    selected: dict[str, str] = field(default_factory=dict)
    space_size: int = 0

    @property
    def coverage(self) -> float:
        counted = self.architectures or len(self.rows)
        return counted / self.space_size if self.space_size else 0.0

    @property
    def replicated(self) -> tuple[int, int]:
        """Architectures measured more than once, and the extra rows they contribute.

        The two are not the same number and were reported as one: ``rows - architectures``
        came to 4 under the name ``replicated_architectures``, and a reader asked where the
        four architectures were. There is one, trained at five seeds for the region of
        practical equivalence, and it contributes four rows beyond its first.
        """
        counts: dict[tuple, int] = {}
        for row in self.rows:
            key = tuple(row.get(gene) for gene in ("h", "d_head", "n", "ffn", "kv"))
            counts[key] = counts.get(key, 0) + 1
        repeated = [n for n in counts.values() if n > 1]
        return len(repeated), sum(repeated) - len(repeated)

    def summary(self) -> dict[str, object]:
        replicated_architectures, replicate_rows = self.replicated
        return {
            "rows": len(self.rows),
            "architectures": self.architectures or len(self.rows),
            "replicated_architectures": replicated_architectures,
            "replicate_rows": replicate_rows,
            "space_size": self.space_size,
            "coverage": round(self.coverage, 4),
            "front_size": self.front_architectures or len(self.front_indices),
            "front_rows": len(self.front_indices),
            "front_is": (
                "exact over the whole space"
                if self.coverage >= 1.0
                else f"exact over the {self.architectures or len(self.rows)} architectures "
                     f"measured so far, which is {self.coverage:.1%} of the space — not the "
                     "same object as the front over all of it"
            ),
            "front_is_over": (
                "one point per architecture, the median perplexity across its replicates. "
                "Computed per run instead, a lucky seed could put an architecture on the "
                "front while its own replicates sit off it, and membership would describe "
                "a seed rather than a design."
            ),
            "excluded": self.excluded,
            "held_constant": self.held_constant,
            "selected": self.selected,
            "objectives": ["params", "kv_cache_bytes", "perplexity"],
            "why_excluded": {
                "other_configuration": (
                "measured under a different corpus or token budget. Perplexity on "
                "different text, or after four times the training, is not the same "
                "quantity — so each configuration gets its own table and its own front."
            ),
            "not_usable_for_benchmark": (
                    "the run did not meet its token budget, so its perplexity answers a "
                    "different question than a complete row's"
                ),
                "duplicate_fingerprint": (
                    "the same measurement completed more than once; the latest is kept"
                ),
            },
        }


def _derive(genotype: Genotype, manifest: dict, on_front: bool) -> dict:
    cost = manifest.get("cost") or {}
    objectives = manifest.get("objectives") or {}
    params = parameter_count(genotype)
    return {
        **{gene: getattr(genotype, gene) for gene in GENES},
        "e": genotype.e,
        "intermediate": genotype.intermediate,
        "params": params,
        "params_non_embedding": params - VOCAB * genotype.e,
        "embedding_share": round(embedding_share(genotype), 6),
        "kv_cache_bytes": kv_cache_bytes(genotype),
        "flops_per_token": forward_flops_per_token(genotype),
        "activation_width": activation_width(genotype),
        "perplexity": objectives.get("perplexity"),
        "bits_per_byte": objectives.get("bits_per_byte"),
        "tokens_consumed": manifest.get("tokens_consumed"),
        "wall_clock_seconds": manifest.get("wall_clock_seconds"),
        "tokens_per_second": cost.get("tokens_per_second"),
        "achieved_tflops": cost.get("achieved_tflops"),
        "micro_batch_used": cost.get("micro_batch_used"),
        "resumes": cost.get("resumes", 0),
        "seed": (manifest.get("config") or {}).get("seed"),
        "precision": manifest.get("precision"),
        "device": manifest.get("device"),
        "data_vintage": manifest.get("data_vintage"),
        "tokeniser_fingerprint": manifest.get("tokeniser_fingerprint"),
        "on_measured_front": int(on_front),
    }


def assemble(manifests: list[dict], space_size: int,
             select: dict[str, object] | None = None) -> Assembly:
    """Build the table and its exact front from run manifests.

    ``manifests`` may contain anything the artifact container holds; this decides what
    belongs. The count of what was dropped and why travels with the result, because a table
    that silently discarded a tenth of its inputs is not a table anyone should trust.

    ``select`` names the configuration to assemble, as manifest field to value. It exists
    because the container now holds several: four corpora from the transfer study and two
    token budgets from the convergence study, all of them legitimate measurements of
    different things. A table is per configuration, and which one has to be chosen rather
    than inferred — picking the most numerous would make the answer depend on how far a
    campaign had got.

    Filtering rather than a parameter per field, so that a field added to the comparability
    rule is selectable without touching this signature. The first version took only
    ``corpus`` and was immediately refused by the rule: the 4x-budget study is the same
    corpus at four times the tokens, and the guard said so in the words it was written
    with — split the table and say which rows belong to which protocol.
    """
    from .search.problem import require_comparable
    from .space.enumerate import non_dominated

    excluded: dict[str, int] = {"not_usable_for_benchmark": 0, "duplicate_fingerprint": 0}

    usable: dict[tuple[int, ...], dict] = {}
    for manifest in manifests:
        if select and any(str(manifest.get(k)) != str(v) for k, v in select.items()):
            excluded.setdefault("other_configuration", 0)
            excluded["other_configuration"] += 1
            continue
        if not manifest.get("usable_for_benchmark"):
            excluded["not_usable_for_benchmark"] += 1
            continue
        values = ((manifest.get("genotype") or {}).get("values")) or []
        if len(values) != len(GENES):
            excluded.setdefault("malformed_genotype", 0)
            excluded["malformed_genotype"] += 1
            continue
        key = tuple(values) + ((manifest.get("config") or {}).get("seed", 0),)
        if key in usable:
            # Counted as a duplicate and then kept anyway: the assignment below used to run
            # unconditionally, so the last manifest in file order won and the table was a
            # function of that order rather than of the runs. It moved the measured front
            # from 220 architectures to 218 and changed 220 perplexities, none of which any
            # check caught, because `verify` seeds the table as an input instead of
            # rebuilding it. The first completed run of an architecture at a seed is the
            # measurement; the 220 that follow it are re-runs.
            excluded["duplicate_fingerprint"] += 1
            continue
        usable[key] = manifest

    if not usable:
        raise NoUsableRows(
            f"{len(manifests)} manifest(s) and none usable. "
            f"{excluded['not_usable_for_benchmark']} did not meet their token budget, "
            "which is a different measurement rather than a worse one."
        )

    kept = list(usable.values())
    held = require_comparable(kept)

    # Three objectives, all minimised: two costs computed in closed form and one measured.
    genotypes = [
        Genotype(**dict(zip(GENES, key[: len(GENES)], strict=True))) for key in usable
    ]
    objectives = np.array([
        [parameter_count(g), kv_cache_bytes(g),
         (m.get("objectives") or {}).get("perplexity", float("inf"))]
        for g, m in zip(genotypes, kept, strict=True)
    ], dtype=float)

    # The front is computed over one point per architecture, not over one point per run.
    #
    # A row is a run and an architecture can have several — the pilot measured one shape at
    # five seeds to derive the ROPE, and the multi-corpus campaign will give every front
    # architecture three. With one point per run, an architecture whose luckiest seed came
    # out 1.6 perplexity ahead of its median could join the front while its own replicates
    # sit off it, which makes front membership a property of a seed rather than of a
    # design. The costs are identical across replicates, so only the measured objective
    # needs reducing, and the median is used because with an odd replicate count it is an
    # observation rather than an interpolation and one preempted-and-resumed run cannot
    # drag it.
    shapes: dict[tuple[int, ...], list[int]] = {}
    for index, genotype in enumerate(genotypes):
        shapes.setdefault(genotype.as_tuple(), []).append(index)

    representative = np.array([
        [objectives[group[0], 0], objectives[group[0], 1],
         float(np.median(objectives[group, 2]))]
        for group in shapes.values()
    ], dtype=float)

    on_front_shapes = {
        list(shapes)[position] for position in non_dominated(representative).tolist()
    }
    front = {
        index for shape in on_front_shapes for index in shapes[shape]
    }
    rows = [
        _derive(g, m, index in front)
        for index, (g, m) in enumerate(zip(genotypes, kept, strict=True))
    ]

    return Assembly(
        rows=rows,
        front_indices=sorted(front),
        front_architectures=len(on_front_shapes),
        architectures=len(shapes),
        excluded={k: v for k, v in excluded.items() if v},
        held_constant=held,
        selected={k: str(v) for k, v in (select or {}).items()},
        space_size=space_size,
    )


def read_manifests(directory: Path) -> list[dict]:
    """Every manifest under a directory. For a local copy of the artifact container."""
    found = []
    for path in sorted(Path(directory).rglob("manifest.json")):
        try:
            found.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return found


#: Columns the analyses need as numbers. Everything else in the table is a label or a
#: provenance field, and coercing those would turn a tokeniser fingerprint into a float.
NUMERIC_COLUMNS: tuple[str, ...] = (
    *GENES, "e", "intermediate", "params", "params_non_embedding", "embedding_share",
    "kv_cache_bytes", "flops_per_token", "activation_width", "perplexity",
    "bits_per_byte", "tokens_consumed", "wall_clock_seconds", "tokens_per_second",
    "achieved_tflops", "micro_batch_used", "resumes", "seed", "on_measured_front",
)


class TableUnreadable(RuntimeError):
    """Raised when the published table is absent or is not the table it claims to be."""


@dataclass(frozen=True)
class LoadedTable:
    """The published benchmark table, read back for analysis.

    This is the input to the verification tier, and the reason it exists as a function
    rather than as a snippet: the surrogate study and the characterisation were producing
    artifacts from ad-hoc code, so the published table had no reproducible path to the
    numbers derived from it. The replication package printed commands for both and neither
    existed. A benchmark nobody can re-analyse from what was published is a benchmark whose
    conclusions rest on the authors' working directory.
    """

    rows: list[dict[str, object]]
    columns: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.rows)

    def column(self, name: str) -> "np.ndarray":  # noqa: UP037 — numpy imported lazily
        import numpy as np

        if name not in self.columns:
            raise TableUnreadable(
                f"the table has no column {name!r}; it has {', '.join(self.columns)}"
            )
        values = [row[name] for row in self.rows]
        if name in NUMERIC_COLUMNS:
            return np.asarray(values, dtype=float)
        return np.asarray(values, dtype=object)

    def genes(self) -> "np.ndarray":  # noqa: UP037
        import numpy as np

        return np.column_stack([self.column(gene) for gene in GENES])

    def held_constant(self) -> dict[str, set]:
        """What the table holds fixed, read from the rows rather than from a summary.

        Recomputed here so a reader who has only the CSV can check the invariant the
        assembly claims, instead of taking the summary's word for it.
        """
        return {
            field: {row[field] for row in self.rows}
            for field in ("precision", "device", "data_vintage", "tokeniser_fingerprint")
        }


def load_table(path: Path) -> LoadedTable:
    """Read the published table, refusing one that is not what it says it is.

    Rows are returned in file order. That order is the assembly's, which sorts by
    fingerprint, so two readers of the same file get the same arrays and any analysis
    downstream is reproducible from the file alone.
    """
    import csv as _csv

    if not path.exists():
        raise TableUnreadable(
            f"{path} does not exist. It is the published benchmark table and the input to "
            "every analysis; run: python -m nas4textgen.cli table"
        )
    with path.open(encoding="utf-8", newline="") as handle:
        reader = _csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        raw = list(reader)

    missing = [column for column in COLUMNS if column not in header]
    if missing:
        raise TableUnreadable(
            f"{path} is missing {len(missing)} declared column(s): {', '.join(missing)}.\n"
            "Either it was written by a different version or it is not the benchmark table."
        )
    if not raw:
        raise TableUnreadable(f"{path} has a header and no rows")

    rows: list[dict[str, object]] = []
    for number, row in enumerate(raw, start=2):
        converted: dict[str, object] = {}
        for column in header:
            value = row[column]
            if column in NUMERIC_COLUMNS:
                try:
                    converted[column] = float(value)
                except (TypeError, ValueError) as exc:
                    raise TableUnreadable(
                        f"{path} line {number}, column {column!r}: {value!r} is not a "
                        "number, and a numeric column that silently became a string would "
                        "make every downstream comparison wrong rather than failing"
                    ) from exc
            else:
                converted[column] = value
        rows.append(converted)

    return LoadedTable(rows=rows, columns=header)
