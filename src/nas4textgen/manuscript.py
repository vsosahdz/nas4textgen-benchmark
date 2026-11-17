"""Every table and figure in the paper, generated from the artifact directory.

Nothing here is transcribed. A number typed into a manuscript is a number that can disagree
with the artifact it came from, and by the time anyone notices, the paper is submitted. So
each table is built by reading the JSON and CSV the pipeline wrote, and a missing input is a
failure rather than a gap left blank.

Two consequences worth stating.

**A table cannot be produced from data that does not exist.** Asking for the depth-versus-
width table before the characterisation has run raises, naming the artifact and the command
that writes it. That is what makes the verification tier a check: it regenerates the paper
from the published benchmark and fails if anything is absent or has moved.

**Coverage travels with every number.** These artifacts are being written while the sweep is
still running, so a table generated today describes a fraction of the space. Each carries the
row count it was built from, and a figure that would read as final if the caption were
dropped is worse than one that cannot.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


#: A paragraph column that does not justify. Needs `array`, which the preamble loads.
RAGGED = r">{\raggedright\arraybackslash}"

#: A cell a table generator marks as the one worth looking at in its column.
EMPHASIS = re.compile(r"\*\*(.+?)\*\*")


class MissingArtifact(RuntimeError):
    """Raised when a table needs an artifact the pipeline has not written."""


@dataclass(frozen=True)
class Table:
    """One manuscript table, with where its numbers came from."""

    number: str
    caption: str
    columns: tuple[str, ...]
    rows: list[dict]
    source: str
    coverage: str = ""
    #: The column specification, when the default will not do. The default suits a table of
    #: numbers: a label on the left and figures right-aligned beside it. A table whose cells
    #: are sentences needs paragraph columns, since a `tabular` cell does not wrap and the
    #: box around it would shrink the whole table until the text was unreadable.
    alignment: str = ""

    def render_latex(self) -> str:
        """The same table as an ACM ``table`` float, for the Overleaf project.

        Generated rather than transcribed for the reason every other artifact here is: a
        number typed into a manuscript can disagree with the source it came from, and a
        LaTeX table typed once and edited later disagrees silently. The caption and the
        source line travel with it, so a table lifted into a draft carries its provenance.
        """
        def escape(value: object) -> str:
            # Maths is held aside before anything is escaped and put back afterwards, so a
            # table can name a gene with the symbol the equations use for it. Without this
            # a cell reading `$k_v$` prints as the literal dollar signs, and a reader
            # comparing Table 1 against the encoding meets two spellings of one variable.
            #
            # Every span is held, where the prose converter holds only the ones that look
            # like maths. The difference is deliberate: prose is written by hand and may
            # quote a sum of money, while these cells are generated, and the generator
            # writes a cost as "USD 590" rather than with a currency sign.
            from .overleaf import MATHS

            kept: list[str] = []

            def hold(match: re.Match) -> str:
                kept.append(match.group(0))
                return f"\x00{len(kept) - 1}\x00"

            def hold_bold(inner: str) -> str:
                """The emphasised cell, with its own contents escaped in turn."""
                kept.append(r"\textbf{" + escape(inner) + "}")
                return f"\x00{len(kept) - 1}\x00"

            # A cell a generator wrapped in `**` is the one worth looking at in its column,
            # and is set in bold. A table of forty numbers in one weight asks the reader to
            # find the result themselves. The markers are held aside with the maths so the
            # escaping below cannot see them.
            text = EMPHASIS.sub(lambda m: hold_bold(m.group(1)), str(value))
            text = MATHS.sub(hold, text)
            for character, replacement in (
                ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
                ("—", "---"), ("–", "--"), ("≤", r"$\leq$"), ("×", r"$\times$"),
            ):
                text = text.replace(character, replacement)
            return re.sub(r"\x00(\d+)\x00", lambda m: kept[int(m.group(1))], text)

        alignment = self.alignment or "l" + "r" * (len(self.columns) - 1)
        # The coverage note goes in the caption. Set below the table as a line of its own it
        # ran the full column width while the table beside it was narrower and centred, so
        # the note and the table did not line up with each other. It is a caveat about the
        # measurement and belongs with the rest of the caption; the artifact path is still
        # left out, since a reader of the journal cannot open it.
        caption = f"{self.caption} {self.coverage}".strip() if self.coverage else self.caption
        lines = [
            # Every slot the page builder has, with the `!` waiving the limits on how much
            # of a page may be float. Five of these tables are cited within six pages of
            # each other, and restricted to the top of a page the queue backs up until a
            # table lands four pages after the sentence that sends the reader to it.
            r"\begin{table}[!htb]",
            r"  \caption{" + escape(caption) + "}",
            r"  \label{tab:" + self.number + "}",
            r"  \centering",
            # Several of these tables have headers that are phrases rather than symbols, and
            # at natural width they run into the right margin. The box shrinks a table only
            # when it is too wide, so the narrow ones keep the document's type size.
            r"  \begin{adjustbox}{max width=\linewidth}",
            r"  \begin{tabular}{" + alignment + "}",
            r"    \toprule",
            "    " + " & ".join(escape(c) for c in self.columns) + r" \\",
            r"    \midrule",
        ]
        for row in self.rows:
            lines.append("    " + " & ".join(
                escape(row.get(column, "")) for column in self.columns) + r" \\")
        lines += [r"    \bottomrule", r"  \end{tabular}", r"  \end{adjustbox}"]
        lines.append(r"  \Description{" + escape(self.caption[:180]) + "}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    def render(self) -> str:
        """Markdown, because it converts to LaTeX and is readable in a diff."""
        widths = [
            max(len(str(column)), *(len(str(row.get(column, ""))) for row in self.rows))
            if self.rows else len(str(column))
            for column in self.columns
        ]
        head = " | ".join(str(c).ljust(w) for c, w in zip(self.columns, widths, strict=True))
        rule = " | ".join("-" * w for w in widths)
        body = [
            " | ".join(str(row.get(c, "")).ljust(w)
                       for c, w in zip(self.columns, widths, strict=True))
            for row in self.rows
        ]
        lines = [
            f"**Table {self.number}.** {self.caption}",
            "",
            f"| {head} |",
            f"| {rule} |",
            *(f"| {b} |" for b in body),
            "",
            f"*Source: {self.source}.*",
        ]
        if self.coverage:
            lines.append(f"*{self.coverage}*")
        return "\n".join(lines)


def _read(root: Path, relative: str, writer: str) -> dict:
    path = root / relative
    if not path.exists():
        raise MissingArtifact(
            f"{relative} has not been written. Run: {writer}\n"
            "A manuscript table is generated from the artifact rather than transcribed, so "
            "a missing input is a failure rather than a gap to fill in later."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(root: Path, relative: str, writer: str) -> list[dict]:
    path = root / relative
    if not path.exists():
        raise MissingArtifact(f"{relative} has not been written. Run: {writer}")
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


#: Each gene, the attribute that holds it, and how the row names it. The symbol is set as
#: maths so that the table and the encoding of the manuscript spell a variable one way.
GENE_LABELS: tuple[tuple[str, str], ...] = (
    ("h", "$h$, query heads"),
    ("d_head", r"$d_{\mathrm{head}}$, head dimension"),
    ("n", "$n$, transformer blocks"),
    ("ffn", r"$\mathit{ffn}$, feed-forward ratio"),
    ("kv", "$k_v$, key-value heads, dividing $h$"),
)


def table_search_space(root: Path) -> Table:
    """The space gene by gene, the anchor it was built around, and what the ceiling removed.

    The anchor column is the point of the table. A space described only by its cardinality
    leaves the reader to take on trust that it contains the architecture it claims to be
    derived from, and this one does contain it exactly: the levels are chosen so that
    DistilGPT2 is a member rather than a neighbour, and the closed-form parameter count
    reproduces the published checkpoint's to the parameter.
    """
    from .space.cost import kv_cache_bytes, parameter_count
    from .space.genotype import DISTILGPT2, LEVELS, REFERENCE_CONTEXT

    summary = _read(root, "space/enumeration_summary.json",
                    "python -m nas4textgen.cli enumerate")

    rows = [
        {"quantity": label,
         "across the space": ", ".join(str(v) for v in LEVELS[gene]),
         "DistilGPT2": str(getattr(DISTILGPT2, gene))}
        for gene, label in GENE_LABELS
    ]
    # The derived width and the two cost objectives, which are functions of the genes above
    # rather than genes themselves. Separated from them because a reader who takes `e` for a
    # sixth gene has misread the space in the way Section 2.4 says the prior one was misread.
    rows += [
        {"quantity": r"$e = h \cdot d_{\mathrm{head}}$, model width (derived)",
         "across the space": " to ".join(
             f"{v:,}" for v in (min(LEVELS["h"]) * min(LEVELS["d_head"]),
                                max(LEVELS["h"]) * max(LEVELS["d_head"]))),
         "DistilGPT2": f"{DISTILGPT2.e:,}"},
        {"quantity": "parameters",
         "across the space": " to ".join(f"{v:,}" for v in summary["params_range"]),
         "DistilGPT2": f"{parameter_count(DISTILGPT2):,}"},
        {"quantity": f"KV cache at {REFERENCE_CONTEXT:,} tokens (bytes)",
         "across the space": " to ".join(f"{v:,}" for v in summary["kv_bytes_range"]),
         "DistilGPT2": f"{kv_cache_bytes(DISTILGPT2):,}"},
        {"quantity": "grid before any constraint",
         "across the space": f"{summary['total_before_ceiling']:,}", "DistilGPT2": ""},
        {"quantity": "removed by the parameter ceiling of "
                     f"{summary['parameter_ceiling']:,}",
         "across the space": f"{summary['excluded']:,} "
                             f"({summary['excluded_fraction']:.1%})",
         "DistilGPT2": ""},
        {"quantity": "architectures measured", "DistilGPT2": "",
         "across the space": f"{summary['cardinality']:,}"},
    ]

    return Table(
        number="1",
        caption=(
            "The search space gene by gene, beside the values the anchor architecture "
            "takes. The first five rows are the searched genes and their admissible "
            "levels; the rows below them are derived or counted."
        ),
        columns=("quantity", "across the space", "DistilGPT2"),
        rows=rows,
        source="artifacts/space/enumeration_summary.json "
               "and src/nas4textgen/space/genotype.py",
    )


def table_benchmark_coverage(root: Path) -> Table:
    """What has been measured, and what was refused."""
    summary = _read(root, "benchmark/summary.json", "python -m nas4textgen.cli table")
    held = summary.get("held_constant", {})
    # `rows` counts training runs and was printed under the label "architectures measured",
    # directly above a space of 1,099 and a coverage of 100 per cent. Three numbers that
    # cannot all be true of the same quantity, and a reader asked where the extra four came
    # from. The two counts are now separate rows and the difference between them is one of
    # them, so the arithmetic is on the page.
    rows = [
        {"quantity": "architectures measured", "value": f"{summary['architectures']:,}"},
        {"quantity": "of a space of", "value": f"{summary['space_size']:,}"},
        {"quantity": "coverage", "value": f"**{summary['coverage']:.1%}**"},
        {"quantity": "training runs", "value": f"{summary['rows']:,}"},
        {"quantity": "runs beyond one per architecture",
         "value": f"{summary['replicate_rows']}"},
        {"quantity": "measured Pareto front", "value": f"**{summary['front_size']}**"},
    ]
    rows += [
        {"quantity": f"held constant: {key}", "value": str(value)}
        for key, value in sorted(held.items())
    ]
    for reason, count in sorted(summary.get("excluded", {}).items()):
        rows.append({"quantity": f"excluded: {reason}", "value": str(count)})

    return Table(
        number="2",
        caption=(
            "The benchmark table. Rows measured under different precision, corpus vintage, "
            "token budget or tokenizer are refused rather than reconciled."
        ),
        columns=("quantity", "value"),
        rows=rows,
        source="artifacts/benchmark/summary.json",
        coverage=(
            f"The front is {summary['front_is']}. The run count exceeds the architecture "
            f"count by the {summary['replicate_rows']} further seeds of the "
            + ("single architecture" if summary["replicated_architectures"] == 1
               else f"{summary['replicated_architectures']} architectures")
            + " replicated to measure the region of practical equivalence."),
    )


def table_depth_versus_width(root: Path) -> Table:
    """The prior work's shape claim, tested."""
    payload = _read(root, "characterisation/depth_vs_width.json",
                    "python -m nas4textgen.cli characterise")
    iso = payload["iso_parameter"]
    # The artifact carries what `test_iso_parameter` computed, nested as that function
    # returns it. An earlier flatter shape came from code written beside the analysis
    # rather than by it, and this table read that shape — so the manuscript depended on an
    # artifact no command produced. The declaration is where the ROPE lives, because a ROPE
    # belongs to the comparison that declared it, not to the verdict it produced.
    bayes = payload["bayesian"]["bayesian"]
    declaration = payload["bayesian"]["declaration"]

    rows = []
    for index, ((lo, hi), deep, wide, delta) in enumerate(zip(
        iso["band_edges_params"], iso["deep_mean_perplexity"],
        iso["wide_mean_perplexity"], iso["delta_deep_minus_wide"], strict=True,
    ), start=1):
        rows.append({
            "band": index,
            "parameters (M)": f"{lo/1e6:.1f}-{hi/1e6:.1f}",
            "deepest third": f"{deep:.1f}",
            "widest third": f"{wide:.1f}",
            "difference": f"{delta:+.1f}",
        })

    return Table(
        number="3",
        caption=(
            "Depth against width at matched parameter count. Positive differences mean the "
            "widest third reached lower perplexity. The prior work reports the opposite. "
            f"Bayesian signed-rank against a ROPE of {declaration['rope']:.3f} "
            f"perplexity, measured from {declaration['rope_source']}: "
            f"{bayes['verdict']} (p={bayes['p(wide better)']:.3f})."
        ),
        columns=("band", "parameters (M)", "deepest third", "widest third", "difference"),
        rows=rows,
        source="artifacts/characterisation/depth_vs_width.json",
    )


def table_gene_effects(root: Path) -> Table:
    """Which genes move measured quality."""
    payload = _read(root, "characterisation/fronts_and_genes.json",
                    "python -m nas4textgen.cli characterise")
    effects = payload["gene_effects"]["first_order_indices"]
    return Table(
        number="4",
        caption=(
            "First-order variance indices on measured quality. These describe the effect on "
            "perplexity, not on cost; the structural gate measured the latter and the two "
            "need not agree. The shortfall from one is where interactions live."
        ),
        columns=("gene", "first-order index"),
        rows=[{"gene": gene, "first-order index": f"{value:.4f}"}
              for gene, value in effects.items()]
        + [{"gene": "sum", "first-order index":
            f"{payload['gene_effects']['sum_of_first_order']:.4f}"}],
        source="artifacts/characterisation/fronts_and_genes.json",
    )


def table_surrogate_slate(root: Path) -> Table:
    """Every predictor, against the two it must beat."""
    candidates = sorted(root.glob("surrogate/slate_at_*.json"),
                        key=lambda p: int(p.stem.split("_")[-1]))
    if not candidates:
        raise MissingArtifact(
            "no surrogate slate has been written. Run: python -m nas4textgen.cli surrogate"
        )
    payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
    # The slate's own `coverage` field divides its row count by the size of the space, which
    # is above one whenever an architecture has been measured twice. The benchmark summary
    # is the artifact that separates the two counts, so the note is built from it.
    summary = _read(root, "benchmark/summary.json", "python -m nas4textgen.cli table")

    rows = []
    for entry in payload["scores"]:
        if "unavailable" in entry:
            rows.append({
                "predictor": entry["name"], "tier": entry["tier"],
                "Kendall tau": "unavailable", "tau (top decile)": "",
                "precision@10": "", "disqualified": "",
            })
            continue
        rows.append({
            "predictor": entry["name"],
            "tier": entry["tier"],
            "Kendall tau": f"{entry['kendall_tau']:.4f}",
            "tau (top decile)": f"{entry['kendall_tau_top_decile']:.4f}",
            "precision@10": f"{entry['precision_at'].get('10', 0):.2f}",
            "disqualified": "yes" if entry["disqualified"] else "",
        })

    floor = payload["trivial_floors"]["sort_by_parameters"]
    # The best rank correlation in the column, which is the number a reader scans for, and
    # the only one of the fourteen they would otherwise have to find by eye.
    ranked = [r for r in rows if r["Kendall tau"] != "unavailable"]
    if ranked:
        best = max(ranked, key=lambda r: float(r["Kendall tau"]))
        best["Kendall tau"] = f"**{best['Kendall tau']}**"
    return Table(
        number="5",
        caption=(
            "Surrogate predictors, cross-validated on held-out perplexity rank. Sorting by "
            f"parameter count reaches {floor:.4f} and needs no training, so a candidate that "
            "outranks neither trivial predictor is disqualified."
        ),
        columns=("predictor", "tier", "Kendall tau", "tau (top decile)",
                 "precision@10", "disqualified"),
        rows=rows,
        source=f"artifacts/{candidates[-1].relative_to(root)}",
        coverage=f"Fitted on {payload['training_rows']:,} measured rows: every one of the "
                 f"{summary['architectures']:,} architectures in the space, plus the "
                 f"{summary['replicate_rows']} further seeds of the one architecture "
                 f"replicated to measure the region of practical equivalence.",
    )


def table_campaign_cost(root: Path) -> Table:
    """What the measurements cost, from what the cloud allocated rather than computed.

    This table read `progress.json` for weeks, which records the time each training process
    ran. A cloud bills allocated node time — image pull, corpus mount, and the scale-down
    window — and the difference here is 54 per cent. A cost table in a paper is exactly
    where that error becomes permanent, so it reads the allocation artifact and reports
    both figures with the gap named.
    """
    progress = _read(root, "campaign/progress.json",
                     "python -m nas4textgen.cli progress")
    cost = _read(root, "campaign/cost.json", "python -m nas4textgen.cli cost")
    model = _read(root, "canary/throughput_model.json",
                  "python -m nas4textgen.cli canary")

    rates = [s["tokens_per_second"] for s in model["samples"]]
    per_run = cost["node_hours_per_run"]
    return Table(
        number="6",
        caption=(
            "What the campaign consumed, measured from allocated node time and not from "
            "process time. Wall-clock time per architecture spans a factor of ten across "
            "the space, so no single per-architecture figure describes it."
        ),
        columns=("quantity", "value"),
        rows=[
            {"quantity": "architectures measured", "value": f"{progress['measured']:,}"},
            {"quantity": "runs that consumed node time",
             "value": f"{cost['runs_in_window']:,}"},
            {"quantity": "node-hours, training process time",
             "value": f"{cost['process_node_hours']:,.0f}"},
            {"quantity": "node-hours, allocated",
             "value": f"{cost['allocated_node_hours']:,.0f}"},
            {"quantity": "allocation overhead",
             "value": f"{cost['allocation_overhead_allocated']:+.1%}"},
            {"quantity": "node-hours per run",
             "value": f"{per_run['allocated']:.2f} allocated "
                      f"({per_run['process_time']:.2f} of process time)"},
            {"quantity": "work recovered after preemption",
             "value": f"{progress['resumes']} resumes"},
            {"quantity": "device", "value": model["device"]},
            {"quantity": "throughput range (tokens/s)",
             "value": f"{min(rates):,.0f} to {max(rates):,.0f}"},
        ],
        source="artifacts/campaign/cost.json, artifacts/campaign/progress.json and "
               "artifacts/canary/throughput_model.json",
        coverage=(
            "Allocation is integrated from Azure Monitor, which retains platform metrics "
            "for 93 days, after which these figures cannot be recomputed from any source."
        ),
    )


#: Every table, in manuscript order. Adding one here is what puts it in the paper.
def table_design_rules(root: Path) -> Table:
    """Rules that survive on both fronts, with the lift each earns on each.

    Two fronts because the joint one counts cache size as an objective, so a rule about the
    gene that minimises cache is partly true by construction. The parameters-only column is
    what remains once that tautology is removed, and a rule is reported only where both
    columns clear the threshold.
    """
    payload = _read(root, "characterisation/design_rules.json",
                    "python -m nas4textgen.cli characterise")
    joint = {r["rule"]: r for r in payload["joint"]["rules"] if r["holds"]}
    parameters = {r["rule"]: r for r in payload["parameters_only"]["rules"] if r["holds"]}
    surviving = [rule for rule in joint if rule in parameters]

    rows = [{
        "rule": rule,
        "lift, joint front": f"{joint[rule]['lift']:.2f}",
        "lift, parameters only": f"{parameters[rule]['lift']:.2f}",
        "covers the parameters front": f"{parameters[rule]['coverage']:.1%}",
    } for rule in sorted(surviving, key=lambda r: -joint[r]["lift"])]

    thresholds = payload["thresholds"]
    return Table(
        number="7",
        caption=(
            f"Design rules surviving on both fronts. {len(surviving)} of "
            f"{len(payload['joint']['rules'])} candidates clear a lift of "
            f"{thresholds['minimum_lift']} and a coverage of "
            f"{thresholds['minimum_coverage']:.0%} on each front. Coverage is the share of "
            "the parameters-only front the rule accounts for."
        ),
        columns=("rule", "lift, joint front", "lift, parameters only",
                 "covers the parameters front"),
        rows=rows,
        source="artifacts/characterisation/design_rules.json",
        coverage=f"Joint front: {payload['joint']['front_size']} architectures; "
                 f"parameters-only front: {payload['parameters_only']['front_size']}.",
    )


def table_label_budget(root: Path) -> Table:
    """Where a trained predictor starts beating the untrained one.

    The question a group with a fixed training budget actually faces. Reported against the
    parameter floor at every budget, because a surrogate's rank correlation read on its own
    looks strong at every size on this space.
    """
    payload = _read(root, "surrogate/budget_curve.json",
                    "python -m nas4textgen.cli surrogate")
    floors = payload["floor_tau_by_size"]
    trivial = {"sort_by_parameters", "linear_on_genes"}

    rows = []
    for size in payload["sizes"]:
        best, who = None, ""
        for name, entry in payload["per_candidate"].items():
            if name in trivial:
                continue
            for point in entry["points"]:
                if point["training_rows"] == size and (
                        best is None or point["kendall_tau"] > best):
                    best, who = point["kendall_tau"], name
        if best is None:
            continue
        floor = floors[str(size)]
        rows.append({
            "labels": str(size),
            "sort by parameters": f"{floor:.4f}",
            "best trained model": f"{best:.4f}",
            "which": who,
            "gain": f"{best - floor:+.4f}",
        })

    crossing = next((r["labels"] for r in rows if not r["gain"].startswith("-")), "never")
    # The row where a trained predictor first beats a predictor that costs nothing is the
    # result the table exists to report, so it is the one marked.
    for row in rows:
        if row["labels"] == crossing:
            row["gain"] = f"**{row['gain']}**"
            row["best trained model"] = f"**{row['best trained model']}**"
            break
    return Table(
        number="8",
        caption=(
            "Rank correlation against the number of trained labels, each budget averaged "
            f"over {payload['repeats_per_size']} draws. The trained predictors first clear "
            f"the untrained floor at {crossing} labels; below that, fitting a regressor is "
            "worse than sorting the space by parameter count, which costs nothing."
        ),
        columns=("labels", "sort by parameters", "best trained model", "which", "gain"),
        rows=rows,
        source="artifacts/surrogate/budget_curve.json",
        coverage=f"Drawn from {payload['total_measured_rows']:,} measured rows, one per "
                 f"training run.",
    )


def table_transfer(root: Path) -> Table:
    """What became of the measured front on corpora it was not selected on."""
    payload = _read(root, "transfer/analysis.json",
                    "python -m nas4textgen.cli transfer-analysis")
    rows = [{
        "corpus": entry["corpus"],
        "seeds": str(entry["seeds"]),
        "Kendall tau": f"{entry['kendall_tau']:.3f}",
        "tau, top decile": f"{entry['kendall_tau_top_decile']:.3f}",
        "still non-dominated": f"{entry['still_non_dominated']} "
                               f"({entry['still_non_dominated_share']:.0%})",
        "top 10 shared": f"{entry['top10_overlap']}/10",
        "ROPE": f"{entry['rope']:.3f}",
    } for entry in payload["corpora"]]

    return Table(
        number="9",
        caption=(
            f"The {payload['front_size']}-architecture front of "
            f"{payload['search_corpus']}, retrained on each further corpus. Rank "
            "correlation is against the ordering on the search corpus; non-dominance is "
            "recomputed among the same architectures on the new corpus, so a front whose "
            "ordering were preserved exactly would retain all of them. Every quantity is a "
            "rank, a share or a dominance relation computed within one corpus."
        ),
        columns=("corpus", "seeds", "Kendall tau", "tau, top decile",
                 "still non-dominated", "top 10 shared", "ROPE"),
        rows=rows,
        source="artifacts/transfer/analysis.json",
        coverage="Only the front was retrained, so no other corpus's exact front is known "
                 "and none is claimed.",
    )


def table_battery(root: Path) -> Table:
    """Each battery measure against the objective it was added to supplement."""
    payload = _read(root, "battery/analysis.json",
                    "python -m nas4textgen.cli battery-analysis")
    rows = [{
        "corpus": entry["corpus"],
        "measure": entry["metric"],
        "reported as": entry["key"],
        "architectures": str(entry["architectures"]),
        "Kendall tau vs perplexity": f"{entry['kendall_tau_against_perplexity']:+.3f}",
        "adds information": "" if entry["redundant"] else "yes",
    } for entry in payload["metrics"]]

    informative = payload["metrics_adding_information"]
    return Table(
        number="10",
        caption=(
            "The reporting battery scored against the search objective, over the measured "
            f"front on each corpus that licenses the measure. {informative} of "
            f"{payload['metrics_measured']} fall below the redundancy threshold of "
            f"{payload['redundant_above']}, declared before the measurements were taken. "
            "Signs follow from each measure's own direction, since higher BLiMP accuracy "
            "and lower calibration error are both better, and carry no separate meaning."
        ),
        columns=("corpus", "measure", "reported as", "architectures",
                 "Kendall tau vs perplexity", "adds information"),
        rows=rows,
        source="artifacts/battery/analysis.json",
        coverage=f"A measure computed on fewer than {payload['minimum_architectures']} "
                 "architectures is not correlated at all: over a handful of points a rank "
                 "correlation reports its sample size.",
    )


def table_cost_matched(root: Path) -> Table:
    """The shape comparison repeated across corpora, seeds and both definitions of cost."""
    payload = _read(root, "characterisation/cost_matched.json",
                    "python -m nas4textgen.cli matched")
    rows = [{
        "corpus": entry["corpus"],
        "seed": "pooled" if entry["seed"] is None else str(entry["seed"]),
        "matched on": entry["cost"].replace("_", " "),
        "architectures": str(entry["architectures"]),
        "bands won by width": f"{entry['bands_won_by_width']} of {entry['bands']}",
    } for entry in payload["comparisons"] if entry["seed"] is None]

    tally = payload["bands_won_by_width"]
    return Table(
        number="11",
        caption=(
            "The widest third against the deepest third within bands of equal cost, on "
            "every corpus and under both cost objectives. The row marked *front only* is "
            "the control: the search corpus restricted to the same 220 architectures the "
            "other corpora ran."
        ),
        columns=("corpus", "seed", "matched on", "architectures", "bands won by width"),
        rows=rows,
        source="artifacts/characterisation/cost_matched.json",
        coverage="Per-seed rows are in the artifact; only the pooled rows are shown here.",
    )


def table_replication_tiers(root: Path) -> Table:
    """What each reproduction tier costs and what it settles.

    Read from the dictionary the replication document is also built from, so the paper and
    the instructions cannot describe a tier differently. They already had: the paper
    promised the verification tier in under two minutes where the code says ten.

    This is the one table with no artifact behind it, because a tier is a declaration about
    what someone else's machine will need rather than a measurement of ours.
    """
    from .replication import TIER_REQUIREMENTS, Tier

    order = (Tier.VERIFY, Tier.SMOKE, Tier.PARTIAL, Tier.FULL)
    return Table(
        number="12",
        caption=(
            "The four tiers the analysis reproduces in, the hardware each one needs and "
            "what each one establishes. The tiers nest: each runs everything the tier "
            "above it runs. What each costs to run is declared in the replication "
            "document published with the benchmark."
        ),
        columns=("tier", "hardware", "wall clock", "what it establishes"),
        rows=[{
            "tier": tier.value,
            "hardware": TIER_REQUIREMENTS[tier]["hardware"],
            "wall clock": TIER_REQUIREMENTS[tier]["compact"]["wall_clock"],
            "what it establishes": TIER_REQUIREMENTS[tier]["compact"]["establishes"],
        } for tier in order],
        source="src/nas4textgen/replication.py",
        # Every cell here is a phrase, so the columns have to wrap, and they are set ragged
        # right: justified text in a column this narrow opens word spaces wide enough to
        # read as gaps in the table. The widths are shares of the text block and sum with
        # the gaps between columns to just under it.
        alignment=(r"@{}l " + RAGGED + r"p{0.25\linewidth} " + RAGGED
                   + r"p{0.15\linewidth} " + RAGGED + r"p{0.45\linewidth}@{}"),
    )


TABLES = (
    table_search_space,
    table_benchmark_coverage,
    table_depth_versus_width,
    table_gene_effects,
    table_surrogate_slate,
    table_campaign_cost,
    table_design_rules,
    table_label_budget,
    table_transfer,
    table_battery,
    table_cost_matched,
    table_replication_tiers,
)


def generate(root: Path) -> tuple[list[Table], list[str]]:
    """Build every table. Returns what was produced and what could not be, with why.

    Partial generation is deliberate. During a campaign some artifacts do not exist yet, and
    a command that refused to produce anything until all of them did would be useless for
    the months that matters. What it must not do is produce a table quietly missing rows.
    """
    produced: list[Table] = []
    missing: list[str] = []
    for build in TABLES:
        try:
            produced.append(build(root))
        except MissingArtifact as exc:
            missing.append(f"{build.__name__}: {str(exc).splitlines()[0]}")
    return produced, missing


def render_all(tables: list[Table], missing: list[str]) -> str:
    parts = [
        "# Manuscript tables",
        "",
        "Generated by `python -m nas4textgen.cli manuscript` from the artifact directory.",
        "Do not edit: a number typed into a manuscript is a number that can disagree with",
        "the artifact it came from, and by the time anyone notices the paper is submitted.",
        "",
    ]
    for table in tables:
        parts += [table.render(), ""]
    if missing:
        parts += [
            "## Not yet available",
            "",
            "These tables need artifacts the pipeline has not written. Each names the",
            "command that produces its input.",
            "",
            *(f"- {entry}" for entry in missing),
            "",
        ]
    return "\n".join(parts)
