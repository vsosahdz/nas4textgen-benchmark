"""Figures, generated from the artifact directory for the same reason the tables are.

The manuscript reached ten tables and no figures, which is unusual for the venue and leaves
the central result — an exact front over an exhaustively measured space — as a column of
numbers. The four here are the ones that say something a table says less well.

**Generated, never drawn.** A figure exported once from a notebook is a figure that disagrees
with its source as soon as a measurement is rerun, and the disagreement is invisible because
nobody re-reads a PDF. Each function below reads an artifact and returns a path; a changed
measurement changes the figure on the next build or fails loudly because the artifact is
missing.

**PDF, not PNG.** The text in a figure should be selectable and the lines should survive
being printed, and a reviewer reading at 400% should not meet pixels.
"""

from __future__ import annotations

import csv
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

#: Figure widths in inches, sized so that nothing has to be scaled in LaTeX: a figure
#: scaled to fit prints its labels at a size nobody chose.
#:
#: ``WIDE_WIDTH`` is the text width of the target class, measured rather than guessed. The
#: ``elsarticle`` preprint layout at 12pt sets 390pt, which is 5.40 inches. Drawn at the
#: earlier 7.0 the wide figures were reduced by 23 per cent to fit, so a label specified at
#: 7.5pt reached the page at 5.8pt, smaller than anything the figure asked for.
COLUMN_WIDTH = 3.4
WIDE_WIDTH = 5.4

#: Everything is grey except what the figure is about. A reader should be able to say what
#: a figure argues before reading its caption.
MUTED = "#c8ccd4"
ACCENT = "#1f4e79"
SECOND = "#b4461e"


class ArtifactMissing(RuntimeError):
    """Raised when a figure's data has not been produced yet."""


@dataclass(frozen=True)
class Figure:
    """One generated figure and what it was drawn from."""

    name: str
    caption: str
    source: str
    path: Path

    def as_latex(self) -> str:
        # The caption goes through the same inline conversion the prose does. Written raw
        # it carried an em dash the document font cannot set, which XeTeX reports as a
        # warning and renders as nothing.
        from .overleaf import _inline

        return "\n".join([
            # Every slot, as the tables use, so that a figure is not pushed pages past the
            # sentence that sends the reader to it when the float queue is busy.
            r"\begin{figure}[!htb]",
            r"  \centering",
            # At its natural size unless that is wider than the text block. Included at
            # `\linewidth`, a figure drawn at 3.4 inches was scaled up by nearly two in a
            # 6.5-inch text block, which printed the axis labels larger than the body text;
            # included at natural size, the wider diagrams ran into the right margin. The
            # box is a ceiling rather than a target, so each figure keeps the type size it
            # was drawn with and only the oversized ones are reduced.
            r"  \adjustbox{max width=\linewidth}{\includegraphics{figures/"
            + self.path.name + "}}",
            r"  \caption{" + _inline(self.caption) + "}",
            r"  \label{fig:" + self.name + "}",
            r"\end{figure}",
        ])


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    return plt


def _read_json(root: Path, relative: str, command: str) -> dict:
    path = root / relative
    if not path.exists():
        raise ArtifactMissing(f"{relative} has not been written. Run: {command}")
    return json.loads(path.read_text(encoding="utf-8"))


def _on_front(row: dict) -> bool:
    """Whether a table row is on the measured front.

    The CSV holds ``1`` and ``0``; comparing against ``"1.0"`` matched nothing and drew a
    front of zero architectures onto the figure whose entire subject is the front. The
    value is parsed rather than string-matched so a change of format cannot empty it again.
    """
    try:
        return float(row.get("on_measured_front", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _read_table(root: Path) -> list[dict]:
    path = root / "benchmark/table.csv"
    if not path.exists():
        raise ArtifactMissing(
            "benchmark/table.csv has not been written. Run: "
            "python -m nas4textgen.cli table")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def figure_exact_front(root: Path, out: Path) -> Figure:
    """Every measured architecture, with the exact front marked.

    The figure the paper exists to be able to draw. Each point is a trained model rather
    than an estimate, so the front is the answer and not the best attempt.
    """
    from .transfer import _non_dominated

    plt = _style()
    rows = _read_table(root)
    measured = [r for r in rows if r.get("perplexity") not in (None, "", "nan")]
    if not measured:
        raise ArtifactMissing("the benchmark table holds no measured perplexities")

    #: One point per architecture, at the median perplexity of its replicates, because that
    #: is what the front was computed over. Plotting the rows instead put the 4 replicated
    #: architectures on the figure twice, and a replicate of a front member can land off the
    #: line that its own architecture is on.
    by_architecture: dict[tuple, list[dict]] = {}
    for row in measured:
        key = (row["h"], row["d_head"], row["n"], row["ffn"], row["kv"])
        by_architecture.setdefault(key, []).append(row)

    #: The front is over three objectives and the plane holds two of them, so 68 of the 220
    #: members sit above the parameters-and-perplexity front drawn through the others. They
    #: are not dominated: each is on the front because of the cache size this plane does not
    #: show. Drawn as one undifferentiated set they read as an error, which is what a reader
    #: reported, so the two groups are marked apart and the caption says why.
    entries = []
    for replicates in by_architecture.values():
        perplexity = sorted(float(r["perplexity"]) for r in replicates)
        middle = len(perplexity) // 2
        median = (perplexity[middle] if len(perplexity) % 2
                  else (perplexity[middle - 1] + perplexity[middle]) / 2)
        entries.append(((float(replicates[0]["params"]), median),
                        any(_on_front(r) for r in replicates)))

    points = [p for p, _ in entries]
    in_plane = set(_non_dominated(points))
    joint = {i for i, (_, on) in enumerate(entries) if on}

    rest = [points[i] for i in range(len(points)) if i not in joint]
    plane = sorted(points[i] for i in in_plane)
    cache = [points[i] for i in sorted(joint - in_plane)]

    # Three markers of different shape, not three colours. The journal is printed in grey
    # as often as in colour, and two tones of the same shape are one group there.
    fig, axes = plt.subplots(figsize=(COLUMN_WIDTH, 2.6))
    axes.scatter([x for x, _ in rest], [y for _, y in rest], s=4, c=MUTED, marker=".",
                 linewidths=0, label=f"dominated ({len(rest):,})")
    axes.scatter([x for x, _ in cache], [y for _, y in cache], s=16, marker="^",
                 facecolors="none", edgecolors=SECOND, linewidths=0.7, zorder=2,
                 label=f"on the front by cache size ({len(cache)})")
    axes.plot([x for x, _ in plane], [y for _, y in plane], "-", color=ACCENT, linewidth=1.1)
    axes.scatter([x for x, _ in plane], [y for _, y in plane], s=9, c=ACCENT, marker="o",
                 linewidths=0, zorder=3,
                 label=f"on the front in this plane ({len(plane)})")
    axes.set_xscale("log")
    axes.set_xlabel("parameters")
    axes.set_ylabel("held-out perplexity")
    axes.legend(frameon=False, loc="upper right")
    path = out / "exact-front.pdf"
    fig.savefig(path)
    plt.close(fig)

    architectures = len({(r["h"], r["d_head"], r["n"], r["ffn"], r["kv"]) for r in rows})
    return Figure(
        name="exact-front",
        caption=(
            f"The measured space, in two of the three objectives. All {architectures:,} "
            f"architectures are trained to the same token budget under one protocol, so the "
            f"{len(plane) + len(cache)} on the front are the Pareto-optimal set rather than "
            f"the best set any search returned. The {len(plane)} drawn as a line are the "
            f"front in this plane; the other {len(cache)} lie above it and are not dominated, "
            "each one reaching the front on the key-value cache size the plane does not "
            "show."),
        source="artifacts/benchmark/table.csv",
        path=path)


def figure_label_budget(root: Path, out: Path) -> Figure:
    """Where a trained surrogate starts beating the untrained one."""
    plt = _style()
    payload = _read_json(root, "surrogate/budget_curve.json",
                         "python -m nas4textgen.cli surrogate")
    sizes = payload["sizes"]
    floors = payload["floor_tau_by_size"]
    trivial = {"sort_by_parameters", "linear_on_genes"}

    best = []
    for size in sizes:
        values = [point["kendall_tau"]
                  for name, entry in payload["per_candidate"].items()
                  if name not in trivial
                  for point in entry["points"] if point["training_rows"] == size]
        best.append(max(values) if values else float("nan"))
    floor = [floors[str(size)] for size in sizes]

    fig, axes = plt.subplots(figsize=(COLUMN_WIDTH, 2.4))
    axes.axhline(floor[0], color=SECOND, linewidth=1.1, linestyle="--",
                 label=f"sort by parameters ({floor[0]:.3f})")
    axes.plot(sizes, best, "-o", color=ACCENT, linewidth=1.1, markersize=3,
              label="best trained predictor")
    crossing = next((s for s, b, f in zip(sizes, best, floor, strict=True) if b > f), None)
    if crossing is not None:
        axes.axvline(crossing, color="#888", linewidth=0.7, linestyle=":")
        # Above the curve and to the right of the crossing: placed at the minimum it sat
        # on top of the legend, which is the one place a label must not go.
        axes.annotate(f"{crossing} labels", xy=(crossing, max(best)),
                      xytext=(5, -4), textcoords="offset points", fontsize=7, color="#555")
    axes.set_xscale("log")
    # Ticks at the budgets actually measured. The log locator puts minor ticks at 3, 4 and
    # 6 times each decade, and at this width their labels overlap into illegibility.
    axes.set_xticks(sizes)
    axes.set_xticklabels([str(s) for s in sizes])
    axes.minorticks_off()
    axes.set_xlabel("trained labels")
    axes.set_ylabel(r"Kendall $\tau$")
    axes.legend(frameon=False, loc="lower right")
    path = out / "label-budget.pdf"
    fig.savefig(path)
    plt.close(fig)

    return Figure(
        name="label-budget",
        caption=(
            "Rank correlation against the number of trained labels. The untrained floor "
            "costs nothing to compute, and no trained predictor clears it below about "
            f"{crossing} labels, which is the regime in which most model-based "
            "searches operate."),
        source="artifacts/surrogate/budget_curve.json",
        path=path)


def figure_battery(root: Path, out: Path) -> Figure:
    """Two battery measures against the objective, one that agrees and one that does not.

    Drawn as a pair after a single-panel version was drafted. On its own, degeneration
    against perplexity shows a tight relationship and reads as an argument that the battery
    is redundant — which is what its rank correlation of 0.793 says. The finding is not that
    the battery disagrees with the objective; it is that *whether* it disagrees depends on
    the measure and the corpus, and a figure showing only the agreeing case argues the
    opposite of the section it illustrates.
    """
    plt = _style()
    from .artifacts import artifacts_root
    from .azure.data import load_manifests
    from .battery import collect

    try:
        manifests = [m for m in load_manifests(artifacts_root()) if m.get("battery")]
    except FileNotFoundError as exc:
        raise ArtifactMissing(str(exc)) from exc

    panels = [
        ("tinystories", "repetition_rate", "repetition_rate_4", "repeated 4-grams"),
        ("wikitext103", "calibration", "expected_calibration_error",
         "expected calibration error"),
    ]
    gathered = []
    for corpus, metric, key, label in panels:
        points = []
        for row in collect(manifests, corpus).values():
            entry = row["battery"].get(metric)
            if isinstance(entry, dict) and isinstance(entry.get(key), (int, float)):
                points.append((sum(row["perplexity"]) / len(row["perplexity"]),
                               float(entry[key])))
        if len(points) < 10:
            raise ArtifactMissing(f"too few battery runs for {corpus}/{metric}")
        gathered.append((corpus, label, points))

    # The two panels are laid out with a gap wide enough for the right one's axis label,
    # which otherwise prints over the left one's points. The tight bounding box trims the
    # canvas after the layout, so the gap costs no width in the compiled page.
    fig, axes = plt.subplots(1, 2, figsize=(WIDE_WIDTH, 2.4), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06, wspace=0.09)
    taus = {}
    for axis, (corpus, label, points) in zip(axes, gathered, strict=True):
        from scipy.stats import kendalltau

        tau, _ = kendalltau([x for x, _ in points], [y for _, y in points])
        taus[corpus] = tau
        axis.scatter([x for x, _ in points], [y for _, y in points], s=7,
                     c=ACCENT if abs(tau) >= 0.6 else SECOND, linewidths=0, alpha=0.75)
        axis.set_xlabel("held-out perplexity")
        axis.set_ylabel(label)
        axis.set_title(rf"{corpus}  ($\tau$ = {tau:+.3f})", fontsize=8)
    path = out / "battery.pdf"
    fig.savefig(path)
    plt.close(fig)

    return Figure(
        name="battery",
        caption=(
            "Two battery measures against the search objective, over the front. Degeneration "
            f"on TinyStories tracks perplexity closely ($\\tau$ = {taus['tinystories']:+.3f}) "
            "and adds little; calibration on the search corpus does not "
            f"($\\tau$ = {taus['wikitext103']:+.3f}), and none of the ten best architectures "
            "by perplexity are among the ten best calibrated. Whether a measure is worth "
            "reporting is a property of the measure and the corpus, not of the battery."),
        source="artifacts/manifests/runs.jsonl",
        path=path)


#: Every length in the protocol figure is in inches, and its axes are made to span the
#: figure with one data unit to the inch, so that a label measured before the layout keeps
#: its width after it. Sizing the boxes in data units instead made the padding depend on
#: the axis range, which the layout itself was changing: four of the five labels crossed
#: their box edge.
PROTOCOL_FONTSIZE = 7.5
PROTOCOL_PAD_X, PROTOCOL_PAD_Y = 0.10, 0.07
PROTOCOL_GAP_X, PROTOCOL_GAP_Y, PROTOCOL_MARGIN = 0.19, 0.16, 0.03


def protocol_layout(stages, branches, sizes):
    """Place the protocol boxes, each one sized to the label it has to hold.

    `sizes` maps a label to the width and height it measures, in inches. Drawing every box
    at one width instead left "1,099 architectures" crossing a box that also had to hold
    "transfer". The three stages run left to right and the two branches stack to the right
    of the last of them.
    """
    height = max(h for _, h in sizes.values()) + 2 * PROTOCOL_PAD_Y
    widths = [sizes[label][0] + 2 * PROTOCOL_PAD_X for label, _ in stages]
    branch_width = max(sizes[label][0] for label, _ in branches) + 2 * PROTOCOL_PAD_X

    total_height = 2 * height + PROTOCOL_GAP_Y + 2 * PROTOCOL_MARGIN
    branch_x = PROTOCOL_MARGIN + sum(widths) + PROTOCOL_GAP_X * len(stages)
    total_width = branch_x + branch_width + PROTOCOL_MARGIN

    placed, x = [], PROTOCOL_MARGIN
    for (label, colour), width in zip(stages, widths, strict=True):
        placed.append((x, total_height / 2 - height / 2, width, label, colour))
        x += width + PROTOCOL_GAP_X
    branch_y = (total_height - PROTOCOL_MARGIN - height, PROTOCOL_MARGIN)
    for (label, colour), y in zip(branches, branch_y, strict=True):
        placed.append((branch_x, y, branch_width, label, colour))
    return placed, height, branch_x, branch_y, (total_width, total_height)


def figure_protocol(root: Path, out: Path) -> Figure:
    """What is measured at each stage, and what each stage refuses to measure.

    Drawn rather than read from an artifact: it describes the protocol, and the protocol is
    a decision rather than a measurement. The counts in it are read from the artifacts so
    that the one part which can drift does not.
    """
    plt = _style()
    space = _read_json(root, "space/enumeration_summary.json",
                       "python -m nas4textgen.cli enumerate")
    rows = _read_table(root)
    front = sum(1 for r in rows if _on_front(r))
    # `cardinality` is the key the enumeration writes. Falling back to the row count gave
    # 1,103 — the table's rows, which include replicates — against the 1,099 architectures
    # the paper states everywhere else.
    total = space["cardinality"]
    measured = len({(r["h"], r["d_head"], r["n"], r["ffn"], r["kv"]) for r in rows})

    fig, axes = plt.subplots(figsize=(WIDE_WIDTH, 1.9))
    axes.set_axis_off()
    axes.set_position((0, 0, 1, 1))

    def measure(label):
        """Width and height of a label in inches."""
        artist = axes.text(0, 0, label, ha="center", va="center",
                           fontsize=PROTOCOL_FONTSIZE)
        fig.canvas.draw()
        box = artist.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        artist.remove()
        return box.width, box.height

    stages = [
        (f"search space\n{total:,} architectures", MUTED),
        (f"exhaustive census\n{measured:,} trained", ACCENT),
        (f"exact front\n{front} architectures", ACCENT),
    ]
    branches = [("battery\n2 corpora", SECOND), ("transfer\n3 corpora, 3 seeds", SECOND)]

    sizes = {label: measure(label) for label, _ in stages + branches}
    placed, height, branch_x, branch_y, size = protocol_layout(stages, branches, sizes)
    total_width, total_height = size
    fig.set_size_inches(total_width, total_height)
    axes.set_xlim(0, total_width)
    axes.set_ylim(0, total_height)
    middle = total_height / 2

    for x, y, width, label, colour in placed:
        axes.add_patch(plt.Rectangle((x, y), width, height, fill=False, edgecolor=colour,
                                     linewidth=1.2))
        axes.text(x + width / 2, y + height / 2, label, ha="center", va="center",
                  fontsize=PROTOCOL_FONTSIZE, color="#222")

    for index in range(len(stages) - 1):
        start = placed[index][0] + placed[index][2]
        axes.annotate("", xy=(start + PROTOCOL_GAP_X, middle), xytext=(start, middle),
                      arrowprops={"arrowstyle": "->", "color": "#666", "linewidth": 0.9})
    last = placed[len(stages) - 1]
    for y in branch_y:
        axes.annotate("", xy=(branch_x, y + height / 2),
                      xytext=(last[0] + last[2], middle),
                      arrowprops={"arrowstyle": "->", "color": "#666", "linewidth": 0.9})
    path = out / "protocol.pdf"
    fig.savefig(path)
    plt.close(fig)

    return Figure(
        name="protocol",
        caption=(
            "The measurement protocol. The census trains every architecture in the space, "
            "so the front it yields is exact. The battery and the transfer study read that "
            "front; neither searches for a front of its own."),
        source="artifacts/space/enumeration_summary.json",
        path=path)


def figure_search_space(root: Path, out: Path) -> Figure:
    """The search space, as a diagram of what is fixed, what is searched and what follows.

    Drawn rather than measured, because a search space is a design decision. The counts in
    it are read from the enumeration so that the one part which can drift does not.
    """
    plt = _style()
    space = _read_json(root, "space/enumeration_summary.json",
                       "python -m nas4textgen.cli enumerate")

    fig, axes = plt.subplots(figsize=(WIDE_WIDTH, 4.2))
    axes.set_axis_off()
    axes.set_xlim(0, 10.1)
    axes.set_ylim(0, 10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    def box(x, y, w, h, label, colour, fontsize=7, radius=0.12):
        axes.add_patch(plt.Rectangle(
            (x, y), w, h, facecolor=colour, edgecolor="#333", linewidth=0.8,
            joinstyle="round"))
        axes.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                  fontsize=fontsize, color="#111", linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        axes.annotate("", xy=(x1, y1), xytext=(x0, y0),
                      arrowprops={"arrowstyle": "-|>", "color": "#444",
                                  "linewidth": 0.9})

    # The stack, read top to bottom, with the searched depth bracketing the blocks.
    column, width, height = 0.5, 2.2, 0.62
    stack = [
        (9.0, "Input", "none"),
        (8.0, "Token embedding", FIXED),
        (7.0, "Positional embedding", FIXED),
        (5.9, "Block 1", SEARCHED),
        (5.0, "Block 2", SEARCHED),
        (3.6, "Block $n$", SEARCHED),
        (2.5, "Final layer norm", FIXED),
        (1.5, "Weight-tied head", FIXED),
        (0.5, "Output", "none"),
    ]
    for y, label, colour in stack:
        if colour == "none":
            axes.add_patch(plt.Polygon(
                [(column + width / 2, y + height), (column + width, y + height / 2),
                 (column + width / 2, y), (column, y + height / 2)],
                facecolor="#f4f4f4", edgecolor="#333", linewidth=0.8))
            axes.text(column + width / 2, y + height / 2, label, ha="center",
                      va="center", fontsize=7)
        else:
            # The longest label in this column is "Positional embedding", which overruns
            # the box at the common size. Sized by the label rather than by hand so that
            # renaming a stage cannot quietly push its text past the border again.
            box(column, y, width, height, label, colour,
                fontsize=6.4 if len(label) <= 18 else 5.7)

    for (y0, label, _), (y1, _, _) in zip(stack, stack[1:], strict=False):
        # The ellipsis marks the blocks the depth gene repeats, and only those. Drawing it
        # wherever two boxes sit more than a unit apart put one between the positional
        # embedding and the first block, where nothing is elided.
        if label == "Block 2":
            axes.text(column + width / 2 + 0.42, (y0 + y1) / 2 + 0.3, "$\\vdots$",
                      ha="center", va="center", fontsize=11, color="#444")
        arrow(column + width / 2, y0, column + width / 2, y1 + height)

    axes.annotate("", xy=(column - 0.25, 3.6), xytext=(column - 0.25, 6.52),
                  arrowprops={"arrowstyle": "<|-|>", "color": "#1f4e79",
                              "linewidth": 0.9})
    axes.text(column - 0.45, 5.0, "depth $n$", rotation=90, ha="center", va="center",
              fontsize=6.2, color="#1f4e79")

    # The block, expanded.
    axes.add_patch(plt.Rectangle((3.4, 3.9), 6.3, 4.4, fill=False,
                                 edgecolor="#777", linewidth=0.8, linestyle=":"))
    axes.text(6.55, 8.05, "Transformer block", ha="center", fontsize=6.8, color="#444")
    arrow(column + width, 5.9 + height / 2, 3.4, 6.3)

    box(3.75, 6.6, 2.6, 1.1, "Grouped-query\nattention", FIXED, fontsize=6.6)
    box(6.65, 6.6, 2.7, 1.1, "heads $h$\nkey-value heads $kv$", SEARCHED, fontsize=6.2)
    arrow(6.35, 7.15, 6.65, 7.15)
    box(3.75, 5.2, 2.6, 1.1, "Feed-forward\n(GELU)", FIXED, fontsize=6.6)
    box(6.65, 5.2, 2.7, 1.1, "ratio $ffn$", SEARCHED, fontsize=6.6)
    arrow(6.35, 5.75, 6.65, 5.75)
    arrow(5.05, 6.6, 5.05, 6.3)
    box(3.75, 4.15, 5.6, 0.8,
        "head dimension $d_{head}$\nwidth $e = h \\times d_{head}$ follows from it",
        DERIVED, fontsize=6.2)

    # The encoding, with the anchor written out in it above a second architecture. The
    # anchor row is the point of the strip: it shows that DistilGPT2 is a member of the
    # space and not the thing the space was built next to.
    from .space.genotype import DISTILGPT2

    genes = ["$h$", "$d_{head}$", "$n$", "$ffn$", "$kv$"]
    anchor = [str(getattr(DISTILGPT2, g))
              for g in ("h", "d_head", "n", "ffn", "kv")]
    example = ["8", "32", "6", "4", "1"]
    cell, left, y = 0.82, 3.45, 3.0
    for index, gene in enumerate(genes):
        box(left + index * cell, y, cell, 0.5, gene, SEARCHED, fontsize=6.6)
    for row, (values, note) in enumerate((
            (anchor, f"DistilGPT2, $e = {DISTILGPT2.e}$"),
            (example, r"another member, $e = 256$"))):
        row_y = y - 0.53 * (row + 1)
        for index, value in enumerate(values):
            box(left + index * cell, row_y, cell, 0.5, value, "#ffffff", fontsize=6.6)
        axes.text(left + 5 * cell + 0.2, row_y + 0.25, note, fontsize=6.2,
                  va="center", color="#444")
    axes.text(left + 5 * cell + 0.2, y + 0.25, "genotype", fontsize=6.2, va="center",
              color="#444")

    # Wrapped to the width that is actually free rather than to wherever the sentence
    # happened to end. Matplotlib does not clip text to the axes, and `savefig` with a
    # tight bounding box grows the figure to contain whatever overhangs it: one long line
    # here made the saved figure 6.9 inches wide, which LaTeX then scaled back to the
    # 5.4-inch text block, shrinking every label in the drawing by a fifth.
    summary = (
        f"{space['total_before_ceiling']:,} combinations on the grid; "
        f"{space['excluded']:,} removed by the "
        f"{space['parameter_ceiling'] / 1e6:.0f}M parameter ceiling and the requirement "
        f"that $kv$ divide $h$, leaving {space['cardinality']:,} valid architectures "
        f"spanning {space['params_range'][0] / 1e6:.2f} to "
        f"{space['params_range'][1] / 1e6:.2f}M parameters."
    )
    axes.text(3.45, 1.45, "\n".join(textwrap.wrap(summary, width=74)),
              fontsize=6.2, va="top", color="#222", linespacing=1.6)

    # A row rather than a column, set in the band above the stack that nothing else
    # occupies. As a column on the right it was the only thing in a fifth of the figure.
    for index, (colour, label) in enumerate((
            (FIXED, "fixed"), (SEARCHED, "searched"), (DERIVED, "derived"))):
        swatch = 3.5 + index * 1.55
        box(swatch, 9.0, 0.45, 0.36, "", colour)
        axes.text(swatch + 0.58, 9.18, label, fontsize=6.6, va="center")

    path = out / "search-space.pdf"
    fig.savefig(path)
    plt.close(fig)

    return Figure(
        name="search-space",
        caption=(
            "The search space. Five genes are searched and the embedding width follows "
            f"from two of them, which is what keeps every one of the {space['cardinality']:,} "
            "architectures internally consistent: a space cannot hold its embedding layers "
            "fixed while searching the dimension they are built from. The genotype strip "
            "writes out the anchor architecture in the encoding, which is the sense in "
            "which the space is based on DistilGPT2: the anchor is a member of it."),
        source="artifacts/space/enumeration_summary.json",
        path=path)


FIGURES = (figure_search_space, figure_exact_front, figure_protocol,
           figure_label_budget, figure_battery)


def generate(root: Path, out: Path) -> tuple[list[Figure], list[str]]:
    """Build every figure. Returns what was produced and what could not be, with why."""
    out.mkdir(parents=True, exist_ok=True)
    produced, missing = [], []
    for build in FIGURES:
        try:
            produced.append(build(root, out))
        except ArtifactMissing as exc:
            missing.append(f"{build.__name__}: {exc}")
    return produced, missing


#: The prior submission's search-space diagram used yellow for fixed components and cyan
#: for searched ones. The convention is kept so a reader of both papers sees the same
#: language, with a third colour for the width this space derives rather than searches:
#: that derivation is the difference between the two spaces and the figure should show it.
FIXED = "#ffe680"
SEARCHED = "#bdeaf2"
DERIVED = "#e8d5f0"
