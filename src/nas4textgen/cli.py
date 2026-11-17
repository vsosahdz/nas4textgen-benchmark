"""Command-line entry points for the stages that need no GPU.

Everything reachable from here is part of the ``verify`` replication tier: it runs on
CPU, on any machine, and reproduces bit-exactly.

    python -m nas4textgen.cli gate
    python -m nas4textgen.cli enumerate
    python -m nas4textgen.cli designs
    python -m nas4textgen.cli fetch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .artifacts import write_json, write_table
from .audit_prior import PRIOR_PROMPT_FRACTION
from .azure.workspace import ImageIsStale
from .corpora.fetch import fetch_all
from .space.cost import embedding_share
from .space.designs import (
    fill_distance,
    latin_hypercube_design,
    maximin_order,
    uniform_random_design,
    waves,
)
from .space.enumerate import (
    PARAMETER_CEILING,
    enumerate_with_ceiling,
    non_dominated,
    reference_point,
)
from .space.gate import run_gate
from .space.genotype import GENES
from .submission import HIGHLIGHT_CHARACTERS

#: Wave sizes for the sweep. Nested by construction, so wave k+1 contains wave k, and
#: each is a usable space-filling sample on its own.
WAVE_SIZES = (150, 300, 600, 1000, 1099)


def cmd_gate(_: argparse.Namespace) -> int:
    report = run_gate()
    print()
    print(report.render())
    path = write_json("gate/structural_gate.json", report.as_dict())
    print(f"\n  written: {path}")
    if not report.passed:
        print("\n  No sweep may be submitted until every check passes.")
        return 1
    return 0


def cmd_enumerate(args: argparse.Namespace) -> int:
    enumeration = enumerate_with_ceiling(ceiling=args.ceiling)
    objectives = enumeration.objectives
    front = non_dominated(objectives)

    rows = []
    on_front = set(int(i) for i in front)
    for index, genotype in enumerate(enumeration.genotypes):
        rows.append(
            {
                **{gene: getattr(genotype, gene) for gene in GENES},
                "e": genotype.e,
                "intermediate": genotype.intermediate,
                "params": int(objectives[index, 0]),
                "kv_bytes": int(objectives[index, 1]),
                "embedding_share": round(embedding_share(genotype), 6),
                "on_cost_front": int(index in on_front),
            }
        )
    columns = [
        *GENES,
        "e",
        "intermediate",
        "params",
        "kv_bytes",
        "embedding_share",
        "on_cost_front",
    ]
    table = write_table("space/enumeration.csv", rows, columns)
    summary = write_json(
        "space/enumeration_summary.json",
        {
            "cardinality": len(enumeration),
            "total_before_ceiling": enumeration.total_before_ceiling,
            "parameter_ceiling": enumeration.ceiling,
            "excluded": enumeration.excluded,
            "excluded_fraction": round(enumeration.excluded_fraction, 6),
            "reference_context": enumeration.context,
            "reference_point": [float(x) for x in reference_point(objectives)],
            "cost_front_size": len(front),
            "cost_front_indices": [int(i) for i in front],
            "params_range": [int(objectives[:, 0].min()), int(objectives[:, 0].max())],
            "kv_bytes_range": [int(objectives[:, 1].min()), int(objectives[:, 1].max())],
        },
    )
    print()
    print(f"  architectures under the ceiling  {len(enumeration):,}")
    print(f"  excluded by the ceiling          {enumeration.excluded:,} "
          f"({enumeration.excluded_fraction:.1%})")
    print(f"  exact cost front                 {len(front)} point"
          f"{'s' if len(front) != 1 else ''}")
    print(f"\n  written: {table}\n           {summary}")
    return 0


def cmd_designs(args: argparse.Namespace) -> int:
    enumeration = enumerate_with_ceiling(ceiling=args.ceiling)
    space = enumeration.genotypes

    order = maximin_order(space, seed=args.seed)
    reproduced = maximin_order(space, seed=args.seed)
    assert np.array_equal(order, reproduced), "maximin order is not reproducible"

    wave_indices = waves(order, list(WAVE_SIZES))
    for smaller, larger in zip(wave_indices[:-1], wave_indices[1:], strict=False):
        assert set(smaller.tolist()) <= set(larger.tolist()), "waves do not nest"

    comparison: dict[str, dict] = {}
    for size in (150, 300, 600):
        lhs, notes = latin_hypercube_design(space, size, seed=args.seed)
        rnd = uniform_random_design(space, size, seed=args.seed)
        comparison[str(size)] = {
            "fill_distance": {
                "maximin": round(fill_distance(space, order[:size]), 5),
                "latin_hypercube": round(fill_distance(space, lhs), 5),
                "uniform_random": round(fill_distance(space, rnd), 5),
            },
            "latin_hypercube_stratification": notes,
        }

    path = write_json(
        "space/sampling_designs.json",
        {
            "cardinality": len(space),
            "seed": args.seed,
            "wave_sizes": list(WAVE_SIZES),
            "maximin_order": [int(i) for i in order],
            "waves_nest": True,
            "order_is_reproducible": True,
            "comparison": comparison,
        },
    )
    print()
    print("  fill distance — lower is more space-filling")
    print(f"  {'budget':>8}{'maximin':>12}{'LHS':>12}{'random':>12}")
    print("  " + "-" * 44)
    for size, entry in comparison.items():
        f = entry["fill_distance"]
        print(f"  {size:>8}{f['maximin']:>12.4f}{f['latin_hypercube']:>12.4f}"
              f"{f['uniform_random']:>12.4f}")
    print("\n  waves nest: yes    order reproducible: yes")
    print(f"  written: {path}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Retrieve corpora into the cache. The only stage that reaches the network."""
    if args.blimp:
        from .corpora.cache import cache_root
        from .evaluation import blimp

        written = blimp.fetch_items(cache_root(), limit_per_paradigm=args.limit)
        print(f"\n  BLiMP: {written:,} minimal pairs cached\n")
        return 0

    outcomes = fetch_all(
        corpora=args.corpus or None, limit=args.limit, force=args.force
    )
    print()
    print(f"  {'corpus':<16}{'split':<12}{'status':<14}{'rows':>10}")
    print("  " + "-" * 60)
    for outcome in outcomes:
        print(f"  {outcome.corpus:<16}{outcome.split:<12}{outcome.status:<14}"
              f"{outcome.rows:>10,}")
        if outcome.status == "unavailable":
            print(f"  {'':<16}{'':<12}{outcome.detail[:56]}")
    print()

    unavailable = [o for o in outcomes if o.status == "unavailable"]
    path = write_json(
        "corpora/retrieval.json",
        {
            "outcomes": [o.as_dict() for o in outcomes],
            "retrieved": sum(o.status == "retrieved" for o in outcomes),
            "already_cached": sum(o.status == "cached" for o in outcomes),
            "unavailable": len(unavailable),
            "limit_applied": args.limit,
        },
    )
    print(f"  written: {path}")
    if unavailable:
        print()
        print(f"  {len(unavailable)} split(s) unavailable. The campaign can proceed on what")
        print("  was retrieved; the manuscript states which corpora its results rest on")
        print("  rather than implying all four.")
    return 0


def cmd_canary(args: argparse.Namespace) -> int:
    """Show, or submit, the four architectures that gate every fan-out."""
    from .azure.canary import select_canary, timeout_source

    members = select_canary()
    print()
    print(f"  {'role':<16}{'genotype':<34}{'params':>13}{'FLOPs/tok':>14}{'n*e':>8}")
    print("  " + "-" * 85)
    for m in members:
        g = m.genotype
        genes = f"h{g.h} d{g.d_head} n{g.n} ffn{g.ffn} kv{g.kv} (e={g.e})"
        print(f"  {m.role:<16}{genes:<34}{m.params:>13,}{m.flops_per_token:>14,}"
              f"{m.activation_width:>8,}")
    source = timeout_source(members)
    print(f"\n  timeout and memory bounds come from {source.role!r}, not from the largest")
    print("  by parameters: parameters here are mostly the 50,257-row embedding, so the")
    print("  parameter extreme is merely the widest model and runs well inside the bound.")

    path = write_json(
        "canary/selection.json",
        {
            "members": [m.as_dict() for m in members],
            "timeout_source": source.role,
            "bounds_rationale": (
                "Parameter count in this space is dominated by the embedding matrix, so it "
                "ranks architectures by width. The duration and memory bounds are taken "
                "from the architecture maximising forward FLOPs per token."
            ),
        },
    )
    print(f"\n  written: {path}")

    if not args.submit:
        print("\n  Nothing submitted. Re-run with --submit to launch, or --submit --dry-run")
        print("  to see exactly what would be launched.")
        return 0

    from .azure.data import require_uploaded
    from .azure.submit import JobSpec, submit_wave

    require_uploaded()
    unknown = set(args.force or ()) - {m.role for m in members}
    if unknown:
        print(f"\n  no canary member named {sorted(unknown)}; roles are "
              f"{[m.role for m in members]}")
        return 1

    specs = [
        JobSpec(
            genotype=m.genotype,
            corpus=args.corpus,
            seed=args.seed,
            token_budget=args.token_budget,
            extra_tags={"wave": "canary", "canary_role": m.role},
        )
        for m in members
    ]
    outcomes = submit_wave(
        specs,
        # The canary is the one wave allowed a provisional limit: it is what measures the
        # real one. Every wave after it is refused until that measurement exists.
        allow_provisional=True,
        cache_vintage=args.vintage,
        dry_run=args.dry_run,
        force_roles=tuple(args.force or ()),
    )
    print()
    for outcome in outcomes:
        print(f"  {outcome.status:<16}{outcome.job_name:<24}{outcome.detail}")
    write_json("canary/submission.json", {"outcomes": [o.as_dict() for o in outcomes]})
    return 0 if all(o.status != "failed" for o in outcomes) else 1


def cmd_tokenise(args: argparse.Namespace) -> int:
    """Encode every cached split once, so no job spends node time doing it again."""
    from .corpora.fetch import SPLITS
    from .corpora.tokenise import tokenise_split

    print()
    print(f"  {'corpus':<16}{'split':<12}{'sequences':>12}{'tokens':>16}")
    print("  " + "-" * 58)
    from .corpora.cache import CacheMiss

    requested = list(args.corpus or [])
    records, absent = [], []
    for corpus in requested or list(SPLITS):
        for split in SPLITS[corpus]:
            try:
                record = tokenise_split(corpus, split, force=args.force)
            except CacheMiss:
                # A corpus that was never fetched is a legitimate state, not a crash. The
                # documented flow is fetch what you need and tokenise what you have, and
                # raising here meant a subset could not be tokenised at all: CI fetched one
                # corpus, this stage encoded it correctly, and then died on the next one
                # with the successful work already done and the exit status saying failure.
                absent.append(f"{corpus}/{split}")
                print(f"  {corpus:<16}{split:<12}{'not cached':>12}{'—':>16}")
                continue
            records.append(record)
            print(f"  {corpus:<16}{split:<12}{record.sequences:>12,}{record.tokens:>16,}")

    path = write_json(
        "corpora/tokenisation.json",
        {
            "splits": [r.as_dict() for r in records],
            "total_tokens": sum(r.tokens for r in records),
            "why": (
                "Encoding once rather than per job. Re-tokenising WikiText-103 costs about "
                "a hundred seconds of node time, which across the sweep is roughly "
                "thirty-three GPU-hours spent producing an identical answer. It also fixes "
                "the token stream: a job that encodes its own input depends on the "
                "tokeniser version in its image, so a rebuild could change the stream "
                "under half a sweep."
            ),
        },
    )
    print(f"\n  written: {path}")

    if absent:
        print(f"\n  {len(absent)} split(s) not cached: {', '.join(absent)}")
        print("  Retrieve them with the corpus fetch step; downstream stages never reach")
        print("  the network.")
    # Asking for a corpus by name and not getting it is a failure; tokenising whatever
    # happens to be cached is the documented default and is not.
    if absent and requested:
        print("\n  Refusing: these were requested by name rather than encountered.")
        return 1
    if not records:
        print("\n  Refusing: nothing was cached, so nothing was encoded.")
        return 1
    return 0


def cmd_budget_study(args: argparse.Namespace) -> int:
    """Retrain a stratified subset at a larger budget, to test the ranking's stability."""
    import csv as _csv

    from .azure.budget_study import plan
    from .space.genotype import Genotype

    table = Path("artifacts/benchmark/table.csv")
    if not table.exists():
        print("\n  no benchmark table yet. Run: python -m nas4textgen.cli table")
        return 1

    with table.open(encoding="utf-8") as handle:
        measured = [
            Genotype(**{gene: int(row[gene]) for gene in GENES})
            for row in _csv.DictReader(handle)
        ]

    detail = plan(measured, args.token_budget, size=args.size)
    print()
    print(f"  {detail['architectures']} architectures, "
          f"{detail['short_budget']:,} -> {detail['long_budget']:,} tokens "
          f"({detail['multiple']}x, {detail['epochs_at_long_budget']} epochs)")
    print(f"  compute span: {detail['compute_span']['decades']} decades")
    print(f"  drawn from {len(measured)} already measured at the sweep's budget")

    path = write_json("budget_study/plan.json", detail)
    print(f"\n  written: {path}")

    if not args.submit:
        print("\n  Nothing submitted. Re-run with --submit, or --submit --dry-run first.")
        return 0

    from .azure.data import require_uploaded
    from .azure.submit import JobSpec, submit_wave

    require_uploaded()
    specs = [
        JobSpec(
            genotype=Genotype(**member["genotype"]),
            corpus=args.corpus,
            seed=args.seed,
            token_budget=detail["long_budget"],
            extra_tags={"wave": "budget4x", "stratum": str(member["stratum"])},
        )
        for member in detail["members"]
    ]
    outcomes = submit_wave(specs, cache_vintage=args.vintage, dry_run=args.dry_run)

    from collections import Counter
    tally = Counter(o.status for o in outcomes)
    print()
    for status, count in tally.most_common():
        print(f"  {status:<18}{count:>5}")
    write_json("budget_study/submission.json",
               {"outcomes": [o.as_dict() for o in outcomes], "tally": dict(tally)})
    return 0 if not tally.get("failed") else 1


def cmd_manuscript(args: argparse.Namespace) -> int:
    """Generate every manuscript table from the artifact directory."""
    from .artifacts import artifacts_root
    from .manuscript import generate, render_all

    tables, missing = generate(artifacts_root())
    target = Path("docs/TABLES.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_all(tables, missing) + "\n", encoding="utf-8")

    print()
    print(f"  {len(tables)} table(s) generated, {len(missing)} not yet available")
    for table in tables:
        print(f"    Table {table.number}  {table.caption[:64]}")
    for entry in missing:
        print(f"    missing   {entry[:72]}")
    print(f"\n  written: {target}")

    if args.require_all and missing:
        print("\n  Refusing: --require-all was given and some artifacts are absent.")
        return 1
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Re-derive the prior protocol's METEOR figures that the manuscript reports.

    The command exists because section 5 of the paper rested on two decimals that no command
    produced. Arguing that the prior result survived review because nobody could check its
    numbers, while reporting unverifiable numbers of one's own, is the one position this
    paper cannot hold.
    """
    from transformers import GPT2TokenizerFast

    from .audit_prior import (
        HYPOTHESES,
        PROTOCOLS,
        CorpusUnavailable,
        audit,
        load_prior_rows,
    )
    from .corpora.tokenise import TOKENISER

    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENISER)
    try:
        rows = load_prior_rows()
    except CorpusUnavailable as exc:
        print(f"\n  the prior corpus could not be loaded: {exc}\n")
        return 1

    report = audit(
        rows,
        encode=lambda s: tokenizer.encode(s),
        decode=lambda ids: tokenizer.decode(ids),
        max_rows=args.limit,
        prompt_fraction=args.prompt_fraction,
    )
    payload = report.as_dict()
    finding = payload["finding"]

    print()
    print(f"  {len(rows):,} rows from {payload['corpus']['config']}, "
          f"{payload['markup_share']:.1%} of evaluated rows are markup")
    print(f"  prompt covers {report.prompt_fraction:.0%} of each row's characters, "
          "as the prior work's Figure B.1 documents and its code implemented")
    print()
    print(f"  {'hypothesis':<20}{'prior':>12}{'prompt removed':>16}{'corrected':>12}")
    print("  " + "-" * 60)
    for hypothesis in HYPOTHESES:
        cells = "".join(
            f"{report.score(hypothesis, p).meteor:>{width}.4f}"
            for p, width in zip(PROTOCOLS, (12, 16, 12), strict=True)
        )
        print(f"  {hypothesis:<20}{cells}")
    print()
    print()
    print(f"  what scoring the prompt is worth:    "
          f"{finding['prompt_defect_contribution']:.4f}")
    print(f"  spread across model-free hypotheses: {finding['floor_spread']:.4f}")
    print(f"  improvement the prior work reports:  "
          f"{finding['claimed_improvement']:.4f}")
    print(f"  the improvement exceeds the spread:  "
          f"{finding['improvement_exceeds_floor_spread']}")
    print()

    path = write_json("audit/prior_protocol.json", payload)
    print(f"  written: {path}")
    print()
    return 0


def cmd_objections(args: argparse.Namespace) -> int:
    """Print the reviewer-objection registry, gaps included.

    Exits non-zero under ``--require-complete``, which is what a resubmission checklist
    would call. It fails today on purpose: the review text was never committed, so the
    claim that every objection is answered is not one this repository can support.
    """
    from .objections import (
        RECORDED,
        REVIEW_TEXT,
        RegistryIncomplete,
        open_objections,
        report,
        require_complete,
    )

    payload = report()
    reviewer = ""
    print()
    for objection in RECORDED:
        if objection.reviewer != reviewer:
            reviewer = objection.reviewer
            print(f"\n  ── Reviewer {reviewer[1:]} "
                  f"{'─' * 56}")
        print(f"  {objection.reviewer} #{objection.number:2s} "
              f"[{objection.disposition:9s}] {objection.summary}")
        if args.verbose:
            print(f"      asked: “{objection.quote[:150]}…”")
            print(f"      {objection.response}")
        for citation in objection.evidence:
            print(f"      {citation}")

    for entry in payload["broken_citations"]:
        print(f"  broken citation: {entry}")
    for entry in payload["unverified_quotes"]:
        print(f"  unverified quote: {entry}")

    print(f"\n  {payload['total']} objections from {len(payload['by_reviewer'])} reviewers, "
          f"quoted from {REVIEW_TEXT}")
    print(f"  {payload['addressed']} addressed, {payload['pending']} still open, "
          f"{payload['declined']} declined")

    still_open = open_objections()
    if still_open:
        print("\n  Open — the mechanism is built and the number is not final:")
        for objection in still_open:
            print(f"    {objection.reviewer} #{objection.number:2s} {objection.summary}")

    write_json("objections/registry.json", payload)

    if args.require_complete:
        try:
            require_complete()
        except RegistryIncomplete as exc:
            print(f"\n  Refusing: {str(exc).splitlines()[0]}")
            return 1
    return 0


def _loaded_table(args: argparse.Namespace):
    """The published table, and how to name it in an artifact.

    The name recorded is the file's, never its absolute path. A path is a fact about the
    machine that ran the command — it differs between the authors' checkout and a reader's
    by construction, so recording it would make every artifact that mentions it fail to
    reproduce for reasons having nothing to do with the analysis.
    """
    from .artifacts import artifacts_root
    from .benchmark import load_table

    path = Path(args.table) if args.table else artifacts_root() / "benchmark/table.csv"
    return load_table(path), path.name


def _measured_rope(path: Path) -> tuple[float, str]:
    """The region of practical equivalence, from the run that measured it.

    Refuses rather than falling back to a default. A comparison whose noise floor was
    invented is a comparison that can declare anything equivalent, and the prior work's
    central defect was a number whose provenance nobody could state.
    """
    if not path.exists():
        raise SystemExit(
            f"\n  Refusing: no measured ROPE at {path}.\n"
            "  It comes from repeated runs of one architecture across seeds. Run the pilot\n"
            "  (python -m nas4textgen.cli pilot --submit) or pass --rope-from.\n"
            "  A ROPE chosen at the command line would let this comparison declare any\n"
            "  difference equivalent, which is what the protocol exists to prevent."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    measured = payload.get("rope") or {}
    if "rope" not in measured or not measured.get("source"):
        raise SystemExit(
            f"\n  Refusing: {path} records no ROPE with a named source."
        )
    return float(measured["rope"]), str(measured["source"])


def cmd_surrogate(args: argparse.Namespace) -> int:
    """Score every predictor on the published table, and trace the budget curve.

    Reads the table rather than the cloud. That is what makes this stage part of the
    verification tier: a reader with the published CSV and no account can rerun it and get
    the same numbers, which is the property the replication package claimed and could not
    deliver, because this command did not exist.
    """
    from .surrogate.budget import measure
    from .surrogate.evaluate import report, score_slate
    from .surrogate.slate import build_slate

    table, path = _loaded_table(args)
    features = table.genes()
    target = table.column("perplexity")
    params = table.column("params")
    costs = np.column_stack([params, table.column("kv_cache_bytes")])

    candidates = build_slate()
    scores = score_slate(candidates, features, target, params,
                         folds=args.folds, seed=args.seed)
    payload = report(scores, rows=len(table), space_size=args.space_size)
    payload["source_table"] = path
    slate_path = write_json(f"surrogate/slate_at_{len(table)}.json", payload)

    curve = measure(candidates, features, target, costs, params, seed=args.seed)
    curve_path = write_json("surrogate/budget_curve.json", curve.as_dict())

    # The slate as a statistical question rather than a leaderboard. A table sorted by tau
    # invites the reading that its top row is the best predictor, and that reading is not
    # available from the numbers: fourteen candidates over five folds produce a winner by
    # arithmetic whether or not one is genuinely better.
    from .surrogate.compare import NotEnoughBlocks, per_fold_taus
    from .surrogate.compare import compare as compare_slate

    folds = per_fold_taus(candidates, features, target, params, seed=args.seed)
    try:
        ranked = compare_slate(folds)
    except NotEnoughBlocks as exc:
        ranked = {"underpowered": str(exc)}
    ranked_path = write_json("surrogate/comparison.json",
                             {**ranked, "source_table": path})

    print()
    print(f"  {len(candidates)} candidates on {len(table)} measured rows")
    for entry in payload["scores"][:6]:
        if "unavailable" in entry:
            print(f"    {entry['name']:<26}unavailable")
            continue
        print(f"    {entry['name']:<26}tau {entry['kendall_tau']:+.4f}"
              f"   top-decile {entry['kendall_tau_top_decile']:+.4f}"
              f"{'   disqualified' if entry['disqualified'] else ''}")
    if "friedman" in ranked:
        best = max(ranked["probability_best"].items(), key=lambda kv: kv[1])
        rejected = sum(1 for x in (ranked["posthoc"] or [])
                       if x["rejected_at_0.05"])
        print(f"\n  ranked over {ranked['power']['blocks']} blocks "
              f"({ranked['power']['design']}): Friedman p = "
              f"{ranked['friedman']['p_value']:.3g}, W = "
              f"{ranked['friedman']['kendalls_w']:.3f}")
        print(f"    p(best) {best[0]} {best[1]:.3f};  "
              f"{rejected} of {len(ranked['posthoc'] or [])} pairs reject after Holm")
    print(f"\n  written: {slate_path}\n           {curve_path}\n           {ranked_path}")
    return 0


def cmd_characterise(args: argparse.Namespace) -> int:
    """Every characterisation of the space, from the published table."""
    from .artifacts import artifacts_root
    from .characterise import (
        MINIMUM_COVERAGE,
        MINIMUM_LIFT,
        design_rules,
        fronts_differ,
        gene_effects_on_quality,
        grouped_query_frontier,
        iso_parameter_depth_vs_width,
        test_iso_parameter,
    )
    from .space.genotype import GENES as GENE_NAMES

    table, path = _loaded_table(args)
    params = table.column("params")
    perplexity = table.column("perplexity")
    depth, width = table.column("n"), table.column("e")

    # The ROPE is read from the pilot rather than taken as a flag default. A width chosen
    # at the command line is a threshold picked by whoever ran the command, and the
    # protocol requires it to be measured and to say where it came from — which is why
    # test_iso_parameter makes the source a required argument rather than an optional note.
    rope, rope_source = _measured_rope(Path(args.rope_from) if args.rope_from
                                       else artifacts_root() / "pilot/results.json")

    iso = iso_parameter_depth_vs_width(params, perplexity, depth, width, bands=args.bands)
    bayes = test_iso_parameter(iso, rope=rope, rope_source=rope_source)
    depth_path = write_json("characterisation/depth_vs_width.json", {
        "iso_parameter": iso.as_dict(),
        "bayesian": bayes,
        "source_table": path,
        "rows": len(table),
    })

    fronts = fronts_differ(params, table.column("kv_cache_bytes"), perplexity)
    effects = gene_effects_on_quality(table.genes(), perplexity, GENE_NAMES)
    genes_path = write_json("characterisation/fronts_and_genes.json", {
        "fronts": fronts.as_dict(),
        "gene_effects": effects,
        "source_table": path,
        "rows": len(table),
    })

    # Rules are scored against two fronts, because one of them would flatter the answer.
    # The joint front counts cache size as an objective, so a rule about the gene that
    # minimises cache is partly true by construction; the parameters-only front removes
    # that objective entirely. Reporting both makes the tautological share visible as the
    # difference rather than leaving a reader to wonder about it.
    from .space.enumerate import non_dominated as _non_dominated

    joint = table.column("on_measured_front").astype(bool)
    params_only = np.isin(
        np.arange(len(table.rows)),
        _non_dominated(np.column_stack([params, perplexity])),
    )
    rule_sets = {
        "joint": design_rules(table.genes(), joint, GENE_NAMES),
        "parameters_only": design_rules(table.genes(), params_only, GENE_NAMES),
    }
    rules_path = write_json("characterisation/design_rules.json", {
        name: {
            "front_size": int(mask.sum()),
            "rules": [r.as_dict() for r in rules],
            "holding": [r.as_dict() for r in rules if r.holds],
        }
        for (name, rules), mask in zip(rule_sets.items(), (joint, params_only), strict=True)
    } | {
        "source_table": path,
        "why_two_fronts": (
            "The joint front counts cache size as an objective, so a rule about the gene "
            "that minimises cache is partly true by construction. The parameters-only "
            "front removes that objective, and the drop in lift between the two is the "
            "size of the tautology."
        ),
        "thresholds": {"minimum_lift": MINIMUM_LIFT, "minimum_coverage": MINIMUM_COVERAGE},
    })

    frontier = grouped_query_frontier(
        table.column("h"), depth, width, table.column("ffn"),
        table.column("kv"), table.column("kv_cache_bytes"), perplexity,
    )
    gqa_path = write_json("characterisation/gqa_frontier.json", {
        **frontier.as_dict(), "source_table": path, "rows": len(table),
    })

    print()
    print(f"  on {len(table)} measured rows")
    print(f"    depth vs width      {bayes['bayesian']['verdict']}")
    print(f"                        rope {rope:.6f} from {rope_source[:44]}")
    print(f"    front agreement     Jaccard {fronts.jaccard:.4f}")
    print("    gene effects        " + "  ".join(
        f"{gene} {value:.3f}" for gene, value in
        effects["first_order_indices"].items()))
    print("    design rules         " + ", ".join(
        f"{r.predicate} (lift {r.lift:.2f})"
        for r in rule_sets["parameters_only"] if r.holds) [:96] or "none hold")
    print(f"\n  written: {depth_path}\n           {genes_path}\n           {gqa_path}"
          f"\n           {rules_path}")
    return 0


#: What a canary manifest has to show before the full wave is submitted. Each entry is a
#: metric and the key that proves it was computed rather than skipped or caught.
BATTERY_EVIDENCE = {
    "calibration": "samples",
    "blimp": "macro_accuracy",
    "distinct_n": "distinct_1",
    "repetition_rate": "repetition_rate_4",
}


def cmd_submission(args: argparse.Namespace) -> int:
    """Write the pieces the submission form asks for, one file each.

    The editorial system takes the abstract, the keywords and the highlights as separate
    plain-text fields. Extracted from the manuscript rather than retyped beside it: a
    retyped abstract is how the PDF and the record that indexes it come to differ.
    """
    from .submission import SectionMissing, report

    source = Path(args.paper)
    if not source.exists():
        print(f"\n  Refusing: no manuscript at {source}.")
        print("  The paper is kept with its authors; run this where it lives.\n")
        return 1

    try:
        payload = report(source.read_text(encoding="utf-8"))
    except SectionMissing as exc:
        print(f"\n  {exc}\n")
        return 1

    out = Path(args.into)
    out.mkdir(parents=True, exist_ok=True)
    (out / "abstract.txt").write_text(payload["abstract"]["text"] + "\n",
                                      encoding="utf-8")
    (out / "keywords.txt").write_text(payload["keywords"]["text"] + "\n",
                                      encoding="utf-8")
    (out / "highlights.txt").write_text(
        "\n".join(h["text"] for h in payload["highlights"]) + "\n", encoding="utf-8")
    for statement in payload["statements"]:
        (out / f"{statement['name']}.txt").write_text(statement["text"] + "\n",
                                                      encoding="utf-8")

    abstract = payload["abstract"]
    print()
    print(f"  abstract.txt    {abstract['words']} words"
          + ("" if abstract["within_limit"]
             else f"  OVER the {abstract['limit']}-word line this journal family "
                  "commonly sets"))
    print(f"  keywords.txt    {payload['keywords']['count']} keywords")
    over = [h for h in payload["highlights"] if not h["within_limit"]]
    print(f"  highlights.txt  {len(payload['highlights'])} highlights"
          + (f", {len(over)} over {HIGHLIGHT_CHARACTERS} characters" if over else ""))
    if not payload["highlights_within_count"]:
        print("    more than five; the form takes five")
    for statement in payload["statements"]:
        marker = "  [pending]" if "pending" in statement["text"] else ""
        print(f"  {statement['name'] + '.txt':<24}{statement['words']} words{marker}")
    print(f"\n  written to {out}\n")
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    """Write the checksum manifest the tagged revision is cited by.

    Refuses while the authorship block is incomplete or a reference still carries its
    marker: a release is what the paper points at, and pointing at one that claims less
    than it should is harder to undo than not cutting it.
    """
    from .artifacts import artifacts_root
    from .authorship import AuthorshipIncomplete, require_complete, require_orcid
    from .authorship import authors as cff_authors
    from .release import NothingToPublish, write

    root = artifacts_root().parent
    try:
        require_complete()
        for author in cff_authors():
            require_orcid(author.fields.get("orcid", ""))
    except (AuthorshipIncomplete, ValueError) as exc:
        print(f"\n  Refusing: {exc}\n")
        return 1

    try:
        path, entries, absent = write(root)
    except NothingToPublish as exc:
        print(f"\n  {exc}\n")
        return 1

    total = sum(e.bytes for e in entries)
    print()
    print(f"  {len(entries)} files, {total / 1e6:.1f} MB")
    if absent:
        print(f"  {len(absent)} declared path(s) absent:")
        for name in absent:
            print(f"    {name}")
    print(f"\n  written: {path}")
    print("  Verify with: sha256sum -c artifacts/release/checksums.txt\n")
    return 0


def cmd_matched(args: argparse.Namespace) -> int:
    """Does the shape conclusion survive the corpus, the seed and the definition of cost."""
    from .artifacts import artifacts_root
    from .azure.data import load_manifests
    from .matched import analyse

    try:
        report = analyse(load_manifests(artifacts_root()), bands=args.bands)
    except FileNotFoundError as exc:
        print(f"\n  {exc}\n")
        return 1

    payload = report.as_dict()
    print()
    print(f"  {'corpus':<26}{'seed':>7}{'cost':>16}{'width wins':>13}")
    print("  " + "-" * 64)
    for entry in payload["comparisons"]:
        seed = "pooled" if entry["seed"] is None else str(entry["seed"])
        mark = "  *" if entry["unanimous"] else ""
        print(f"  {entry['corpus']:<26}{seed:>7}{entry['cost']:>16}"
              f"{entry['bands_won_by_width']:>8} of {entry['bands']}{mark}")
    print()
    for cost, tally in payload["bands_won_by_width"].items():
        print(f"  {cost:<18}{tally['total']} of {tally['of']} bands")
    print()

    path = write_json("characterisation/cost_matched.json", payload)
    print(f"  written: {path}\n")
    return 0


def cmd_battery_analysis(args: argparse.Namespace) -> int:
    """Score every battery metric against the objective it was added to supplement."""
    from .artifacts import artifacts_root
    from .azure.data import load_manifests
    from .battery import NoBatteryRuns, analyse

    try:
        report = analyse(load_manifests(artifacts_root()))
    except (FileNotFoundError, NoBatteryRuns) as exc:
        print(f"\n  {exc}\n")
        return 1

    payload = report.as_dict()
    print()
    print(f"  {'corpus':<14}{'metric':<18}{'key':<28}{'tau':>8}{'arch':>7}")
    print("  " + "-" * 76)
    for entry in payload["metrics"]:
        mark = "  redundant" if entry["redundant"] else ""
        print(f"  {entry['corpus']:<14}{entry['metric']:<18}{entry['key']:<28}"
              f"{entry['kendall_tau_against_perplexity']:>8.3f}"
              f"{entry['architectures']:>7}{mark}")
    thin = payload["measured_but_underpowered"]
    if thin:
        print()
        print(f"  {len(thin)} metric(s) were measured on fewer than "
              f"{payload['minimum_architectures']} architectures and are not correlated:")
        for entry in thin:
            print(f"    {entry['corpus']:<14}{entry['metric']}")
        print("  A rank correlation over a handful of points reports its sample size.")

    print()
    if payload["metrics_measured"]:
        print(f"  {payload['metrics_adding_information']} of "
              f"{payload['metrics_measured']} metrics order the front differently enough "
              f"from perplexity to be worth reporting (threshold |tau| < "
              f"{report.threshold})")
    else:
        print("  Nothing to report yet: no metric has reached "
              f"{payload['minimum_architectures']} architectures. The wave is still "
              "running, or the manifest cache needs `cli manifests`.")
    print()

    path = write_json("battery/analysis.json", payload)
    print(f"  written: {path}\n")
    return 0


def cmd_battery_verify(args: argparse.Namespace) -> int:
    """Read the canary's manifests and say whether the battery actually measured anything.

    The gate the fan-out waits on. A job that finishes successfully while its battery caught
    an exception and recorded `unavailable` looks identical from the outside to one that
    worked, and 440 of those would cost the whole budget to produce a column of error
    strings. So the check is on the values, not on the job status.
    """
    from .artifacts import artifacts_root
    from .azure.data import load_manifests
    from .corpora.registry import CORPORA

    try:
        manifests = load_manifests(artifacts_root())
    except FileNotFoundError as exc:
        print(f"\n  {exc}\n")
        return 1

    runs = [m for m in manifests if (m.get("battery") or {})]
    if not runs:
        print("\n  No manifest carries a battery. Either the canary has not finished, or "
              "the cache is stale — run `cli manifests` and look again.\n")
        return 1

    print()
    print(f"  {len(runs)} run(s) with a battery\n")
    problems: list[str] = []
    for manifest in sorted(runs, key=lambda m: (m.get("corpus", ""), str(m.get("genotype")))):
        battery = manifest["battery"]
        corpus = manifest.get("corpus", "?")
        entry = CORPORA.get(corpus)
        readable = (manifest.get("genotype") or {}).get("readable", "?")
        print(f"  {corpus:<14}{readable}")

        for metric, evidence in BATTERY_EVIDENCE.items():
            value = battery.get(metric)
            licensed = entry.licenses(metric) if entry else False
            if not licensed:
                print(f"      {metric:<18}not licensed here")
                continue
            if not isinstance(value, dict) or evidence not in value:
                reason = (value or {}).get("unavailable") or (value or {}).get(
                    "not_implemented") or "absent"
                print(f"      {metric:<18}MISSING — {str(reason)[:60]}")
                problems.append(f"{corpus}/{readable}: {metric} — {str(reason)[:80]}")
                continue
            print(f"      {metric:<18}{evidence} = {value[evidence]}")

        blimp = battery.get("blimp")
        if isinstance(blimp, dict) and "underpowered_paradigms" in blimp:
            thin = blimp["underpowered_paradigms"]
            count = len(thin) if isinstance(thin, (list, tuple)) else thin
            if count:
                problems.append(
                    f"{corpus}/{readable}: {count} BLiMP paradigm(s) underpowered")
        print()

    if problems:
        print("  Not ready to fan out:")
        for problem in problems:
            print(f"    {problem}")
        print("\n  Fix these before submitting the wave; a defect here is paid for 440 "
              "times.\n")
        return 1

    print("  Every licensed metric produced a value. The wave can go out.\n")
    return 0


def cmd_battery(args: argparse.Namespace) -> int:
    """Retrain the measured front with the reporting battery switched on.

    The battery was implemented and never ran: every manifest in the campaign holds
    perplexity and nothing else, and the trained weights are deleted on completion, so the
    only way to obtain the measurements is to train again. This submits that.

    **Two corpora, chosen by what each licenses.** The search corpus admits calibration and
    BLiMP, which is what the front itself can be described with. TinyStories additionally
    licenses the degeneration measures, because it is the only corpus here where models at
    this scale produce text coherent enough for them to mean anything. Running the battery
    where a corpus does not license a metric would produce a number with no referent, which
    is the failure the registry exists to prevent.
    """
    from collections import Counter

    from .azure.data import require_uploaded
    from .azure.submit import JobSpec, submit_wave
    from .corpora.cache import cache_root
    from .corpora.registry import CORPORA
    from .evaluation import blimp
    from .space.genotype import GENES, Genotype

    # Before anything is submitted: the metric's own data has to be in the cache the nodes
    # mount, and at the count its own module calls informative. A wave that trains 440
    # models and then finds no items would cost the whole budget to learn it.
    try:
        items = blimp.load_items(cache_root())
    except Exception as exc:  # noqa: BLE001
        print(f"\n  Refusing: BLiMP is not in the cache ({exc}).")
        print("  Run: python -m nas4textgen.cli fetch --blimp --limit 100\n")
        return 1
    per_paradigm = Counter(item.paradigm for item in items)
    if per_paradigm and min(per_paradigm.values()) < blimp.MINIMUM_ITEMS:
        print(f"\n  Refusing: the thinnest paradigm has {min(per_paradigm.values())} "
              f"pairs, under the {blimp.MINIMUM_ITEMS} this module calls informative.")
        print("  Every paradigm would be reported as underpowered.\n")
        return 1

    table, source = _loaded_table(args)
    unique, seen = [], set()
    for row in table.rows:
        if row["on_measured_front"] != 1.0:
            continue
        genotype = Genotype(**{gene: int(row[gene]) for gene in GENES})
        if genotype.as_tuple() not in seen:
            seen.add(genotype.as_tuple())
            unique.append(genotype)

    corpora = args.corpus or list(args.default_corpora)
    if args.canary:
        # Span rather than a random draw: the cheapest and the most expensive architecture
        # on the front put the battery's memory use at both ends, and the median is the one
        # most like the rest. The battery runs after training, when the model and its
        # optimiser state are still resident, so it is the peak-memory moment of the job.
        by_cost = sorted(unique, key=lambda g: (g.n * g.e, g.e))
        unique = [by_cost[0], by_cost[len(by_cost) // 2], by_cost[-1]][: args.canary]

    print()
    print(f"  {len(unique)} architectures x {len(corpora)} corpora = "
          f"{len(unique) * len(corpora)} runs"
          + ("   (canary)" if args.canary else ""))
    print(f"  drawn from {source}")
    print(f"  BLiMP: {len(items):,} pairs over {len(per_paradigm)} paradigms")
    for corpus in corpora:
        entry = CORPORA[corpus]
        measured = [m for m in ("calibration", "blimp", "distinct_n", "repetition_rate")
                    if entry.licenses(m)]
        print(f"    {corpus:<16}{', '.join(measured)}")
    if args.canary:
        for genotype in unique:
            print(f"      h{genotype.h} d{genotype.d_head} n{genotype.n} "
                  f"f{genotype.ffn} kv{genotype.kv}  (e={genotype.e})")

    plan = write_json("battery/canary_plan.json" if args.canary else "battery/plan.json", {
        "canary": bool(args.canary),
        "architectures": len(unique),
        "corpora": corpora,
        "runs": len(unique) * len(corpora),
        "token_budget": args.token_budget,
        "blimp_pairs": len(items),
        "source_table": source,
        "why_retrained": (
            "The battery was implemented in task 8.5 and never invoked by a run. "
            "Checkpoints are deleted on successful completion, so the measurements cannot "
            "be recovered from the campaign and the front has to be trained again."
        ),
    })
    print(f"\n  written: {plan}")

    if not args.submit:
        print("\n  Nothing submitted. Re-run with --submit, or --submit --dry-run first.")
        return 0

    require_uploaded()
    total: Counter = Counter()
    for corpus in corpora:
        specs = [
            JobSpec(
                genotype=genotype,
                corpus=corpus,
                seed=args.seed,
                token_budget=args.token_budget,
                run_battery=True,
                extra_tags={
                    "wave": ("battery-canary" if args.canary else f"battery-{corpus}"),
                    "battery_corpus": corpus,
                },
            )
            for genotype in unique
        ]
        outcomes = submit_wave(specs, cache_vintage=args.vintage, dry_run=args.dry_run)
        tally = Counter(o.status for o in outcomes)
        total.update(tally)
        print(f"  {corpus:<16}" +
              "  ".join(f"{status} {count}" for status, count in tally.most_common()))
        write_json(f"battery/{corpus}_submission.json",
                   {"outcomes": [o.as_dict() for o in outcomes], "tally": dict(tally)})
        if tally.get("failed"):
            print(f"\n  Stopping: {tally['failed']} job(s) were rejected. The remaining "
                  "corpora are not submitted, so this can be fixed before more is queued.")
            return 1

    print()
    for status, count in total.most_common():
        print(f"  {status:<18}{count:>6}")
    return 0


def cmd_transfer_analysis(args: argparse.Namespace) -> int:
    """Read the completed transfer runs and say whether the front survived the change."""
    from .artifacts import artifacts_root
    from .azure.data import load_manifests
    from .transfer import NotEnoughRuns, analyse

    try:
        report = analyse(load_manifests(artifacts_root()))
    except (FileNotFoundError, NotEnoughRuns) as exc:
        print(f"\n  {exc}\n")
        return 1

    print()
    print(f"  {report.front_size} architectures of the {report.search_corpus} front, "
          "retrained elsewhere")
    print()
    print(f"  {'corpus':<15}{'tau':>8}{'tau top 10%':>13}{'still on front':>16}"
          f"{'top 10 shared':>15}")
    print("  " + "-" * 65)
    for corpus in report.corpora:
        print(f"  {corpus.corpus:<15}{corpus.kendall_tau:>8.3f}"
              f"{corpus.tau_top_decile:>13.3f}"
              f"{corpus.retained:>10} ({corpus.retained_share:.0%})"
              f"{corpus.top10_overlap:>12}/10")
    print()
    print("  design rules, against the front membership they were derived from")
    for corpus in report.corpora:
        cells = "   ".join(f"{r.rule} {r.lift:.2f}" for r in corpus.rules)
        print(f"    {corpus.corpus:<15}{cells}")
    print()

    path = write_json("transfer/analysis.json", report.as_dict())
    print(f"  written: {path}")
    print()
    return 0


def cmd_transfer(args: argparse.Namespace) -> int:
    """Train the measured front on the other corpora, to see whether it transfers.

    The front was measured on one corpus, and a front that only holds on Wikipedia is a
    fact about Wikipedia. This trains every architecture on it against each remaining
    corpus at several seeds, which is what lets a difference between corpora be separated
    from the noise of a single run.

    **Submitted one corpus-and-seed at a time.** A single call queueing two thousand jobs
    is how 213 of wave four's 395 came back marked Completed at the instant they were
    created, having never been scheduled. Whatever the cause on the scheduler's side, a
    smaller batch is the part of it this end controls, and each batch is verified before
    the next goes out.
    """
    from collections import Counter

    from .azure.data import require_uploaded
    from .azure.submit import JobSpec, submit_wave
    from .corpora.registry import CORPORA, SEARCH_CORPUS
    from .space.genotype import GENES, Genotype

    table, source = _loaded_table(args)
    front = [
        Genotype(**{gene: int(row[gene]) for gene in GENES})
        for row in table.rows if row["on_measured_front"] == 1.0
    ]
    # One entry per architecture: the front marks every replicate row of a front shape, so
    # taking rows directly would submit the same architecture more than once.
    unique, seen = [], set()
    for genotype in front:
        if genotype.as_tuple() not in seen:
            seen.add(genotype.as_tuple())
            unique.append(genotype)

    corpora = args.corpus or [key for key in CORPORA if key != SEARCH_CORPUS]
    seeds = list(range(args.seeds))

    if args.canary:
        # Every corpus here is a path the cluster has never run. TinyStories licenses
        # generation metrics, which the sweep's corpus does not, so a code path that has
        # never executed in a job runs for the first time; BabyLM derives its splits
        # locally from a single upstream one. A full batch would pay for whichever of
        # those is broken two hundred and twenty times, and 220 jobs was submitted before
        # this gate existed — which is the reason it exists.
        #
        # Architectures are chosen for span rather than at random: the cheapest and the
        # most expensive on the front exercise the memory probe at both ends, and the
        # median is the one most like the rest.
        by_cost = sorted(unique, key=lambda g: (g.n * g.e, g.e))
        picks = [by_cost[0], by_cost[len(by_cost) // 2], by_cost[-1]][: args.canary]
        seeds = [0]
        batches = [(corpus, 0) for corpus in corpora]
        unique = picks
        print()
        print(f"  canary: {len(picks)} architectures on each of {len(corpora)} corpora, "
              f"{len(picks) * len(corpora)} runs")
        for genotype in picks:
            print(f"    h{genotype.h} d{genotype.d_head} n{genotype.n} "
                  f"f{genotype.ffn} kv{genotype.kv}  (e={genotype.e})")
    else:
        batches = [(corpus, seed) for corpus in corpora for seed in seeds]

    print()
    if not args.canary:
        print(f"  {len(unique)} front architectures x {len(corpora)} corpora x "
              f"{len(seeds)} seeds = {len(unique) * len(batches)} runs")
    print(f"  drawn from {source}")
    for corpus in corpora:
        entry = CORPORA[corpus]
        print(f"    {corpus:<16}{entry.role}")

    plan = write_json("transfer/canary_plan.json" if args.canary else "transfer/plan.json", {
        "canary": bool(args.canary),
        "front_architectures": len(unique),
        "corpora": corpora,
        "seeds": seeds,
        "runs": len(unique) * len(batches),
        "token_budget": args.token_budget,
        "source_table": source,
        "batching": (
            "one corpus and seed per submission, verified before the next. A single call "
            "queueing every run is how 213 jobs came back Completed having never been "
            "scheduled."
        ),
    })
    print(f"\n  written: {plan}")

    if not args.submit:
        print("\n  Nothing submitted. Re-run with --submit, or --submit --dry-run first.")
        return 0

    require_uploaded()
    total: Counter = Counter()
    for corpus, seed in batches:
        specs = [
            JobSpec(
                genotype=genotype,
                corpus=corpus,
                seed=seed,
                token_budget=args.token_budget,
                extra_tags={
                "wave": ("transfer-canary" if args.canary
                         else f"transfer-{corpus}-s{seed}"),
                "transfer_corpus": corpus,
            },
            )
            for genotype in unique
        ]
        outcomes = submit_wave(specs, cache_vintage=args.vintage, dry_run=args.dry_run)
        tally = Counter(o.status for o in outcomes)
        total.update(tally)
        print(f"  {corpus:<16}seed {seed}   " +
              "  ".join(f"{status} {count}" for status, count in tally.most_common()))
        write_json(f"transfer/{corpus}_s{seed}_submission.json",
                   {"outcomes": [o.as_dict() for o in outcomes], "tally": dict(tally)})
        if tally.get("failed"):
            print(f"\n  Stopping: {tally['failed']} job(s) were rejected in this batch. "
                  "The remaining batches are not submitted, so the campaign can be fixed "
                  "before more of it is queued.")
            return 1

    print()
    for status, count in total.most_common():
        print(f"  {status:<18}{count:>6}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Run every algorithm on the complete table and compare them against the exact front.

    The measurement the census makes possible: with all 1,099 architectures trained, the
    front is exact, so the distance to it is a number rather than a comparison against
    whichever run happened to be best.
    """
    import numpy as np

    from .search.compare import BUDGETS, NotEnoughRuns, compare, run_slate
    from .search.problem import BenchmarkTable
    from .search.run import SLATE
    from .space.genotype import GENES

    table_rows, source = _loaded_table(args)
    rows = {
        tuple(int(row[gene]) for gene in GENES): np.array(
            [row["params"], row["kv_cache_bytes"], row["perplexity"]], dtype=float
        )
        for row in table_rows.rows
    }
    table = BenchmarkTable(
        rows, ("params", "kv_cache_bytes", "perplexity"), (True, True, True),
        source=source,
    )

    names = tuple(args.algorithm or [entry.name for entry in SLATE])
    print()
    print(f"  {len(names)} algorithms x {args.seeds} seeds on {len(rows):,} architectures")
    print("  front is exact over the whole space, so IGD+ is a distance and not a ranking")

    seeds = tuple(range(args.seeds))
    results, excluded = run_slate(table, names, seeds=seeds)
    for name, reason in excluded.items():
        print(f"    excluded  {name}: {reason[:88]}")

    payload = {"runs": [r.as_dict() for r in results], "excluded": excluded,
               "source_table": source, "budgets": list(BUDGETS)}

    print()
    print(f"  {'algorithm':<24}" + "".join(f"{b:>12}" for b in BUDGETS))
    print("  " + "-" * (24 + 12 * len(BUDGETS)))
    for name in names:
        runs = [r for r in results if r.algorithm == name]
        if not runs:
            continue
        cells = []
        for budget in BUDGETS:
            values = [r.igd_plus[budget] for r in runs if budget in r.igd_plus]
            cells.append(f"{np.median(values):>12.5f}" if values else f"{'—':>12}")
        print(f"  {name:<24}" + "".join(cells))
    print("  median IGD+ to the exact front, lower is better")

    for budget in BUDGETS:
        try:
            payload[f"comparison_at_{budget}"] = compare(results, budget)
        except NotEnoughRuns as exc:
            payload[f"comparison_at_{budget}"] = {"undecided": str(exc)}
            print(f"\n  budget {budget}: undecided — {str(exc).splitlines()[0][:88]}")
            continue
        entry = payload[f"comparison_at_{budget}"]
        print(f"\n  budget {budget}:  Friedman p = {entry['friedman']['p_value']:.4g}"
              f"   rope {entry['rope']['rope']:.5f} (measured)")
        best = sorted(entry["probability_best"].items(), key=lambda kv: -kv[1])[:3]
        print("    p(best): " + "   ".join(f"{k} {v:.3f}" for k, v in best))

    path = write_json("search/comparison.json", payload)
    print(f"\n  written: {path}")
    return 0


def cmd_sampling(args: argparse.Namespace) -> int:
    """Which sampling design recovers the exact front best, at a matched budget."""
    from .sampling import BUDGETS, compare_designs, test_designs
    from .space.enumerate import enumerate_with_ceiling
    from .space.genotype import GENES as GENE_NAMES

    table, source = _loaded_table(args)
    measured = {
        tuple(int(row[gene]) for gene in GENE_NAMES):
        (row["params"], row["kv_cache_bytes"], row["perplexity"])
        for row in table.rows
    }
    space = [g for g in enumerate_with_ceiling().genotypes if g.as_tuple() in measured]
    objectives = np.array([measured[g.as_tuple()] for g in space], dtype=float)

    comparison = compare_designs(space, objectives, seeds=tuple(range(args.seeds)))
    designs = sorted({r.design for r in comparison.results})

    print()
    print(f"  {len(space):,} architectures, exact front of {comparison.front_size}")
    for measure, direction in (("igd_plus", "lower is better"),
                               ("front_recall", "higher is better"),
                               ("fill_distance", "lower is better")):
        print(f"\n  {measure} — {direction}")
        print(f"  {'budget':>8}" + "".join(f"{d:>18}" for d in designs))
        for budget, row in comparison.medians(measure).items():
            print(f"  {budget:>8}" + "".join(f"{row[d]:>18.5f}" for d in designs))

    payload = {
        "results": [r.as_dict() for r in comparison.results],
        "space_size": comparison.space_size,
        "front_size": comparison.front_size,
        "source_table": source,
        "medians": {m: comparison.medians(m)
                    for m in ("igd_plus", "front_recall", "fill_distance")},
    }
    for measure in ("igd_plus", "front_recall"):
        for budget in BUDGETS:
            payload[f"{measure}_at_{budget}"] = test_designs(comparison, budget, measure)

    print()
    for budget in BUDGETS:
        entry = payload[f"igd_plus_at_{budget}"]
        if "underpowered" in entry:
            print(f"  budget {budget}: underpowered")
            continue
        best = max(entry["probability_best"].items(), key=lambda kv: kv[1])
        print(f"  budget {budget}:  Friedman p = {entry['friedman']['p_value']:.3g}"
              f"   p(best) {best[0]} {best[1]:.3f}")

    path = write_json("sampling/designs.json", payload)
    print(f"\n  written: {path}")
    return 0


def cmd_cost(args: argparse.Namespace) -> int:
    """What the campaign cost, from what Azure allocated rather than what it computed."""
    from datetime import UTC, datetime

    from .artifacts import artifacts_root
    from .azure.workspace import resolve_platform
    from .cost import MetricsUnavailable, build_report

    platform = resolve_platform()
    resource = (f"/subscriptions/{platform.subscription_id}"
                f"/resourceGroups/{platform.resource_group}"
                f"/providers/Microsoft.MachineLearningServices/workspaces/"
                f"{platform.workspace}")

    progress = json.loads(
        (artifacts_root() / "campaign/progress.json").read_text(encoding="utf-8")
    )
    # Runs, not architectures of the sweep. The window holds the census, the 4x-budget
    # study and the transfer, and every one of them consumed node time — dividing by the
    # census alone charged the whole campaign to it.
    from .azure.submit import written_results

    runs_in_window = args.runs or len(written_results(platform))
    start = datetime.fromisoformat(args.since).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.until).replace(tzinfo=UTC) if args.until else \
        datetime.now(UTC)

    try:
        report = build_report(
            resource, start, end,
            process_node_hours=float(progress["node_hours"]),
            runs=runs_in_window,
        )
    except MetricsUnavailable as exc:
        print(f"\n  Refusing: {exc}")
        return 1

    print()
    print(report.render())
    if args.remaining:
        print(f"\n  {args.remaining} run(s) still to go            "
              f"${report.forecast(args.remaining):,.0f}")
        print(f"  projected total                        "
              f"${report.usd + report.forecast(args.remaining):,.0f}")
    payload = write_json("campaign/cost.json", report.as_dict())
    print(f"\n  written: {payload}")
    print("  Platform metrics are retained 93 days; after that this cannot be recomputed.")
    return 0


def cmd_references(args: argparse.Namespace) -> int:
    """Check the carried-over reference list against the record, entry by entry.

    Reviewer R3 found three errors in this list. Three verified errors means an unknown
    number of unverified ones, so every entry is checked rather than the three that were
    caught. Nothing is edited: each entry gets a proposal and the similarity it rests on.
    """
    from .overleaf import import_cited
    from .references import apply, check_all, render_bibliography, report

    bibliography = Path(args.bibliography)
    manuscript = Path(args.cited_from)
    if not bibliography.exists() or not manuscript.exists():
        print(f"\n  Refusing: need both {bibliography} and {manuscript}.\n")
        return 1

    entries, absent = import_cited(
        bibliography.read_text(encoding="utf-8", errors="replace"),
        manuscript.read_text(encoding="utf-8", errors="replace"))
    if args.limit:
        entries = entries[: args.limit]

    print(f"\n  checking {len(entries)} cited entries against Crossref and arXiv\n")
    findings = check_all(entries, pause=args.pause)
    payload = report(findings)

    for status in ("conflict", "needs a human", "not found", "unreachable", "matched"):
        count = payload["by_status"].get(status, 0)
        if count:
            print(f"  {status:<16}{count:>4}")

    conflicts = [f for f in findings if f.status == "conflict"]
    if conflicts:
        print("\n  Disagreements with the record — each needs a decision:")
        for finding in conflicts:
            print(f"\n    {finding.key}")
            for name, (ours, theirs) in finding.conflicts.items():
                print(f"      {name}:")
                print(f"        ours     {ours[:68]}")
                print(f"        registry {theirs[:68]}")

    if absent:
        print(f"\n  {len(absent)} cited key(s) absent from the bibliography: "
              f"{', '.join(absent[:5])}")

    path = write_json("references/verification.json", payload)
    print(f"\n  written: {path}")

    if args.apply:
        updated = apply(entries, findings,
                        accept_conflicts=args.prefer_published)
        settled = sum(1 for e in updated if e.verified)
        header = (
            "% Carried from the prior submission and checked against Crossref, arXiv and\n"
            "% OpenAlex by `python -m nas4textgen.cli references --apply`. Fields the\n"
            "% registries supplied have been filled in. A field the registry contradicts is\n"
            + ("% Where a registry contradicted a field, the registry won: the policy is\n"
               "% to cite the published version of record.\n" if args.prefer_published
               else "% A field the registry contradicts is left as it was and reported in\n"
                    "% artifacts/references/verification.json.\n")
            + 
            "% An entry still marked UNVERIFIED is one a human has to settle.\n")
        target = Path(args.into) / "references.bib"
        if target.parent.exists():
            target.write_text(render_bibliography(updated, header), encoding="utf-8")
            print(f"  written: {target}")
            print(f"  {settled} of {len(updated)} entries are now complete and "
                  "uncontradicted; the rest keep their marker")
        else:
            print(f"\n  {target.parent} does not exist; run `cli overleaf` first")
            return 1
    else:
        print("  Nothing was edited. Re-run with --apply to fill the gaps the "
              "registries closed.")
    print()
    return 0


def _verified_references(references):
    """Apply the recorded verification, if one has been run.

    Returns the references unchanged when no verification artifact exists, so a checkout
    that has never run `cli references` still builds.
    """
    from .artifacts import artifacts_root
    from .references import Candidate, Finding, apply

    record = artifacts_root() / "references/verification.json"
    if not record.exists():
        return references

    payload = json.loads(record.read_text(encoding="utf-8"))
    findings = [
        Finding(
            key=item["key"], title=item["title"], status=item["status"],
            candidate=Candidate(item["candidate"]["source"], item["candidate"]["title"],
                                item["candidate"]["similarity"],
                                item["candidate"]["fields"])
            if item["candidate"] else None,
            additions=item["additions"],
            conflicts={k: (v["ours"], v["registry"])
                       for k, v in item["conflicts"].items()},
            missing_after=tuple(item["still_missing"]))
        for item in payload["findings"]
    ]
    return apply(references, findings, accept_conflicts=True)


def cmd_overleaf(args: argparse.Namespace) -> int:
    """Build the Overleaf project: one folder, ready to upload."""
    import shutil

    from .artifacts import artifacts_root
    from .authorship import authors as cff_authors
    from .authorship import contacts, require_human
    from .manuscript import generate
    from .overleaf import (
        VENUES,
        drop_front_block,
        drop_sections,
        parse_bibtex,
        place_floats,
        preamble,
        rewrite_references,
        section_labels,
        split_abstract,
        to_latex,
    )

    require_human()
    out = Path(args.into)
    if out.exists() and not args.force:
        print(f"\n  Refusing: {out} exists. Pass --force to rebuild it.")
        print("  It is an output directory — anything edited inside it is lost on rebuild.")
        return 1
    shutil.rmtree(out, ignore_errors=True)
    (out / "tables").mkdir(parents=True)

    source = Path(args.paper)
    if not source.exists():
        # Not an error in the pipeline. The manuscript is kept with its authors rather
        # than in this repository, so a clone that has never held it reaches this line
        # legitimately and should be told what it is missing, not handed a traceback.
        print(f"\n  Refusing: no manuscript at {source}.")
        print("  The paper is not kept in this repository, which holds the code and the")
        print("  experiments. Run this where the manuscript lives, or pass --paper.\n")
        return 1

    # Figures from the artifacts, for the same reason the tables are: a figure exported
    # once disagrees with its source as soon as a measurement is rerun, and nobody
    # re-reads a PDF to notice.
    from .figures import generate as generate_figures

    figures, figures_missing = generate_figures(artifacts_root(), out / "figures")

    # Tables from the artifacts, one file each, pulled in with \input.
    tables, missing = generate(artifacts_root())
    for table in tables:
        (out / "tables" / f"table{table.number}.tex").write_text(
            table.render_latex() + "\n", encoding="utf-8")

    listed = cff_authors()
    contact = contacts()
    corresponding = (f"{contact[0].get('given-names','')} "
                     f"{contact[0].get('family-names','')}".strip() if contact else "")
    title = next(
        (line.strip() for line in source.read_text(encoding="utf-8").splitlines()
         if line.startswith("# ")), "# Untitled"
    )[2:]

    venue = VENUES[args.venue]
    markdown = source.read_text(encoding="utf-8")
    # The abstract moves into the frontmatter for a class that has one, and stays a section
    # for a class that does not. Left in the body under elsarticle it prints after the
    # author list as ordinary prose.
    abstract, remainder = (split_abstract(markdown) if venue.frontmatter
                           else ("", markdown))
    body = to_latex(drop_sections(drop_front_block(remainder)))
    # Each float goes after the paragraph that first names it. Appending them all after the
    # body left LaTeX to place them, and LaTeX put every figure on pages 33 to 35 of a
    # 35-page article while the prose cited them from page 7.
    # The prose writes "Table 9"; LaTeX numbers floats by the order they appear, which is
    # the order of first mention once they are placed. The two agree only by accident, so
    # each reference becomes a \ref and LaTeX resolves it. Written as a literal number,
    # the transfer table printed as Table 7 while three sentences called it Table 9.
    floats, references = {}, {}
    for table in tables:
        floats[f"tab:{table.number}"] = r"\input{tables/table" + table.number + "}"
        references[f"Table {table.number}"] = f"tab:{table.number}"
    for position, figure in enumerate(figures, start=1):
        floats[f"fig:{figure.name}"] = figure.as_latex()
        references[f"Figure {position}"] = f"fig:{figure.name}"
    # Sections too. The manuscript numbers its own headings and the prose refers to them by
    # those numbers, so moving a section leaves every sentence pointing at whatever now
    # holds the old number. A \ref cannot be wrong: it prints what LaTeX numbered.
    references.update(section_labels(remainder))
    body = rewrite_references(body, references)
    body, unplaced = place_floats(body, floats)
    inputs = ""
    if unplaced:
        # Never referred to in the prose, so there is nowhere it belongs. Kept at the end
        # and reported, because a table nobody cites is either a missing sentence or a
        # table that should not have been generated.
        print(f"\n  {len(unplaced)} float(s) never referred to: {', '.join(unplaced)}")
        inputs = "\n\n% Never referred to in the prose.\n" + "\n\n".join(
            floats[label] for label in unplaced)
    # acmart needs \maketitle; elsarticle typesets the frontmatter itself and errors on it.
    title_command = "" if venue.frontmatter else "\n" + r"\maketitle"
    (out / "main.tex").write_text(
        preamble(title, [a.fields for a in listed], corresponding,
                 venue=venue, abstract=abstract)
        + title_command + "\n\n" + body
        + "\n\n% Tables, generated from the artifact directory.\n" + inputs
        + "\n\n" + r"\bibliographystyle{" + venue.bibliography_style + "}" + "\n"
        + r"\bibliography{references}" + "\n\n" + r"\end{document}" + "\n",
        encoding="utf-8")

    # The bibliography, carried over and marked rather than trusted.
    # A missing source here produced an empty bibliography in silence, which is worse
    # than an error: the project compiles, the citations are gone, and nothing said so.
    # This paper's own reference list, used as written. It was built from the prior
    # submission's bibliography, corrected against three registries and frozen here as
    # source; the corrections are reproducible with `cli references` but are not re-run to
    # compile a paper, and the project that held the original list stays out of this
    # repository, which holds the code and the experiments.
    bibliography = Path(args.references) if args.references else None
    if bibliography is None:
        # Omitting the flag once wrote an empty bibliography in silence, which is how a
        # rebuild lost 59 citations while `\bibliography{references}` stayed in the
        # preamble and the project still compiled.
        print("\n  Refusing: no reference list given, and the document cites one.")
        print("  Pass --references, or --references '' to build without a bibliography.")
        return 1
    if not bibliography.exists():
        print(f"\n  Refusing: no reference list at {bibliography}.")
        print("  An absent file would produce an empty bibliography that compiles.")
        return 1

    text = bibliography.read_text(encoding="utf-8", errors="replace")
    entries = parse_bibtex(text)
    if not entries:
        print(f"\n  Refusing: {bibliography} yielded no entries.")
        return 1
    shutil.copyfile(bibliography, out / "references.bib")

    references = list(entries.values())
    unverified = sum(1 for r in references if not r.verified)
    (out / "README.md").write_text(_overleaf_readme(
        len(tables), missing, len(references), unverified,
        [a for a in listed if not a.complete], venue=venue), encoding="utf-8")

    print()
    print(f"  {out}")
    print(f"    main.tex          generated from {source}")
    print(f"    tables/           {len(tables)} table(s) from the artifact directory")
    print(f"    references.bib    {len(references)} entries, {unverified} unverified")
    print("    README.md         what to upload and what regenerates")
    if missing:
        for entry in missing:
            print(f"    missing table: {entry[:70]}")
    incomplete = [a for a in listed if not a.complete]
    if incomplete:
        print(f"\n  {len(incomplete)} author(s) still incomplete: "
              + ", ".join(a.display for a in incomplete))
    print(f"  {unverified} reference(s) unverified — not submission-ready")

    # Kept after the build rather than instead of it: the output folder is regenerated in
    # place, so the only way to still have v01 after building v02 is to have copied it.
    if getattr(args, "snapshot", False):
        from .overleaf import snapshot

        print(f"\n  version kept: {snapshot(Path.cwd(), out)}")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """Keep the built project as a numbered version.

    Separate from the build, and run after it rather than during it, because the thing
    worth keeping is the compiled document. ``overleaf --snapshot`` copies the folder
    before tectonic has written a PDF into it, which is why v01 holds sources and no paper.
    """
    from .overleaf import next_version, snapshot

    built = Path(args.built)
    if not (built / "main.tex").exists():
        print(f"\n  Refusing: no build at {built}.")
        print("  Run `cli overleaf` first, then compile, then keep the result.\n")
        return 1

    label = args.label or next_version(Path.cwd())
    if not (built / "main.pdf").exists():
        print(f"\n  Note: {built}/main.pdf does not exist, so {label} will hold sources")
        print("  only. Compile before keeping a version if you want the document itself.")

    try:
        target = snapshot(Path.cwd(), built, label=args.label, note=args.note or "")
    except FileExistsError as exc:
        print(f"\n  Refusing: {exc}\n")
        return 1

    files = sum(1 for p in target.rglob("*") if p.is_file())
    print(f"\n  {label} kept at {target.relative_to(Path.cwd())}")
    print(f"    {files} file(s), "
          f"{'with' if (target / 'main.pdf').exists() else 'without'} the compiled PDF")
    print(f"    next version will be {next_version(Path.cwd())}\n")
    return 0


def _missing_note(missing: list[str]) -> str:
    if not missing:
        return ""
    return (f"- **{len(missing)} table(s) could not be generated** — the artifacts they "
            "read are absent, and each names the command that writes its input.")


def _overleaf_readme(tables: int, missing: list[str], references: int,
                     unverified: int, incomplete: list, venue=None) -> str:
    from .overleaf import DEFAULT_VENUE

    venue = venue or DEFAULT_VENUE
    return f"""# Overleaf project

Target venue: **{venue.name}**. Upload this whole folder to Overleaf (New Project ->
Upload Project, as a zip). `{venue.document_class}` is installed there, so nothing else is
needed. Rebuild for the other venue with `--venue`; the generator carries both.

## This folder is an output

`main.tex` and `tables/` are generated. Editing them works until the next rebuild, and then
the edit is gone. Change the source instead:

| to change | edit |
|---|---|
| the prose | `docs/manuscript/paper.md` |
| any number in a table | nothing — rerun the pipeline stage that measures it |
| the author block | `CITATION.cff` |

Then run `python -m nas4textgen.cli overleaf --force` again.

The prose lives in Markdown rather than here because writing the paper twice — once
readable, once for the compiler — gives two files that drift, and the drift shows up when a
reviewer reads the version nobody edited.

## What is not ready

- **{unverified} of {references} references are unverified.** They are carried over from the
  prior submission, where reviewer R3 found three outright errors including a citation to
  DistilBERT where DistilGPT2 was meant. Each is a `@misc` with a marker in its note field:
  check it against its source, give it the right entry type and fields, and delete the
  marker.
- **{len(incomplete)} author(s) lack an ORCID**, and a release cannot be minted without
  them: `python -m nas4textgen.authorship` refuses.
- **The CRediT statement is blank on purpose.** It is the one item here that only the authors
  can write.
- **Inline Markdown tables in the prose** are commented out in `main.tex` with a TODO. They
  are the small comparison tables in the results text; move each to a generated table or
  typeset it by hand.
{_missing_note(missing)}

## Files

- `main.tex` — the paper, converted from the Markdown draft
- `tables/table1.tex` … `table{tables}.tex` — generated from `artifacts/`, pulled in with
  `\\input`
- `references.bib` — carried over, unverified
"""


def cmd_verify(args: argparse.Namespace) -> int:
    """Re-derive the published analysis into a scratch tree and compare."""
    import tempfile

    from .artifacts import artifacts_root
    from .replication import Tier
    from .verify import RootsCoincide, run

    published = Path(args.published) if args.published else artifacts_root()
    scratch = Path(args.into) if args.into else Path(tempfile.mkdtemp(prefix="nas4tg-verify-"))

    try:
        report = run(published, scratch, tier=Tier(args.tier), only=tuple(args.stage or ()))
    except RootsCoincide as exc:
        print(f"\n  Refusing: {exc}")
        return 1

    print()
    print(report.render())
    payload = write_json("verify/report.json", report.as_dict())
    print(f"\n  written: {payload}")
    return 0 if report.passed else 1


def cmd_replication(args: argparse.Namespace) -> int:
    """Regenerate the replication instructions from the declaration in code."""
    from .replication import STAGES, Determinism, manifest, render_markdown

    target = Path("docs/REPLICATION.md")
    rendered = render_markdown() + "\n"

    if args.check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != rendered:
            print("\n  docs/REPLICATION.md is out of date with the declaration in code.")
            print("  Regenerate it with: python -m nas4textgen.cli replication")
            return 1
        print("\n  docs/REPLICATION.md matches the declaration.")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    payload = write_json("replication/manifest.json", manifest())

    exact = sum(s.determinism is Determinism.BIT_EXACT for s in STAGES)
    print()
    print(f"  {len(STAGES)} stages: {exact} bit-exact, {len(STAGES) - exact} tolerance-exact")
    print(f"  written: {target}\n           {payload}")
    return 0


def cmd_manifests(args: argparse.Namespace) -> int:
    """Mirror run manifests from the artifact container into a local cache.

    Every command that reads manifests was downloading all of them on each run. This makes
    that cost payable once, and leaves a file that re-derives the manifest-backed tables
    without the subscription.
    """
    from .artifacts import artifacts_root
    from .azure.data import MANIFEST_CACHE, sync_manifests

    root = artifacts_root()
    print(f"\n  syncing into {root / MANIFEST_CACHE}")
    fetched, total = sync_manifests(root)
    print(f"\n  {fetched:,} newly fetched, {total:,} cached in all\n")
    return 0


def cmd_table(args: argparse.Namespace) -> int:
    """Assemble the benchmark table and its exact measured front."""
    from .artifacts import artifacts_root
    from .azure.data import load_manifests
    from .benchmark import COLUMNS, assemble
    from .space.enumerate import enumerate_with_ceiling

    # From the local cache rather than the container. This command used to list and download
    # every manifest on each invocation, which at 3,341 runs is ten minutes before any
    # analysis starts, and made assembling a table per corpus cost a pass each.
    try:
        manifests = load_manifests(artifacts_root())
    except FileNotFoundError as exc:
        print(f"\n  {exc}\n")
        return 1

    result = assemble(
        manifests, space_size=len(enumerate_with_ceiling().genotypes),
        select={"corpus": args.corpus, "token_budget": args.token_budget},
    )
    report = result.summary()

    print()
    print(f"  {report['architectures']:,} architectures of {report['space_size']:,} "
          f"({report['coverage']:.1%} of the space)")
    if report["replicated_architectures"]:
        print(f"  {report['rows']:,} rows, so "
              f"{report['replicated_architectures']} run(s) are seed replicates")
    print(f"  measured front: {report['front_size']} architectures "
          f"({report['front_rows']} rows)")
    print(f"    {report['front_is']}")
    if report["excluded"]:
        print("\n  excluded")
        for reason, count in report["excluded"].items():
            print(f"    {count:>5}  {reason}")
    print("\n  held constant across every row")
    for key, value in report["held_constant"].items():
        print(f"    {key:<24}{value}")

    table = write_table("benchmark/table.csv", result.rows, list(COLUMNS))
    summary = write_json("benchmark/summary.json", report)
    print(f"\n  written: {table}\n           {summary}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Submit one wave of the sweep. Waves nest, so each is a usable benchmark on its own."""
    from .azure.sweep import WAVE_SIZES, cost, load_order, waves
    from .azure.throughput import load as load_model
    from .space.enumerate import enumerate_with_ceiling

    space = enumerate_with_ceiling().genotypes
    plan = waves(load_order(), space, WAVE_SIZES)
    model = load_model()

    print()
    print(f"  {'wave':>5}{'cumulative':>12}{'new':>7}{'node-h':>10}{'USD':>9}{'days':>7}")
    print("  " + "-" * 52)
    for wave in plan:
        if model is None:
            print(f"  {wave.number:>5}{wave.size:>12}{len(wave.new):>7}"
                  f"{'?':>10}{'?':>9}{'?':>7}")
            continue
        c = cost(wave, model, args.token_budget, args.nodes)
        print(f"  {wave.number:>5}{wave.size:>12}{c.new_runs:>7}"
              f"{c.node_hours:>10,.0f}{c.usd:>9,.0f}{c.days_at_nodes:>7.1f}")

    if model is None:
        print("\n  No throughput model recorded, so no cost can be stated and no limit")
        print("  derived. Run the pilot first.")
        return 1

    chosen = next((w for w in plan if w.number == args.wave), None)
    if chosen is None:
        print(f"\n  no wave {args.wave}; waves are 1 to {len(plan)}")
        return 1

    c = cost(chosen, model, args.token_budget, args.nodes)
    print(f"\n  wave {chosen.number}: {c.new_runs} new architectures, "
          f"{c.node_hours:,.0f} node-hours, ${c.usd:,.0f}, "
          f"{c.days_at_nodes:.1f} days on {args.nodes} nodes")

    path = write_json(f"sweep/wave{chosen.number}_plan.json",
                      {**chosen.as_dict(), "cost": c.as_dict()})
    print(f"  written: {path}")

    if not args.submit:
        print("\n  Nothing submitted. Re-run with --submit, or --submit --dry-run first.")
        return 0

    from .azure.data import require_uploaded
    from .azure.submit import JobSpec, submit_wave

    require_uploaded()
    specs = [
        JobSpec(
            genotype=genotype,
            corpus=args.corpus,
            seed=args.seed,
            token_budget=args.token_budget,
            extra_tags={"wave": f"sweep{chosen.number}"},
        )
        for genotype in chosen.new
    ]
    outcomes = submit_wave(specs, cache_vintage=args.vintage, dry_run=args.dry_run)

    from collections import Counter
    tally = Counter(o.status for o in outcomes)
    print()
    for status, count in tally.most_common():
        print(f"  {status:<18}{count:>5}")
    for outcome in outcomes:
        if outcome.status == "failed":
            print(f"    {outcome.job_name}  {outcome.detail}")

    write_json(f"sweep/wave{chosen.number}_submission.json",
               {"outcomes": [o.as_dict() for o in outcomes], "tally": dict(tally)})
    return 0 if not tally.get("failed") else 1


def cmd_pilot(args: argparse.Namespace) -> int:
    """Show, or submit, the runs that turn the sweep's cost from a range into a number."""
    from .azure.pilot import plan, summary
    from .space.cost import forward_flops_per_token

    members = plan(size=args.size, seeds=args.seeds)
    report = summary(members)

    print()
    print(f"  {report['runs']} runs: {report['calibration']['architectures']} for the "
          f"throughput fit, {report['variance']['seeds']} seeds for the ROPE")
    print(f"  compute range spans {report['calibration']['decades_covered']} decades\n")
    print(f"  {'purpose':<13}{'genotype':<34}{'FLOPs/tok':>14}{'seed':>6}")
    print("  " + "-" * 68)
    for member in members:
        g = member.genotype
        genes = f"h{g.h} d{g.d_head} n{g.n} ffn{g.ffn} kv{g.kv} (e={g.e})"
        print(f"  {member.purpose:<13}{genes:<34}"
              f"{forward_flops_per_token(g):>14,}{member.seed:>6}")

    path = write_json("pilot/plan.json", report)
    print(f"\n  written: {path}")

    if not args.submit:
        print("\n  Nothing submitted. Re-run with --submit, or --submit --dry-run first.")
        return 0

    from .azure.data import require_uploaded
    from .azure.submit import JobSpec, submit_wave

    require_uploaded()
    specs = [
        JobSpec(
            genotype=member.genotype,
            corpus=args.corpus,
            seed=member.seed,
            token_budget=args.token_budget,
            extra_tags={"wave": "pilot", "pilot_purpose": member.purpose},
        )
        for member in members
    ]
    outcomes = submit_wave(specs, cache_vintage=args.vintage, dry_run=args.dry_run)

    print()
    for outcome in outcomes:
        print(f"  {outcome.status:<16}{outcome.job_name:<26}{outcome.detail}")
    write_json("pilot/submission.json", {"outcomes": [o.as_dict() for o in outcomes]})
    return 0 if all(o.status != "failed" for o in outcomes) else 1


def cmd_progress(args: argparse.Namespace) -> int:
    """Where the campaign has got to. Reads only; a running wave cannot notice this."""
    from .azure.progress import collect
    from .space.enumerate import enumerate_with_ceiling

    planned = args.planned or len(enumerate_with_ceiling().genotypes)
    progress = collect(planned, concurrency=args.nodes)
    print(progress.render())

    path = write_json("campaign/progress.json", progress.as_dict())
    print(f"\n  written: {path}")
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    """Mirror the local corpus cache into project storage, then verify it arrived."""
    from .azure.data import ensure_datastores, upload_cache, verify_uploaded

    for name, container in ensure_datastores():
        print(f"  datastore {name} -> {container}")

    uploaded = upload_cache(force=args.force)
    print()
    print(f"  {'corpus':<16}{'file':<30}{'state':<17}{'MB':>10}")
    print("  " + "-" * 74)
    for record in uploaded:
        state = "already present" if record.already_present else "uploaded"
        print(f"  {record.corpus:<16}{record.split:<30}{state:<17}"
              f"{record.bytes_uploaded / 1e6:>10.1f}")

    problems = verify_uploaded()
    print()
    if problems:
        for problem in problems:
            print(f"  MISMATCH  {problem}")
        print("\n  The remote cache is not the measured one. Do not submit against it.")
        return 1
    print("  Every split matches the hash recorded when it was cached.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Every command, in one place.

    Separate from ``main`` so the command set can be inspected without running anything.
    The replication instructions print a command per stage, and two of them named
    subcommands that did not exist — a reader following the document reached "invalid
    choice". A test now walks the declaration and asserts each command is real, which
    needs the parser as a value rather than as a local variable.
    """
    parser = argparse.ArgumentParser(prog="nas4textgen")
    sub = parser.add_subparsers(dest="command", required=True)

    gate = sub.add_parser("gate", help="run the structural gate over the search space")
    gate.set_defaults(func=cmd_gate)

    enum = sub.add_parser("enumerate", help="enumerate the space and the exact cost front")
    enum.add_argument("--ceiling", type=int, default=PARAMETER_CEILING)
    enum.set_defaults(func=cmd_enumerate)

    des = sub.add_parser("designs", help="generate the execution order and comparison designs")
    des.add_argument("--ceiling", type=int, default=PARAMETER_CEILING)
    des.add_argument("--seed", type=int, default=0)
    des.set_defaults(func=cmd_designs)

    fetch = sub.add_parser("fetch", help="retrieve corpora into the cache")
    fetch.add_argument("--corpus", action="append", help="repeatable; default is all")
    fetch.add_argument("--limit", type=int, help="rows per split, for a quick check")
    fetch.add_argument("--force", action="store_true", help="re-retrieve even if cached")
    fetch.add_argument("--blimp", action="store_true",
                       help="retrieve BLiMP minimal pairs instead of a corpus")
    fetch.set_defaults(func=cmd_fetch)

    tok = sub.add_parser("tokenise", help="encode cached corpora once, into the cache")
    tok.add_argument("--corpus", action="append", help="repeatable; default is all")
    tok.add_argument("--force", action="store_true", help="re-encode even if current")
    tok.set_defaults(func=cmd_tokenise)

    budget = sub.add_parser(
        "budget-study", help="retrain a stratified subset at 4x budget"
    )
    budget.add_argument("--submit", action="store_true")
    budget.add_argument("--dry-run", action="store_true")
    budget.add_argument("--size", type=int, default=50)
    budget.add_argument("--corpus", default="wikitext103")
    budget.add_argument("--seed", type=int, default=0)
    budget.add_argument("--token-budget", type=int, default=100_000_000,
                        help="the sweep's budget; the study runs 4x this")
    budget.add_argument("--vintage", default="unknown")
    budget.set_defaults(func=cmd_budget_study)

    manu = sub.add_parser("manuscript", help="generate the manuscript tables")
    manu.add_argument("--require-all", action="store_true",
                      help="fail if any table's artifact is missing; for the final run")
    manu.set_defaults(func=cmd_manuscript)

    aud = sub.add_parser("audit", help="re-derive the prior protocol's METEOR figures")
    aud.add_argument("--limit", type=int, help="rows to score, for a quick check")
    aud.add_argument("--prompt-fraction", type=float, default=PRIOR_PROMPT_FRACTION,
                     help="share of each row's characters given as the prompt; the prior "
                          "work documents 0.7 and the figure is sensitive to it")
    aud.set_defaults(func=cmd_audit)

    obj = sub.add_parser("objections", help="the reviewer-objection registry and its gaps")
    obj.add_argument("--require-complete", action="store_true",
                     help="fail while any objection is still open; for the resubmission")
    obj.add_argument("--verbose", action="store_true",
                     help="print what each reviewer asked and the full response")
    obj.set_defaults(func=cmd_objections)

    sur = sub.add_parser("surrogate", help="score every predictor on the published table")
    sur.add_argument("--table", help="the published table (default: artifacts/benchmark)")
    sur.add_argument("--folds", type=int, default=5)
    sur.add_argument("--seed", type=int, default=0)
    sur.add_argument("--space-size", type=int, default=1099)
    sur.set_defaults(func=cmd_surrogate)

    cha = sub.add_parser("characterise", help="characterise the space from the table")
    cha.add_argument("--table", help="the published table (default: artifacts/benchmark)")
    cha.add_argument("--bands", type=int, default=10)
    cha.add_argument("--rope-from", help="artifact holding the measured ROPE "
                                         "(default: artifacts/pilot/results.json)")
    cha.set_defaults(func=cmd_characterise)

    tr = sub.add_parser("transfer", help="train the measured front on the other corpora")
    tr.add_argument("--table", help="the published table (default: artifacts/benchmark)")
    tr.add_argument("--corpus", action="append",
                    help="repeatable; default is every corpus but the search one")
    tr.add_argument("--seeds", type=int, default=3,
                    help="seeds per architecture per corpus")
    tr.add_argument("--token-budget", type=int, default=100_000_000)
    tr.add_argument("--vintage", default="unknown")
    tr.add_argument("--canary", type=int, nargs="?", const=3, default=0,
                    metavar="N",
                    help="prove each corpus path with N architectures before any full "
                         "batch; every corpus here is a path the cluster has never run")
    tr.add_argument("--submit", action="store_true")
    tr.add_argument("--dry-run", action="store_true")
    tr.set_defaults(func=cmd_transfer)

    srch = sub.add_parser("search", help="compare every algorithm against the exact front")
    srch.add_argument("--table", help="the published table (default: artifacts/benchmark)")
    srch.add_argument("--algorithm", action="append", help="repeatable; default is the slate")
    srch.add_argument("--seeds", type=int, default=20,
                      help="blocks for the comparison; eight is the point "
                           "where a moderate difference becomes detectable")
    srch.set_defaults(func=cmd_search)

    samp = sub.add_parser("sampling", help="compare sampling designs against the exact front")
    samp.add_argument("--table", help="the published table (default: artifacts/benchmark)")
    samp.add_argument("--seeds", type=int, default=20)
    samp.set_defaults(func=cmd_sampling)

    cost = sub.add_parser("cost", help="what the campaign cost, from allocated node time")
    cost.add_argument("--since", default="2026-08-13", help="campaign start")
    cost.add_argument("--until", help="default is now")
    cost.add_argument("--runs", type=int,
                      help="runs that consumed node time in the window; default counts "
                           "every manifest in the container, since the census is not the "
                           "only thing that ran")
    cost.add_argument("--remaining", type=int, default=0,
                      help="runs still to go, costed on allocation rather than process time")
    cost.set_defaults(func=cmd_cost)

    ref = sub.add_parser("references",
                         help="check the carried reference list against the record")
    ref.add_argument("--bibliography",
                     default="docs/prior/latex/ieee-access/references.bib")
    ref.add_argument("--cited-from",
                     default="docs/prior/latex/ieee-access/access.tex")
    ref.add_argument("--limit", type=int, help="first N entries, for a quick look")
    ref.add_argument("--apply", action="store_true",
                     help="fill in the fields the registries supplied; conflicts are "
                          "left alone and reported")
    ref.add_argument("--prefer-published", action="store_true",
                     help="where the registry contradicts a field, take the registry\'s "
                          "value: it is the published record, and the alternative is "
                          "citing a preprint\'s year for a journal article")
    ref.add_argument("--into", default="submission-neurocomputing",
                     help="the project folder whose references.bib is rewritten")
    ref.add_argument("--pause", type=float, default=0.4,
                     help="seconds between requests; the registries are shared")
    ref.set_defaults(func=cmd_references)

    ovl = sub.add_parser("overleaf", help="build the Overleaf project folder")
    ovl.add_argument("--into", default="submission-neurocomputing", help="output folder")
    ovl.add_argument("--paper", default="docs/manuscript/paper.md")
    ovl.add_argument("--references", default="docs/manuscript/references.bib",
                     help="this paper's reference list. It is source, not a build "
                          "product: omitting it once rebuilt the project with an empty "
                          "bibliography that compiled cleanly")
    ovl.add_argument("--cited-from", default="access.tex",
                     help="the manuscript whose \\cite keys select entries, when "
                          "--references is a .bib; resolved beside it")
    ovl.add_argument("--snapshot", action="store_true",
                     help="keep a numbered copy of this build under "
                          "docs/manuscript/versions, so an earlier version survives the "
                          "next rebuild")
    ovl.add_argument("--venue", default="neurocomputing", choices=("neurocomputing", "telo"),
                     help="which journal's LaTeX conventions to generate")
    ovl.add_argument("--force", action="store_true",
                     help="rebuild, discarding anything edited inside the output folder")
    ovl.set_defaults(func=cmd_overleaf)

    ver = sub.add_parser("verify", help="re-derive the published analysis and compare")
    ver.add_argument("--tier", default="verify",
                 choices=["verify", "smoke", "partial", "full"])
    ver.add_argument("--published", help="the artifact tree to check (default: artifacts/)")
    ver.add_argument("--into", help="where to regenerate; must not be the published tree")
    ver.add_argument("--stage", action="append", help="restrict to one stage, repeatable")
    ver.set_defaults(func=cmd_verify)

    repl = sub.add_parser("replication", help="regenerate the replication instructions")
    repl.add_argument("--check", action="store_true",
                      help="fail if the document is out of date rather than rewriting it")
    repl.set_defaults(func=cmd_replication)

    bat = sub.add_parser("battery",
                         help="retrain the front with the reporting battery switched on")
    bat.add_argument("--corpus", action="append",
                     help="repeatable; the default is the search corpus and TinyStories")
    bat.add_argument("--seed", type=int, default=0)
    bat.add_argument("--token-budget", type=int, default=100_000_000)
    bat.add_argument("--table", help="the published table (default: artifacts/benchmark)")
    bat.add_argument("--vintage", default="unknown")
    bat.add_argument("--canary", nargs="?", type=int, const=3,
                     help="prove the battery on N architectures before the full wave")
    bat.add_argument("--submit", action="store_true")
    bat.add_argument("--dry-run", action="store_true")
    bat.set_defaults(func=cmd_battery, default_corpora=("wikitext103", "tinystories"))

    vrs = sub.add_parser("version",
                         help="keep the compiled build as v01, v02, v03 and so on")
    vrs.add_argument("--built", default="submission-neurocomputing",
                     help="the build folder to keep")
    vrs.add_argument("--label", help="the version label; the next free one by default")
    vrs.add_argument("--note", help="one line on what changed, written into VERSION.md")
    vrs.set_defaults(func=cmd_version)

    sbm = sub.add_parser("submission",
                         help="the abstract, keywords and highlights as separate files")
    sbm.add_argument("--paper", default="docs/manuscript/paper.md")
    sbm.add_argument("--into", default="submission-neurocomputing")
    sbm.set_defaults(func=cmd_submission)

    rel = sub.add_parser("release",
                         help="write the checksum manifest for a tagged revision")
    rel.set_defaults(func=cmd_release)

    mch = sub.add_parser("matched",
                         help="the shape comparison across corpora, seeds and both costs")
    mch.add_argument("--bands", type=int, default=5)
    mch.set_defaults(func=cmd_matched)

    ban = sub.add_parser("battery-analysis",
                         help="score each battery metric against the objective")
    ban.set_defaults(func=cmd_battery_analysis)

    bvf = sub.add_parser("battery-verify",
                         help="check the canary measured what it claims, before the wave")
    bvf.set_defaults(func=cmd_battery_verify)

    tan = sub.add_parser("transfer-analysis",
                         help="whether the measured front survives the other corpora")
    tan.set_defaults(func=cmd_transfer_analysis)

    man = sub.add_parser("manifests",
                         help="mirror run manifests from the container into a cache")
    man.set_defaults(func=cmd_manifests)

    table = sub.add_parser("table", help="assemble the benchmark table and exact front")
    table.add_argument("--corpus", default="wikitext103",
                       help="a table is per corpus, since perplexity on different text is "
                            "a different quantity")
    table.add_argument("--token-budget", type=int, default=100_000_000,
                       help="and per budget: the convergence study measured the same "
                            "corpus at four times the tokens")
    table.set_defaults(func=cmd_table)

    sweep = sub.add_parser("sweep", help="submit one nested wave of the sweep")
    sweep.add_argument("--wave", type=int, default=1, help="which wave; they nest")
    sweep.add_argument("--submit", action="store_true")
    sweep.add_argument("--dry-run", action="store_true")
    sweep.add_argument("--nodes", type=int, default=5, help="for the wall-clock estimate")
    sweep.add_argument("--corpus", default="wikitext103")
    sweep.add_argument("--seed", type=int, default=0)
    sweep.add_argument("--token-budget", type=int, default=100_000_000)
    sweep.add_argument("--vintage", default="unknown")
    sweep.set_defaults(func=cmd_sweep)

    pilot = sub.add_parser("pilot", help="the runs that fix the sweep's cost and the ROPE")
    pilot.add_argument("--submit", action="store_true")
    pilot.add_argument("--dry-run", action="store_true")
    pilot.add_argument("--size", type=int, default=20, help="architectures for the fit")
    pilot.add_argument("--seeds", type=int, default=5, help="seeds for the ROPE")
    pilot.add_argument("--corpus", default="wikitext103")
    pilot.add_argument("--token-budget", type=int, default=100_000_000)
    pilot.add_argument("--vintage", default="unknown")
    pilot.set_defaults(func=cmd_pilot)

    prog = sub.add_parser("progress", help="how far the campaign has got; read-only")
    prog.add_argument("--planned", type=int, help="rows expected; default is the space size")
    prog.add_argument("--nodes", type=int, default=5, help="concurrency the projection assumes")
    prog.set_defaults(func=cmd_progress)

    upload = sub.add_parser("upload", help="mirror the corpus cache into project storage")
    upload.add_argument("--force", action="store_true", help="re-upload even if hashes match")
    upload.set_defaults(func=cmd_upload)

    canary = sub.add_parser("canary", help="show or submit the wave that gates every fan-out")
    canary.add_argument("--submit", action="store_true", help="launch it; otherwise only show")
    canary.add_argument("--dry-run", action="store_true", help="with --submit, launch nothing")
    canary.add_argument(
        "--force",
        action="append",
        metavar="ROLE",
        help=(
            "Re-measure one canary member by role, even though it already completed. "
            "Repeatable, and deliberately not a blanket flag: a boolean --force resubmitted "
            "all four members when only one needed it, spending six node-hours re-measuring "
            "rows that were already valid. Naming the member is the difference between a "
            "correction and a re-run."
        ),
    )
    canary.add_argument("--corpus", default="wikitext103")
    canary.add_argument("--seed", type=int, default=0)
    canary.add_argument("--token-budget", type=int, default=100_000_000)
    canary.add_argument("--vintage", default="unknown", help="data vintage recorded on each job")
    canary.set_defaults(func=cmd_canary)

    return parser


def commands() -> set[str]:
    """The subcommand names this CLI accepts."""
    for action in build_parser()._subparsers._group_actions:  # noqa: SLF001
        if hasattr(action, "choices"):
            return set(action.choices)
    return set()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ImageIsStale as exc:
        # Reported rather than raised. Wave two's submission was refused correctly and the
        # traceback went into a pipe, so the wave simply did not exist and nothing said why.
        print(f"\n  refused: the published image is not this code.\n\n  {exc}\n")
        return 4


if __name__ == "__main__":
    sys.exit(main())
