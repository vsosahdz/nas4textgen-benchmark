"""Command-line entry point for one training run. The image's ENTRYPOINT.

One architecture, one corpus, one seed, one row of the benchmark. Every job in the sweep
is this module with different arguments, which is what keeps a thousand runs comparable:
there is one code path, and the things held constant are held constant by being defaults
here rather than by being remembered at submission time.

    python -m nas4textgen.train --h 12 --d-head 64 --n 6 --ffn 4 --kv 2 \\
        --corpus wikitext103 --output /outputs

Exit status is the contract with the scheduler:

    0   the run met its budget and produced a benchmark row
    2   the run stopped early — measured and written, but not comparable
    3   the run failed
    4   the environment was wrong before any compute was spent
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .checkpoint import CheckpointCorrupt
from .corpora.cache import CacheCorrupt, CacheMiss, load
from .corpora.registry import CORPORA, SEARCH_CORPUS
from .corpora.tokenise import load_tokenised, tokeniser_fingerprint
from .space.genotype import GENES, LEVELS, Genotype
from .training import (
    AcceleratorMissing,
    ArchitectureDoesNotFit,
    OutputsNotWritable,
    TrainingConfig,
    train_and_evaluate,
)

EXIT_OK = 0
EXIT_TRUNCATED = 2
EXIT_FAILED = 3
EXIT_ENVIRONMENT = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nas4textgen.train", description=__doc__)

    genes = parser.add_argument_group("genotype")
    for gene in GENES:
        genes.add_argument(
            f"--{gene.replace('_', '-')}",
            dest=gene,
            type=int,
            required=True,
            choices=LEVELS[gene],
            help=f"one of {LEVELS[gene]}",
        )

    run = parser.add_argument_group("run")
    run.add_argument("--corpus", default=SEARCH_CORPUS, choices=sorted(CORPORA))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--cache",
        type=Path,
        help=(
            "Corpus cache directory. On a cluster this is the mounted input, whose path "
            "Azure ML chooses per run, so it is passed rather than assumed. Defaults to "
            "whatever NAS4TG_CACHE names."
        ),
    )
    run.add_argument("--train-split", default="train")
    run.add_argument("--eval-split", default="validation")

    budget = parser.add_argument_group("budget — identical across the sweep by default")
    defaults = TrainingConfig()
    budget.add_argument("--token-budget", type=int, default=defaults.token_budget)
    budget.add_argument("--block-size", type=int, default=defaults.block_size)
    budget.add_argument("--micro-batch-size", type=int, default=defaults.micro_batch_size)
    budget.add_argument("--seed", type=int, default=defaults.seed)
    budget.add_argument(
        "--allow-cpu",
        action="store_true",
        help=(
            "Train on the CPU when no GPU is present, instead of failing. For the smoke "
            "tier only: on the cluster a silent CPU fallback runs about sixty times slower "
            "on hardware billed for its GPU, and nothing about the result reveals it."
        ),
    )
    budget.add_argument(
        "--soft-deadline-seconds",
        type=int,
        default=defaults.soft_deadline_seconds,
        help="stop training, still evaluate and write",
    )
    budget.add_argument(
        "--run-battery",
        action="store_true",
        help=(
            "Also compute the reporting battery — calibration, BLiMP, and the degeneration "
            "measures the corpus licenses — after training. Off by default: the census pays "
            "for the objective only, and 1,099 runs each scoring 1,340 minimal pairs is "
            "node time no objective consumes."
        ),
    )
    budget.add_argument(
        "--hard-deadline-seconds",
        type=int,
        default=defaults.hard_deadline_seconds,
        help="fires regardless of what the process is doing, so the node is released",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Before anything reads the cache: the corpus loader resolves NAS4TG_CACHE at call
    # time, so the mounted path has to be in place first.
    if args.cache is not None:
        os.environ["NAS4TG_CACHE"] = str(args.cache)

    genotype = Genotype(**{gene: getattr(args, gene) for gene in GENES})
    if not genotype.is_valid():
        print(f"invalid genotype: {genotype} — kv must divide h", file=sys.stderr)
        return EXIT_ENVIRONMENT

    config = TrainingConfig(
        token_budget=args.token_budget,
        block_size=args.block_size,
        micro_batch_size=args.micro_batch_size,
        seed=args.seed,
        soft_deadline_seconds=args.soft_deadline_seconds,
        hard_deadline_seconds=args.hard_deadline_seconds,
        require_accelerator=not args.allow_cpu,
        run_battery=args.run_battery,
    )

    # The corpus is read from the cache and never fetched. A miss here is a setup error,
    # not a signal to reach the network: a run that downloaded its own data could be
    # training on content that moved since the rest of the sweep was measured.
    try:
        train_sequences, train_meta = load_tokenised(args.corpus, args.train_split)
        eval_sequences, _ = load_tokenised(args.corpus, args.eval_split)
    except (CacheMiss, CacheCorrupt) as exc:
        print(f"corpus unavailable: {exc}", file=sys.stderr)
        return EXIT_ENVIRONMENT

    # Byte lengths for bits-per-byte, recovered from the text rather than from the ids: a
    # token count is not a byte count, and the metric is defined on bytes so that models
    # with different tokenisers could in principle be compared.
    eval_text, eval_record = load(args.corpus, args.eval_split, verify=False)
    eval_bytes = [len(row.encode("utf-8")) for row in eval_text if row.strip()]

    # The vintage of the text these ids came from, not of the ids. Two runs are comparable
    # only if they saw the same corpus, and the corpus is the text.
    vintage = eval_record.vintage

    print(f"  architecture   {genotype}")
    print(f"  corpus         {args.corpus}, {train_meta.tokens:,} training tokens")
    print(f"  tokeniser      {train_meta.tokeniser} via tokenizers "
          f"{train_meta.tokenizers_version}, vocab {tokeniser_fingerprint()}")
    print(f"  data vintage   {vintage}")
    print(f"  budget         {config.token_budget:,} tokens")
    print(f"  deadlines      soft {config.soft_deadline_seconds}s / "
          f"hard {config.hard_deadline_seconds}s")

    try:
        outcome = train_and_evaluate(
            genotype=genotype,
            train_sequences=train_sequences,
            eval_sequences=eval_sequences,
            output_dir=args.output,
            config=config,
            corpus=args.corpus,
            eval_byte_lengths=eval_bytes,
            data_vintage=vintage,
            tokeniser_fingerprint=tokeniser_fingerprint(),
        )
    except OutputsNotWritable as exc:
        print(f"outputs unusable: {exc}", file=sys.stderr)
        return EXIT_ENVIRONMENT
    except AcceleratorMissing as exc:
        print(f"wrong hardware: {exc}", file=sys.stderr)
        return EXIT_ENVIRONMENT
    except ArchitectureDoesNotFit as exc:
        print(f"does not fit: {exc}", file=sys.stderr)
        return EXIT_ENVIRONMENT
    except CheckpointCorrupt as exc:
        # A setup error, not a training failure: this output directory belongs to a
        # different experiment, and nothing about running longer would fix that.
        print(f"wrong checkpoint: {exc}", file=sys.stderr)
        return EXIT_ENVIRONMENT

    print(f"  device         {outcome.extra.get('device')} "
          f"({outcome.extra.get('precision')})")
    print(f"  micro-batch    {outcome.extra.get('micro_batch_used')} "
          f"x {config.block_size} tokens, {outcome.extra.get('accumulation_steps')} accumulated")
    print(f"  throughput     {outcome.extra.get('tokens_per_second'):,} tokens/s "
          f"({outcome.extra.get('achieved_tflops')} TFLOP/s)")
    print(f"  status         {outcome.status}")
    print(f"  tokens         {outcome.tokens_consumed:,} in {outcome.steps} steps")
    print(f"  wall clock     {outcome.wall_clock_seconds:.1f}s")
    if outcome.likelihood is not None:
        print(f"  perplexity     {outcome.likelihood.perplexity:.4f}")

    if outcome.usable_for_benchmark:
        return EXIT_OK
    if outcome.status in ("soft_deadline", "hard_deadline"):
        # Distinguished from failure on purpose: the measurement exists and is written, it
        # simply is not comparable to runs that met the budget. The sweep records it as a
        # gap with a reason rather than as a success or a crash.
        print(f"  not comparable: {outcome.status}", file=sys.stderr)
        return EXIT_TRUNCATED
    print(f"  failed: {outcome.error}", file=sys.stderr)
    return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
