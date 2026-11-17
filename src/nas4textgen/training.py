"""The training entry point: one architecture in, one benchmark row out.

Takes a genotype, a corpus, a token budget and a seed; returns trained weights, a
perplexity, and a manifest that records everything needed to reproduce or reject the
result. Every architecture in the sweep goes through this function, so its guarantees are
the benchmark's guarantees.

Three of those guarantees exist because of specific failures rather than good practice.

**The budget is in tokens, not epochs.** The prior work compared a baseline trained for
three epochs on the full split against candidates trained for 1.5 epochs on half of it — a
four-fold difference in favour of the model it claimed to beat. Counting tokens makes the
comparison an architecture comparison.

**The wall-clock limit is enforced inside this process.** A limit that lives only in the
submission spec does not release a node when the process itself hangs: the node stays
marked busy, the cluster never scales down, and it bills until someone notices. A soft
deadline stops training and still writes what exists; a hard deadline fires regardless of
what the process is doing.

**Writability is checked before training, not after.** A permissions problem discovered
after twenty minutes of GPU time has cost twenty minutes. Discovered in the first second,
it has cost nothing.
"""

from __future__ import annotations

import json
import math
import random
import signal
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import checkpoint
from .artifacts import provenance
from .checkpoint import AddressableStream, run_fingerprint
from .evaluation.perplexity import LikelihoodResult, evaluate_likelihood
from .space.cost import forward_flops_per_token, kv_cache_bytes, parameter_count
from .space.genotype import Genotype


@dataclass(frozen=True)
class TrainingConfig:
    """Everything that is held identical across the sweep.

    Identical for every architecture, so a difference in the objective is attributable to
    the architecture. Recorded in every manifest for the same reason.
    """

    #: Whether to compute the reporting battery after training. Off by default: the census
    #: needs only the objective, and computing generation measures for 1,099 architectures
    #: on a corpus that does not license them would spend node time producing numbers with
    #: no referent. The front is retrained with it on.
    run_battery: bool = False
    token_budget: int = 100_000_000
    block_size: int = 512
    effective_batch_tokens: int = 65_536
    #: Sequences per forward pass. Does not change the mathematics — the effective batch
    #: is fixed above and gradient accumulation makes up the difference — only how much of
    #: the card is busy at once. Eight sequences of 512 tokens left a T4 at roughly one per
    #: cent of its arithmetic peak. Halved automatically when an architecture cannot fit,
    #: and whatever was used is recorded in the manifest.
    micro_batch_size: int = 32
    #: Smallest micro-batch worth retrying at. Below this the run is not memory-bound, it
    #: is impossible, and failing says so rather than thrashing.
    min_micro_batch_size: int = 1
    #: Share of the card the probe must leave free after a full step.
    #:
    #: A probe runs one step, and one step cannot detect growth. Wave one of the sweep lost
    #: nineteen per cent of its jobs to memory errors at step 19 and step 427 — long after a
    #: probe that had passed. Rather than guess a third time at what accumulates, the probe
    #: requires the step to fit with room to spare, so late growth has somewhere to go.
    #: The failure is then a smaller micro-batch and a slower run instead of a lost job.
    memory_headroom_fraction: float = 0.25
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_fraction: float = 0.02
    grad_clip: float = 1.0
    seed: int = 20260101
    #: Training stops here and still writes results. Derived from the duration the canary
    #: measures at the largest architecture, with margin.
    soft_deadline_seconds: int = 7200
    #: Fires regardless of what the process is doing. The guarantee that a hung job
    #: releases its node rather than holding it busy indefinitely.
    hard_deadline_seconds: int = 8100
    #: Whether a GPU is mandatory. True for anything on the cluster: a silent CPU fallback
    #: on GPU-priced hardware is the failure that cost the first canary forty minutes.
    require_accelerator: bool = True
    #: Shortest gap between checkpoints. The actual gap is this or longer — see
    #: ``checkpoint_overhead_fraction``.
    checkpoint_interval_seconds: int = 600
    #: Share of runtime the checkpointing may consume, which is what actually sets the
    #: interval.
    #:
    #: A fixed interval is the wrong control here because the cost is not fixed: a
    #: checkpoint holds the parameters and both Adam moments, so it is roughly twelve bytes
    #: per parameter — 40 MB for the smallest architecture in this space and about 1.5 GB
    #: for the heaviest. Writing 1.5 GB to a blob mount every ten minutes would spend a
    #: noticeable fraction of a three-hour run on insurance.
    #:
    #: So the first write is timed and the interval is stretched until the overhead fits
    #: this budget. Cheap checkpoints stay frequent; expensive ones space themselves out.
    checkpoint_overhead_fraction: float = 0.05

    def as_dict(self) -> dict[str, object]:
        return {
            "token_budget": self.token_budget,
            "block_size": self.block_size,
            "effective_batch_tokens": self.effective_batch_tokens,
            "micro_batch_size": self.micro_batch_size,
            "min_micro_batch_size": self.min_micro_batch_size,
            "memory_headroom_fraction": self.memory_headroom_fraction,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_fraction": self.warmup_fraction,
            "grad_clip": self.grad_clip,
            "seed": self.seed,
            "require_accelerator": self.require_accelerator,
            "checkpoint_interval_seconds": self.checkpoint_interval_seconds,
            "checkpoint_overhead_fraction": self.checkpoint_overhead_fraction,
            "soft_deadline_seconds": self.soft_deadline_seconds,
            "hard_deadline_seconds": self.hard_deadline_seconds,
        }


class HardDeadlineExceeded(RuntimeError):
    """Raised by the hard deadline. The node is released rather than held busy."""


class OutputsNotWritable(RuntimeError):
    """Raised when the output location cannot be written, before any training happens."""


def set_determinism(seed: int, on_gpu: bool = False) -> None:
    """Seed every generator this pipeline draws from, at the strength the tier claims.

    The strength differs by tier and the difference is deliberate. The verify tier runs on
    CPU and claims bit-exactness, so it pays for deterministic kernels. The training tier
    does not: the spec states that GPU nondeterminism prevents bit-exact reproduction and
    that trained stages reproduce within a tolerance derived from measured seed variance.

    Forcing deterministic algorithms on the GPU therefore buys a guarantee the project
    explicitly declines to make, while replacing tuned kernels with slower ones. The
    seeding stays either way — what changes is only whether kernels are constrained.
    """
    import numpy as np
    import torch

    # The legacy global seeders on purpose, not a Generator. transformers, datasets and
    # torch's own samplers read the global numpy state; seeding a local Generator would
    # leave every one of them unseeded while looking as though determinism was handled.
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if on_gpu:
        # Let cuDNN autotune. Shapes are constant for a whole run — one architecture, one
        # block size, one batch size — so the search happens once and every step after
        # uses the winner.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.use_deterministic_algorithms(True, warn_only=True)


def autocast_dtype(device) -> tuple[bool, object]:
    """Which mixed precision this card actually accelerates.

    ``torch.cuda.is_bf16_supported()`` answers a different question than the one that
    matters: it returns True when PyTorch can *represent* bfloat16, including by emulation,
    not when the hardware has tensor cores for it. On a T4 — Turing, compute capability
    7.5 — it returns True and the resulting run reached 0.37 TFLOP/s against the card's 65
    TFLOP/s of fp16 tensor-core throughput.

    bfloat16 tensor cores arrive with Ampere, compute capability 8.0. Below that, float16
    is the format the silicon accelerates, at the cost of needing loss scaling. Asking for
    the capability directly is the check that matches the hardware.
    """
    import torch

    if device.type != "cuda":
        return False, torch.float32

    major, _ = torch.cuda.get_device_capability(device)
    return True, torch.bfloat16 if major >= 8 else torch.float16


def _release(model, optimiser=None, *tensors) -> None:
    """Drop everything the probe allocated, so the next attempt starts from a clean card."""
    import torch

    model.zero_grad(set_to_none=True)
    if optimiser is not None:
        optimiser.state.clear()
        del optimiser
    for tensor in tensors:
        del tensor
    torch.cuda.empty_cache()


class ArchitectureDoesNotFit(RuntimeError):
    """Raised when even a single sequence will not fit in memory."""


#: Substrings that identify a CUDA error which is memory exhaustion arriving from a library
#: outside PyTorch's caching allocator. cuBLAS and cuDNN call ``cudaMalloc`` themselves for
#: their handles and workspaces, so when the allocator has already reserved the card they
#: fail with their own status code rather than with ``torch.cuda.OutOfMemoryError``.
#:
#: Matched by message rather than by type because PyTorch surfaces every CUDA status as a
#: bare ``RuntimeError``. The list is deliberately short: an allocation failure is returned
#: by the API and leaves the context usable, whereas an illegal access or a launch failure
#: poisons it, and catching those would turn a corrupted run into a plausible-looking
#: result. Anything not named here still propagates.
_ALLOCATION_FAILURES = (
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUDNN_STATUS_ALLOC_FAILED",
    "out of memory",
)


def _is_allocation_failure(error: BaseException) -> bool:
    """Whether a CUDA RuntimeError is memory exhaustion under another name."""
    message = str(error)
    return any(marker in message for marker in _ALLOCATION_FAILURES)


def warm_cublas(device) -> None:
    """Make cuBLAS create its handle while the card is still empty.

    The handle and its workspace come from ``cudaMalloc``, not from PyTorch's caching
    allocator, so they have to be found in whatever the allocator has not already reserved.
    Deferring that to the first matmul of the probe means asking for it at the exact moment
    the card is fullest — which is how a 5.6 M-parameter architecture managed to fail at
    ``cublasCreate`` on a card with 16 GB. One tiny matmul first costs microseconds and
    moves the allocation to the only point in the job where it cannot fail.
    """
    import torch

    if device.type != "cuda":
        return
    probe = torch.zeros(8, 8, device=device)
    (probe @ probe).sum().item()
    del probe


def training_step(
    model, optimiser, scaler, batches, device, use_amp: bool, amp_dtype, grad_clip: float
) -> int:
    """One optimiser step over a list of micro-batches. Returns tokens consumed.

    The only place a step is performed. The probe and the training loop both call this,
    because every time they were separate implementations they diverged and the difference
    was a memory allocation the probe did not make.

    First the probe did forward and backward only, so it missed AdamW's two moment buffers
    — allocated lazily on the first step — and accepted a micro-batch that died at step
    zero asking for 1.54 GiB. Then it did forward, backward and step, so it missed the
    gradient clipping, which allocates temporaries across every parameter, and accepted a
    micro-batch that died at step one asking for 1.54 GiB again. Two different omissions
    with the same symptom.

    A shared implementation cannot be missing something the real step does, which is a
    stronger guarantee than remembering to add the next piece.
    """
    import torch

    optimiser.zero_grad(set_to_none=True)
    tokens = 0
    accum = len(batches)

    for batch in batches:
        ids = torch.tensor(batch, dtype=torch.long, device=device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(ids, labels=ids)
        scaler.scale(out.loss / accum).backward()
        tokens += ids.numel()

    # Unscale before clipping: clipping scaled gradients would clip to a threshold that
    # means something different on every step. This is also the part the probe used to
    # skip, and it is not free — the norm is computed across every parameter.
    scaler.unscale_(optimiser)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimiser)
    scaler.update()
    return tokens


def fit_micro_batch(model, config: TrainingConfig, device, use_amp: bool, amp_dtype) -> int:
    """Largest micro-batch this architecture fits, found by performing a real step.

    A fixed micro-batch cannot serve a space this wide. The architectures here span 0.1 to
    91 million non-embedding parameters and 2 to 24 layers; a size that keeps the largest
    from running out of memory leaves the card almost idle on the smallest, and the sweep
    would pay for that on every one of a thousand jobs.

    Probing rather than estimating: peak memory depends on the attention implementation, on
    what autocast keeps in which precision, on the optimiser's moment buffers, on the
    temporaries gradient clipping allocates, and on allocator fragmentation. No formula
    predicts that well enough to bet an hour of GPU time on, and two attempts at
    enumerating the contributions both missed one.

    So the probe runs ``training_step`` — the same function the loop runs — rather than an
    approximation of it. Anything the real step allocates, the probe allocates.
    """
    import torch
    from torch.optim import AdamW

    if device.type != "cuda":
        return config.micro_batch_size

    total = torch.cuda.get_device_properties(device).total_memory
    budget = total * (1.0 - config.memory_headroom_fraction)
    warm_cublas(device)

    size = config.micro_batch_size
    last_failure = "the probe's peak exceeded the head-room budget at every size"
    while size >= config.min_micro_batch_size:
        probe_optimiser = AdamW(model.parameters(), lr=config.learning_rate)
        probe_scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)
        try:
            torch.cuda.reset_peak_memory_stats(device)
            # One micro-batch is enough for the step's own peak: accumulation reuses the
            # same gradient buffers, so it does not grow with the accumulation count.
            batch = [[0] * config.block_size for _ in range(size)]
            training_step(
                model, probe_optimiser, probe_scaler, [batch],
                device, use_amp, amp_dtype, config.grad_clip,
            )
            peak = torch.cuda.max_memory_allocated(device)
            _release(model, probe_optimiser)

            if peak <= budget:
                return size
            # Fits, but with no room for whatever grows later. Treated exactly like not
            # fitting, because the outcome of accepting it is a lost job rather than a slow
            # one.
            size //= 2
        except torch.cuda.OutOfMemoryError:
            _release(model, probe_optimiser)
            last_failure = "PyTorch's allocator ran out at every size"
            size //= 2
        except RuntimeError as error:
            # The same condition arriving from cuBLAS or cuDNN instead of from PyTorch's
            # allocator. Halving is the right response to both; the only difference is
            # which library ran out first. Any other CUDA error propagates, because a
            # poisoned context that kept training would produce a number rather than a
            # failure, and a wrong number is the worse outcome.
            if not _is_allocation_failure(error):
                raise
            _release(model, probe_optimiser)
            last_failure = (
                f"a library outside PyTorch's allocator ran out at every size: "
                f"{str(error).splitlines()[0]}. On a small architecture that usually means "
                "the card was not empty, so this is worth checking before the result is "
                "read as a property of the architecture"
            )
            size //= 2

    raise ArchitectureDoesNotFit(
        f"a single sequence of {config.block_size} tokens will not fit on "
        f"{torch.cuda.get_device_name(device)}. This architecture cannot be measured on "
        "this hardware, which is a fact about the pair and belongs in the results rather "
        f"than being worked around. What ended the probe: {last_failure}."
    )


class Deadline:
    """A soft stop that preserves work, and a hard stop that guarantees release."""

    def __init__(self, soft_seconds: int, hard_seconds: int) -> None:
        if hard_seconds <= soft_seconds:
            raise ValueError("the hard deadline must leave room after the soft one")
        self.started = time.monotonic()
        self.soft_seconds = soft_seconds
        self.hard_seconds = hard_seconds
        self._armed = False

    def __enter__(self) -> Deadline:
        # SIGALRM fires even when the main thread is blocked inside a native call, which
        # is where a hang is most likely and least visible.
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, self._fire)
            signal.alarm(self.hard_seconds)
            self._armed = True
        return self

    def __exit__(self, *_) -> None:
        if self._armed:
            signal.alarm(0)

    @staticmethod
    def _fire(_signum, _frame):
        raise HardDeadlineExceeded(
            "hard wall-clock deadline reached; terminating so the node is released "
            "instead of remaining marked busy"
        )

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def soft_expired(self) -> bool:
        return self.elapsed >= self.soft_seconds


def pack_blocks(sequences: Sequence[Sequence[int]], block_size: int) -> list[list[int]]:
    """Concatenate token streams and cut them into fixed-length blocks.

    Every block is exactly ``block_size`` long, so token accounting is exact: the budget
    is a count of tokens the model actually saw, not an estimate from a padded batch.
    """
    stream: list[int] = []
    for seq in sequences:
        stream.extend(seq)
    usable = (len(stream) // block_size) * block_size
    return [stream[i : i + block_size] for i in range(0, usable, block_size)]




def _confidence_and_correctness(model, blocks, device, limit: int = 64):
    """Per-token top-prediction probability and whether it was right.

    What `calibration.evaluate` consumes, and deliberately not the probability of the
    observed token — that is what perplexity already measures. Calibration asks whether a
    model's stated confidence matches how often it is right, which needs the probability it
    assigned to its *own* choice.

    Capped at a fixed number of blocks. The measure is a distribution over tokens and
    converges long before the eval split is exhausted, while running the whole split for
    every architecture would add node time to a quantity that stops moving.
    """
    import torch

    confidences: list[float] = []
    correct: list[bool] = []
    model.eval()
    with torch.no_grad():
        for block in blocks[:limit]:
            ids = torch.tensor([list(block)], dtype=torch.long, device=device)
            logits = model(ids).logits.float()
            probabilities = torch.softmax(logits[:, :-1, :], dim=-1)
            top_probability, top_index = probabilities.max(dim=-1)
            targets = ids[:, 1:]
            confidences.extend(top_probability.flatten().tolist())
            correct.extend((top_index == targets).flatten().tolist())
    return confidences, correct


def evaluate_battery(model, eval_blocks, corpus: str, device) -> dict[str, object]:
    """Reporting measures beyond the objective, for the corpora that license each.

    The battery was implemented and tested months before this was written, and never ran:
    the entry point computed likelihood and stopped, so two reviewer objections were
    answered by describing code rather than by reporting measurements. Wiring it here is
    what turns those answers into claims about numbers.

    **Licensing is enforced per metric, not assumed.** A generation score on the search
    corpus at this parameter scale measures noise and reports it as a number — the exact
    failure the prior work's METEOR result was — so a metric the corpus does not admit is
    recorded as unlicensed rather than computed and captioned. "Not measured here, and why"
    is information; a number with no referent is not.
    """
    from .corpora.registry import CORPORA
    from .evaluation import blimp, calibration, degeneration

    entry = CORPORA.get(corpus)
    if entry is None:
        return {"unavailable": f"{corpus} is not in the registry"}

    results: dict[str, object] = {"corpus": corpus,
                                  "licensed": list(entry.licensed_metrics)}

    if entry.licenses("calibration"):
        try:
            confidences, correct = _confidence_and_correctness(
                model, eval_blocks, device)
            results["calibration"] = calibration.evaluate(
                confidences, correct).as_dict()
        except Exception as exc:  # noqa: BLE001
            results["calibration"] = {"unavailable": f"{type(exc).__name__}: {exc}"[:200]}
    else:
        results["calibration"] = {"unlicensed": entry.role}

    if entry.licenses("blimp"):
        # BLiMP is the metric that works where generation does not: a forced choice between
        # two given sentences is defined for every architecture in this space, including
        # the ones whose samples are noise.
        try:
            from transformers import GPT2TokenizerFast

            from .corpora.cache import cache_root
            from .corpora.tokenise import TOKENISER

            # The same tokeniser the corpus was encoded with. A different one would make
            # the two sentences of a pair different lengths for a reason having nothing to
            # do with grammar, and the summed log-probability would follow the tokenisation
            # rather than the judgement.
            tokenizer = GPT2TokenizerFast.from_pretrained(TOKENISER)
            items = blimp.load_items(cache_root())
            report = blimp.score_items(
                model, items, encode=lambda s: tokenizer.encode(s),
                device=device, bos_token_id=tokenizer.eos_token_id,
            )
            results["blimp"] = report.as_dict()
        except Exception as exc:  # noqa: BLE001
            results["blimp"] = {"unavailable": f"{type(exc).__name__}: {exc}"[:200]}
    else:
        results["blimp"] = {"unlicensed": entry.role}

    # The degeneration measures need generated text, so the samples are produced once and
    # scored by every metric that wants them. Generating per metric would pay for decoding
    # several times over and — worse — score each metric on a different set of samples.
    degeneration_metrics = ("distinct_n", "repetition_rate")
    if any(entry.licenses(metric) for metric in degeneration_metrics):
        try:
            samples = _generation_samples(model, eval_blocks)
            report = degeneration.evaluate(samples).as_dict()
            for metric in degeneration_metrics:
                results[metric] = report if entry.licenses(metric) else {
                    "unlicensed": entry.role}
        except Exception as exc:  # noqa: BLE001
            for metric in degeneration_metrics:
                results[metric] = {"unavailable": f"{type(exc).__name__}: {exc}"[:200]}
    else:
        for metric in degeneration_metrics:
            results[metric] = {"unlicensed": entry.role}

    # Licensed by the corpus and not implemented in this codebase. Recorded as such rather
    # than omitted: a metric missing from a report reads as a metric that was not licensed,
    # and the two are different claims. Each of these needs a dependency the project does
    # not carry — `mauve-text` with its feature model and faiss index, `sacrebleu` and
    # `nltk` for the overlap family — and none has been added, so none has a value.
    for metric in ("mauve", "self_bleu", "bleu", "rouge", "meteor", "gleu",
                   "syntactic_validity", "parse_rate", "ewok", "glue"):
        if metric in results:
            continue
        results[metric] = ({"not_implemented": "no implementation in this codebase"}
                           if entry.licenses(metric) else {"unlicensed": entry.role})
    return results


def _generation_samples(model, eval_blocks, prompts: int = 32,
                        prompt_tokens: int = 32, new_tokens: int = 64):
    """Greedy continuations of held-out prompts, as token ids.

    Greedy on purpose. Sampling temperature is a knob that moves every degeneration measure,
    so comparing architectures under sampled decoding compares decoding settings as much as
    models. Greedy has no knob, which makes the comparison across 220 architectures one
    where the only thing differing is the architecture.
    """
    from .evaluation import generation

    config = generation.DecodingConfig(max_new_tokens=new_tokens)
    pad = model.config.eos_token_id
    samples = []
    for block in eval_blocks[:prompts]:
        prompt = list(block)[:prompt_tokens]
        if len(prompt) < prompt_tokens:
            continue
        samples.append(
            generation.generate_continuation(model, prompt, config, pad_token_id=pad))
    return samples


@dataclass
class TrainingOutcome:
    """What a run produced, including how it ended."""

    genotype: Genotype
    status: str  # completed | soft_deadline | hard_deadline | failed
    tokens_consumed: int
    steps: int
    wall_clock_seconds: float
    likelihood: LikelihoodResult | None = None
    #: Reporting measures beyond the objective, keyed by metric. Empty when the battery was
    #: not asked for, and carrying an `unlicensed` note per metric the corpus does not
    #: admit — so a reader can tell "not measured" from "measured and absent".
    battery: dict[str, object] = field(default_factory=dict)
    error: str | None = None
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def usable_for_benchmark(self) -> bool:
        """Only a run that consumed its whole budget belongs in the table.

        A truncated run is a different experiment, and the prior work's central defect was
        exactly this kind of silent incomparability.
        """
        return self.status == "completed" and self.likelihood is not None


class AcceleratorMissing(RuntimeError):
    """Raised when a job that expects a GPU cannot see one."""


def resolve_device(require_accelerator: bool):
    """Pick the device, and refuse to train slowly on the wrong one without saying so.

    This exists because the canary ran forty minutes on the CPU of a GPU node. The model
    was simply never moved, so every tensor stayed where it was built; nothing failed,
    nothing warned, and the run produced a perfectly valid perplexity roughly sixty times
    slower than it should have. Across a sweep that is the difference between two days and
    four months, discovered by paying for it.

    A missing accelerator is therefore an error, not a fallback. The fallback is what made
    the original bug invisible: CPU training works, so there was nothing to notice.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_accelerator:
        raise AcceleratorMissing(
            "no CUDA device, and this job requires one. Training would fall back to the "
            "CPU and finish roughly sixty times slower on hardware billed for the GPU, "
            "which is how the first canary spent forty minutes producing six per cent of "
            "its token budget.\n"
            "Pass --allow-cpu for a deliberate CPU run, such as the smoke tier."
        )
    return torch.device("cpu")


def _verify_writable(output_dir: Path) -> None:
    """Fail in the first second rather than after twenty minutes of GPU time."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:  # noqa: BLE001
        raise OutputsNotWritable(
            f"cannot write to {output_dir}: {exc}. Refusing to train, because a result "
            "that cannot be written is a result that does not exist."
        ) from exc


def train_and_evaluate(
    genotype: Genotype,
    train_sequences: Sequence[Sequence[int]],
    eval_sequences: Sequence[Sequence[int]],
    output_dir: Path,
    # Safe as a default despite B008: TrainingConfig is a frozen dataclass, so the shared
    # instance cannot be mutated by a caller and every run gets the same budget.
    config: TrainingConfig = TrainingConfig(),  # noqa: B008
    corpus: str = "unknown",
    eval_byte_lengths: Sequence[int] | None = None,
    data_vintage: str = "unknown",
    tokeniser_fingerprint: str = "unknown",
) -> TrainingOutcome:
    """Train one architecture to its token budget and measure its perplexity."""
    import torch
    from torch.optim import AdamW

    from .model import build_model, count_parameters

    output_dir = Path(output_dir)
    _verify_writable(output_dir)

    # Declared before training, so an interrupted run leaves evidence of what it intended.
    (output_dir / "manifest.partial.json").write_text(
        json.dumps(
            {
                "status": "started",
                "genotype": genotype.as_tuple(),
                "corpus": corpus,
                "config": config.as_dict(),
                "provenance": provenance(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Device before seeding: how strongly determinism is enforced depends on which tier
    # this run belongs to, and that is decided by the hardware.
    device = resolve_device(config.require_accelerator)
    set_determinism(config.seed, on_gpu=device.type == "cuda")

    model = build_model(genotype).to(device)
    model.train()

    # Chosen from the device rather than configured: the right answer is a property of the
    # silicon, and a wrong setting here is not a crash but a silent sixty-fold slowdown.
    use_amp, amp_dtype = autocast_dtype(device)
    # float16 needs loss scaling to keep small gradients from flushing to zero; bfloat16
    # has the exponent range to do without.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype is torch.float16
    )

    blocks = pack_blocks(train_sequences, config.block_size)
    if not blocks:
        raise ValueError("the training corpus yielded no full blocks")

    micro_batch = fit_micro_batch(model, config, device, use_amp, amp_dtype)

    # The probe took a real optimiser step, so the weights have moved. Rebuild from the
    # same seed rather than training from a perturbed start: a benchmark row must depend
    # on the genotype and the seed, not on how many sizes the probe happened to try.
    del model
    set_determinism(config.seed, on_gpu=device.type == "cuda")
    model = build_model(genotype).to(device)
    model.train()

    accum = max(1, config.effective_batch_tokens // (micro_batch * config.block_size))
    total_steps = max(1, config.token_budget // config.effective_batch_tokens)
    warmup_steps = max(1, int(total_steps * config.warmup_fraction))

    optimiser = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return config.learning_rate * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return config.learning_rate * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    stream = AddressableStream(blocks, micro_batch, config.seed)

    # Resume before the first step, so a reclaimed node picks up where it stopped rather
    # than repeating work already paid for.
    checkpoint_dir = output_dir / "checkpoint"
    fingerprint = run_fingerprint(genotype, corpus, config)
    resumed = checkpoint.load(checkpoint_dir, fingerprint, model, optimiser, scaler)

    started = time.monotonic()
    tokens = resumed.tokens if resumed else 0
    step = resumed.step if resumed else 0
    batch_index = resumed.micro_batch_index if resumed else 0
    prior_seconds = resumed.elapsed_seconds if resumed else 0.0
    resumes = resumed.resumes if resumed else 0
    last_checkpoint = time.monotonic()
    checkpoint_interval = float(config.checkpoint_interval_seconds)
    checkpoint_seconds = 0.0
    checkpoints_written = 0
    if resumed:
        print(f"  resumed        step {step:,}, {tokens:,} tokens, "
              f"{prior_seconds:.0f}s already spent (resume #{resumes})")

    checkpoint_interval = float(config.checkpoint_interval_seconds)
    checkpoint_seconds = 0.0
    checkpoints_written = 0

    status = "completed"
    error: str | None = None

    try:
        with Deadline(config.soft_deadline_seconds, config.hard_deadline_seconds) as clock:
            while tokens < config.token_budget:
                if clock.soft_expired:
                    status = "soft_deadline"
                    break
                for group in optimiser.param_groups:
                    group["lr"] = lr_at(step)

                tokens += training_step(
                    model, optimiser, scaler,
                    [stream.at(batch_index + i) for i in range(accum)],
                    device, use_amp, amp_dtype, config.grad_clip,
                )
                batch_index += accum
                step += 1

                if time.monotonic() - last_checkpoint >= checkpoint_interval:
                    write_started = time.monotonic()
                    checkpoint.save(
                        checkpoint_dir,
                        checkpoint.Checkpoint(
                            step=step, tokens=tokens, micro_batch_index=batch_index,
                            elapsed_seconds=prior_seconds + (time.monotonic() - started),
                            resumes=resumes, fingerprint=fingerprint,
                        ),
                        model, optimiser, scaler,
                    )
                    took = time.monotonic() - write_started
                    checkpoint_seconds += took
                    checkpoints_written += 1

                    # Timed rather than assumed. A checkpoint costs roughly twelve bytes
                    # per parameter, so what is cheap for the smallest architecture here is
                    # 1.5 GB for the heaviest, and a fixed interval would quietly spend a
                    # large share of the longest runs on insurance.
                    checkpoint_interval = max(
                        float(config.checkpoint_interval_seconds),
                        took / max(config.checkpoint_overhead_fraction, 1e-6),
                    )
                    last_checkpoint = time.monotonic()
    except HardDeadlineExceeded as exc:
        status, error = "hard_deadline", str(exc)
    except Exception as exc:  # noqa: BLE001
        status, error = "failed", f"{type(exc).__name__}: {exc}"

    elapsed = prior_seconds + (time.monotonic() - started)

    # A run stopped by its soft deadline is going to be retried, so its last minutes are
    # worth keeping. One written here is the difference between resuming near the end and
    # resuming from the last periodic write.
    if status in ("soft_deadline", "hard_deadline"):
        try:
            checkpoint.save(
                checkpoint_dir,
                checkpoint.Checkpoint(
                    step=step, tokens=tokens, micro_batch_index=batch_index,
                    elapsed_seconds=elapsed, resumes=resumes, fingerprint=fingerprint,
                ),
                model, optimiser, scaler,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  could not checkpoint on stop: {exc}")

    likelihood = None
    battery: dict[str, object] = {}
    if status in ("completed", "soft_deadline"):
        eval_blocks = pack_blocks(eval_sequences, config.block_size)
        model.eval()
        likelihood = evaluate_likelihood(
            model,
            [eval_blocks[i : i + micro_batch]
             for i in range(0, len(eval_blocks), micro_batch)],
            byte_lengths=eval_byte_lengths,
        )
        if config.run_battery:
            battery = evaluate_battery(model, eval_blocks, corpus, device)

    outcome = TrainingOutcome(
        genotype=genotype,
        status=status,
        tokens_consumed=tokens,
        steps=step,
        wall_clock_seconds=round(elapsed, 3),
        likelihood=likelihood,
        battery=battery,
        error=error,
        extra={
            "parameters_closed_form": parameter_count(genotype),
            "parameters_instantiated": count_parameters(model),
            "kv_cache_bytes": kv_cache_bytes(genotype),
            "accumulation_steps": accum,
            "micro_batch_used": micro_batch,
            "resumes": resumes,
            "checkpoints_written": checkpoints_written,
            "checkpoint_seconds": round(checkpoint_seconds, 2),
            "checkpoint_overhead": (
                round(checkpoint_seconds / elapsed, 5) if elapsed > 0 else 0.0
            ),
            "checkpoint_interval_used": round(checkpoint_interval, 1),
            "resumed_from_step": resumed.step if resumed else None,
            "planned_steps": total_steps,
            # Recorded, not inferred. A run that trained on the CPU of a GPU node produces
            # a perfectly valid perplexity and nothing else distinguishes it, so the
            # benchmark table has to carry the device it was measured on.
            "device": device.type,
            # Recorded so a memory failure becomes measurable. Two rounds of guessing what
            # a training step allocates both missed something, and the third response is to
            # measure rather than to guess again.
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
            "device_memory_bytes": (
                int(torch.cuda.get_device_properties(device).total_memory)
                if device.type == "cuda" else None
            ),
            "precision": str(amp_dtype).replace("torch.", "") if use_amp else "float32",
            "tokens_per_second": round(tokens / elapsed, 1) if elapsed > 0 else 0.0,
            # Achieved arithmetic rate, so underutilisation is visible in the table rather
            # than inferred later from wall clock. The first GPU canary reached 0.57% of
            # its card's peak and nothing in the output said so.
            "achieved_tflops": round(
                3 * forward_flops_per_token(genotype) * tokens / elapsed / 1e12, 4
            ) if elapsed > 0 else 0.0,
        },
    )
    # Only once the row exists. Keeping them would leave a gigabyte per architecture in the
    # artifact container for runs whose results are already written.
    if outcome.usable_for_benchmark:
        checkpoint.clear(checkpoint_dir)

    write_run_manifest(
        outcome, output_dir, config, corpus,
        data_vintage=data_vintage,
        tokeniser_fingerprint=tokeniser_fingerprint,
    )
    return outcome


def write_run_manifest(
    outcome: TrainingOutcome,
    output_dir: Path,
    config: TrainingConfig,
    corpus: str,
    data_vintage: str = "unknown",
    tokeniser_fingerprint: str = "unknown",
) -> Path:
    """Write the manifest, and fail loudly if it cannot be written.

    A run whose manifest is missing cannot be placed in the benchmark, verified, or
    reproduced. Losing it quietly would mean losing the run without anyone noticing, so
    the failure is raised rather than logged.

    The fields a comparability check needs are promoted to the top level rather than left
    inside ``cost``. The canary produced three rows in float16 and one in emulated
    bfloat16, all of them complete and valid, with nothing in the perplexities to say that
    one came from different arithmetic. A check for that can only exist if the manifest
    states the arithmetic, the corpus vintage and the vocabulary as first-class facts about
    the run — see ``search.problem.require_comparable``.
    """
    payload = {
        "status": outcome.status,
        "usable_for_benchmark": outcome.usable_for_benchmark,
        "genotype": {
            "values": outcome.genotype.as_tuple(),
            "readable": str(outcome.genotype),
        },
        "corpus": corpus,
        # Promoted, not buried: these are what make one row comparable to another, and a
        # table is assembled by reading them rather than by trusting that nothing moved.
        "precision": outcome.extra.get("precision", "unknown"),
        "device": outcome.extra.get("device", "unknown"),
        "data_vintage": data_vintage,
        "tokeniser_fingerprint": tokeniser_fingerprint,
        "config": config.as_dict(),
        "tokens_consumed": outcome.tokens_consumed,
        "token_budget": config.token_budget,
        "budget_met": outcome.tokens_consumed >= config.token_budget,
        "steps": outcome.steps,
        "wall_clock_seconds": outcome.wall_clock_seconds,
        "objectives": (
            outcome.likelihood.as_dict() if outcome.likelihood is not None else None
        ),
        # Separate from `objectives` on purpose: these are reported, never searched on.
        # Folding them in would invite a later reader to optimise one.
        "battery": outcome.battery or None,
        "cost": outcome.extra,
        "error": outcome.error,
        "provenance": provenance(),
    }
    target = output_dir / "manifest.json"
    try:
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        raise OutputsNotWritable(
            f"could not write the run manifest to {target}: {exc}. The run is lost: a "
            "result without a manifest cannot be placed in the benchmark, verified or "
            "reproduced."
        ) from exc
    partial = output_dir / "manifest.partial.json"
    if partial.exists():
        partial.unlink()
    return target
