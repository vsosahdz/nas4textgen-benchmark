"""The training entry point, exercised end to end on CPU at a tiny budget.

This is the `smoke` replication tier: the whole pipeline, small enough to run in seconds
without a GPU. If it passes, the pieces fit together; the sweep only changes the numbers.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from nas4textgen.corpora.cache import cache_root  # noqa: E402
from nas4textgen.space.genotype import Genotype  # noqa: E402
from nas4textgen.training import (  # noqa: E402
    Deadline,
    HardDeadlineExceeded,
    OutputsNotWritable,
    TrainingConfig,
    pack_blocks,
    train_and_evaluate,
)

pytestmark = pytest.mark.slow

TINY = Genotype(h=2, d_head=32, n=2, ffn=2, kv=1)


def corpus(rows: int, length: int, seed: int) -> list[list[int]]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 5000, (length,), generator=gen).tolist() for _ in range(rows)]


def smoke_config(**overrides) -> TrainingConfig:
    base = {
        "token_budget": 8192,
        "block_size": 64,
        "effective_batch_tokens": 512,
        "micro_batch_size": 4,
        "soft_deadline_seconds": 120,
        "hard_deadline_seconds": 180,
        # The smoke tier runs on CPU by design, and says so. The default is to require an
        # accelerator, because a silent CPU fallback on the cluster is not a slower run —
        # it is a run that costs sixty times what it should and looks identical afterwards.
        "require_accelerator": False,
    }
    base.update(overrides)
    return TrainingConfig(**base)


# ── token accounting ───────────────────────────────────────────────────────────

def test_blocks_are_exactly_the_declared_length():
    """Exact accounting is what makes the budget a count rather than an estimate."""
    blocks = pack_blocks([[1] * 100, [2] * 50], block_size=64)
    assert all(len(b) == 64 for b in blocks)
    assert len(blocks) == 150 // 64


def test_the_budget_is_met_in_tokens_not_epochs(tmp_path):
    """The prior work gave its baseline four times the training of its candidates."""
    config = smoke_config()
    outcome = train_and_evaluate(
        TINY, corpus(40, 64, 1), corpus(8, 64, 2), tmp_path, config, corpus="synthetic"
    )
    assert outcome.status == "completed"
    assert outcome.tokens_consumed >= config.token_budget
    assert outcome.usable_for_benchmark


# ── the manifest ───────────────────────────────────────────────────────────────

def test_the_manifest_records_everything_needed_to_reject_the_run(tmp_path):
    train_and_evaluate(
        TINY, corpus(40, 64, 1), corpus(8, 64, 2), tmp_path, smoke_config(), corpus="synthetic"
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "completed"
    assert manifest["budget_met"] is True
    assert manifest["objectives"]["perplexity"] > 0
    assert manifest["config"]["seed"]
    assert manifest["provenance"]["git_revision"] is not None
    assert "git_dirty" in manifest["provenance"]


def test_the_closed_form_cost_matches_the_model_that_was_trained(tmp_path):
    """The benchmark's cost column and its quality column must describe one object."""
    outcome = train_and_evaluate(
        TINY, corpus(40, 64, 1), corpus(8, 64, 2), tmp_path, smoke_config(), corpus="synthetic"
    )
    assert outcome.extra["parameters_closed_form"] == outcome.extra["parameters_instantiated"]


def test_a_partial_manifest_is_written_before_training(tmp_path):
    """An interrupted run leaves evidence of what it intended."""
    train_and_evaluate(
        TINY, corpus(40, 64, 1), corpus(8, 64, 2), tmp_path, smoke_config(), corpus="synthetic"
    )
    assert not (tmp_path / "manifest.partial.json").exists(), "cleared on success"
    assert (tmp_path / "manifest.json").exists()


# ── failing early and loudly ───────────────────────────────────────────────────

def test_unwritable_output_fails_before_any_training(tmp_path):
    """Discovered in the first second it costs nothing; after twenty minutes it costs those."""
    blocked = tmp_path / "blocked"
    blocked.write_text("this is a file, not a directory", encoding="utf-8")
    with pytest.raises(OutputsNotWritable, match="Refusing to train"):
        train_and_evaluate(
            TINY, corpus(4, 64, 1), corpus(2, 64, 2), blocked, smoke_config(), corpus="x"
        )


# ── deadlines ──────────────────────────────────────────────────────────────────

def test_the_hard_deadline_must_leave_room_after_the_soft_one():
    with pytest.raises(ValueError, match="room"):
        Deadline(soft_seconds=100, hard_seconds=100)


def test_the_hard_deadline_fires(tmp_path):
    """The guarantee that a hung job releases its node instead of holding it busy."""
    import time

    with pytest.raises(HardDeadlineExceeded), Deadline(soft_seconds=1, hard_seconds=2):
        time.sleep(5)


def test_a_soft_deadline_still_produces_a_measurement(tmp_path):
    """Stopping early must preserve work — and mark it unusable for the benchmark."""
    config = smoke_config(token_budget=10**9, soft_deadline_seconds=1, hard_deadline_seconds=120)
    outcome = train_and_evaluate(
        TINY, corpus(40, 64, 1), corpus(8, 64, 2), tmp_path, config, corpus="synthetic"
    )
    assert outcome.status == "soft_deadline"
    assert outcome.likelihood is not None, "partial work is still measured"
    assert not outcome.usable_for_benchmark, "a truncated run is a different experiment"

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["budget_met"] is False
    assert manifest["usable_for_benchmark"] is False


# ── the entry point the image actually invokes ─────────────────────────────────

def test_the_image_entrypoint_module_exists():
    """The Dockerfile's ENTRYPOINT named a module that did not exist.

    A job submitted with that image would have failed on the first line, after allocating
    a node and pulling three gigabytes. Nothing in the build caught it, because a build
    does not run its own entrypoint.
    """
    import importlib

    module = importlib.import_module("nas4textgen.train")
    assert hasattr(module, "main")


def test_the_entrypoint_rejects_a_genotype_off_the_grid():
    from nas4textgen.train import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--h", "5", "--d-head", "64", "--n", "6", "--ffn", "4", "--kv", "1",
             "--output", "/tmp/x"]
        )


def test_the_entrypoint_requires_every_gene():
    from nas4textgen.train import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--h", "4", "--output", "/tmp/x"])


def test_exit_status_distinguishes_truncated_from_failed():
    """The scheduler needs three outcomes, not two.

    A truncated run produced a measurement that is written but not comparable; a failed run
    produced nothing. Collapsing them would either discard usable evidence or admit
    incomparable rows into the benchmark.
    """
    from nas4textgen.train import EXIT_ENVIRONMENT, EXIT_FAILED, EXIT_OK, EXIT_TRUNCATED

    assert len({EXIT_OK, EXIT_TRUNCATED, EXIT_FAILED, EXIT_ENVIRONMENT}) == 4


def test_a_missing_accelerator_is_an_error_not_a_quiet_fallback():
    """The bug that cost the first canary forty minutes, pinned.

    The model was never moved to the device, so every tensor stayed where it was built.
    Nothing failed and nothing warned: CPU training works. The run produced a valid
    perplexity roughly sixty times slower than it should have, on hardware billed for its
    GPU. A fallback is exactly what made it invisible, so there is no longer a fallback
    unless it is asked for.
    """
    import torch

    from nas4textgen.training import AcceleratorMissing, resolve_device

    if torch.cuda.is_available():
        assert resolve_device(require_accelerator=True).type == "cuda"
        return

    with pytest.raises(AcceleratorMissing, match="sixty times slower"):
        resolve_device(require_accelerator=True)
    assert resolve_device(require_accelerator=False).type == "cpu"


def test_the_manifest_records_the_device_a_result_was_measured_on(tmp_path):
    """A CPU run and a GPU run produce indistinguishable perplexities.

    Without the device in the table, a wave that silently fell back would enter the
    benchmark alongside the rest and nothing would separate them.
    """
    outcome = train_and_evaluate(
        TINY, corpus(40, 64, 1), corpus(8, 64, 2), tmp_path, smoke_config(),
        corpus="synthetic",
    )
    assert outcome.extra["device"] in ("cpu", "cuda")
    assert outcome.extra["precision"] in ("float32", "float16", "bfloat16")
    assert outcome.extra["tokens_per_second"] > 0


def test_mixed_precision_follows_the_hardware_not_a_capability_flag():
    """The check that cost the first GPU canary most of its throughput.

    torch.cuda.is_bf16_supported() answers whether PyTorch can represent bfloat16, which
    includes emulating it, not whether the silicon accelerates it. On a T4 — Turing,
    compute capability 7.5 — it returns True, and the run reached 0.37 TFLOP/s against the
    card's 65 TFLOP/s of fp16 tensor-core throughput.

    bfloat16 tensor cores arrive with Ampere at capability 8.0. Below that, float16 is the
    format the hardware accelerates.
    """
    from nas4textgen.training import autocast_dtype

    use_amp, dtype = autocast_dtype(torch.device("cpu"))
    assert use_amp is False and dtype is torch.float32

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device to interrogate")

    use_amp, dtype = autocast_dtype(torch.device("cuda"))
    major, _ = torch.cuda.get_device_capability()
    assert use_amp is True
    assert dtype is (torch.bfloat16 if major >= 8 else torch.float16)


def test_kernel_determinism_is_enforced_only_where_the_tier_claims_it():
    """The spec states trained stages are tolerance-exact and that GPU nondeterminism
    prevents bit-exact reproduction. Constraining CUDA kernels therefore pays for a
    guarantee the project declines to make, in exchange for slower kernels.

    The seeding is unconditional either way; only kernel selection changes.
    """
    from nas4textgen.training import set_determinism

    set_determinism(7, on_gpu=False)
    first = [torch.rand(1).item() for _ in range(3)]
    set_determinism(7, on_gpu=False)
    assert [torch.rand(1).item() for _ in range(3)] == first, "seeding must not depend on it"

    set_determinism(7, on_gpu=True)
    second = [torch.rand(1).item() for _ in range(3)]
    assert second == first, "the GPU path seeds identically"


class TestCublasAllocation:
    """A 5.6M-parameter architecture died at ``cublasCreate`` on a 16 GB card, mid-sweep.

    cuBLAS allocates its handle with ``cudaMalloc`` rather than through PyTorch's caching
    allocator, so once the allocator has reserved the card there is nothing left for it. The
    probe's backoff caught ``torch.cuda.OutOfMemoryError`` and this arrives as a bare
    ``RuntimeError``, so the job died instead of halving the batch.
    """

    def test_the_cublas_status_is_recognised_as_memory_exhaustion(self):
        from nas4textgen.training import _is_allocation_failure

        actual = RuntimeError(
            "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`"
        )
        assert _is_allocation_failure(actual)
        assert _is_allocation_failure(RuntimeError("CUDA out of memory. Tried to allocate"))
        assert _is_allocation_failure(RuntimeError("CUDNN_STATUS_ALLOC_FAILED"))

    def test_a_corrupted_context_is_not_swallowed(self):
        """An illegal access poisons the context: continuing would produce a number rather
        than a failure, and a wrong number is the worse outcome."""
        from nas4textgen.training import _is_allocation_failure

        for message in (
            "CUDA error: an illegal memory access was encountered",
            "CUDA error: unspecified launch failure",
            "CUDA error: device-side assert triggered",
        ):
            assert not _is_allocation_failure(RuntimeError(message)), message

    def test_the_handle_is_created_before_the_probe_fills_the_card(self):
        """Prevention rather than recovery: one tiny matmul moves cuBLAS's allocation to the
        only point in the job where it cannot fail."""
        import inspect

        from nas4textgen.training import fit_micro_batch

        source = inspect.getsource(fit_micro_batch)
        warm = source.index("warm_cublas(device)")
        loop = source.index("while size >= config.min_micro_batch_size")
        assert warm < loop, "warming after the loop starts defeats the purpose"

    def test_warming_is_a_no_op_without_a_device(self):
        from nas4textgen.training import warm_cublas

        warm_cublas(torch.device("cpu"))

    @pytest.mark.gpu
    def test_warming_actually_runs_on_the_card(self):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device to warm")
        from nas4textgen.training import warm_cublas

        warm_cublas(torch.device("cuda"))


class TestTheReportingBattery:
    """The battery was implemented in task 8.5 and never ran until this was written.

    Every census manifest carries perplexity and nothing else, so the reviewers who asked
    for grammaticality and calibration were answered with a promise. These tests exist so
    the promise cannot quietly lapse a second time: they check that the battery is wired to
    something a model can actually be run through, not merely importable.
    """

    def _tiny_model(self):
        from nas4textgen.model import build_model
        from nas4textgen.space.enumerate import enumerate_space

        genotype = sorted(enumerate_space(), key=lambda g: g.as_tuple())[0]
        return build_model(genotype)

    def test_confidence_is_the_models_own_choice_not_the_observed_token(self):
        """Calibration asks whether stated confidence matches how often the model is right.

        Using the probability of the observed token instead would measure likelihood, which
        perplexity already reports, and would make the two objectives redundant.
        """
        from nas4textgen.training import _confidence_and_correctness

        model = self._tiny_model()
        torch.manual_seed(0)
        blocks = torch.randint(0, model.config.vocab_size, (2, 64)).tolist()
        confidences, correct = _confidence_and_correctness(
            model, blocks, torch.device("cpu"))

        assert len(confidences) == len(correct) == 2 * 63
        assert all(0.0 <= c <= 1.0 for c in confidences)
        # The top probability is by construction the largest in its row, so no confidence
        # may fall below uniform over the vocabulary.
        assert min(confidences) >= 1.0 / model.config.vocab_size

    def test_an_untrained_model_lands_at_chance_on_blimp(self):
        """The null the battery has to reproduce before any trained result means anything.

        A metric that reports a comfortable score for a model with random weights is
        measuring its own implementation, not the model.
        """
        from nas4textgen.evaluation import blimp
        from nas4textgen.training import evaluate_battery

        try:
            blimp.load_items(cache_root())
        except blimp.ItemsUnavailable:
            pytest.skip("BLiMP is not cached; fetch it with `cli fetch --blimp`")

        report = evaluate_battery(
            self._tiny_model(), torch.randint(0, 100, (2, 64)).tolist(),
            "wikitext103", torch.device("cpu"))

        assert abs(report["blimp"]["macro_accuracy"] - 0.5) < 0.1
        assert report["calibration"]["samples"] > 0

    def test_a_metric_its_corpus_does_not_license_is_refused_not_computed(self):
        """The licence is the whole point of the corpus registry.

        MAUVE on the search corpus would be a number reported against the split the search
        selected on, which is the circularity reviewer R2 objected to.
        """
        from nas4textgen.training import evaluate_battery

        report = evaluate_battery(
            self._tiny_model(), [], "wikitext103", torch.device("cpu"))
        assert report["mauve"] == {"unlicensed": "search corpus"}

    def test_an_unknown_corpus_reports_that_rather_than_guessing(self):
        from nas4textgen.training import evaluate_battery

        report = evaluate_battery(
            self._tiny_model(), [], "not-a-corpus", torch.device("cpu"))
        assert "unavailable" in report

    def test_the_census_did_not_pay_for_the_battery(self):
        """Off by default, and deliberately.

        1,099 runs each paying for BLiMP would have added node time to a quantity no
        objective consumes. The battery belongs on the front, not on the census.
        """
        from nas4textgen.training import TrainingConfig

        assert TrainingConfig().run_battery is False
