"""Resuming must be the same experiment, not merely a cheaper one.

On spot compute a reclaim is normal, so without checkpoints a preemption late in a
three-hour run costs three hours and buys nothing. But a resumed run that diverges from an
uninterrupted one is worse than the waste it saves: the benchmark would then contain rows
produced under conditions that differ by whether a node happened to be reclaimed, which is
the class of defect this whole project exists to correct.

So the central test is not "resume works" but "resume is indistinguishable".
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from nas4textgen import checkpoint as ck  # noqa: E402
from nas4textgen.space.genotype import Genotype  # noqa: E402
from nas4textgen.training import TrainingConfig, train_and_evaluate  # noqa: E402

pytestmark = pytest.mark.slow

TINY = Genotype(h=2, d_head=32, n=2, ffn=2, kv=1)


def corpus(rows: int, length: int, seed: int) -> list[list[int]]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 5000, (length,), generator=gen).tolist() for _ in range(rows)]


def config(**overrides) -> TrainingConfig:
    base = {
        "token_budget": 8192, "block_size": 64, "effective_batch_tokens": 512,
        "micro_batch_size": 4, "soft_deadline_seconds": 120,
        "hard_deadline_seconds": 180, "require_accelerator": False,
        "checkpoint_interval_seconds": 10**6,  # off unless a test asks for it
    }
    base.update(overrides)
    return TrainingConfig(**base)


class TestAddressableStream:
    """Position is addressable so that resuming needs one integer rather than a
    generator's unserialisable internal state."""

    def stream(self, seed=0, micro_batch=4):
        return ck.AddressableStream([[i] * 8 for i in range(40)], micro_batch, seed)

    def test_the_same_index_always_yields_the_same_batch(self):
        first, second = self.stream(), self.stream()
        for index in (0, 1, 7, 9, 10, 25, 100):
            assert first.at(index) == second.at(index)

    def test_reading_out_of_order_does_not_change_what_is_read(self):
        """The property a stateful generator cannot offer, and the one resume needs."""
        forward = [self.stream().at(i) for i in range(20)]
        jumped = self.stream()
        assert jumped.at(19) == forward[19]
        assert jumped.at(3) == forward[3], "going back must not depend on going forward"

    def test_an_epoch_is_a_permutation_of_every_block(self):
        """No block is skipped or repeated within an epoch, which is what makes the token
        budget a pass over the data rather than a sample of it."""
        stream = self.stream()
        seen = [b for i in range(stream.per_epoch) for b in stream.at(i)]
        assert len(seen) == 40
        assert sorted(x[0] for x in seen) == list(range(40))

    def test_consecutive_epochs_do_not_repeat_the_same_order(self):
        stream = self.stream()
        first = [stream.at(i) for i in range(stream.per_epoch)]
        second = [stream.at(stream.per_epoch + i) for i in range(stream.per_epoch)]
        assert first != second

    def test_different_seeds_give_different_orders(self):
        assert self.stream(seed=0).at(0) != self.stream(seed=1).at(0)

    def test_a_micro_batch_larger_than_the_corpus_is_refused(self):
        with pytest.raises(ValueError, match="cannot fill"):
            ck.AddressableStream([[1, 2]], micro_batch=4, seed=0)


def interrupt_after(monkeypatch, steps: int) -> None:
    """Make the soft deadline fire after a fixed number of steps.

    A deterministic interruption rather than a race against a wall clock, and it exercises
    exactly the path a spot reclaim takes: the run stops, evaluates what it has, and writes
    a checkpoint on the way out.
    """
    from nas4textgen.training import Deadline

    seen = {"checks": 0}

    def expired(self) -> bool:
        seen["checks"] += 1
        return seen["checks"] > steps

    monkeypatch.setattr(Deadline, "soft_expired", property(expired))


class TestResumeIsIndistinguishable:
    def test_a_resumed_run_matches_an_uninterrupted_one(self, tmp_path, monkeypatch):
        """The test the whole module exists for.

        Bit-exact here because the smoke tier runs on CPU, where the project does claim
        bit-exactness. On GPU a resumed run differs from an uninterrupted one exactly as
        much as two uninterrupted GPU runs differ from each other — resume adds no new
        source of variation, and the spec already declares that tier tolerance-exact.
        """
        train, evaluate = corpus(80, 64, 1), corpus(8, 64, 2)

        whole = train_and_evaluate(
            TINY, train, evaluate, tmp_path / "whole", config(), corpus="synthetic"
        )
        assert whole.status == "completed"

        split = tmp_path / "split"
        interrupt_after(monkeypatch, steps=5)
        stopped = train_and_evaluate(
            TINY, train, evaluate, split, config(), corpus="synthetic"
        )
        assert stopped.status == "soft_deadline"
        assert 0 < stopped.steps < whole.steps, "the interruption must land mid-run"
        assert (split / "checkpoint" / ck.POINTER).exists(), (
            "a reclaimed run must leave something to resume from"
        )

        monkeypatch.undo()
        resumed = train_and_evaluate(
            TINY, train, evaluate, split, config(), corpus="synthetic"
        )

        assert resumed.status == "completed"
        assert resumed.extra["resumes"] == 1
        assert resumed.tokens_consumed == whole.tokens_consumed
        assert resumed.steps == whole.steps
        assert resumed.likelihood.perplexity == pytest.approx(
            whole.likelihood.perplexity, rel=1e-9
        ), "a reclaimed node must not change the number the benchmark records"

    def test_the_manifest_says_a_run_was_resumed(self, tmp_path, monkeypatch):
        """A row produced across a preemption should be identifiable as one, even though it
        is equivalent — an equivalence that is asserted is worth more than one assumed."""
        train, evaluate = corpus(80, 64, 1), corpus(8, 64, 2)

        interrupt_after(monkeypatch, steps=4)
        train_and_evaluate(TINY, train, evaluate, tmp_path, config(), corpus="synthetic")
        monkeypatch.undo()
        train_and_evaluate(TINY, train, evaluate, tmp_path, config(), corpus="synthetic")

        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["cost"]["resumes"] == 1
        assert manifest["cost"]["resumed_from_step"] > 0

    def test_time_already_spent_is_carried_across_the_resume(self, tmp_path, monkeypatch):
        """Otherwise the recorded wall clock is the last leg only, and the cost of a
        preempted architecture reads as cheaper than an uninterrupted one."""
        train, evaluate = corpus(80, 64, 1), corpus(8, 64, 2)

        interrupt_after(monkeypatch, steps=4)
        first = train_and_evaluate(TINY, train, evaluate, tmp_path, config(),
                                   corpus="synthetic")
        monkeypatch.undo()
        second = train_and_evaluate(TINY, train, evaluate, tmp_path, config(),
                                    corpus="synthetic")

        assert second.wall_clock_seconds >= first.wall_clock_seconds


class TestSafety:
    def test_a_checkpoint_from_another_experiment_is_refused(self, tmp_path, monkeypatch):
        """Loading it would train the wrong architecture and report the right one, which no
        downstream check could detect: the manifest would describe what was asked for and
        the weights would be something else."""
        train, evaluate = corpus(80, 64, 1), corpus(8, 64, 2)

        interrupt_after(monkeypatch, steps=4)
        train_and_evaluate(TINY, train, evaluate, tmp_path, config(), corpus="synthetic")
        monkeypatch.undo()
        assert (tmp_path / "checkpoint" / ck.POINTER).exists()

        # Raised rather than recorded as a failed run: this output directory belongs to a
        # different experiment, and no amount of training would make it this one's.
        other = Genotype(h=4, d_head=32, n=2, ffn=2, kv=1)
        with pytest.raises(ck.CheckpointCorrupt, match="wrong architecture"):
            train_and_evaluate(other, train, evaluate, tmp_path, config(),
                               corpus="synthetic")

    def test_the_entry_point_treats_it_as_an_environment_error(self):
        """Exit 4, not exit 3. The distinction matters to the scheduler: a failed run is
        worth another node, a wrong output directory is not."""
        import inspect

        from nas4textgen import train as entry

        assert entry.EXIT_ENVIRONMENT == 4 and entry.EXIT_FAILED == 3
        source = inspect.getsource(entry.main)
        assert "CheckpointCorrupt" in source
        handler = source[source.index("except CheckpointCorrupt"):]
        assert "EXIT_ENVIRONMENT" in handler[:400]

    def test_a_kill_during_a_write_leaves_the_previous_checkpoint_usable(self, tmp_path):
        """The property the slot alternation and the write ordering exist for.

        Tested by behaviour rather than by reading the source: the payload goes to the slot
        the pointer is not using, and the pointer is written last, so a process killed
        between them leaves the earlier checkpoint still named and still complete.
        """
        model = torch.nn.Linear(4, 4)
        optimiser = torch.optim.AdamW(model.parameters())

        ck.save(tmp_path, ck.Checkpoint(10, 100, 10, 1.0, 0, "fp"), model, optimiser, None)
        good = json.loads((tmp_path / ck.POINTER).read_text(encoding="utf-8"))

        # A second write that dies after the payload and before the pointer: write the
        # payload into the other slot by hand and leave the pointer alone.
        other = "b" if good["slot"] == "a" else "a"
        (tmp_path / f"state.{other}.pt").write_bytes(b"half a tensor")

        recovered = ck.load(tmp_path, "fp", model, optimiser, None)
        assert recovered is not None and recovered.step == 10

    def test_writes_alternate_between_slots(self, tmp_path):
        """So a kill during a write cannot destroy the only good checkpoint."""
        model = torch.nn.Linear(4, 4)
        optimiser = torch.optim.AdamW(model.parameters())

        def write(step):
            ck.save(tmp_path, ck.Checkpoint(step, step * 10, step, 1.0, 0, "fp"),
                    model, optimiser, None)
            return json.loads((tmp_path / ck.POINTER).read_text(encoding="utf-8"))["slot"]

        assert write(1) != write(2)
        assert (tmp_path / "state.a.pt").exists()
        assert (tmp_path / "state.b.pt").exists()

    def test_a_completed_run_leaves_no_checkpoint_behind(self, tmp_path):
        """A gigabyte per architecture, kept for runs whose results are already written,
        would be more storage than the benchmark itself."""
        outcome = train_and_evaluate(
            TINY, corpus(80, 64, 1), corpus(8, 64, 2), tmp_path,
            config(checkpoint_interval_seconds=0), corpus="synthetic",
        )
        assert outcome.usable_for_benchmark
        assert not (tmp_path / "checkpoint" / ck.POINTER).exists()

    def test_nothing_to_resume_is_not_an_error(self, tmp_path):
        model = torch.nn.Linear(4, 4)
        optimiser = torch.optim.AdamW(model.parameters())
        assert ck.load(tmp_path, "fp", model, optimiser, None) is None


class TestCheckpointCost:
    """The insurance has a premium, and it is measured rather than assumed."""

    def test_the_interval_stretches_when_a_write_is_expensive(self, tmp_path, monkeypatch):
        """A checkpoint is roughly twelve bytes per parameter — parameters plus both Adam
        moments — so what costs 40 MB for the smallest architecture in this space costs
        about 1.5 GB for the heaviest. A fixed interval would spend a large share of the
        longest runs on insurance.
        """
        from nas4textgen import checkpoint as module

        real_save = module.save

        def slow_save(*args, **kwargs):
            import time as clock

            clock.sleep(0.3)
            return real_save(*args, **kwargs)

        monkeypatch.setattr("nas4textgen.training.checkpoint.save", slow_save)

        outcome = train_and_evaluate(
            TINY, corpus(160, 64, 1), corpus(8, 64, 2), tmp_path,
            config(token_budget=32768, checkpoint_interval_seconds=0,
                   checkpoint_overhead_fraction=0.05),
            corpus="synthetic",
        )

        assert outcome.extra["checkpoints_written"] >= 1
        # 0.3 s of writing against a 5% budget implies at least six seconds between writes.
        assert outcome.extra["checkpoint_interval_used"] >= 5.0

        # The stretch is what the test is about, and the evidence for it is that a config
        # asking for a checkpoint every step produced one and then stopped. The share of
        # wall clock spent writing is not evidence: it is 0.3 s over however long this
        # machine took to train a tiny model for 32,768 tokens, so the same correct
        # behaviour reads as 0.4 on a loaded laptop and 0.73 on an idle one. Bounding it
        # bounds the hardware, and this assertion failed four runs in five once the
        # machine went quiet.
        assert outcome.extra["checkpoints_written"] == 1, (
            "the interval did not stretch: a second write happened inside the window the "
            "first one bought")

    def test_the_manifest_records_what_the_insurance_cost(self, tmp_path):
        """So a slow run can be attributed rather than guessed at."""
        outcome = train_and_evaluate(
            TINY, corpus(80, 64, 1), corpus(8, 64, 2), tmp_path,
            config(checkpoint_interval_seconds=0), corpus="synthetic",
        )
        for key in ("checkpoints_written", "checkpoint_seconds", "checkpoint_overhead"):
            assert key in outcome.extra


def test_the_probe_and_the_loop_run_the_same_step():
    """The property that stops them diverging, asserted structurally.

    Two OOMs came from the probe being an approximation of a training step. First it did
    forward and backward only, missing AdamW's moment buffers, and accepted a micro-batch
    that died at step zero. Then it did forward, backward and step, missing the gradient
    clipping temporaries, and accepted one that died at step one — the same 1.54 GiB, a
    different omission.

    Enumerating the contributions failed twice, so the probe now calls the function the
    loop calls. A shared implementation cannot be missing something the real step does.
    """
    import inspect

    from nas4textgen import training

    probe = inspect.getsource(training.fit_micro_batch)
    loop = inspect.getsource(training.train_and_evaluate)

    assert "training_step(" in probe
    assert "training_step(" in loop

    # Neither may perform the step itself, or the sharing is decorative.
    for name, body in (("probe", probe), ("loop", loop)):
        for performed in ("clip_grad_norm_", "scaler.step(", ".backward()"):
            assert performed not in body, (
                f"{name} performs {performed} directly instead of through training_step"
            )


def test_a_step_consumes_exactly_the_tokens_it_was_given(tmp_path):
    """Accounting has to be exact for the budget to be a count rather than an estimate."""
    from nas4textgen.model import build_model
    from nas4textgen.training import training_step

    model = build_model(TINY)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    batches = [[[1] * 64 for _ in range(4)] for _ in range(3)]
    tokens = training_step(
        model, optimiser, scaler, batches,
        torch.device("cpu"), False, torch.float32, 1.0,
    )
    assert tokens == 3 * 4 * 64


def test_the_probe_demands_head_room_rather_than_a_bare_fit():
    """One step cannot detect growth, so it requires room for growth instead.

    Wave one of the sweep lost nineteen per cent of its jobs to memory errors at step 19
    and step 427 — long after a probe that had passed a full step including the optimiser
    and the clipping. Two rounds of enumerating what a step allocates both missed
    something, so the third response is to stop enumerating: require the step to fit with a
    quarter of the card free, and the consequence of being wrong becomes a smaller
    micro-batch instead of a lost job.
    """
    import inspect

    from nas4textgen.training import TrainingConfig, fit_micro_batch

    assert 0 < TrainingConfig().memory_headroom_fraction < 1

    source = inspect.getsource(fit_micro_batch)
    assert "max_memory_allocated" in source, "the probe must measure its own peak"
    assert "memory_headroom_fraction" in source
    # Exceeding the budget must halve exactly as an OOM does.
    assert source.count("size //= 2") >= 2


def test_the_manifest_records_what_memory_was_used(tmp_path):
    """So the next occurrence is measurable rather than guessable."""
    outcome = train_and_evaluate(
        TINY, corpus(80, 64, 1), corpus(8, 64, 2), tmp_path, config(),
        corpus="synthetic",
    )
    assert "peak_memory_bytes" in outcome.extra
    assert "device_memory_bytes" in outcome.extra
