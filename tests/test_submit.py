"""Submission is where a mistake costs money rather than a test run.

Every assertion here is about a failure that would be paid for: a duplicated wave, a node
that never releases, a job pinned to a tag that moved, a corpus fetched from the network
instead of read from the measured cache. None of it needs a subscription — the job is
assembled locally and inspected before anything is sent.
"""

from __future__ import annotations

import pytest

from nas4textgen.azure.submit import (
    CACHE_MOUNT,
    OUTPUT_MOUNT,
    PROVISIONAL_TIMEOUT_SECONDS,
    TIMEOUT_FIXED_OVERHEAD_SECONDS,
    TIMEOUT_SAFETY_FACTOR,
    JobSpec,
    TimeoutNotMeasured,
    build_job,
    resolve_timeout,
)
from nas4textgen.azure.workspace import Platform
from nas4textgen.space.genotype import DISTILGPT2, Genotype

pytest.importorskip("azure.ai.ml", reason="the azure extra is not installed")

PLATFORM = Platform(
    workspace="ws", resource_group="rg", location="southcentralus",
    sweep_cluster="sweep", measure_cluster="measure",
    storage_account="acct", cache_container="cache", artifact_container="artifacts",
    registry="acr.azurecr.io", subscription_id="00000000-0000-0000-0000-000000000000",
)
IMAGE = "acr.azurecr.io/nas4textgen-train@sha256:" + "ab" * 32


def spec(**overrides) -> JobSpec:
    base = {"genotype": DISTILGPT2, "corpus": "wikitext103", "seed": 0,
            "token_budget": 50_000_000}
    return JobSpec(**{**base, **overrides})


def test_fingerprint_identifies_the_measurement_not_the_submission():
    """Two submissions of the same measurement collide; that is what prevents paying twice."""
    assert spec().fingerprint == spec().fingerprint
    assert spec().name() == spec().name()
    assert spec().name(2) != spec().name(1), 'attempts must be distinguishable'


@pytest.mark.parametrize(
    "overrides",
    [
        {"genotype": Genotype(h=12, d_head=64, n=6, ffn=4, kv=2)},
        {"corpus": "tinystories"},
        {"seed": 1},
        {"token_budget": 100_000_000},
        {"eval_split": "test"},
    ],
)
def test_anything_that_changes_the_number_changes_the_fingerprint(overrides):
    assert spec(**overrides).fingerprint != spec().fingerprint


def test_tags_do_not_change_the_fingerprint():
    """A label is not a measurement. Re-tagging a wave must not re-run it."""
    assert spec(extra_tags={"wave": "3"}).fingerprint == spec().fingerprint


def test_timeout_is_refused_until_the_canary_measures_it():
    """A guessed limit either protects nothing or kills healthy runs; both cost money."""
    with pytest.raises(TimeoutNotMeasured):
        resolve_timeout(None)
    assert resolve_timeout(None, allow_provisional=True) == PROVISIONAL_TIMEOUT_SECONDS

    # A multiplier and a fixed addition, because the costs a limit absorbs are of two
    # kinds. A flat 3x on a three-and-a-half-hour job is ten hours, which protects
    # nothing; the fixed term is what covers pulling a six-gigabyte image.
    assert resolve_timeout(600) == int(600 * TIMEOUT_SAFETY_FACTOR
                                       + TIMEOUT_FIXED_OVERHEAD_SECONDS)
    short, long = resolve_timeout(600), resolve_timeout(12_000)
    assert short / 600 > long / 12_000, "head-room is proportionally wider for short jobs"


def test_every_job_carries_a_wall_clock_limit():
    """The property that stops a hung spot node billing overnight."""
    job = build_job(spec(), PLATFORM, IMAGE, "sweep", 1800, "2026-08-13")
    assert job.limits is not None and job.limits.timeout == 1800


def test_job_runs_the_image_by_digest_never_a_tag():
    job = build_job(spec(), PLATFORM, IMAGE, "sweep", 1800, "2026-08-13")
    assert "@sha256:" in job.environment.image
    assert ":latest" not in job.environment.image


def test_corpus_is_mounted_read_only_from_the_cache_container():
    """Read-only and mounted: a job cannot alter the corpus, and does not copy it per node."""
    job = build_job(spec(), PLATFORM, IMAGE, "sweep", 1800, "2026-08-13")
    cache = job.inputs["cache"]
    assert cache.mode == "ro_mount"
    # A datastore path, never an https:// URL. Azure ML reads http(s) as public storage and
    # rejects it as an output, which is how the first canary submission failed.
    assert str(cache.path).startswith("azureml://datastores/")
    assert str(cache.path).endswith("/paths/corpora")
    # The command asks Azure ML where it mounted the input rather than naming a path. A
    # fixed path is a different directory that happens to exist, and the job stopped with
    # "corpus unavailable" when it was one.
    assert f"--cache {CACHE_MOUNT}" in job.command
    assert "NAS4TG_CACHE" not in job.environment_variables


def test_job_cannot_reach_the_network_for_data():
    """Offline flags are set, so a cache miss fails loudly instead of silently downloading
    a corpus that may have moved since the rest of the sweep was measured."""
    job = build_job(spec(), PLATFORM, IMAGE, "sweep", 1800, "2026-08-13")
    assert job.environment_variables["HF_HUB_OFFLINE"] == "1"
    assert job.environment_variables["TRANSFORMERS_OFFLINE"] == "1"


def test_outputs_land_in_the_artifact_container_under_the_fingerprint():
    job = build_job(spec(), PLATFORM, IMAGE, "sweep", 1800, "2026-08-13")
    out = job.outputs["outputs"]
    assert str(out.path).startswith("azureml://datastores/")
    assert str(out.path).endswith(f"/paths/runs/{spec().fingerprint}")
    assert not str(out.path).startswith("http")
    assert out.mode == "rw_mount"
    assert OUTPUT_MOUNT in job.command


def test_command_passes_the_genotype_the_entry_point_requires():
    """The submitted command must parse against the image's own argument parser."""
    from nas4textgen.train import build_parser

    job = build_job(spec(), PLATFORM, IMAGE, "sweep", 1800, "2026-08-13")
    arguments = job.command.replace("python -m nas4textgen.train ", "").split()
    parsed = build_parser().parse_args(arguments)
    assert str(parsed.cache) == CACHE_MOUNT

    assert Genotype(h=parsed.h, d_head=parsed.d_head, n=parsed.n,
                    ffn=parsed.ffn, kv=parsed.kv) == DISTILGPT2
    assert parsed.corpus == "wikitext103"
    assert str(parsed.output) == OUTPUT_MOUNT


def test_tags_record_what_produced_the_number():
    """The workspace answers provenance on its own, without a spreadsheet kept alongside."""
    job = build_job(spec(), PLATFORM, IMAGE, "sweep", 1800, "2026-08-13")
    assert job.tags["image"] == IMAGE
    assert job.tags["data_vintage"] == "2026-08-13"
    assert job.tags["fingerprint"] == spec().fingerprint
    assert job.tags["h"] == "12" and job.tags["kv"] == "12"


def test_determinism_settings_survive_into_the_job():
    job = build_job(spec(), PLATFORM, IMAGE, "sweep", 1800, "2026-08-13")
    assert job.environment_variables["PYTHONHASHSEED"] == "0"
    assert job.environment_variables["OMP_NUM_THREADS"] == "1"


def test_canary_bounds_come_from_the_architecture_that_binds_them():
    """The correction this module exists for.

    Parameter count in this space is mostly the 50,257-row embedding, so ranking by it
    returns the widest model, not the most expensive one. A canary that set the job timeout
    from the parameter extreme would measure a run at a fraction of the real worst case,
    pass, and hand the sweep a limit too tight for the architectures it had not seen.
    """
    from nas4textgen.azure.canary import select_canary, timeout_source
    from nas4textgen.space.cost import activation_width, forward_flops_per_token

    members = select_canary()
    assert {m.role for m in members} == {"smallest", "median", "largest_params", "heaviest"}

    source = timeout_source(members)
    assert source.role == "heaviest"

    largest = next(m for m in members if m.role == "largest_params")
    assert largest.params > source.params, "the parameter extreme should still be the widest"
    assert source.flops_per_token > largest.flops_per_token, (
        "the timeout source must be the more expensive architecture to run"
    )
    assert source.activation_width > largest.activation_width

    from nas4textgen.space.enumerate import enumerate_with_ceiling

    space = enumerate_with_ceiling().genotypes
    assert source.flops_per_token == max(forward_flops_per_token(g) for g in space)
    assert source.activation_width == max(
        activation_width(g) for g in space
        if forward_flops_per_token(g) == source.flops_per_token
    )


def test_canary_members_are_distinct_measurements():
    """Four members, four fingerprints — a canary that re-ran one architecture would
    report three passes and cover three cases."""
    from nas4textgen.azure.canary import select_canary

    specs = [spec(genotype=m.genotype) for m in select_canary()]
    assert len({s.fingerprint for s in specs}) == len(specs)


def test_sweep_inherits_the_canary_timeout_instead_of_a_retyped_number(tmp_path, monkeypatch):
    """A wave submitted next week must carry the limit measured today.

    Passing it by hand would work until someone retypes it, and a timeout that is too tight
    kills healthy runs at the large end of the space — where jobs are most expensive, so the
    error costs most exactly where it is least affordable.
    """
    monkeypatch.setenv("NAS4TG_ARTIFACTS", str(tmp_path))
    from nas4textgen.azure.canary import select_canary, timeout_source
    from nas4textgen.azure.submit import (
        measured_timeout,
        record_measured_timeout,
        resolve_timeout,
    )

    assert measured_timeout() is None
    with pytest.raises(TimeoutNotMeasured):
        resolve_timeout(measured_timeout())

    source = timeout_source(select_canary())
    record_measured_timeout(742.0, source.role, source.genotype, "nas-abc123")

    assert measured_timeout() == 742.0
    expected = int(742.0 * TIMEOUT_SAFETY_FACTOR + TIMEOUT_FIXED_OVERHEAD_SECONDS)
    assert resolve_timeout(measured_timeout()) == expected

    job = build_job(spec(), PLATFORM, IMAGE, "sweep",
                    resolve_timeout(measured_timeout()), "2026-08-13")
    assert job.limits.timeout == expected


def test_process_deadlines_sit_inside_the_scheduler_limit():
    """The ordering that makes the graceful stop reachable at all.

    TrainingConfig defaulted the soft stop to 7200s and the hard stop to 8100s while the job
    limit was 5400s, so Azure ML would have killed every long run before either fired: the
    stop-training-but-still-evaluate-and-write path was unreachable, and a run that hit the
    wall lost everything it had computed. The deadlines are derived from the limit now, so
    they cannot be ordered wrongly.
    """
    from nas4textgen.azure.submit import process_deadlines

    for limit in (600, 1800, 5400, 12_000):
        soft, hard = process_deadlines(limit)
        assert soft < hard < limit, f"deadlines escape the {limit}s limit"
        assert limit - hard >= 0.1 * limit, "no room left for the scheduler to act last"

    job = build_job(spec(), PLATFORM, IMAGE, "sweep", 1800, "2026-08-13")
    from nas4textgen.train import build_parser

    parsed = build_parser().parse_args(
        job.command.replace("python -m nas4textgen.train ", "").split()
    )
    assert parsed.soft_deadline_seconds < parsed.hard_deadline_seconds < job.limits.timeout


def test_mount_paths_are_bindings_rather_than_fixed_directories():
    """The failure a fixed output path produces is silence, which is why this is asserted.

    A wrong input path stops the job in its first second with "corpus unavailable" — loud
    and cheap. A wrong output path does not fail at all: local disk is writable, so the run
    trains, evaluates, writes its manifest, exits zero, and the results vanish with the spot
    node. Nothing distinguishes that from a successful run except the missing artifact.
    """
    job = build_job(spec(), PLATFORM, IMAGE, "sweep", 1800, "2026-08-13")

    assert "${{inputs.cache}}" in job.command
    assert "${{outputs.outputs}}" in job.command
    for fixed in ("/mnt/cache", "/mnt/outputs", "/tmp/outputs"):
        assert fixed not in job.command, f"{fixed} is a path the job was never given"


class TestRetryBudget:
    """Retries are counted per image, because a fixed harness is a new experiment.

    The canary reached four failed attempts on four different images — a corpus path, a
    mount binding, a dtype, a memory probe — and each fix made the next attempt genuinely
    new. A budget counted per fingerprint would have refused to run the version that works,
    which is the opposite of what the guard is for.
    """

    class FakeJob:
        def __init__(self, status, image):
            self.status = status
            self.tags = {"image": image}

    class FakeJobs:
        def __init__(self, existing):
            self.existing = existing

        def get(self, name):
            if name not in self.existing:
                raise KeyError(name)
            return self.existing[name]

    class FakeClient:
        def __init__(self, existing):
            self.jobs = TestRetryBudget.FakeJobs(existing)

    def test_failures_on_older_images_do_not_block_a_fix(self):
        from nas4textgen.azure.submit import _next_attempt

        s = spec()
        old = "acr/img@sha256:" + "00" * 32
        existing = {
            s.name(n): self.FakeJob("Failed", old) for n in range(1, 5)
        }
        attempt, reason = _next_attempt(self.FakeClient(existing), s, image=IMAGE)

        assert attempt == 5, reason
        assert reason is None

    def test_repeated_failure_on_the_same_image_is_refused(self):
        from nas4textgen.azure.submit import _next_attempt

        s = spec()
        existing = {s.name(n): self.FakeJob("Failed", IMAGE) for n in range(1, 5)}
        attempt, reason = _next_attempt(self.FakeClient(existing), s, image=IMAGE)

        assert attempt is None
        assert "on this exact image" in reason

    def test_a_completed_attempt_stops_everything_regardless_of_image(self):
        """A measurement that exists must never be paid for twice, even after a rebuild:
        the fingerprint says the run measures the same thing."""
        from nas4textgen.azure.submit import _next_attempt

        s = spec()
        old = "acr/img@sha256:" + "11" * 32
        existing = {s.name(1): self.FakeJob("Completed", old)}
        attempt, reason = _next_attempt(self.FakeClient(existing), s, image=IMAGE)

        assert attempt is None
        assert "Completed" in reason

    def test_a_total_ceiling_still_bounds_a_resubmit_loop(self):
        from nas4textgen.azure.submit import MAX_ATTEMPTS_TOTAL, _next_attempt

        s = spec()
        existing = {
            s.name(n): self.FakeJob("Failed", f"acr/img@sha256:{n:064d}")
            for n in range(1, MAX_ATTEMPTS_TOTAL + 1)
        }
        attempt, reason = _next_attempt(self.FakeClient(existing), s, image=IMAGE)

        assert attempt is None
        assert "in total" in reason


def test_force_names_a_member_rather_than_resubmitting_everything():
    """A boolean force flag resubmitted all four canary members when one needed it,
    spending six node-hours re-measuring rows that were already valid.

    Naming the member is the difference between a correction and a re-run.
    """
    import inspect

    from nas4textgen.azure.submit import submit_wave

    parameters = inspect.signature(submit_wave).parameters
    assert "force_roles" in parameters
    assert "force" not in parameters, "the blanket flag must not come back"
    assert parameters["force_roles"].default == ()


_ABSENT = object()


class TestImageFreshness:
    """The lock hash does not cover the source, and neither question is the repo revision.

    Wave one of the sweep went out on an image built before a fix to the memory probe, and
    the pre-flight passed: it verified a digest was recorded, which is a different question
    from whether the digest contains the current code.

    The first fix compared the repository revision and refused wave two — for commits that
    had touched only tests and task files. A guard that fires on changes it does not care
    about is a guard that gets bypassed, so what is compared now is a hash of the paths that
    actually go into the image.
    """

    def record(self, tmp_path, **overrides):
        import json

        from nas4textgen.azure.workspace import image_input_hash

        payload = {
            "registry": "acr.azurecr.io", "repository": "nas4textgen-train",
            "digest": "sha256:" + "ab" * 32, "input_hash": image_input_hash(),
        }
        payload.update(overrides)
        payload = {k: v for k, v in payload.items() if v is not _ABSENT}
        path = tmp_path / "published.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_an_image_matching_this_checkout_is_accepted(self, tmp_path):
        from nas4textgen.azure.workspace import require_current_image

        assert require_current_image(self.record(tmp_path))

    def test_an_image_built_from_different_inputs_is_refused(self, tmp_path):
        from nas4textgen.azure.workspace import ImageIsStale, require_current_image

        stale = self.record(tmp_path, input_hash="0" * 16)
        with pytest.raises(ImageIsStale, match="not the code here"):
            require_current_image(stale)

    def test_an_image_that_records_no_hash_is_refused(self, tmp_path):
        """That silence is what let wave one through."""
        from nas4textgen.azure.workspace import ImageIsStale, require_current_image

        with pytest.raises(ImageIsStale, match="records no hash"):
            require_current_image(self.record(tmp_path, input_hash=_ABSENT))

    def test_only_what_a_job_runs_counts(self, tmp_path):
        """Editing a training module must invalidate the image; editing an analysis module
        must not.

        Two earlier versions were both too broad. Comparing HEAD refused a sweep wave for
        commits touching only tests; hashing all of src/ refused two more for edits to the
        surrogate slate and the characterisation, which no training job imports and which
        are in the image only because the package is copied whole.
        """
        from nas4textgen.azure.workspace import image_input_hash

        root = tmp_path
        (root / "src" / "nas4textgen" / "space").mkdir(parents=True)
        (root / "src" / "nas4textgen" / "surrogate").mkdir(parents=True)
        (root / "src" / "nas4textgen" / "space" / "genotype.py").write_text(
            "LEVELS = {}\n", encoding="utf-8"
        )
        (root / "src" / "nas4textgen" / "surrogate" / "slate.py").write_text(
            "SLATE = []\n", encoding="utf-8"
        )
        before = image_input_hash(root)

        (root / "src" / "nas4textgen" / "surrogate" / "slate.py").write_text(
            "SLATE = [1, 2, 3]\n", encoding="utf-8"
        )
        assert image_input_hash(root) == before, "analysis code is not in the image's inputs"

        (root / "src" / "nas4textgen" / "space" / "genotype.py").write_text(
            "LEVELS = {'h': (2,)}\n", encoding="utf-8"
        )
        assert image_input_hash(root) != before, "training code is"

    @pytest.mark.slow
    def test_the_declared_module_list_matches_what_a_run_actually_loads(self, tmp_path):
        """Measured rather than inferred, and re-measured here so it cannot drift.

        Static imports are not enough: training.py imports nas4textgen.model inside a
        function, so the module defining the architecture never appears in train.py's
        import closure at load time. A run is the only way to see it.
        """
        import json
        import subprocess
        import sys

        from nas4textgen.azure.workspace import TRAINING_MODULES

        script = f"""
import json, sys, runpy
sys.argv = ["nas4textgen.train", "--h","2","--d-head","32","--n","2","--ffn","2","--kv","1",
            "--corpus","wikitext103","--token-budget","20000","--allow-cpu",
            "--cache",{str(tmp_path / "cache")!r},"--output",{str(tmp_path / "out")!r}]
try:
    runpy.run_module("nas4textgen.train", run_name="__main__")
except BaseException:
    pass
print("CLOSURE=" + json.dumps(sorted(
    m for m in sys.modules if m.startswith("nas4textgen"))))
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        line = next(
            (x for x in result.stdout.splitlines() if x.startswith("CLOSURE=")), None
        )
        if line is None:
            pytest.skip("the training entry point could not run here")

        loaded = set(json.loads(line.removeprefix("CLOSURE=")))
        declared = set(TRAINING_MODULES)
        missing = loaded - declared
        assert not missing, (
            f"a training run loads {sorted(missing)}, which the image-input hash does not "
            "cover — a change to any of them would not invalidate the image"
        )

    def test_bytecode_caches_do_not_change_the_hash(self, tmp_path):
        """Otherwise the hash moves on its own between two identical checkouts."""
        from nas4textgen.azure.workspace import image_input_hash

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "m.py").write_text("VALUE = 1\n", encoding="utf-8")
        before = image_input_hash(tmp_path)

        cache = tmp_path / "src" / "__pycache__"
        cache.mkdir()
        (cache / "m.cpython-312.pyc").write_bytes(b"\x00compiled")
        assert image_input_hash(tmp_path) == before

    def test_submission_checks_before_queueing_anything(self):
        import inspect

        from nas4textgen.azure.submit import submit_wave

        source = inspect.getsource(submit_wave)
        assert "require_current_image()" in source
        assert source.index("require_current_image()") < source.index("client.jobs")

    def test_the_cli_reports_the_refusal_rather_than_raising(self):
        """Wave two's submission was refused correctly and the traceback went into a pipe,
        so the wave simply did not exist and nothing said why."""
        import inspect

        from nas4textgen import cli

        source = inspect.getsource(cli.main)
        assert "ImageIsStale" in source
        assert "refused" in source


class TestTheBatteryIsPartOfTheMeasurement:
    """A battery run produces a different set of measurements, so it is a different run.

    The fingerprint decides whether a job is submitted or skipped as already done. If a
    battery run fingerprinted the same as the census run of that architecture, every one of
    them would be skipped and the wave would submit nothing, silently.
    """

    def _spec(self, **kwargs):
        from nas4textgen.azure.submit import JobSpec
        from nas4textgen.space.genotype import Genotype

        return JobSpec(
            genotype=Genotype(h=8, d_head=32, n=6, ffn=4, kv=1),
            corpus="wikitext103", seed=0, token_budget=100_000_000, **kwargs)

    def test_a_battery_run_is_not_the_same_measurement_as_the_run_without_it(self):
        assert self._spec().fingerprint != self._spec(run_battery=True).fingerprint

    def test_the_fingerprint_without_a_battery_is_unchanged_by_the_field_existing(self):
        """The guard on 3,123 completed runs.

        Appending the term unconditionally would change every fingerprint minted before the
        battery existed, and the campaign would offer to measure the whole census again.
        The expected value is pinned so a later edit to the fingerprint cannot drift it.
        """
        assert self._spec().fingerprint == "3e5931877d1f547c"

    def test_the_flag_reaches_the_job_command(self):
        """A fingerprint that differs while the command does not would record a battery run
        that never computed one."""
        assert "--run-battery" in self._spec(run_battery=True).command_arguments(3600)
        assert "--run-battery" not in self._spec().command_arguments(3600)

    def test_the_entry_point_accepts_what_the_job_passes(self):
        """The two halves are edited in different files and nothing else connects them."""
        from nas4textgen.train import build_parser

        args = build_parser().parse_args(
            ["--h", "8", "--d-head", "32", "--n", "6", "--ffn", "4", "--kv", "1",
             "--corpus", "wikitext103", "--output", "/tmp/out", "--run-battery"])
        assert args.run_battery is True
