"""Perplexity as a search objective: deterministic, and invariant to batching."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from nas4textgen.evaluation.perplexity import evaluate_likelihood  # noqa: E402
from nas4textgen.model import build_model  # noqa: E402
from nas4textgen.space.genotype import Genotype  # noqa: E402

pytestmark = pytest.mark.slow

GENOTYPE = Genotype(h=4, d_head=32, n=2, ffn=2, kv=2)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(GENOTYPE).eval()


@pytest.fixture(scope="module")
def sequences():
    rng = torch.Generator().manual_seed(7)
    return [torch.randint(0, 5000, (12,), generator=rng).tolist() for _ in range(12)]


def test_repeated_evaluation_is_bit_identical(model, sequences):
    """No sampling anywhere, so the objective is a property of the weights alone."""
    first = evaluate_likelihood(model, [sequences[:6], sequences[6:]])
    second = evaluate_likelihood(model, [sequences[:6], sequences[6:]])
    assert first.total_nll == second.total_nll
    assert first.perplexity == second.perplexity


def test_perplexity_is_invariant_to_batch_size(model, sequences):
    """The property that keeps the objective independent of available memory.

    Averaging per-batch losses and then averaging batches would weight a short batch
    equally with a full one, making the objective depend on how the data happened to be
    grouped — and therefore on the hardware that grouped it.
    """
    one_batch = evaluate_likelihood(model, [sequences])
    two_batches = evaluate_likelihood(model, [sequences[:6], sequences[6:]])
    uneven = evaluate_likelihood(model, [sequences[:1], sequences[1:5], sequences[5:]])

    assert one_batch.total_tokens == two_batches.total_tokens == uneven.total_tokens
    assert abs(one_batch.nll_per_token - two_batches.nll_per_token) < 1e-5
    assert abs(one_batch.nll_per_token - uneven.nll_per_token) < 1e-5


def test_padding_does_not_contribute(model):
    """A padded position has no target, so it must not enter the average."""
    short, long = [1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11]
    mixed = evaluate_likelihood(model, [[short, long]], pad_token_id=0)
    separate_short = evaluate_likelihood(model, [[short]], pad_token_id=0)
    separate_long = evaluate_likelihood(model, [[long]], pad_token_id=0)

    assert mixed.total_tokens == separate_short.total_tokens + separate_long.total_tokens
    assert abs(
        mixed.total_nll - (separate_short.total_nll + separate_long.total_nll)
    ) < 1e-3


def test_sequences_too_short_to_predict_are_counted_not_dropped(model):
    """A one-token sequence yields no prediction; the fact is recorded."""
    result = evaluate_likelihood(model, [[[42], [1, 2, 3, 4]]])
    assert result.skipped_sequences == 1
    assert result.total_tokens == 3


def test_bits_per_byte_is_infinite_without_byte_counts(model, sequences):
    """Reported as unavailable rather than silently wrong.

    Perplexity per token is not comparable across tokenizers, so bits-per-byte is the
    figure that would matter if the vocabulary entered the search space — and a wrong
    value there would be worse than no value.
    """
    result = evaluate_likelihood(model, [sequences])
    assert result.bits_per_byte == float("inf")

    with_bytes = evaluate_likelihood(model, [sequences], byte_lengths=[400])
    assert with_bytes.bits_per_byte < float("inf")


def test_the_definition_travels_with_the_result(model, sequences):
    """So a number in a table can be checked against how it was computed."""
    payload = evaluate_likelihood(model, [sequences]).as_dict()
    assert "batch-invariant" in payload["definition"]
    assert payload["total_tokens"] > 0
