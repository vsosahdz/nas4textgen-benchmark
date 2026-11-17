"""Perplexity: the single search objective.

Chosen because it is the only quality measure that survives the requirements a search
objective has to meet. No decoding, so no sampling and no decoding configuration to
forget. One forward pass, so it is cheap enough to compute for every architecture. No
generated text, so there is nothing to inflate — the failure mode that made the prior
objective unusable does not exist here. And comparable across the space because the
vocabulary is fixed, which is why fixing it mattered.

**Batch invariance is a correctness property, not an optimisation.** Averaging per-batch
losses and then averaging the batches weights short batches equally with full ones, so the
result depends on batch size — and a batch size chosen to fit available memory would make
the objective depend on the hardware it ran on. Token losses are summed and divided by the
token count once, which is the only definition that gives the same number regardless of
how the data was grouped.

Its known weakness is that perplexity is not generation quality. That is not hidden: the
correlation between this objective and the generation battery is measured across every
trained architecture, and either outcome is a result worth reporting.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class LikelihoodResult:
    """Everything needed to reproduce and interpret a likelihood measurement."""

    total_nll: float
    total_tokens: int
    total_bytes: int
    batches: int
    #: Number of sequences whose loss contributed nothing because they were too short to
    #: yield a single prediction. Recorded rather than dropped silently.
    skipped_sequences: int = 0

    @property
    def nll_per_token(self) -> float:
        """Mean negative log-likelihood in nats. Batch-invariant by construction."""
        return self.total_nll / self.total_tokens if self.total_tokens else float("inf")

    @property
    def perplexity(self) -> float:
        return math.exp(self.nll_per_token)

    @property
    def bits_per_byte(self) -> float:
        """Tokenizer-independent likelihood.

        Reported alongside perplexity so the measurement stays meaningful if the
        vocabulary ever enters the search space — perplexity per token is not comparable
        across tokenizers, since a smaller vocabulary splits the same text into more
        tokens and lowers the per-token figure without the model being better.
        """
        if not self.total_bytes:
            return float("inf")
        return self.total_nll / (self.total_bytes * math.log(2))

    def as_dict(self) -> dict[str, object]:
        return {
            "perplexity": round(self.perplexity, 6),
            "nll_per_token": round(self.nll_per_token, 8),
            "bits_per_byte": round(self.bits_per_byte, 8),
            "total_tokens": self.total_tokens,
            "total_bytes": self.total_bytes,
            "batches": self.batches,
            "skipped_sequences": self.skipped_sequences,
            "definition": (
                "sum of per-token negative log-likelihood divided by token count; "
                "batch-invariant"
            ),
        }


def evaluate_likelihood(
    model,
    batches: Iterable[Sequence[Sequence[int]]],
    byte_lengths: Iterable[int] | None = None,
    pad_token_id: int | None = None,
) -> LikelihoodResult:
    """Sum token-level negative log-likelihood over a held-out split.

    ``batches`` yields lists of token-id sequences. ``byte_lengths`` yields the UTF-8 byte
    count of each batch's raw text, needed for bits-per-byte; omit it and that figure is
    reported as infinite rather than silently wrong.

    The model is put in evaluation mode and gradients are disabled, so dropout is inactive
    and the result is a property of the weights alone. No sampling is involved anywhere,
    which is what makes repeated calls bit-identical.
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    batch_count = 0
    skipped = 0

    byte_iter = iter(byte_lengths) if byte_lengths is not None else None

    # Follow the model rather than defaulting to CPU. Building the batch on one device and
    # the weights on another either raises, or — with a model that was itself never moved —
    # runs correctly and sixty times too slowly, which is how a canary spent forty minutes
    # producing six per cent of its budget.
    device = next(model.parameters()).device

    with torch.no_grad():
        for sequences in batches:
            usable = [s for s in sequences if len(s) >= 2]
            skipped += len(sequences) - len(usable)
            if byte_iter is not None:
                total_bytes += next(byte_iter, 0)
            if not usable:
                continue

            width = max(len(s) for s in usable)
            pad = pad_token_id if pad_token_id is not None else 0
            ids = torch.full((len(usable), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros((len(usable), width), dtype=torch.bool, device=device)
            for row, seq in enumerate(usable):
                ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                mask[row, : len(seq)] = True

            logits = model(ids).logits

            # Predict position t+1 from position t, so the last position has no target and
            # the first has no prediction.
            shift_logits = logits[:, :-1, :]
            shift_targets = ids[:, 1:]
            shift_mask = mask[:, 1:] & mask[:, :-1]

            per_token = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_targets.reshape(-1),
                reduction="none",
            ).view(shift_targets.shape)

            total_nll += float((per_token * shift_mask).sum().item())
            total_tokens += int(shift_mask.sum().item())
            batch_count += 1

    return LikelihoodResult(
        total_nll=total_nll,
        total_tokens=total_tokens,
        total_bytes=total_bytes,
        batches=batch_count,
        skipped_sequences=skipped,
    )
