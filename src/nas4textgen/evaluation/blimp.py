"""Grammatical judgement by forced choice over minimal pairs.

Each BLiMP item is two sentences differing in one place, one grammatical and one not. The
model never generates anything: it scores both and the item is correct when the grammatical
one gets the higher log-probability. That property is why this metric is in the battery at
all — every generation-quality measure needs the model to produce coherent text, and at 0.1
to 91 million parameters trained on 100 million tokens most of this space does not. A
forced choice between two given sentences is defined for every architecture in the space,
including the ones whose samples are noise.

It also comes with something the overlap metrics had to have measured for them: **a chance
floor known in advance.** Two options, one correct, so 0.5 is chance and no baseline run is
needed to establish it. What still has to be measured is whether a paradigm can be passed
without grammar at all, and for that there are two degenerate strategies below.

**Sum, not mean.** Log-probabilities are summed over the sentence rather than averaged over
tokens. A minimal pair differs by a substitution, so the two sentences are usually the same
length and the choice does not matter; where it does matter, averaging introduces a bias
that has nothing to do with grammar, because dividing by length rewards whichever sentence
tokenises longer. Length parity is recorded per paradigm rather than assumed, so a paradigm
where it fails is visible instead of quietly scored under a different rule.

**Every paradigm is reported.** BLiMP's headline number averages sixty-seven distinct
phenomena, and an aggregate can move while every individual paradigm sits at chance —
subject-verb agreement improving and island effects degrading cancel in the mean and are
opposite findings. The aggregate is reported because it is comparable with the literature;
the per-paradigm table is reported because it is the one that says anything.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

#: Chance accuracy. Known before any measurement, because the task is binary forced choice.
#: Stated as a constant so nothing has to derive it and no report can omit it.
CHANCE = 0.5

#: Below this many items a paradigm's accuracy is too noisy to read. BLiMP ships 1,000 per
#: paradigm; a subsampled run that falls under this is reported with the count and marked
#: rather than silently averaged in.
MINIMUM_ITEMS = 100


class NotAMinimalPair(ValueError):
    """Raised when an item's two sentences are identical or one is empty."""


@dataclass(frozen=True)
class Item:
    """One minimal pair: the grammatical sentence and its ungrammatical twin."""

    good: str
    bad: str
    paradigm: str

    def __post_init__(self) -> None:
        if not self.good.strip() or not self.bad.strip():
            raise NotAMinimalPair(f"{self.paradigm}: an empty sentence is not a judgement")
        if self.good == self.bad:
            raise NotAMinimalPair(
                f"{self.paradigm}: the two sentences are identical, so the item scores "
                "whichever way floating-point comparison happens to fall"
            )


@dataclass(frozen=True)
class ParadigmResult:
    """One phenomenon's accuracy, with what could have produced it without grammar."""

    paradigm: str
    items: int
    correct: int
    #: Items where the two sentences tokenise to the same length. Where this is below the
    #: item count, a length-based strategy has something to exploit.
    length_matched: int
    #: What a model that always prefers the shorter, or the longer, sentence would score.
    prefer_shorter: float
    prefer_longer: float
    #: Items the model scored exactly equally. Counted as incorrect, and reported, because
    #: a tie is not a judgement and a metric that broke them arbitrarily would hide it.
    ties: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.items if self.items else float("nan")

    @property
    def above_chance(self) -> float:
        return self.accuracy - CHANCE

    @property
    def length_confounded(self) -> bool:
        """Whether a degenerate length preference beats chance on this paradigm.

        When it does, the paradigm's accuracy is not evidence of grammatical knowledge on
        its own: part of it is available to a model that has learned only that one of the
        two sentences tends to be shorter.
        """
        return max(self.prefer_shorter, self.prefer_longer) > CHANCE + 0.05

    @property
    def underpowered(self) -> bool:
        return self.items < MINIMUM_ITEMS

    def as_dict(self) -> dict[str, object]:
        return {
            "paradigm": self.paradigm,
            "items": self.items,
            "accuracy": round(self.accuracy, 6),
            "above_chance": round(self.above_chance, 6),
            "ties": self.ties,
            "length_matched": self.length_matched,
            "floor_prefer_shorter": round(self.prefer_shorter, 6),
            "floor_prefer_longer": round(self.prefer_longer, 6),
            "length_confounded": self.length_confounded,
            "underpowered": self.underpowered,
        }


@dataclass
class BlimpReport:
    """Every paradigm, and the aggregate the literature quotes."""

    paradigms: list[ParadigmResult] = field(default_factory=list)
    scoring: str = "sum of token log-probabilities, no length normalisation"

    @property
    def items(self) -> int:
        return sum(p.items for p in self.paradigms)

    @property
    def macro_accuracy(self) -> float:
        """Mean over paradigms, each weighted equally.

        The macro average rather than the micro one, because BLiMP ships an equal number of
        items per paradigm by design and a subsampled run should not silently reweight the
        phenomena by how many items happened to survive.
        """
        scores = [p.accuracy for p in self.paradigms]
        return statistics.fmean(scores) if scores else float("nan")

    @property
    def paradigms_above_chance(self) -> int:
        return sum(p.accuracy > CHANCE for p in self.paradigms)

    @property
    def confounded_paradigms(self) -> list[str]:
        return [p.paradigm for p in self.paradigms if p.length_confounded]

    def as_dict(self) -> dict[str, object]:
        return {
            "macro_accuracy": round(self.macro_accuracy, 6),
            "chance": CHANCE,
            "paradigms": len(self.paradigms),
            "paradigms_above_chance": self.paradigms_above_chance,
            "items": self.items,
            "scoring": self.scoring,
            "length_confounded_paradigms": self.confounded_paradigms,
            "underpowered_paradigms": [p.paradigm for p in self.paradigms if p.underpowered],
            "per_paradigm": [p.as_dict() for p in self.paradigms],
            "why_per_paradigm": (
                "The aggregate averages sixty-seven distinct phenomena and can move while "
                "every one of them sits at chance. Subject-verb agreement improving and "
                "island effects degrading cancel in the mean and are opposite findings."
            ),
        }


def _sentence_logprob(model, token_ids: Sequence[int], device=None) -> float:
    """Total log-probability of a tokenised sentence, in nats.

    The first token contributes nothing — there is no context to predict it from — so a
    caller must prepend a beginning-of-sequence token if it wants that position scored.
    Both sentences of a pair are treated identically, so the omission cancels in the
    comparison.
    """
    import torch

    if len(token_ids) < 2:
        raise NotAMinimalPair(
            f"a sentence of {len(token_ids)} token(s) yields no prediction to score"
        )

    model.eval()
    ids = torch.tensor([list(token_ids)], dtype=torch.long)
    if device is not None:
        ids = ids.to(device)
    with torch.no_grad():
        logits = model(ids).logits.float()
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = ids[:, 1:]
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(gathered.sum().item())


def score_items(
    model,
    items: Sequence[Item],
    encode,
    device=None,
    bos_token_id: int | None = None,
) -> BlimpReport:
    """Score every item and group the outcome by paradigm.

    ``encode`` turns a sentence into token ids. ``bos_token_id`` is prepended when given, so
    the sentence's first real token is scored too; without it that token is free for both
    sides of the pair and the comparison is still fair, just slightly less informative.
    """
    by_paradigm: dict[str, list[tuple[bool, int, int, bool]]] = {}

    for item in items:
        good_ids = list(encode(item.good))
        bad_ids = list(encode(item.bad))
        if bos_token_id is not None:
            good_ids = [bos_token_id, *good_ids]
            bad_ids = [bos_token_id, *bad_ids]

        good_score = _sentence_logprob(model, good_ids, device)
        bad_score = _sentence_logprob(model, bad_ids, device)

        # Exact equality, deliberately: a tie here means the model assigned the two
        # sentences the same total, not that they are close. A tie is then counted as
        # incorrect rather than broken arbitrarily — splitting ties towards the
        # grammatical sentence would hand an untrained model exactly chance by
        # construction and hide that it has no preference at all.
        tie = good_score == bad_score
        correct = good_score > bad_score
        by_paradigm.setdefault(item.paradigm, []).append(
            (correct, len(good_ids), len(bad_ids), tie)
        )

    report = BlimpReport()
    for paradigm, outcomes in sorted(by_paradigm.items()):
        total = len(outcomes)
        good_shorter = sum(o[1] < o[2] for o in outcomes)
        good_longer = sum(o[1] > o[2] for o in outcomes)
        matched = sum(o[1] == o[2] for o in outcomes)
        report.paradigms.append(ParadigmResult(
            paradigm=paradigm,
            items=total,
            correct=sum(o[0] for o in outcomes),
            length_matched=matched,
            # A strategy that always picks the shorter sentence is correct on the items
            # whose grammatical sentence is shorter, and has no preference at all on the
            # length-matched ones — where it guesses, so it earns chance rather than zero.
            # Crediting it zero there would understate the floor and make a paradigm look
            # harder to game than it is. The two floors sum to one by construction.
            prefer_shorter=(good_shorter + CHANCE * matched) / total,
            prefer_longer=(good_longer + CHANCE * matched) / total,
            ties=sum(o[3] for o in outcomes),
        ))
    return report


def require_informative(report: BlimpReport) -> None:
    """Refuse a report whose paradigms a length preference could have produced.

    Not a failure of the model — a failure of the measurement to mean what it says. If most
    paradigms are length-confounded then the aggregate is partly a statistic about sentence
    length, and quoting it alongside a grammar claim is the error this project exists to
    correct, in a new metric.
    """
    confounded = report.confounded_paradigms
    if len(confounded) > len(report.paradigms) / 2:
        raise ValueError(
            f"{len(confounded)} of {len(report.paradigms)} paradigms are length-confounded: "
            "a model preferring the shorter sentence beats chance on them, so the aggregate "
            "is partly a statistic about sentence length rather than about grammar.\n"
            f"{', '.join(confounded[:8])}"
        )


#: Where the minimal pairs live once retrieved. BLiMP is not a training corpus and does not
#: belong in the corpus registry — it brings its own sentences and measures the model rather
#: than the data it was trained on — so it is cached beside them under its own name.
BLIMP_DATASET = "nyu-mll/blimp"
CACHE_NAME = "blimp"


class ItemsUnavailable(RuntimeError):
    """Raised when the minimal pairs have not been retrieved."""


def fetch_items(cache_root: Path, limit_per_paradigm: int | None = None) -> int:
    """Retrieve BLiMP into the cache. The only part of this metric that needs the network.

    Written because the battery was implemented and could not run: `score_items` takes
    `Item` objects and nothing produced any. A metric with no way to obtain its own data is
    a metric that never runs, which is what happened here for months.
    """
    from datasets import get_dataset_config_names, load_dataset

    target = cache_root / CACHE_NAME
    target.mkdir(parents=True, exist_ok=True)
    written = 0
    with (target / "items.jsonl").open("w", encoding="utf-8") as handle:
        for paradigm in get_dataset_config_names(BLIMP_DATASET):
            rows = load_dataset(BLIMP_DATASET, paradigm, split="train")
            for index, row in enumerate(rows):
                if limit_per_paradigm is not None and index >= limit_per_paradigm:
                    break
                handle.write(json.dumps({
                    "paradigm": paradigm,
                    "good": row["sentence_good"],
                    "bad": row["sentence_bad"],
                }, ensure_ascii=False) + "\n")
                written += 1
    return written


def load_items(cache_root: Path) -> list[Item]:
    """The cached minimal pairs, refusing rather than returning an empty list.

    An empty list would make `score_items` return a report of zero paradigms, and a
    macro-average over nothing is `nan` — which flows into a manifest looking like a
    measurement that came out undefined rather than one that never ran.
    """
    path = cache_root / CACHE_NAME / "items.jsonl"
    if not path.exists():
        raise ItemsUnavailable(
            f"no minimal pairs at {path}. Retrieve them once with: "
            "python -m nas4textgen.cli fetch --blimp"
        )
    items = []
    with path.open(encoding="utf-8", newline="\n") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                items.append(Item(good=row["good"], bad=row["bad"],
                                  paradigm=row["paradigm"]))
            except NotAMinimalPair:
                # A pair whose sentences are identical carries no judgement; counted by
                # its absence from the paradigm rather than scored as a coin flip.
                continue
    if not items:
        raise ItemsUnavailable(f"{path} is present but holds no usable pairs")
    return items
