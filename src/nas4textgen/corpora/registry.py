"""The four corpora, each with a stated role and the metrics it licenses.

A corpus is not neutral evidence. At the parameter scale under study, models trained on
Wikipedia do not produce coherent text, so generation-quality metrics computed there
measure noise and report it as a number. The prior work's central defect was a metric
that measured something other than what it claimed; the defence is not a better metric
but declaring, per corpus, which measurements are meaningful on it.

Each entry therefore carries a **role** and an explicit list of licensed metrics. A
metric asked for on a corpus that does not license it is refused rather than computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Metrics that need only a forward pass. Meaningful anywhere.
#:
#: ``blimp`` belongs here rather than to the corpus whose evaluation suite ships it. It
#: brings its own minimal pairs, so it measures the model rather than the corpus, and a
#: forced choice between two given sentences is defined even where the model generates
#: nothing worth reading. That is what makes it the second quality axis for the sweep: the
#: sweep trains on Wikipedia, where every generation metric is withheld, and perplexity was
#: otherwise the only signal available there.
LIKELIHOOD = ("perplexity", "bits_per_byte", "calibration", "blimp")

#: Metrics that require the model to generate text worth reading. Only meaningful where
#: models at this scale are coherent.
GENERATION = ("mauve", "distinct_n", "self_bleu", "repetition_rate")

#: Overlap metrics against a reference continuation. Carried for continuity with the
#: prior work, never as a search objective, and always beside their trivial floor.
OVERLAP = ("bleu", "rouge", "meteor", "gleu")


@dataclass(frozen=True)
class Corpus:
    key: str
    identifier: str
    config: str | None
    #: Column holding the text. Declared rather than guessed: a corpus whose column is
    #: inferred can silently yield the wrong field after an upstream schema change.
    text_column: str
    role: str
    licensed_metrics: tuple[str, ...]
    notes: str
    #: Rows that are structural markup rather than prose. WikiText-family corpora carry
    #: section headers whose reference continuation is a fragment of "= = =", and those
    #: rows dominated the prior work's reported score.
    has_markup_rows: bool = False
    #: Some corpora publish a single split. Deriving train and validation from it locally
    #: is legitimate, but the result must never be mistaken for an upstream split — the
    #: cache records the derivation so a reader knows which boundary is theirs.
    single_upstream_split: bool = False
    revision: str | None = None
    licence: str = "see the dataset card"
    strata: dict[str, float] = field(default_factory=dict)

    def licenses(self, metric: str) -> bool:
        return metric in self.licensed_metrics


WIKITEXT_103 = Corpus(
    key="wikitext103",
    identifier="Salesforce/wikitext",
    config="wikitext-103-raw-v1",
    text_column="text",
    role="search corpus",
    licensed_metrics=LIKELIHOOD + OVERLAP,
    has_markup_rows=True,
    notes=(
        "The corpus the sweep trains on. Continuity with the prior work, and large "
        "enough to remove the objection that WikiText-2 is too small. Generation "
        "metrics are NOT licensed here: at this parameter scale, models trained on "
        "Wikipedia do not generate coherent text, so a generation score would be "
        "noise presented as a measurement. Overlap metrics are licensed only for "
        "comparison against their trivial floors, never as an objective."
    ),
)

TINYSTORIES = Corpus(
    key="tinystories",
    identifier="roneneldan/TinyStories",
    config=None,
    text_column="text",
    role="makes generation measurable",
    licensed_metrics=LIKELIHOOD + GENERATION,
    notes=(
        "The only corpus here where models at this scale produce coherent text, which "
        "is what makes the generation battery informative rather than a measurement of "
        "noise. Synthetic and simple by construction — that is the property being used, "
        "not a limitation being tolerated."
    ),
)

BABYLM = Corpus(
    key="babylm",
    # The challenge's own organisation, strict track. An earlier version of this entry
    # pointed at a community mirror named `Sree1994/babylm_100M`, and the name was wrong:
    # it holds 13.6M tokens, which is the *strict-small* track of 10M words, not the 100M
    # one. Training on it and reporting "BabyLM 100M" would have been a false statement
    # about which corpus produced the numbers — caught by counting tokens rather than by
    # trusting a repository name, after the count showed seven epochs of repetition where
    # one was expected.
    #
    # The official repository resolves and needs no loading script; the note that it did
    # not was itself out of date.
    identifier="BabyLM-community/BabyLM-2026-Strict",
    config=None,
    text_column="text",
    role="standardised small-scale evaluation",
    licensed_metrics=LIKELIHOOD + ("ewok", "glue"),
    single_upstream_split=True,
    notes=(
        "Brings an evaluation suite designed for this scale, with public baselines to "
        "compare against. Using it removes the argument about which downstream tasks "
        "are appropriate: the suite is already agreed on by a community working at the "
        "same size. The strict track is roughly 136M tokens, which is what makes it "
        "usable at this project's 100M-token budget without repeating data — the "
        "strict-small track would need seven passes and would confound corpus with "
        "repetition."
    ),
)

STACK_PYTHON = Corpus(
    key="stack_python",
    # The Stack requires accepting an access agreement, which a replication package cannot
    # assume a reader has done. CodeParrot's cleaned Python is ungated and serves the same
    # purpose: the point is a parser, not a particular corpus.
    identifier="codeparrot/codeparrot-clean-valid",
    config=None,
    text_column="content",
    single_upstream_split=True,
    role="a quality metric that cannot be gamed",
    licensed_metrics=LIKELIHOOD + ("syntactic_validity", "parse_rate") + GENERATION,
    notes=(
        "Generated code either parses or it does not. After a metric failure of the "
        "kind this project exists to correct, at least one quality signal must be "
        "machine-verifiable and independent of overlap heuristics. A parser does not "
        "negotiate, cannot be inflated by verbosity, and has no trivial floor above "
        "zero."
    ),
)

CORPORA: dict[str, Corpus] = {
    c.key: c for c in (WIKITEXT_103, TINYSTORIES, BABYLM, STACK_PYTHON)
}

#: The corpus the sweep trains on. The others receive only the measured front.
SEARCH_CORPUS = WIKITEXT_103.key


class MetricNotLicensed(RuntimeError):
    """Raised when a metric is requested on a corpus that does not license it."""


def require_licensed(corpus_key: str, metric: str) -> None:
    """Refuse a metric the corpus does not license.

    This is deliberately an exception rather than a warning. A generation score
    computed on Wikipedia at this scale is not a weak result — it is a number with no
    referent, and the prior work shows how readily such a number becomes a headline.
    """
    corpus = CORPORA[corpus_key]
    if not corpus.licenses(metric):
        raise MetricNotLicensed(
            f"{metric!r} is not licensed on {corpus_key!r} (role: {corpus.role}).\n"
            f"Licensed here: {', '.join(corpus.licensed_metrics)}.\n"
            f"Why: {corpus.notes}"
        )
