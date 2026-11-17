"""Every objection the prior submission drew, and where this work answers it.

A resubmission that says "we addressed the reviewers' comments" without saying where is
asking the same reviewers to find the answers themselves. So each objection is recorded with
the artifact, test or design decision that answers it, and a test asserts that every entry
has evidence which actually exists — a citation to a file that has been renamed is worse
than no citation, because it looks like diligence.

The reviews are quoted rather than paraphrased. The text they are quoted from is kept
locally rather than committed — referee reports are third-party writing addressed to an
editor, and this repository holds the code and the experiments — so the quote check runs for
whoever has them and reports its own absence for everyone else.
An earlier version of this file was a reconstruction assembled from the passing references
the design and the code happened to make, because the review text had arrived in
conversation and was never written down. That is the failure mode the quotes now prevent:
answering a question the reviewer did not ask reads, to that reviewer, as not having read
them.

**Three dispositions, and the boundary between them is what makes the count mean anything.**

``addressed``
    The repository contains what answers it, and the answer does not depend on measurements
    still running.

``pending``
    The mechanism is in place and tested; the number it produces is not final. Most of these
    wait on the sweep finishing or on the manuscript being written. Counting them is the
    honest progress signal — ``require_complete`` refuses to certify the set while any
    remain.

**The work is no longer going back to IEEE Access.** That removes the procedural reason to
answer these and leaves the substantive one, which was always the stronger of the two: three
referees read the prior submission carefully and most of what they found was right. A new
venue will not have seen their comments, so nothing here is owed to anyone — which is
exactly why the registry stays. An objection answered because a reviewer will check is
answered until they stop checking; an objection answered because it was correct stays
answered.

Two entries are affected by the move rather than by any work. R3 #13 and R3 #14 concern the
prior manuscript's bibliography and typography, and a manuscript written for a different
venue inherits neither its reference list nor its template remnants. They remain listed
because the underlying discipline — verify every reference against its source, proofread —
transfers even when the specific defects do not.

``declined``
    Argued not to apply, with the condition that would bring it back. "It does not apply" is
    an assertion; "it applies again the moment Lex-MODES returns to the slate" is an
    argument a reviewer can check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Where the reviews live, verbatim. Every quote below is checked against this file.
REVIEW_TEXT = "docs/reviews/ieee-access-prior-submission.md"

#: Cited by objections whose answer is a section of a manuscript that is not written yet.
TASKS = "openspec/changes/mo-nas-textgen-benchmark/tasks.md"

DISPOSITIONS = ("addressed", "pending", "declined")


class EvidenceMissing(RuntimeError):
    """Raised when an objection cites evidence that is not there."""


class ReviewsUnavailable(RuntimeError):
    """Raised when the review text is not in this checkout.

    The reviews are referee reports on an earlier submission: third-party text written for
    an editor, used here to drive the rebuild and kept out of the repository, which holds
    the code and the experiments. Verification is therefore possible for whoever has them
    and impossible for everyone else, and saying which of those two situations applies is
    better than a failure that reads like a defect in the registry.
    """


class QuoteNotFound(RuntimeError):
    """Raised when an objection quotes words the reviews do not contain."""


class RegistryIncomplete(RuntimeError):
    """Raised when the registry is certified while objections are still open."""


@dataclass(frozen=True)
class Objection:
    """One reviewer comment and what answers it."""

    reviewer: str
    number: str
    #: The reviewer's own words, or a contiguous excerpt of them. Checked against the
    #: committed review text, so a summary cannot drift away from what was asked.
    quote: str
    summary: str
    disposition: str
    response: str
    #: Files that carry the answer. Checked to exist, because a citation to a moved file
    #: reads as diligence and is not.
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.disposition not in DISPOSITIONS:
            raise ValueError(
                f"{self.reviewer} #{self.number}: disposition must be one of "
                f"{DISPOSITIONS}. A fourth option would let an objection be neither "
                "answered, nor argued with, nor counted as outstanding."
            )
        if self.disposition == "declined" and "would" not in self.response:
            raise ValueError(
                f"{self.reviewer} #{self.number}: a declined objection has to state the "
                "condition under which it would apply. 'It does not apply' is an assertion; "
                "'it applies again the moment Lex-MODES returns' is an argument."
            )
        if not self.evidence:
            raise ValueError(
                f"{self.reviewer} #{self.number}: no evidence cited. Saying an objection is "
                "answered without saying where asks the reviewer to find the answer."
            )
        if len(self.quote) < 25:
            raise ValueError(
                f"{self.reviewer} #{self.number}: the quote is too short to identify what "
                "was asked, and a registry keyed to a fragment can be pointed at the wrong "
                "objection."
            )

    @property
    def is_open(self) -> bool:
        return self.disposition == "pending"

    def as_dict(self) -> dict[str, object]:
        return {
            "reviewer": self.reviewer,
            "number": self.number,
            "quote": self.quote,
            "summary": self.summary,
            "disposition": self.disposition,
            "response": self.response,
            "evidence": list(self.evidence),
        }


#: Shared citations, named so a rename shows up in one place.
_SEARCH = ("src/nas4textgen/search/run.py", "src/nas4textgen/search/local.py",
           "src/nas4textgen/search/scoring.py")
_SURROGATE = ("src/nas4textgen/surrogate/slate.py", "src/nas4textgen/surrogate/evaluate.py",
              "src/nas4textgen/surrogate/budget.py")
_STATISTICS = ("src/nas4textgen/statistics/frequentist.py",
               "src/nas4textgen/statistics/bayesian.py",
               "src/nas4textgen/statistics/rope.py",
               "src/nas4textgen/statistics/comparison.py")


RECORDED: tuple[Objection, ...] = (
    # ── Reviewer 1 ────────────────────────────────────────────────────────────────────
    Objection(
        reviewer="R1",
        number="1",
        quote=(
            "the paper would be much stronger if it included more rigorous comparisons with "
            "existing NAS methods (e.g., GPT-NAS, TextNAS, random search, or standard "
            "evolutionary NAS) to demonstrate the advantages of the proposed optimization "
            "strategy"
        ),
        summary="Compare against existing NAS methods and add an ablation study.",
        disposition="addressed",
        response=(
            "The comparison is rebuilt on a different footing. Rather than quoting published "
            "numbers from other search spaces — which compares spaces, not searches — every "
            "algorithm runs on the same enumerated table of 1,099 measured architectures "
            "whose Pareto front is known exactly, so the gap to the true front is a "
            "measurement rather than an estimate. The slate is seven entries, each with a "
            "recorded role: random search as the declared floor, SMS-EMOA as the reference, "
            "NSGA-II as the comparator the literature uses, AGE-MOEA-II for front geometry, "
            "Pareto local search as the method a discrete grid actually admits, ParEGO as "
            "the surrogate-assisted entry, and MOEA/D excluded with the reason recorded "
            "rather than left as a silent omission. GPT-NAS and TextNAS are not in the slate "
            "and cannot be: they search different spaces, so running them here would mean "
            "reimplementing them against this genotype and calling the result theirs. "
            "Ablations are the surrogate budget curve, the representation study, and the "
            "trivial floors every predictor must outrank.\n\n"
            "Run on the closed census over 20 seeds: the omnibus rejects at every budget "
            "(p from 1.6e-3 to 1.2e-5) and pairwise comparisons survive Holm — two of "
            "fifteen at 25 architectures rising to seven of ten at 200."
        ),
        evidence=(*_SEARCH, "src/nas4textgen/search/problem.py",
                  "src/nas4textgen/search/compare.py",
                  "src/nas4textgen/surrogate/budget.py"),
    ),

    # ── Reviewer 2 ────────────────────────────────────────────────────────────────────
    Objection(
        reviewer="R2",
        number="1",
        quote=(
            "Redraw Figure 1 as a framework diagram, not a model structure diagram. "
            "Additionally, add an intuitive diagram explaining the rationale behind the "
            "heuristic rules of the Lex-MODES algorithm."
        ),
        summary="Redraw Figure 1 as a framework diagram; add a Lex-MODES rationale diagram.",
        disposition="pending",
        response=(
            "The second half is moot: Lex-MODES is not in this work. It was dropped as a "
            "first-generation lexicographic ordering in favour of SMS-EMOA, whose "
            "steady-state hypervolume survival makes population hypervolume non-decreasing "
            "by construction — which is what turns a convergence curve into an instrument. "
            "The framework diagram belongs to the manuscript and is written against the "
            "generated tables rather than drawn by hand."
        ),
        evidence=("src/nas4textgen/search/run.py", "src/nas4textgen/manuscript.py", TASKS),
    ),
    Objection(
        reviewer="R2",
        number="2",
        quote=(
            "a comparison is made between the proposed Lex-MODES model and the baseline "
            "SMS-EMOA model. Given that they are presented together, it is advisable to "
            "conduct a detailed comparative analysis. Additionally, a comparison table "
            "should be included"
        ),
        summary="Give Lex-MODES against SMS-EMOA a detailed analysis and a comparison table.",
        disposition="pending",
        response=(
            "The pairing the objection is about no longer exists — SMS-EMOA is now the "
            "reference algorithm rather than a baseline, and Lex-MODES is gone. What "
            "replaces it is a seven-entry slate compared on IGD+ and on hypervolume reported "
            "both normalised and raw, with the population snapshotted at every evaluation "
            "rather than only the archive, because the archive's hypervolume rises "
            "monotonically for every algorithm including random search and therefore says "
            "nothing about the search. The comparison table is generated from the artifacts."
        ),
        evidence=(*_SEARCH, "src/nas4textgen/statistics/comparison.py",
                  "src/nas4textgen/manuscript.py"),
    ),
    Objection(
        reviewer="R2",
        number="3",
        quote=(
            "This paper compares four types of proxy models (SVR/RFR/XGB/MLP) and recommends "
            "incorporating more complex architectures, such as Transformers. Additionally, "
            "when selecting a model, the relationships between inputs and between inputs and "
            "outputs should be considered."
        ),
        summary="Add more complex surrogate architectures; consider input relationships.",
        disposition="addressed",
        response=(
            "Four candidates became fourteen, organised in tiers with a stated purpose each: "
            "trivial floors that need no training, linear and kernel models, gradient "
            "boosting, neural, and an uncertainty tier — Gaussian process, quantile "
            "regression forest, NGBoost, Bayesian ridge — because a point prediction cannot "
            "answer a question about calibration. A sequence or graph model over the "
            "architecture is deliberately absent and the reason is recorded in the slate: "
            "the genotype is a fixed-length vector of five genes over a stack of identical "
            "blocks, so a graph network would receive the same graph 1,099 times with only "
            "node features differing. It would belong the moment the space admits varying "
            "topology, which is what the follow-up change proposes. The relationships the "
            "objection asks about are measured directly instead of assumed: first-order "
            "variance indices per gene, and the shortfall from one as where interactions "
            "live."
        ),
        evidence=(*_SURROGATE, "src/nas4textgen/characterise.py"),
    ),
    Objection(
        reviewer="R2",
        number="4",
        quote=(
            "The experimental evaluation is limited to WikiText-2, which makes it difficult "
            "to assess whether the proposed NAS framework generalizes to other "
            "text-generation datasets, domains, or generation tasks."
        ),
        summary="One corpus cannot support a claim about generalisation.",
        disposition="addressed",
        response=(
            "Four corpora, each admitted for a stated reason and each licensing a different "
            "part of the battery. The sweep trains on WikiText-103 rather than WikiText-2 — "
            "large enough that the size objection goes away — and generation metrics are "
            "explicitly *not* licensed there, because models at this parameter scale trained "
            "on Wikipedia do not produce coherent text and a generation score would be noise "
            "presented as a measurement. A synthetic corpus carries the generation battery, "
            "a small-scale suite with public baselines carries the downstream comparison, "
            "and a code corpus carries a quality signal a parser can verify. All 220 "
            "architectures of the measured front were retrained on the other three at three "
            "seeds each, 1,980 runs, and the result is reported with its limit stated: the "
            "ordering transfers nearly intact, at a rank correlation of 0.929 to 0.965, "
            "while only 39 to 51 per cent of front membership survives. What that does not "
            "establish is what a second exhaustive census would find, since only the front "
            "was retrained and not the remaining 879 architectures. The claim made is "
            "therefore about how far this front travels rather than about the space on "
            "another corpus, and Section 6 says so."
        ),
        evidence=("src/nas4textgen/corpora/registry.py", "src/nas4textgen/benchmark.py",
                  TASKS),
    ),
    Objection(
        reviewer="R2",
        number="5",
        quote=(
            "Generation quality is evaluated only with METEOR. This metric alone cannot "
            "adequately capture fluency, coherence, factuality, diversity, or human "
            "preference; additional automatic metrics and human evaluation are needed."
        ),
        summary="METEOR alone is not enough; add metrics and human evaluation.",
        disposition="addressed",
        response=(
            "METEOR is no longer an objective. Held-out perplexity is, because it is defined "
            "for every architecture in the space and cannot be inflated by verbosity. Around "
            "it sits a battery: degeneration measures with an explicit length-confound "
            "check, calibration error with the bin counts it came from, a parser for the "
            "code corpus, and overlap metrics kept only as comparisons against their own "
            "trivial floors — an empty continuation, the prompt's tail repeated, the most "
            "frequent tokens — with a metric refused outright when it cannot separate a "
            "model from those floors. The battery is measured, not merely implemented: "
            "all 220 architectures of the front were retrained on two corpora with it "
            "enabled, 440 runs, and every licensed metric produced a value in every run. "
            "It is reported against the objective rather than beside it, because a metric "
            "that ranks architectures the way perplexity already does answers the objection "
            "with a column rather than a fact: five of the six metric-and-corpus pairs fall "
            "below a redundancy threshold fixed at 0.9 before the runs were submitted. The "
            "sharpest case is calibration on the search corpus, where none of the ten best "
            "architectures by perplexity are among the ten best calibrated and the lowest "
            "perplexity on the front ranks 133rd of 220. Human evaluation is not run and "
            "this registry will not claim it is: raters scoring 0.1-to-91M-parameter models "
            "trained on 100M tokens would be ranking degrees of incoherence, and the parser "
            "is the substitute — machine-verifiable, not negotiable, with no trivial floor "
            "above zero."
        ),
        evidence=("src/nas4textgen/evaluation/generation.py",
                  "src/nas4textgen/evaluation/degeneration.py",
                  "src/nas4textgen/evaluation/perplexity.py",
                  "src/nas4textgen/evaluation/syntax.py",
                  "src/nas4textgen/corpora/registry.py"),
    ),
    Objection(
        reviewer="R2",
        number="6",
        quote=(
            "The performance predictor is trained on only 300 fully trained sampled "
            "architectures. The manuscript does not provide detailed held-out prediction "
            "errors, calibration results, or evidence that the predictor reliably ranks the "
            "architectures encountered during NAS."
        ),
        summary="No held-out error, calibration, or ranking evidence for the predictor.",
        disposition="addressed",
        response=(
            "Three measures declared before any model was fitted, all about ranking rather "
            "than fit: Kendall tau on held-out architectures, tau restricted to the top "
            "decile — because a predictor that orders the bad ones well and the good ones "
            "badly is useless for search — and precision@k. Every candidate is scored "
            "against two trivial floors, sorting by parameter count and a linear model on "
            "the genes, and one that outranks neither is disqualified by a rule fixed in "
            "advance. The budget curve answers the objection's own number directly: it "
            "measures how many labels a surrogate needs before it beats sorting by "
            "parameters, on a common held-out set. At 176 measured rows the best model's "
            "margin over the floor was +0.055; at 321 rows it was +0.075."
        ),
        evidence=(*_SURROGATE, "src/nas4textgen/surrogate/compare.py"),
    ),
    Objection(
        reviewer="R2",
        number="7",
        quote=(
            "The search objective uses parameter count as the only efficiency proxy, while "
            "the final results show that parameter count does not consistently correspond to "
            "memory use, CPU demand, or power consumption. This weakens the hardware-aware "
            "motivation of the method."
        ),
        summary="Parameter count is the only efficiency proxy and does not track hardware.",
        disposition="addressed",
        response=(
            "The objection is correct and became a design constraint. The efficiency "
            "objective is KV-cache bytes in closed form, not parameter count, because that "
            "is the quantity that governs what a deployment can hold. The two are kept "
            "deliberately in conflict: the structural gate measures the rank correlation "
            "between parameters and cache bytes over the enumerated space and the space is "
            "accepted only because it is +0.615 rather than near one — a space where the two "
            "objectives agree has no trade-off to search for, and reporting a front over it "
            "would be reporting a sorted list. Deployment quantities are measured separately "
            "under a frozen protocol rather than proxied at all."
        ),
        evidence=("src/nas4textgen/space/cost.py", "src/nas4textgen/space/gate.py",
                  "src/nas4textgen/hardware.py"),
    ),
    Objection(
        reviewer="R2",
        number="8",
        quote=(
            "Hardware evaluation lacks essential deployment measures such as inference "
            "latency, throughput, energy per generated token, measurement repetitions, and "
            "variability. GPU utilization and CPU frequency alone are insufficient"
        ),
        summary="No latency, throughput, energy per token, repetitions, or variability.",
        disposition="declined",
        response=(
            "Declined because this paper no longer makes the claim the objection is about. "
            "The prior submission called itself hardware-aware and measured GPU utilisation "
            "and a CPU frequency in kHz on a Jetson — the metric reviewer R3 said could not "
            "be interpreted. This work's objectives are perplexity, parameter count and "
            "key-value cache bytes, and all 1,099 architectures were measured without any "
            "such device. Withdrawing the claim rather than half-substantiating it is the "
            "same move made for Lex-MODES: the objection is answered by removing what it "
            "was about.\n\n"
            "What would make it apply again is a deployment claim, and one is planned: the "
            "protocol this objection asks for is built, frozen and ROPE-derived, and it "
            "moves to a second paper whose subject is exactly this — decode latency, "
            "throughput, peak memory and joules per token over five repetitions, measured "
            "on the exactly-known front rather than on whatever a search happened to find. "
            "The objection becomes that paper's motivation instead of this one's debt.\n\n"
            "The related objection about the efficiency proxy, R2 #7, stays answered and "
            "needs no device: the efficiency objective is cache bytes in closed form, and "
            "the space is accepted only because cache and parameters disagree."
        ),
        evidence=(
            "src/nas4textgen/hardware.py",
            "openspec/changes/hardware-aware-deployment/proposal.md",
            "openspec/changes/mo-nas-textgen-benchmark/proposal.md",
        ),
    ),
    Objection(
        reviewer="R2",
        number="9",
        quote=(
            "The final comparison includes only four NAS-selected models, and Model 4 was "
            "explicitly not the best architecture from a single configuration. The "
            "model-selection rule is therefore not fully objective and may introduce "
            "selection bias."
        ),
        summary="Four hand-picked models introduce selection bias.",
        disposition="addressed",
        response=(
            "There is no selection rule to be biased, because nothing is selected. The space "
            "is enumerated under a parameter ceiling to 1,099 architectures and every one of "
            "them is trained and measured at an identical budget, so the Pareto front is "
            "computed from the full table rather than chosen from a shortlist. That is the "
            "structural answer to the objection: a benchmark cannot report selectively about "
            "a census. The census is complete: 1,099 of 1,099 architectures measured, "
            "with a front of 220 that is exact over the whole space."
        ),
        evidence=("src/nas4textgen/space/enumerate.py", "src/nas4textgen/benchmark.py",
                  "src/nas4textgen/azure/sweep.py"),
    ),
    Objection(
        reviewer="R2",
        number="10",
        quote=(
            "The reported METEOR improvements are relatively small, but no statistical "
            "significance tests or confidence intervals are provided for the final quality "
            "comparisons. It is unclear whether the gains over the baseline are robust."
        ),
        summary="Small reported gains with no significance tests or intervals.",
        disposition="addressed",
        response=(
            "Friedman with Holm correction for the frequentist side, Bayesian signed-rank "
            "and Plackett-Luce for the Bayesian one, and the region of practical equivalence "
            "derived from measured seed-to-seed noise rather than chosen — which is the part "
            "that answers the objection rather than decorating it, since a threshold picked "
            "after seeing the differences can make any gain significant. Averaging p-values "
            "across problems is refused outright by the comparison module rather than left "
            "to discipline. The comparison design is declared before the data it will read."
        ),
        evidence=(*_STATISTICS, "src/nas4textgen/azure/seeds.py"),
    ),
    Objection(
        reviewer="R2",
        number="11",
        quote=(
            "the search varies embedding dimensions while embedding and positional "
            "components are described as fixed; the manuscript should clearly explain how "
            "dimensional compatibility and any projection layers are implemented"
        ),
        summary="Embeddings are called fixed while the embedding dimension is searched.",
        disposition="addressed",
        response=(
            "The contradiction is removed rather than explained: nothing is described as "
            "fixed. The genotype is five genes and the embedding width is not one of them — "
            "it is derived, e = h x d_head, so the token and position matrices, the "
            "normalisation layers and the block widths all follow from the genotype with no "
            "projection layers anywhere and no compatibility question to answer. The block "
            "family is GPT-2 with one substitution, grouped-query attention, so key/value "
            "heads are searchable; kv == h reduces to ordinary multi-head attention exactly, "
            "asserted against an unmodified GPT-2 in the tests. Two independent checks keep "
            "the description honest: the closed-form parameter count must equal the built "
            "model's, and the space must reproduce DistilGPT2 at its exact 81,912,576 "
            "parameters."
        ),
        evidence=("src/nas4textgen/model.py", "src/nas4textgen/space/genotype.py",
                  "src/nas4textgen/space/cost.py", "tests/test_model.py"),
    ),
    Objection(
        reviewer="R2",
        number="12",
        quote=(
            "The proposed Lex-MODES method appears strongly biased toward parameter "
            "minimization, rather than maintaining a balanced multi-objective search. More "
            "ablation studies are needed to isolate the contributions of Lex-MODES, "
            "self-adaptation, the predictor, and the proposed encoding."
        ),
        summary="Lex-MODES is biased toward parameter minimisation; isolate each component.",
        disposition="declined",
        response=(
            "Declined on its subject rather than its substance: Lex-MODES and its "
            "self-adaptation are not part of this work, so there is no bias of that method "
            "left to measure. The objection would apply again in full the moment a "
            "lexicographic or self-adaptive operator returns to the slate, and the diagnosis "
            "behind it is kept as a standing check — hypervolume is reported raw as well as "
            "normalised precisely so that a search collapsing onto one objective is visible "
            "rather than hidden by rescaling. The two components that did survive are "
            "ablated: the predictor by the budget curve against trivial floors, and the "
            "encoding by running the same algorithms under vector and categorical "
            "representations of the same space."
        ),
        evidence=("src/nas4textgen/search/run.py", "src/nas4textgen/search/scoring.py",
                  "src/nas4textgen/search/problem.py", "src/nas4textgen/surrogate/budget.py"),
    ),

    # ── Reviewer 3 ────────────────────────────────────────────────────────────────────
    Objection(
        reviewer="R3",
        number="1",
        quote=(
            "fails to specify the baseline reference point, sampling rate, measurement "
            "duration, number of repetitions, warm‑up procedures, the state of the "
            "device prior to execution, or whether the reported values represent absolute "
            "measurements or deltas relative to idle conditions"
        ),
        summary="The hardware measurement protocol is unreproducible as described.",
        disposition="declined",
        response=(
            "The objection was correct and the protocol it asked for exists: every item on "
            "its list is a field of a frozen object, declared before measurement and "
            "refusing values that would make it meaningless — a warm-up under five seconds "
            "measures the device clocking up, fewer than three repetitions cannot support a "
            "spread, an unstated power mode makes a figure comparable to nothing. Warm-up "
            "30 s, sampling 10 Hz, 60 s per repetition, five repetitions, device idle at "
            "least 60 s beforehand, and figures reported as deltas against a measured idle "
            "baseline. The kHz CPU metric the objection could not interpret is gone; it was "
            "never a deployment quantity.\n\n"
            "It is declined here rather than answered because this paper reports no hardware "
            "measurement at all, so there is no protocol of its own for a reader to "
            "reproduce. Publishing the protocol beside numbers it did not produce would be "
            "the appearance of rigour without the substance. It would apply again the moment "
            "this work reports a deployment figure, and the second paper is where that "
            "happens — with the protocol already fixed, which is the point of having frozen "
            "it before any device was available."
        ),
        evidence=(
            "src/nas4textgen/hardware.py",
            "tests/test_hardware.py",
            "openspec/changes/hardware-aware-deployment/proposal.md",
        ),
    ),
    Objection(
        reviewer="R3",
        number="2",
        quote=(
            "for one and the same DistilGPT2 architecture, the memory indicator varies from "
            "247.87 to 54,269.24 KB depending on the number of training epochs, even though "
            "the model structure remains unchanged"
        ),
        summary="Memory figures are internally inconsistent and never defined.",
        disposition="addressed",
        response=(
            "Three different quantities were sharing one column. They are separated and each "
            "is defined where it is computed. The memory *objective* is KV-cache bytes at a "
            "declared context length, closed-form from the genotype and independent of "
            "training entirely, which is why it cannot vary with epochs. Peak training "
            "memory is recorded per run as an operational fact about the job. Deployment "
            "memory is measured under the hardware protocol with its own repetitions. No "
            "figure is reported without its units and the quantity it names."
        ),
        evidence=("src/nas4textgen/space/cost.py", "src/nas4textgen/hardware.py",
                  "src/nas4textgen/benchmark.py", "src/nas4textgen/training.py"),
    ),
    Objection(
        reviewer="R3",
        number="3",
        quote=(
            "The authors must therefore specify unambiguously what is meant by a "
            "“fixed” component: an unchanged structure, unchanged dimensions, or "
            "retention of pre‑trained weights."
        ),
        summary="\"Fixed\" is never defined while the embedding dimension varies.",
        disposition="addressed",
        response=(
            "Answered by removing the word rather than defining it. No component is "
            "described as fixed: the whole model is constructed from the genotype, every "
            "width follows from e = h x d_head, and no weights are retained from anywhere "
            "because every architecture trains from scratch. What is held constant across "
            "the benchmark is stated explicitly and enforced — corpus, vintage, tokeniser, "
            "token budget, precision — and rows differing in any of them are refused rather "
            "than reconciled."
        ),
        evidence=("src/nas4textgen/model.py", "src/nas4textgen/space/genotype.py",
                  "src/nas4textgen/benchmark.py"),
    ),
    Objection(
        reviewer="R3",
        number="4",
        quote=(
            "models with an embedding dimension different from 768 cannot directly utilise "
            "the majority of the pre‑trained weights of DistilGPT2 ... Under divergent "
            "initialisation conditions, any conclusion regarding the pure effect of "
            "architecture becomes invalid."
        ),
        summary="Comparing against a pre-trained DistilGPT2 invalidates the architecture claim.",
        disposition="addressed",
        response=(
            "The objection is decisive and the design concedes it: every architecture, the "
            "anchor included, is trained from scratch on an identical token budget of 100 "
            "million tokens, the same corpus at the same recorded vintage, the same "
            "tokeniser identified by hash, and the same precision. The budget is met in "
            "tokens rather than epochs so that a smaller model does not quietly receive more "
            "passes over the data. Nothing is initialised from pre-trained weights, so there "
            "is no divergent initialisation left for the conclusion to depend on. The "
            "comparability rule is code that refuses rows, not a paragraph in a methods "
            "section."
        ),
        evidence=("src/nas4textgen/benchmark.py", "src/nas4textgen/training.py",
                  "src/nas4textgen/replication.py", "tests/test_benchmark.py"),
    ),
    Objection(
        reviewer="R3",
        number="5",
        quote=(
            "no a‑priori decision rule is defined that would justify selecting MLP as "
            "the overall best compromise"
        ),
        summary="Surrogate quality is unverifiable and the winner was chosen after the fact.",
        disposition="addressed",
        response=(
            "The decision rule now precedes the data. Three measures are named in advance, "
            "the disqualification threshold is the trivial floors rather than a number "
            "chosen later, and the region of practical equivalence is computed from measured "
            "replicate noise instead of picked. The Bayesian comparison reports the "
            "posterior over which model wins along with the rope it was evaluated against, "
            "and the slate records every candidate including those that could not be fitted, "
            "so the reported winner is the survivor of a stated procedure rather than the "
            "best row of a table read afterwards."
        ),
        evidence=(*_SURROGATE, "src/nas4textgen/statistics/rope.py",
                  "src/nas4textgen/statistics/bayesian.py"),
    ),
    Objection(
        reviewer="R3",
        number="6",
        quote=(
            "the evolutionary algorithm may have optimised the surrogate error rather than "
            "the true generative quality"
        ),
        summary="A predictor fitted on 300 architectures may not generalise where NAS searches.",
        disposition="addressed",
        response=(
            "Structurally impossible here, which is a stronger answer than more validation. "
            "The algorithms do not query a surrogate; they query the measured table. Every "
            "architecture a search evaluates has a real trained-and-measured objective "
            "vector, so a search cannot optimise prediction error because prediction is not "
            "in the loop. The surrogate is studied as a separate object — how many labels it "
            "needs before it beats sorting by parameter count, and how well it ranks the top "
            "decile — and ParEGO is the one entry that does consult a model, precisely so "
            "the cost of doing so is visible against entries that do not."
        ),
        evidence=("src/nas4textgen/search/problem.py", "src/nas4textgen/search/local.py",
                  "src/nas4textgen/surrogate/budget.py", "src/nas4textgen/benchmark.py"),
    ),
    Objection(
        reviewer="R3",
        number="7",
        quote=(
            "three models improve the score only from 0.6961 to 0.7000, yet standard "
            "deviations, confidence intervals, the number of independent training runs, and "
            "significance tests are absent"
        ),
        summary="Improvements of 0.004 reported with no variance and no test.",
        disposition="pending",
        response=(
            "The number of seeds is fixed in advance and recorded per run rather than left "
            "implicit, and the seed-to-seed spread is not merely reported — it is what "
            "defines the region of practical equivalence, so a difference of the size the "
            "objection quotes is measured against the noise of the process that produced it "
            "rather than against zero. Whether a gain survives that comparison is then a "
            "Bayesian signed-rank verdict with the rope stated, alongside Friedman with Holm "
            "for the frequentist reading. Pending until the final architectures are trained "
            "across seeds."
        ),
        evidence=(*_STATISTICS, "src/nas4textgen/azure/seeds.py", TASKS),
    ),
    Objection(
        reviewer="R3",
        number="8",
        quote=(
            "six unique best architectures were found, yet only four are included in the "
            "final comparison. Model 4 was never the best in any configuration ... such a "
            "selection rule was not specified in advance and introduces a risk of selective "
            "reporting."
        ),
        summary="Post hoc selection of which architectures to report.",
        disposition="addressed",
        response=(
            "The census removes the possibility rather than defending against it: all 1,099 "
            "architectures under the parameter ceiling are trained and measured, so there is "
            "no shortlist and no rule to specify. Which rows are comparable is decided by "
            "declared invariants — precision, vintage, budget, tokeniser — and enforced by "
            "code that refuses the row, not by judgement applied afterwards. Coverage travels "
            "with every table, so a partial result cannot read as a final one."
        ),
        evidence=("src/nas4textgen/space/enumerate.py", "src/nas4textgen/benchmark.py",
                  "src/nas4textgen/manuscript.py"),
    ),
    Objection(
        reviewer="R3",
        number="9",
        quote=(
            "A random‑search baseline is essential. It has not been shown that the "
            "evolutionary methods outperform random sampling or direct selection from the "
            "300 already fully trained architectures."
        ),
        summary="No random-search baseline; evolution never shown to beat sampling.",
        disposition="addressed",
        response=(
            "Random search is in the slate as the declared floor, and the framing is taken "
            "seriously rather than included for form: on 1,099 points that floor is not low, "
            "and an algorithm that does not beat it has bought nothing. An early run of this "
            "comparison appeared to show random beating NSGA-II fivefold, which turned out "
            "to be a defect in the budget accounting — queries were counted instead of "
            "distinct architectures, and NSGA-II's archive re-queries what it already knows. "
            "Correcting it left NSGA-II at 23 distinct architectures in 158 queries against "
            "random's 115 in 149, a different comparison entirely. Sorting by parameter count "
            "is the second floor, on the surrogate side.\n\n"
            "The answer on the closed census is quantitative and uncomfortable in both "
            "directions. Random search has the *best* mean rank of six algorithms at a "
            "budget of 25 architectures and falls to second-worst at 200; the evolutionary "
            "methods overtake it between 50 and 100. At 200, AGE-MOEA-II beats it at an "
            "adjusted p of 0.0049 — so the objection's premise is answered, and the answer "
            "is that the advantage exists and only above a budget the prior work never "
            "reached."
        ),
        evidence=("src/nas4textgen/search/run.py", "src/nas4textgen/search/local.py",
                  "src/nas4textgen/search/scoring.py", "src/nas4textgen/surrogate/slate.py"),
    ),
    Objection(
        reviewer="R3",
        number="10",
        quote=(
            "Relying solely on METEOR is insufficient for evaluating open‑ended text "
            "generation. Perplexity, additional continuation quality metrics such as ROUGE "
            "and BERTScore or another contextual metric, generated text examples, and "
            "ideally human evaluation should be supplied."
        ),
        summary="METEOR alone cannot evaluate open-ended generation.",
        disposition="addressed",
        response=(
            "Perplexity is the quality objective, exactly as the objection proposes, and it "
            "is measured for all 1,099 architectures. The overlap metrics the objection "
            "names are kept only under a condition: each is scored against trivial floors — "
            "an empty continuation, the prompt's tail, the most frequent tokens, a "
            "deliberately overlong one — and a metric that cannot separate a real model "
            "from those floors is refused rather than reported. The licensing half of that "
            "is enforced: generation metrics are licensed on the synthetic corpus and "
            "withheld on Wikipedia, where models this size produce nothing coherent to "
            "score, and a run asking for an unlicensed metric receives a refusal rather "
            "than a number. The measurement is done. On TinyStories, where generation is "
            "licensed, distinct-1 runs from 0.069 to 0.645 across the front and the share "
            "of repeated 4-grams from 0.033 to 0.926 — a separation far wider than the "
            "perplexities of the same architectures, and one a likelihood averaged over a "
            "held-out split registers only faintly. The prior work\'s own metric is "
            "answered separately: its figures are re-derived by `cli audit`, which shows an "
            "empty generation scoring 0.3088 under its protocol against a reported "
            "improvement of 0.024."
        ),
        evidence=("src/nas4textgen/evaluation/perplexity.py",
                  "src/nas4textgen/evaluation/generation.py",
                  "src/nas4textgen/evaluation/degeneration.py",
                  "src/nas4textgen/corpora/registry.py"),
    ),
    Objection(
        reviewer="R3",
        number="11",
        quote=(
            "It would be beneficial to discuss whether the proposed NAS system could be "
            "leveraged for specialised generative models ... metrics such as syntactic "
            "correctness, compilation and synthesis success, and functional correctness "
            "verified through testbenches would become essential."
        ),
        summary="Discuss transfer to specialised generation, with machine-checkable metrics.",
        disposition="addressed",
        response=(
            "The mechanism the objection describes is already in the design, on code rather "
            "than on Verilog: a code corpus is in the registry specifically because generated "
            "code either parses or it does not, which gives at least one quality signal that "
            "a parser settles — not inflatable by verbosity and with no trivial floor above "
            "zero. That is the same argument the objection makes for syntactic and "
            "functional correctness, applied to a corpus this project can actually run. The "
            "reviewer also offers a reference of their own as illustration; whether the "
            "manuscript cites it is an editorial decision, and it is recorded here so the "
            "decision is visible rather than silent."
        ),
        evidence=("src/nas4textgen/corpora/registry.py",
                  "src/nas4textgen/evaluation/syntax.py", REVIEW_TEXT),
    ),
    Objection(
        reviewer="R3",
        number="12",
        quote=(
            "The experimental section omits several mandatory reproducibility parameters, "
            "including the optimizer, learning rate, batch size, exact number of epochs for "
            "the final architectures, initialisation strategy, random seeds, decoding "
            "parameters, output length, numerical precision, software versions"
        ),
        summary="The reproducibility parameters are missing.",
        disposition="addressed",
        response=(
            "Handled as an artifact rather than a paragraph, because a paragraph is where "
            "these go stale. Replication instructions are generated from a declaration in "
            "code, with each stage marked bit-exact or tolerance-exact and the reason it can "
            "claim only that. Every run writes a manifest recording device, precision, image "
            "digest, dependency lock hash, corpus vintage, tokeniser hash, seed and token "
            "budget, so the workspace itself can answer which code and which data produced "
            "any number. The image is pinned by digest rather than tag, and a pre-flight "
            "refuses to submit when the image's contents differ from the checkout — a guard "
            "added after a wave went out on stale code."
        ),
        evidence=("src/nas4textgen/replication.py", "docs/REPLICATION.md",
                  "src/nas4textgen/azure/submit.py", "src/nas4textgen/azure/workspace.py"),
    ),
    Objection(
        reviewer="R3",
        number="13",
        quote=(
            "Reference [12] contains incorrect author formatting, [30] pertains to "
            "DistilBERT rather than DistilGPT2, and [19] lists the conference name in place "
            "of the publication title."
        ),
        summary="Specific bibliographic errors, and others besides.",
        disposition="pending",
        response=(
            "A manuscript-level defect with no code to answer it, and the three named "
            "instances are the reason the whole bibliography is rebuilt rather than patched: "
            "a reference list with three verified errors has an unknown number of unverified "
            "ones. Each entry will be checked against its source when the manuscript is "
            "written, and this registry entry is what makes that a task rather than an "
            "intention."
        ),
        evidence=(TASKS, REVIEW_TEXT),
    ),
    Objection(
        reviewer="R3",
        number="14",
        quote=(
            "the text requires thorough language and formatting revision. Instances "
            "include “DistilGP2,” ... Template remnants such as “Author et al.,” "
            "“VOLUME 11, 2023,” fictitious publication dates, and a placeholder DOI "
            "remain in the manuscript."
        ),
        summary="Typos, template remnants, a placeholder DOI and fictitious dates.",
        disposition="pending",
        response=(
            "Nothing here can be inherited: the manuscript is rewritten against generated "
            "tables, so the placeholder DOI, the template volume line and the unreplaced "
            "citation marker have no path into the new text. The tables in particular are "
            "generated from the artifact directory rather than transcribed, because a number "
            "typed into a manuscript can disagree with the artifact it came from and by the "
            "time anyone notices the paper is submitted. Pending until the text exists to be "
            "proofread."
        ),
        evidence=(TASKS, "src/nas4textgen/manuscript.py", REVIEW_TEXT),
    ),
)


#: Evidence paths that are deliberately not in the repository, which carries the
#: implementation and the results. A path under here is reported rather than counted
#: missing, because "it is kept elsewhere" and "it does not exist" are different claims.
#:
#: The check passed on the authors' machine while these were untracked, because the files
#: were still on disk — so it would have failed for the first person to clone the
#: repository and nowhere before that.
KEPT_LOCALLY = (
    "openspec/",            # the plan: what was intended, and in what order
    "docs/reviews/",        # the referee reports, which are not ours to publish
    "docs/manuscript/",     # the paper, which lives with its authors
    "docs/REPLICATION.md",  # generated from replication.py, which is the evidence
    "docs/TABLES.md",       # generated from the artifacts, which are the evidence
)


def verify_evidence(root: Path | None = None) -> list[str]:
    """Check that every cited file exists. Returns the citations that do not."""
    root = root or Path(__file__).resolve().parents[2]
    broken = []
    for objection in RECORDED:
        for citation in objection.evidence:
            if citation.startswith(KEPT_LOCALLY):
                continue
            if not (root / citation).exists():
                broken.append(f"{objection.reviewer} #{objection.number} -> {citation}")
    return broken


def evidence_kept_locally() -> list[str]:
    """Citations that point outside this repository, and which objections make them.

    Reported rather than hidden. An objection whose only evidence is a plan is an
    objection with no artifact behind it yet, and that is worth seeing in the listing.
    """
    found = []
    for objection in RECORDED:
        for citation in objection.evidence:
            if citation.startswith(KEPT_LOCALLY):
                found.append(f"{objection.reviewer} #{objection.number} -> {citation}")
    return found


def require_evidence(root: Path | None = None) -> None:
    """Refuse a registry whose citations have rotted."""
    broken = verify_evidence(root)
    if broken:
        raise EvidenceMissing(
            "these objections cite files that do not exist:\n  " + "\n  ".join(broken)
            + "\nA citation to a moved file reads as diligence and is not."
        )


def _normalise(text: str) -> str:
    """Collapse whitespace so a quote wrapped across source lines still matches."""
    return " ".join(text.split())


def verify_quotes(root: Path | None = None) -> list[str]:
    """Check every quote against the committed reviews. Returns those not found.

    Quotes are stored with ``...`` where an ellipsis skips text, so each side of the
    ellipsis is checked separately — matching the whole string would fail on an elision the
    reviewer never wrote.
    """
    root = root or Path(__file__).resolve().parents[2]
    source = root / REVIEW_TEXT
    if not source.exists():
        raise ReviewsUnavailable(
            f"{REVIEW_TEXT} is not in this checkout. The reviews are kept locally rather "
            "than in the repository; with them present this command checks all "
            f"{len(RECORDED)} quotes against them."
        )
    reviews = _normalise(source.read_text(encoding="utf-8"))
    missing = []
    for objection in RECORDED:
        for fragment in _normalise(objection.quote).split("..."):
            fragment = fragment.strip()
            if fragment and fragment not in reviews:
                missing.append(f"{objection.reviewer} #{objection.number}: {fragment[:60]}...")
    return missing


def require_quotes(root: Path | None = None) -> None:
    """Refuse a registry that answers words the reviewers did not write."""
    missing = verify_quotes(root)
    if missing:
        raise QuoteNotFound(
            "these objections quote text the reviews do not contain:\n  "
            + "\n  ".join(missing)
            + "\nAnswering a question the reviewer did not ask reads, to that reviewer, as "
              "not having read them."
        )


def open_objections() -> tuple[Objection, ...]:
    """Everything still waiting on a measurement or on the manuscript."""
    return tuple(o for o in RECORDED if o.is_open)


def require_complete() -> None:
    """Refuse to certify the registry while objections are still open.

    Called before a resubmission claims the reviewers' comments are addressed. It fails
    today, for a reason that is now a fact about the campaign rather than about this file:
    the sweep is still measuring and the manuscript is not written.
    """
    still_open = open_objections()
    if still_open:
        raise RegistryIncomplete(
            f"{len(still_open)} of {len(RECORDED)} objections are still open:\n  "
            + "\n  ".join(f"{o.reviewer} #{o.number}: {o.summary}" for o in still_open)
        )


def report(root: Path | None = None) -> dict[str, object]:
    """The registry, with its own gaps in it."""
    by_reviewer: dict[str, int] = {}
    for objection in RECORDED:
        by_reviewer[objection.reviewer] = by_reviewer.get(objection.reviewer, 0) + 1

    # The quote check needs the reviews, which are kept with the authors. A report that
    # raised here would be unusable in any checkout without them; one that silently
    # returned an empty list would claim every quote verified while checking none. It
    # says which of the two situations it is in.
    try:
        unverified = verify_quotes(root)
        quotes_checked = True
    except ReviewsUnavailable:
        unverified, quotes_checked = [], False

    return {
        "recorded": [objection.as_dict() for objection in RECORDED],
        "total": len(RECORDED),
        "by_reviewer": by_reviewer,
        "addressed": sum(o.disposition == "addressed" for o in RECORDED),
        "pending": sum(o.disposition == "pending" for o in RECORDED),
        "declined": sum(o.disposition == "declined" for o in RECORDED),
        "broken_citations": verify_evidence(root),
        "unverified_quotes": unverified,
        "quotes_checked": quotes_checked,
        "evidence_kept_locally": evidence_kept_locally(),
        "review_text": REVIEW_TEXT,
        "complete": not open_objections(),
        "why_open_is_reported": (
            "An objection whose mechanism exists but whose number is not final is not "
            "answered yet, and counting it as answered is how a cover letter comes to claim "
            "more than the work supports. The open count is the honest progress signal."
        ),
    }
