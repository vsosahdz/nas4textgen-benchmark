"""Every predictor the study compares, and the two it must beat to matter.

The slate is chosen for the geometry of this problem rather than assembled from what is
popular: five categorical or ordinal genes over 1,099 points, with a few hundred labels at
most. That rules some families in and others out, and D17 records which and why.

**The trivial predictors are not filler.** Sorting by parameter count answers "does this beat
bigger-is-better", and linear regression on the genes answers "is anything nonlinear
happening". A candidate that cannot outrank both is disqualified, and reporting that is more
useful than hiding it — in this space the parameter floor reaches Kendall tau 0.89, so a
surrogate reporting 0.93 without the floor beside it would look far stronger than it is.

**Selection is by rank, so at least one candidate is fit on rank.** Fitting for squared
error and selecting on Kendall tau is a mismatch, and it is the mismatch the prior work
made. The slate includes a pairwise ranking objective so the comparison contains a model
that does not make it.

**Uncertainty is required rather than optional.** A wave-based sweep can use predictive
variance to choose what to measure next, and the reviewers asked for calibration. A point
prediction cannot answer either.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: Genes, in the order the design matrix uses them.
GENE_ORDER = ("h", "d_head", "n", "ffn", "kv")


@dataclass(frozen=True)
class Candidate:
    """One predictor, with what it is for and what it costs to include."""

    name: str
    tier: str
    build: Callable[[], object] | None
    #: Whether the model reports a predictive distribution rather than a point.
    uncertainty: bool = False
    #: Whether it is fit on a ranking objective rather than squared error.
    ranking_loss: bool = False
    #: Predictors that need no fitting at all — the trivial floors.
    closed_form: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    why: str = ""

    @property
    def is_trivial(self) -> bool:
        return self.tier == "trivial"

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "tier": self.tier,
            "uncertainty": self.uncertainty,
            "ranking_loss": self.ranking_loss,
            "why": self.why,
        }


def _by_parameter_count(_features: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Predict rank from cost alone: bigger is better.

    Negated because larger models have lower perplexity and the study predicts perplexity,
    where lower is better. This is the floor a surrogate has to clear, and in this space it
    is high — parameter count is dominated by the embedding matrix, which tracks capacity.
    """
    return -params


def build_slate() -> list[Candidate]:
    """The full slate. Missing backends are reported rather than silently skipped."""
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from sklearn.linear_model import BayesianRidge, LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    candidates: list[Candidate] = [
        Candidate(
            "sort_by_parameters", "trivial", None,
            closed_form=_by_parameter_count,
            why="does the predictor beat bigger-is-better? In this space that floor is high",
        ),
        Candidate(
            "linear_on_genes", "trivial",
            lambda: LinearRegression(),
            why="is anything nonlinear happening? If this wins, nothing is",
        ),
        Candidate(
            "random_forest", "tier1",
            lambda: RandomForestRegressor(400, random_state=0, n_jobs=-1),
            why="the default for small tabular problems, and the reference to beat",
        ),
        Candidate(
            "gradient_boosting", "tier1",
            lambda: GradientBoostingRegressor(random_state=0),
            why="boosting on squared error, for comparison against the ranking objective",
        ),
        Candidate(
            "gaussian_process", "tier1",
            lambda: make_pipeline(
                StandardScaler(),
                GaussianProcessRegressor(
                    # Matern rather than RBF: the genes are ordinal with as few as three
                    # levels, so the response is not smooth enough for RBF's assumptions.
                    kernel=ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
                    normalize_y=True, random_state=0, n_restarts_optimizer=2,
                ),
            ),
            uncertainty=True,
            why="the standard surrogate in expensive multi-objective search, for its "
                "calibrated variance; absent from the prior slate",
        ),
        Candidate(
            "bayesian_ridge", "tier2",
            lambda: make_pipeline(StandardScaler(), BayesianRidge()),
            uncertainty=True,
            why="a linear model that reports uncertainty, so the uncertainty tier has a "
                "floor of its own",
        ),
        Candidate(
            "svr_rbf", "tier4",
            lambda: make_pipeline(StandardScaler(), SVR(C=10.0)),
            why="carried from the prior work, so its domination is shown rather than "
                "assumed by dropping it",
        ),
        Candidate(
            "mlp", "tier4",
            lambda: make_pipeline(
                StandardScaler(),
                # Every argument by keyword. In scikit-learn 1.9 the first positional
                # parameter of MLPRegressor is `loss`, not `hidden_layer_sizes`, so
                # MLPRegressor((64, 64)) passes the tuple as the loss and fails with a
                # message about loss that reads like a library problem. It was mine.
                MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=4000, random_state=0),
            ),
            why="carried from the prior work for the same reason",
        ),
    ]

    candidates.extend(_optional_candidates())
    return candidates


def _optional_candidates() -> list[Candidate]:
    """Candidates whose backends may be absent. Reported as unavailable, never skipped.

    A slate that quietly shrinks to what happened to be installed is a slate whose result
    depends on the machine. Each of these appears in the report either with a score or with
    the reason it has none.
    """
    found: list[Candidate] = []

    try:
        from xgboost import XGBRanker, XGBRegressor

        found.append(Candidate(
            "xgboost", "tier1", lambda: XGBRegressor(
                n_estimators=400, max_depth=4, learning_rate=0.08,
                random_state=0, n_jobs=-1, verbosity=0,
            ),
            why="the boosting baseline most NAS surrogate papers report",
        ))
        found.append(Candidate(
            "xgboost_ranker", "tier3", lambda: XGBRanker(
                objective="rank:pairwise", n_estimators=400, max_depth=4,
                learning_rate=0.08, random_state=0, n_jobs=-1, verbosity=0,
            ),
            ranking_loss=True,
            why="fit on a pairwise ranking loss, because selection is by Kendall tau and "
                "fitting for squared error while selecting on rank is the mismatch the "
                "prior work made",
        ))
    except ImportError:
        found.append(Candidate("xgboost", "tier1", None, why="xgboost is not installed"))

    try:
        from lightgbm import LGBMRegressor

        found.append(Candidate(
            "lightgbm", "tier1", lambda: LGBMRegressor(
                n_estimators=400, num_leaves=15, learning_rate=0.08,
                random_state=0, n_jobs=-1, verbose=-1,
            ),
            why="frequently beats XGBoost at a few hundred samples",
        ))
    except ImportError:
        found.append(Candidate("lightgbm", "tier1", None, why="lightgbm is not installed"))

    try:
        from catboost import CatBoostRegressor

        # CatBoost refuses float columns declared categorical, and the design matrix is
        # float because every other candidate wants it that way. Wrapping the cast in the
        # estimator keeps one matrix for the whole slate — the alternative is a second
        # matrix that could drift out of step with the first.
        class _CatBoostOnGenes(CatBoostRegressor):
            def fit(self, X, y=None, **kwargs):  # noqa: N803
                return super().fit(np.asarray(X).astype(int), y, **kwargs)

            def predict(self, X, **kwargs):  # noqa: N803
                return super().predict(np.asarray(X).astype(int), **kwargs)

        found.append(Candidate(
            "catboost", "tier1", lambda: _CatBoostOnGenes(
                iterations=500, depth=5, learning_rate=0.08,
                random_seed=0, verbose=False,
                # Every gene is categorical or ordinal, and this is the one library that
                # takes them as such rather than through a one-hot expansion.
                cat_features=list(range(len(GENE_ORDER))),
            ),
            why="native categorical handling, which is what every gene here is",
        ))
    except ImportError:
        found.append(Candidate("catboost", "tier1", None, why="catboost is not installed"))

    try:
        from quantile_forest import RandomForestQuantileRegressor

        found.append(Candidate(
            "quantile_forest", "tier2",
            lambda: RandomForestQuantileRegressor(n_estimators=400, random_state=0),
            uncertainty=True,
            why="predictive quantiles without a distributional assumption, so calibration "
                "can be checked against held-out architectures",
        ))
    except ImportError:
        found.append(Candidate(
            "quantile_forest", "tier2", None, why="quantile-forest is not installed"
        ))

    try:
        from ngboost import NGBRegressor
        from sklearn.tree import DecisionTreeRegressor

        found.append(Candidate(
            "ngboost", "tier2",
            # The base learner is seeded explicitly. NGBoost's own ``random_state`` seeds
            # its subsampling but not the estimator it is handed, and the default is
            # ``DecisionTreeRegressor(max_depth=3)`` with no seed at all. sklearn's tree
            # breaks ties between equally-good splits with its RNG, and on five integer
            # genes over small subsets those ties are common — so two identical runs
            # produced different curves. Caught by the verification tier: this stage
            # declares bit-exact reproduction and was the only candidate not delivering it.
            lambda: NGBRegressor(
                n_estimators=400, random_state=0, verbose=False,
                Base=DecisionTreeRegressor(max_depth=3, random_state=0),
            ),
            uncertainty=True,
            why="boosting that predicts a distribution rather than a point",
        ))
    except ImportError:
        found.append(Candidate("ngboost", "tier2", None, why="ngboost is not installed"))

    return found


#: Families the design rules out, with the condition that would bring each back. Recorded
#: because reviewer R2 asked for them, and "we did not try it" is a weaker answer than "it
#: does not apply, and here is when it would".
EXCLUDED: dict[str, str] = {
    "graph_neural_network": (
        "The genotype is a fixed-length vector of five genes, not a graph. Every "
        "architecture here has the same topology — a stack of identical blocks — so a GCN "
        "would receive the same graph 1,099 times with only node features differing, which "
        "is a multilayer perceptron with extra steps. Applicable the moment the space "
        "admits varying topology: skip connections, branching, or per-layer heterogeneity."
    ),
    "sequence_model": (
        "An LSTM or transformer over the architecture string treats five genes as a "
        "sequence whose order carries no meaning. Applicable if the encoding becomes "
        "variable-length, as it would with per-layer genes."
    ),
    "zero_cost_proxy": (
        "Gradient-based proxies at initialisation are a different experiment rather than a "
        "different model: they predict from an untrained network instead of from the "
        "genotype, so comparing them here would confound the question. Worth its own study "
        "against this table, which is one of the things a published benchmark enables."
    ),
}
