"""Search-algorithm comparison over the measured benchmark table.

Once the sweep has measured every architecture, an "evaluation" is a lookup rather than a
training run. That is what makes a slate of seven algorithms over many seeds affordable,
and it is what makes the comparison exact: the true Pareto front is known, so a run is
scored against the answer rather than against the best front anyone happened to find.
"""

from __future__ import annotations

from .problem import (
    BenchmarkTable,
    IncomparableRows,
    LookupProblem,
    UnmeasuredArchitecture,
    build_choice_problem,
    build_vector_problem,
    decode_vector,
    require_comparable,
)
from .run import SLATE, AlgorithmSpec, build_algorithm, run_algorithm
from .scoring import QueryTrace, igd_plus, normalise, score_trace

__all__ = [
    "SLATE",
    "AlgorithmSpec",
    "QueryTrace",
    "build_algorithm",
    "require_comparable",
    "run_algorithm",
    "BenchmarkTable",
    "IncomparableRows",
    "LookupProblem",
    "UnmeasuredArchitecture",
    "build_choice_problem",
    "build_vector_problem",
    "decode_vector",
    "igd_plus",
    "normalise",
    "score_trace",
]
