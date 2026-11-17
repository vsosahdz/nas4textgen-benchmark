"""Verify the environment can actually run, not merely that it installed.

A successful ``pip install`` does not prove a wheel's native library can load.
XGBoost and LightGBM install cleanly on macOS and then fail at import because they
link against OpenMP, which Apple does not ship. This check catches that class of
failure before it reaches a reader.

Run with::

    python -m nas4textgen.check_env

Exits non-zero if a base dependency cannot be imported, or if the determinism
environment required by the ``verify`` tier is not set. Surrogate candidates whose
backend is unavailable are reported as unavailable rather than silently skipped —
see specs/replication-package/spec.md.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys

# Base dependencies. A failure here is fatal: the verify tier cannot run.
BASE_MODULES: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("matplotlib", "matplotlib"),
    ("pyyaml", "yaml"),
    ("tqdm", "tqdm"),
    ("scikit-learn", "sklearn"),
    ("scikit-posthocs", "scikit_posthocs"),
    ("pymc", "pymc"),
    ("arviz", "arviz"),
]

# Surrogate candidates with a native backend that can fail to load even when the
# wheel installed. Reported, not fatal — but the surrogate comparison must record
# any that are unavailable.
NATIVE_MODULES: list[tuple[str, str, str]] = [
    ("xgboost", "xgboost", "macOS: brew install libomp"),
    ("lightgbm", "lightgbm", "macOS: brew install libomp"),
    ("catboost", "catboost", ""),
    ("ngboost", "ngboost", ""),
    ("quantile-forest", "quantile_forest", ""),
]

DETERMINISM_VARS = [
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
]


def _probe(module: str) -> tuple[bool, str]:
    try:
        m = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 — any failure is a failure to report
        return False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:70]}"
    return True, getattr(m, "__version__", "?")


def _exercise_slate() -> list[tuple[str, str]]:
    """Fit every available surrogate candidate on a small synthetic problem.

    Importing is not enough: a native backend can import and still fail on first
    use. The slate is only trustworthy if every candidate has produced a
    prediction.
    """
    import numpy as np
    from scipy.stats import kendalltau

    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, size=(120, 5)).astype(float)
    y = x @ np.array([2.0, -1.5, 0.8, 0.3, -0.6]) + rng.normal(0, 0.4, 120)
    x_tr, y_tr, x_te, y_te = x[:90], y[:90], x[90:], y[90:]

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.linear_model import BayesianRidge, LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR

    candidates: dict[str, object] = {
        "linear (trivial)": LinearRegression(),
        "random forest": RandomForestRegressor(n_estimators=50, random_state=0),
        "gaussian process": GaussianProcessRegressor(normalize_y=True, random_state=0),
        "bayesian ridge": BayesianRidge(),
        "SVR (carried)": SVR(),
        "MLP (carried)": MLPRegressor(max_iter=1000, random_state=0),
    }

    def _try_add(name: str, build) -> None:
        try:
            candidates[name] = build()
        except Exception as exc:  # noqa: BLE001
            results.append((name, f"unavailable — {type(exc).__name__}"))

    results: list[tuple[str, str]] = []

    _try_add("catboost", lambda: __import__("catboost").CatBoostRegressor(
        iterations=50, verbose=0, random_state=0))
    _try_add("quantile forest", lambda: __import__(
        "quantile_forest").RandomForestQuantileRegressor(n_estimators=50, random_state=0))
    _try_add("ngboost", lambda: __import__("ngboost").NGBRegressor(
        n_estimators=50, verbose=False, random_state=0))
    _try_add("xgboost (squared)", lambda: __import__("xgboost").XGBRegressor(
        n_estimators=50, verbosity=0, random_state=0))
    _try_add("xgboost (rank:pairwise)", lambda: __import__("xgboost").XGBRanker(
        objective="rank:pairwise", n_estimators=50, verbosity=0, random_state=0))
    _try_add("lightgbm", lambda: __import__("lightgbm").LGBMRegressor(
        n_estimators=50, verbose=-1, random_state=0))

    for name, model in candidates.items():
        try:
            if "rank" in name:
                model.fit(x_tr, y_tr, group=[len(x_tr)])  # type: ignore[attr-defined]
            else:
                model.fit(x_tr, y_tr)  # type: ignore[attr-defined]
            tau = kendalltau(model.predict(x_te), y_te).statistic  # type: ignore[attr-defined]
            results.append((name, f"tau {tau:+.3f}"))
        except Exception as exc:  # noqa: BLE001
            results.append((name, f"FAILED — {type(exc).__name__}"))
    return results


def main() -> int:
    print(f"platform  {platform.platform()}")
    print(f"python    {platform.python_version()}  ({sys.executable})")
    print()

    failures = 0

    print("base dependencies")
    for name, module in BASE_MODULES:
        ok, detail = _probe(module)
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<20} {detail}")
        failures += not ok

    print()
    print("native-backend dependencies")
    for name, module, hint in NATIVE_MODULES:
        ok, detail = _probe(module)
        suffix = "" if ok or not hint else f"   → {hint}"
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<20} {detail}{suffix}")

    print()
    print("determinism environment (required for the verify tier's bit-exactness)")
    for var in DETERMINISM_VARS:
        value = os.environ.get(var)
        state = "ok  " if value is not None else "MISS"
        failures += value is None
        print(f"  {state}  {var:<24} {value if value is not None else 'not set'}")
    if any(os.environ.get(v) is None for v in DETERMINISM_VARS):
        print("        → cp .env.example .env && set -a; . ./.env; set +a")

    print()
    print("surrogate slate")
    for name, detail in _exercise_slate():
        marker = "ok  " if detail.startswith("tau") else "WARN"
        print(f"  {marker}  {name:<26} {detail}")

    print()
    if failures:
        print(f"{failures} problem(s). The verify tier will not reproduce until they are fixed.")
        return 1
    print("Environment ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
