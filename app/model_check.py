"""
Model check for the dashboard: how far off was each method, one month ahead?

Reads data/processed/trained_vs_baselines.csv (made by `python -m src.models.trained_v0`) and
averages the miss (|forecast - actual|, in new EVs per month) for each method, separately for
the development period and the protected 2026 holdout.
"""
import csv

LABEL = {"naive": "Last value", "naive3": "3-month average", "snaive": "Same month last year",
         "cagr": "Growth rate", "ridge_lag": "Ridge (learned)", "gbm_growth": "Boosted trees (learned)"}
LEARNED = {"ridge_lag", "gbm_growth"}

def mae_table(path: str, horizon: int = 1) -> dict[str, dict[str, tuple[float, int]]]:
    """period -> method -> (average miss, number of forecasts scored), for one forecast horizon."""
    total: dict[tuple[str, str], list[float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["h"]) != horizon:
                continue
            total.setdefault((r["period"], r["method"]), []).append(abs(float(r["y_pred"]) - float(r["y_true"])))
    out: dict[str, dict[str, tuple[float, int]]] = {}
    for (period, method), errs in total.items():
        out.setdefault(period, {})[method] = (sum(errs) / len(errs), len(errs))
    return out